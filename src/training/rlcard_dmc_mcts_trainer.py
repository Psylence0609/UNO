"""
RLCard DMC Trainer with MCTS Reward Shaping
Attempts to integrate RLCard's built-in DMCTrainer with MCTS reward shaping.
Note: This is experimental as RLCard's trainer has its own training loop.
"""

import os
import sys
import rlcard
import torch
import numpy as np
import yaml

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from rlcard.agents.dmc_agent.trainer import DMCTrainer
from src.environments.uno_env import UnoEnvironment
from src.mcts.proper_mcts import ProperMCTS, calculate_mcts_reward


class MCTSShapedUnoEnv:
    """
    Wrapper around RLCard UNO environment that applies MCTS reward shaping.
    This wrapper intercepts rewards and adds MCTS-based intermediate rewards.
    """
    
    def __init__(self, base_env, mcts_shaper, config):
        """
        Initialize MCTS-shaped environment wrapper.
        
        Args:
            base_env: RLCard UNO environment
            mcts_shaper: MCTS reward shaper instance
            config: Configuration dictionary
        """
        self.env = base_env
        self.mcts_shaper = mcts_shaper
        self.config = config
        self.last_state = None
        self.last_action = None
        self.last_player_id = None
        
        # Environment properties (delegate to base env)
        self.num_actions = self.env.num_actions
        self.num_players = self.env.num_players
        self.state_shape = self.env.state_shape
        
        # RLCard's trainer expects action_shape
        # For UNO, action_shape is typically [None] (one-hot encoding for all players)
        # RLCard will convert this to [[num_actions], [num_actions]] for 2 players
        if hasattr(self.env, 'action_shape'):
            self.action_shape = self.env.action_shape
        else:
            # Default: one-hot encoding (None means one-hot)
            # RLCard expects a list where None indicates one-hot encoding
            # It will expand this to per-player action shapes
            self.action_shape = [None]
    
    def reset(self):
        """Reset environment and return initial state."""
        state, player_id = self.env.reset()
        self.last_state = state
        self.last_player_id = player_id
        self.last_action = None
        return state, player_id
    
    def step(self, action, raw_action=False):
        """
        Take a step and apply MCTS reward shaping.
        
        Args:
            action: Action to take
            raw_action: Whether action is in raw format
            
        Returns:
            tuple: (next_state, next_player_id)
        """
        # Store previous state for reward shaping
        prev_state = self.last_state
        prev_player_id = self.last_player_id
        
        # Take step in base environment
        next_state, next_player_id = self.env.step(action, raw_action=raw_action)
        
        # Apply MCTS reward shaping if we have previous state
        if prev_state is not None and self.last_action is not None:
            # Calculate base reward (sparse - only at game end)
            base_reward = 0.0
            if self.env.is_over():
                payoffs = self.env.get_payoffs()
                base_reward = 1.0 if payoffs[prev_player_id] > 0 else -1.0
            
            # Calculate MCTS-shaped reward
            # Note: This modifies the environment's internal reward, which RLCard's
            # trainer may not use directly. This is a limitation of the integration.
            try:
                shaped_reward = calculate_mcts_reward(
                    self.env, prev_state, self.last_action, next_state,
                    prev_player_id, base_reward, self.mcts_shaper
                )
                # Store shaped reward (RLCard's trainer may not use this directly)
                self._last_shaped_reward = shaped_reward
            except Exception as e:
                print(f"  MCTS reward calculation failed: {e}")
                self._last_shaped_reward = base_reward
        else:
            self._last_shaped_reward = 0.0
        
        # Update state tracking
        self.last_state = next_state
        self.last_player_id = next_player_id
        self.last_action = action
        
        return next_state, next_player_id
    
    def is_over(self):
        """Check if game is over."""
        return self.env.is_over()
    
    def get_payoffs(self):
        """Get final payoffs."""
        return self.env.get_payoffs()
    
    def get_perfect_information(self):
        """Get perfect information (if available)."""
        if hasattr(self.env, 'get_perfect_information'):
            return self.env.get_perfect_information()
        return None
    
    def seed(self, seed=None):
        """
        Set random seed for the environment.
        Required by RLCard's trainer for actor processes.
        
        Args:
            seed: Random seed value
            
        Returns:
            list: List of seeds used
        """
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
        else:
            # If base env doesn't have seed, try to set it via config
            # RLCard environments typically support seeding via reset
            return [seed] if seed is not None else None
    
    def set_agents(self, agents):
        """
        Set agents for the environment.
        Required by RLCard's trainer.
        
        Args:
            agents: List of agents
        """
        if hasattr(self.env, 'set_agents'):
            return self.env.set_agents(agents)
        else:
            # Store agents if needed
            self._agents = agents
    
    def run(self, is_training=False):
        """
        Run a complete game.
        Required by RLCard's trainer for actor processes.
        
        Args:
            is_training: Whether this is a training run
            
        Returns:
            tuple: (trajectories, payoffs)
        """
        if hasattr(self.env, 'run'):
            return self.env.run(is_training=is_training)
        else:
            # Fallback: manually run game using reset/step
            trajectories = [[] for _ in range(self.num_players)]
            state, player_id = self.reset()
            
            while not self.is_over():
                # Get current agent
                if hasattr(self, '_agents') and self._agents:
                    agent = self._agents[player_id]
                    action = agent.step(state)
                else:
                    # Fallback to random if no agents set
                    legal_actions = list(state['legal_actions'].keys()) if 'legal_actions' in state else list(range(self.num_actions))
                    action = np.random.choice(legal_actions)
                
                # Store trajectory
                trajectories[player_id].append(state)
                trajectories[player_id].append(action)
                
                # Take step
                state, player_id = self.step(action)
            
            # Get final payoffs
            payoffs = self.get_payoffs()
            return trajectories, payoffs
    
    def get_action_feature(self, action):
        """
        Get action feature representation.
        Required by RLCard's trainer.
        
        Args:
            action: Action value
            
        Returns:
            np.array: Action feature vector
        """
        if hasattr(self.env, 'get_action_feature'):
            return self.env.get_action_feature(action)
        else:
            # Fallback: one-hot encoding
            action_feature = np.zeros(self.num_actions)
            if isinstance(action, (int, np.integer)):
                if 0 <= action < self.num_actions:
                    action_feature[action] = 1.0
            return action_feature


def main():
    """Main function to start RLCard DMC training with MCTS reward shaping."""
    print(" RLCard DMC + MCTS Training (Experimental)")
    print("=" * 50)
    print("  Note: This is experimental. RLCard's DMCTrainer has its own")
    print("   training loop, so MCTS integration may be limited.")
    print("=" * 50)
    print()
    
    # Load configuration
    config_path = os.path.join(project_root, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize base UNO environment
    print(" Initializing UNO environment...")
    base_env = rlcard.make('uno', config={'seed': config['environment']['seed']})
    print(f" Base environment created: {base_env.num_actions} actions")
    
    # Initialize MCTS shaper
    print(" Initializing MCTS reward shaper...")
    # Create a wrapper environment for MCTS (needs UnoEnvironment interface)
    uno_env_wrapper = UnoEnvironment(seed=config['environment']['seed'])
    mcts_shaper = ProperMCTS(
        env=uno_env_wrapper,
        config=config.get('mcts', {})
    )
    print(f" MCTS shaper created: {mcts_shaper.num_simulations} simulations")
    
    # Wrap environment with MCTS reward shaping
    print(" Wrapping environment with MCTS reward shaping...")
    env = MCTSShapedUnoEnv(base_env, mcts_shaper, config)
    print(" MCTS-shaped environment created")
    print()
    
    # Check device availability
    # Note: RLCard's DMCTrainer only supports CUDA or CPU, not MPS
    if torch.cuda.is_available():
        device = "0"
        device_for_rlcard = "0"
        print(f" Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = ""
        device_for_rlcard = ""  # Fall back to CPU - RLCard doesn't support MPS
        print(f"  MPS (Apple Silicon) detected but RLCard's DMCTrainer doesn't support it")
        print(f"   Falling back to CPU (training will be slower)")
    else:
        device = ""
        device_for_rlcard = ""
        print(f"  Using CPU (training will be slower)")
    
    print(f"Device for RLCard: {device_for_rlcard if device_for_rlcard else 'CPU'}")
    print()
    
    # Training configuration
    TOTAL_FRAMES = 100_000_000  # 100M frames
    SAVE_INTERVAL = 30  # minutes
    
    print(" Training Configuration:")
    print(f"   Total Frames: {TOTAL_FRAMES:,}")
    print(f"   MCTS Simulations: {mcts_shaper.num_simulations}")
    print(f"   MCTS Max Depth: {config['mcts']['max_depth']}")
    print(f"   Save Interval: {SAVE_INTERVAL} minutes")
    print(f"   Batch Size: 32")
    print(f"   Learning Rate: 0.0001")
    print()
    
    # Create RLCard's DMCTrainer with MCTS-shaped environment
    print(" Creating RLCard DMCTrainer with MCTS...")
    trainer = DMCTrainer(
        env=env,  # Use MCTS-shaped environment
        cuda=device_for_rlcard,
        is_pettingzoo_env=False,
        load_model=False,
        xpid='uno_rlcard_dmc_mcts',  # Experiment ID
        save_interval=SAVE_INTERVAL,
        num_actor_devices=1,
        num_actors=5,  # Reduced for MCTS overhead
        training_device=device_for_rlcard,
        savedir=os.path.join(project_root, 'experiments/rlcard_dmc_mcts_uno_100M'),
        total_frames=TOTAL_FRAMES,
        exp_epsilon=0.01,
        batch_size=32,
        unroll_length=100,
        num_buffers=50,
        num_threads=4,
        max_grad_norm=40,
        learning_rate=0.0001,
        alpha=0.99,
        momentum=0,
        epsilon=1e-05,
    )
    
    print(" Trainer created successfully")
    print()
    print("  Important Notes:")
    print("   - MCTS reward shaping is applied via environment wrapper")
    print("   - RLCard's trainer may not directly use shaped rewards")
    print("   - This is experimental and may require custom modifications")
    print("   - Consider using custom DMCTrainer for full MCTS integration")
    print()
    print(" Starting training...")
    print("=" * 50)
    
    # Start training
    try:
        trainer.start()
    except KeyboardInterrupt:
        print("\n  Training interrupted by user")
        print(" Checkpoint should be saved in experiments/rlcard_dmc_mcts_uno/")
    except Exception as e:
        print(f"\n Training error: {e}")
        print(" Tip: RLCard's trainer may not be compatible with MCTS wrapper.")
        print("   Consider using the custom DMCTrainer for full MCTS support.")
        raise
    
    print("\n" + "=" * 50)
    print(" TRAINING COMPLETED!")
    print("=" * 50)
    print(f" Results saved to: {os.path.join(project_root, 'experiments/rlcard_dmc_mcts_uno')}")
    print("=" * 50)


if __name__ == "__main__":
    main()

