"""
Training script for DMC agent with opponent modeling on UNO.
"""

import os
import sys
import yaml
import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.uno_env import UnoEnvironment
from src.agents.dmc_agent_with_opponent import DMCAgentWithOpponentModeling
from src.agents.random_agent import RandomAgent
from src.mcts.proper_mcts import ProperMCTS, calculate_mcts_reward
from src.training.logger import TrainingLogger
from src.evaluation.evaluator import Evaluator


class DMCTrainerWithOpponent:
    """Trainer for DMC agent with opponent modeling and MCTS reward shaping."""
    
    def __init__(self, config_path="config.yaml"):
        """
        Initialize the trainer.
        
        Args:
            config_path (str): Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup environment
        self.env = UnoEnvironment(seed=self.config['environment']['seed'])
        
        # Calculate state size
        sample_state, _ = self.env.reset()
        features = []
        if 'obs' in sample_state:
            features.extend(sample_state['obs'].flatten())
        features.extend(np.zeros(self.env.num_actions))
        self.state_size = len(features)
        
        # Setup DMC agent with opponent modeling
        self.dmc_agent = DMCAgentWithOpponentModeling(
            state_size=self.state_size,
            action_size=self.env.num_actions,
            config=self.config
        )
        
        # Reset opponent tracking
        self.dmc_agent.reset_opponent_tracking(num_players=self.env.num_players)
        
        # Setup MCTS reward shaper (optional, can be disabled)
        mcts_config = self.config.get('mcts', {})
        if mcts_config.get('enabled', True):
            self.mcts_shaper = ProperMCTS(
                env=self.env,
                config=mcts_config
            )
            self.use_mcts_reward = True
        else:
            self.mcts_shaper = None
            self.use_mcts_reward = False
        
        # Setup opponents
        self.random_opponent = RandomAgent(self.env.num_actions)
        
        # Setup logger
        self.logger = TrainingLogger(
            log_dir=self.config['paths']['logs'],
            experiment_name=f"dmc_opponent_modeling_{self.config['environment']['seed']}"
        )
        
        # Setup evaluator
        self.evaluator = Evaluator(self.env)
        
        # Training counters
        self.episode = 0
        self.total_steps = 0
        
    def _process_state(self, state):
        """Process state for the agent."""
        return self.dmc_agent._process_state(state)
    
    def train_episode(self):
        """Train for one episode with opponent tracking."""
        state, player_id = self.env.reset()
        
        # Reset opponent tracking for new episode
        self.dmc_agent.reset_opponent_tracking(num_players=self.env.num_players)
        
        episode_reward = 0
        episode_length = 0
        
        agents = [self.dmc_agent, self.random_opponent]
        
        while not self.env.is_over():
            # Get current agent
            current_agent = agents[player_id]
            
            # Extract opponent features before action (for our agent)
            if player_id == 0:  # DMC agent's turn
                opponent_features = self.dmc_agent.extract_opponent_features(state)
                prev_state = state
            
            # Take action
            action = current_agent.use_raw(state)
            next_state, next_player_id = self.env.step(action)
            episode_length += 1
            self.total_steps += 1
            
            # Update opponent history after opponent's action
            if player_id == 1:  # Opponent's turn
                # Get opponent hand size if available
                opponent_hand_size = None
                if 'raw_obs' in next_state and 'num_cards' in next_state['raw_obs']:
                    opponent_hand_size = next_state['raw_obs']['num_cards'].get(1)
                
                # Update opponent history
                self.dmc_agent.update_opponent_history(
                    action=action,
                    opponent_id=1,
                    hand_size=opponent_hand_size
                )
            
            # Calculate reward and store experience (only for DMC agent)
            if player_id == 0:  # DMC agent's turn
                # Base sparse reward
                base_reward = 0.0
                if self.env.is_over():
                    payoffs = self.env.get_payoffs()
                    base_reward = 1.0 if payoffs[0] > 0 else -1.0
                
                # MCTS-shaped reward (if enabled)
                if self.use_mcts_reward:
                    shaped_reward = calculate_mcts_reward(
                        self.env, state, action, next_state, player_id, 
                        base_reward, self.mcts_shaper
                    )
                else:
                    shaped_reward = base_reward
                
                episode_reward += shaped_reward
                
                # Extract opponent features for next state
                next_opponent_features = self.dmc_agent.extract_opponent_features(next_state) if not self.env.is_over() else opponent_features
                
                # Store transition with opponent features
                self.dmc_agent.store_transition(
                    state=prev_state,
                    action=action,
                    reward=shaped_reward,
                    next_state=next_state if not self.env.is_over() else prev_state,
                    done=self.env.is_over(),
                    opponent_features=opponent_features
                )
            
            # Update state and player
            state = next_state
            player_id = next_player_id
        
        # Learn from episode
        learning_stats = self.dmc_agent.learn_episode()
        
        # Log episode data
        episode_data = {
            'reward': episode_reward,
            'length': episode_length,
            'epsilon': self.dmc_agent.epsilon,
            'total_steps': self.total_steps
        }
        
        # Add learning statistics
        episode_data.update(learning_stats)
        
        return episode_data
    
    def evaluate_agent(self, num_games=1000):
        """
        Evaluate the DMC agent with opponent modeling against different opponents.
        
        Args:
            num_games (int): Number of games to evaluate
            
        Returns:
            dict: Evaluation results
        """
        results = {}
        
        # Set agent to evaluation mode
        self.dmc_agent.eval()
        original_epsilon = self.dmc_agent.epsilon
        self.dmc_agent.epsilon = 0.0  # No exploration during evaluation
        
        # Evaluate against random agent
        agents = [self.dmc_agent, self.random_opponent]
        random_results = self.evaluator.evaluate_agents(agents, num_games, verbose=False)
        
        results['vs_random'] = {
            'win_rate': random_results['win_rates'][0],
            'avg_game_length': random_results['avg_game_length'],
            'wins': random_results['wins'][0],
            'losses': random_results['wins'][1]
        }
        
        # Restore training mode
        self.dmc_agent.train()
        self.dmc_agent.epsilon = original_epsilon
        
        return results
    
    def save_model(self, filepath=None):
        """Save the trained model."""
        if filepath is None:
            os.makedirs(os.path.join(self.config['paths']['models'], "custom"), exist_ok=True)
            filepath = os.path.join(
                self.config['paths']['models'],
                "custom",
                f"dmc_opponent_episode_{self.episode}.pth"
            )
        
        self.dmc_agent.save(filepath)
        self.logger.save_checkpoint(self.episode, filepath)
        return filepath
    
    def train(self):
        """Main training loop."""
        self.logger.start_training(self.config)
        
        num_episodes = self.config['training']['episodes']
        eval_freq = self.config['evaluation']['eval_freq']
        save_freq = self.config['logging']['save_freq']
        
        eval_data = None
        
        try:
            for episode in tqdm(range(num_episodes), desc="Training DMC with Opponent Modeling"):
                self.episode = episode
                
                # Train one episode
                self.logger.start_episode(episode)
                episode_data = self.train_episode()
                self.logger.end_episode(episode, episode_data)
                
                # Evaluate periodically
                if episode % eval_freq == 0 and episode > 0:
                    eval_data = self.evaluate_agent(
                        num_games=self.config['evaluation']['eval_episodes']
                    )
                    
                    eval_data_to_log = {
                        'random_win_rate': eval_data['vs_random']['win_rate'],
                        'random_avg_length': eval_data['vs_random']['avg_game_length']
                    }
                    
                    self.logger.log_evaluation(episode, eval_data_to_log)
                    
                    # Early stopping if excellent performance achieved
                    if eval_data['vs_random']['win_rate'] >= 0.70:  # 70% target
                        self.logger.logger.info(f"Excellent performance achieved! Stopping training.")
                        break
                
                # Save model periodically
                if episode % save_freq == 0 and episode > 0:
                    self.save_model()
        
        except KeyboardInterrupt:
            self.logger.logger.info("Training interrupted by user")
        
        finally:
            # Final evaluation
            self.logger.logger.info("Performing final evaluation...")
            final_eval = self.evaluate_agent(
                num_games=self.config['evaluation']['eval_episodes']
            )
            
            # Log final results
            final_eval_data = {
                'random_win_rate': final_eval['vs_random']['win_rate'],
                'random_avg_length': final_eval['vs_random']['avg_game_length']
            }
            
            self.logger.log_evaluation(self.episode, final_eval_data)
            
            # Save final model
            os.makedirs(os.path.join(self.config['paths']['models'], "custom"), exist_ok=True)
            final_model_path = self.save_model(
                os.path.join(self.config['paths']['models'], "custom", "dmc_opponent_modeling_final.pth")
            )
            
            self.logger.end_training()
            
            return final_eval, final_model_path


def main():
    """Main function to start training."""
    print(" UNO DMC + Opponent Modeling Training")
    print("=" * 50)
    
    # Check device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Create trainer and start training
    trainer = DMCTrainerWithOpponent()
    
    print(f"Training DMC agent with opponent modeling...")
    print(f"State size: {trainer.state_size}")
    print(f"Action size: {trainer.env.num_actions}")
    print(f"Opponent feature size: {trainer.dmc_agent.opponent_feature_size}")
    print(f"Strategy dimension: {trainer.dmc_agent.strategy_dim}")
    print(f"Episodes: {trainer.config['training']['episodes']}")
    if trainer.use_mcts_reward:
        print(f"MCTS simulations: {trainer.mcts_shaper.num_simulations}")
    else:
        print(f"MCTS reward shaping: Disabled")
    print()
    
    # Start training
    final_eval, model_path = trainer.train()
    
    # Print final results
    print("\n TRAINING COMPLETED!")
    print("=" * 50)
    print(f"Final win rate vs Random: {final_eval['vs_random']['win_rate']:.1%}")
    print(f"Final model saved to: {model_path}")
    print(f"Logs saved to: {trainer.logger.log_dir}")


if __name__ == "__main__":
    main()

