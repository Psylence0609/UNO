"""
Training script for DMC agent with MCTS reward shaping on UNO.
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
from src.agents.dmc_agent import DMCAgent
from src.agents.dqn_agent_new import DQNAgent
from src.agents.random_agent import RandomAgent
from src.mcts.proper_mcts import ProperMCTS, calculate_mcts_reward
from src.training.logger import TrainingLogger
from src.evaluation.evaluator import Evaluator

class DMCTrainer:
    """Trainer for DMC agent with MCTS reward shaping."""
    
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
        
        # Setup DMC agent
        self.dmc_agent = DMCAgent(
            state_size=self.state_size,
            action_size=self.env.num_actions,
            config=self.config
        )
        
        # Setup MCTS reward shaper
        self.mcts_shaper = ProperMCTS(
            env=self.env,
            config=self.config.get('mcts', {})
        )
        
        # Setup opponents
        self.random_opponent = RandomAgent(self.env.num_actions)
        
        # Setup baseline DQN (if available)
        try:
            self.dqn_agent = DQNAgent(
                state_size=self.state_size,
                action_size=self.env.num_actions,
                config=self.config
            )
            # Try to load pre-trained DQN
            if os.path.exists("models/dqn_mcts_final.pth"):
                self.dqn_agent.load("models/dqn_mcts_final.pth")
                self.dqn_agent.epsilon = 0.0  # No exploration for comparison
                print("✅ Loaded pre-trained DQN for comparison")
        except Exception as e:
            print(f"⚠️  Could not load DQN baseline: {e}")
            self.dqn_agent = None
        
        # Setup logger
        self.logger = TrainingLogger(
            log_dir=self.config['paths']['logs'],
            experiment_name=f"dmc_mcts_vs_random_{self.config['environment']['seed']}"
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
        """Train for one episode."""
        state, player_id = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        agents = [self.dmc_agent, self.random_opponent]
        
        while not self.env.is_over():
            # Get current agent
            current_agent = agents[player_id]
            
            # Store previous state for learning (only for DMC agent)
            if player_id == 0:  # DMC agent's turn
                prev_state = self._process_state(state)
            
            # Take action
            action = current_agent.use_raw(state)
            next_state, next_player_id = self.env.step(action)
            episode_length += 1
            self.total_steps += 1
            
            # Calculate reward and store experience (only for DMC agent)
            if player_id == 0:  # DMC agent's turn
                # Base sparse reward
                base_reward = 0.0
                if self.env.is_over():
                    payoffs = self.env.get_payoffs()
                    base_reward = 1.0 if payoffs[0] > 0 else -1.0
                
                # MCTS-shaped reward
                shaped_reward = calculate_mcts_reward(
                    self.env, state, action, next_state, player_id, 
                    base_reward, self.mcts_shaper
                )
                
                episode_reward += shaped_reward
                
                # Store transition for episode-based learning
                next_state_processed = self._process_state(next_state) if not self.env.is_over() else prev_state
                self.dmc_agent.store_transition(
                    prev_state, action, shaped_reward, next_state_processed, self.env.is_over()
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
        Evaluate the DMC agent against different opponents.
        
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
        
        # Evaluate against DQN if available
        if self.dqn_agent is not None:
            agents = [self.dmc_agent, self.dqn_agent]
            dqn_results = self.evaluator.evaluate_agents(agents, num_games, verbose=False)
            
            results['vs_dqn'] = {
                'win_rate': dqn_results['win_rates'][0],
                'avg_game_length': dqn_results['avg_game_length'],
                'wins': dqn_results['wins'][0],
                'losses': dqn_results['wins'][1]
            }
        
        # Restore training mode
        self.dmc_agent.train()
        self.dmc_agent.epsilon = original_epsilon
        
        return results
    
    def save_model(self, filepath=None):
        """Save the trained model."""
        if filepath is None:
            os.makedirs(self.config['paths']['models'], exist_ok=True)
            filepath = os.path.join(
                self.config['paths']['models'],
                f"dmc_episode_{self.episode}.pth"
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
        
        try:
            for episode in tqdm(range(num_episodes), desc="Training DMC+MCTS"):
                self.episode = episode
                
                # Train one episode
                self.logger.start_episode(episode)
                episode_data = self.train_episode()
                self.logger.end_episode(episode, episode_data)
                
                # Evaluate periodically
                eval_data = None
                if episode % eval_freq == 0 and episode > 0:
                    eval_data = self.evaluate_agent(
                        num_games=self.config['evaluation']['eval_episodes']
                    )
                    
                    # Log evaluation results
                    combined_eval_data = {
                        'random_win_rate': eval_data['vs_random']['win_rate'],
                        'random_avg_length': eval_data['vs_random']['avg_game_length']
                    }
                    
                    if 'vs_dqn' in eval_data:
                        combined_eval_data.update({
                            'dqn_win_rate': eval_data['vs_dqn']['win_rate'],
                            'dqn_avg_length': eval_data['vs_dqn']['avg_game_length']
                        })
                    
                    self.logger.log_evaluation(episode, combined_eval_data)
                
                # Save model periodically
                if episode % save_freq == 0 and episode > 0:
                    self.save_model()
                
                # Early stopping if excellent performance achieved
                if eval_data is not None and eval_data['vs_random']['win_rate'] >= 0.70:  # 70% target
                    self.logger.logger.info(f"Excellent performance achieved! Stopping training.")
                    break
        
        except KeyboardInterrupt:
            self.logger.logger.info("Training interrupted by user")
        
        finally:
            # Final evaluation
            self.logger.logger.info("Performing final evaluation...")
            final_eval = self.evaluate_agent(
                num_games=self.config['evaluation']['eval_episodes']
            )
            
            # Log final results
            combined_eval_data = {
                'random_win_rate': final_eval['vs_random']['win_rate'],
                'random_avg_length': final_eval['vs_random']['avg_game_length']
            }
            
            if 'vs_dqn' in final_eval:
                combined_eval_data.update({
                    'dqn_win_rate': final_eval['vs_dqn']['win_rate'],
                    'dqn_avg_length': final_eval['vs_dqn']['avg_game_length']
                })
            
            self.logger.log_evaluation(self.episode, combined_eval_data)
            
            # Save final model
            final_model_path = self.save_model(
                os.path.join(self.config['paths']['models'], "dmc_mcts_final.pth")
            )
            
            self.logger.end_training()
            
            return final_eval, final_model_path


def main():
    """Main function to start DMC+MCTS training."""
    print("🚀 UNO DMC + MCTS Training")
    print("=" * 50)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create trainer and start training
    trainer = DMCTrainer()
    trainer.dmc_agent.set_device(device)
    
    print(f"Training DMC agent with MCTS reward shaping...")
    print(f"State size: {trainer.state_size}")
    print(f"Action size: {trainer.env.num_actions}")
    print(f"Episodes: {trainer.config['training']['episodes']}")
    print(f"MCTS simulations: {trainer.mcts_shaper.num_simulations}")
    print()
    
    # Start training
    final_eval, model_path = trainer.train()
    
    # Print final results
    print("\n🏆 TRAINING COMPLETED!")
    print("=" * 50)
    print(f"Final win rate vs Random: {final_eval['vs_random']['win_rate']:.1%}")
    if 'vs_dqn' in final_eval:
        print(f"Final win rate vs DQN: {final_eval['vs_dqn']['win_rate']:.1%}")
    print(f"Final model saved to: {model_path}")
    print(f"Logs saved to: {trainer.logger.log_dir}")


if __name__ == "__main__":
    main()