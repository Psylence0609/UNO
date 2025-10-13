"""
Training script for DQN agent with MCTS reward shaping on UNO.
Fair comparison setup with DMC+MCTS using same architecture and rewards.
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
from src.agents.dqn_agent_new import DQNAgent
from src.agents.random_agent import RandomAgent
from src.mcts.proper_mcts import ProperMCTS, calculate_mcts_reward
from src.training.logger import TrainingLogger
from src.evaluation.evaluator import Evaluator

class DQNTrainer:
    """Trainer for DQN agent with MCTS reward shaping."""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize the trainer."""
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
        
        # Setup DQN agent
        self.dqn_agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.env.num_actions,
            config=self.config
        )
        
        # Setup MCTS reward shaper
        self.mcts_shaper = ProperMCTS(
            env=self.env,
            config=self.config.get('mcts', {})
        )
        
        # Setup opponent
        self.random_opponent = RandomAgent(self.env.num_actions)
        
        # Setup logger
        self.logger = TrainingLogger(
            log_dir=self.config['paths']['logs'],
            experiment_name=f"dqn_mcts_vs_random_{self.config['environment']['seed']}"
        )
        
        # Setup evaluator
        self.evaluator = Evaluator(self.env)
        
        # Training counters
        self.episode = 0
        self.total_steps = 0
        
    def train_episode(self):
        """Train for one episode."""
        state, player_id = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        agents = [self.dqn_agent, self.random_opponent]
        
        while not self.env.is_over():
            # Get current agent
            current_agent = agents[player_id]
            
            # Store previous state for learning (only for DQN agent)
            if player_id == 0:  # DQN agent's turn
                prev_state = state
            
            # Take action
            action = current_agent.use_raw(state)
            next_state, next_player_id = self.env.step(action)
            episode_length += 1
            self.total_steps += 1
            
            # Calculate reward and store experience (only for DQN agent)
            if player_id == 0:  # DQN agent's turn
                # Base sparse reward
                base_reward = 0.0
                if self.env.is_over():
                    payoffs = self.env.get_payoffs()
                    base_reward = 1.0 if payoffs[0] > 0 else -1.0
                
                # MCTS-shaped reward (same as DMC for fair comparison)
                shaped_reward = calculate_mcts_reward(
                    self.env, prev_state, action, next_state, player_id, 
                    base_reward, self.mcts_shaper
                )
                
                episode_reward += shaped_reward
                
                # Store experience in DQN replay buffer
                self.dqn_agent.step(
                    prev_state, action, shaped_reward, next_state, self.env.is_over()
                )
            
            # Update state and player
            state = next_state
            player_id = next_player_id
        
        # Episode data
        episode_data = {
            'reward': episode_reward,
            'length': episode_length,
            'epsilon': self.dqn_agent.epsilon,
            'total_steps': self.total_steps
        }
        
        return episode_data
    
    def evaluate_agent(self, num_games=1000):
        """Evaluate the DQN agent against random opponent."""
        # Set agent to evaluation mode
        self.dqn_agent.eval()
        original_epsilon = self.dqn_agent.epsilon
        self.dqn_agent.epsilon = 0.0  # No exploration during evaluation
        
        # Evaluate against random agent
        agents = [self.dqn_agent, self.random_opponent]
        results = self.evaluator.evaluate_agents(agents, num_games, verbose=False)
        
        eval_results = {
            'win_rate': results['win_rates'][0],
            'avg_game_length': results['avg_game_length'],
            'wins': results['wins'][0],
            'losses': results['wins'][1]
        }
        
        # Restore training mode
        self.dqn_agent.train()
        self.dqn_agent.epsilon = original_epsilon
        
        return eval_results
    
    def save_model(self, filepath=None):
        """Save the trained model."""
        if filepath is None:
            os.makedirs(self.config['paths']['models'], exist_ok=True)
            filepath = os.path.join(
                self.config['paths']['models'],
                f"dqn_mcts_episode_{self.episode}.pth"
            )
        
        self.dqn_agent.save(filepath)
        self.logger.save_checkpoint(self.episode, filepath)
        return filepath
    
    def train(self):
        """Main training loop."""
        self.logger.start_training(self.config)
        
        num_episodes = self.config['training']['episodes']
        eval_freq = self.config['evaluation']['eval_freq']
        save_freq = self.config['logging']['save_freq']
        
        try:
            for episode in tqdm(range(num_episodes), desc="Training DQN+MCTS"):
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
                    eval_log_data = {
                        'random_win_rate': eval_data['win_rate'],
                        'random_avg_length': eval_data['avg_game_length']
                    }
                    
                    self.logger.log_evaluation(episode, eval_log_data)
                
                # Save model periodically
                if episode % save_freq == 0 and episode > 0:
                    self.save_model()
                
                # Early stopping if excellent performance achieved
                if eval_data is not None and eval_data['win_rate'] >= 0.70:  # 70% target
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
            final_eval_data = {
                'random_win_rate': final_eval['win_rate'],
                'random_avg_length': final_eval['avg_game_length']
            }
            
            self.logger.log_evaluation(self.episode, final_eval_data)
            
            # Save final model
            final_model_path = self.save_model(
                os.path.join(self.config['paths']['models'], "dqn_mcts_final.pth")
            )
            
            self.logger.end_training()
            
            return final_eval, final_model_path


def main():
    """Main function to start DQN+MCTS training."""
    print("🚀 UNO DQN + MCTS Training")
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
    trainer = DQNTrainer()
    
    print(f"Training DQN agent with MCTS reward shaping...")
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
    print(f"Final win rate vs Random: {final_eval['win_rate']:.1%}")
    print(f"Final model saved to: {model_path}")
    print(f"Logs saved to: {trainer.logger.log_dir}")


if __name__ == "__main__":
    main()