"""
Training script for DQN agent on UNO.
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
from src.agents.dqn_agent import DQNAgent
from src.agents.random_agent import RandomAgent
from src.utils.replay_buffer import ReplayBuffer
from src.training.logger import TrainingLogger, calculate_reward
from src.evaluation.evaluator import Evaluator

class DQNTrainer:
    """Trainer for DQN agent."""
    
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
        
        # Calculate state size by processing a sample state
        sample_state, _ = self.env.reset()
        # Process state manually to get size (before agent is created)
        features = []
        if 'obs' in sample_state:
            features.extend(sample_state['obs'].flatten())
        # Add legal actions mask
        features.extend(np.zeros(self.env.num_actions))
        self.state_size = len(features)
        
        # Setup agent
        self.agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.env.num_actions,
            config=self.config
        )
        
        # Setup replay buffer
        self.replay_buffer = ReplayBuffer(
            buffer_size=self.config['training']['memory_size'],
            batch_size=self.config['training']['batch_size'],
            seed=self.config['environment']['seed']
        )
        
        # Setup opponent (random agent for now)
        self.opponent = RandomAgent(self.env.num_actions)
        
        # Setup logger
        self.logger = TrainingLogger(
            log_dir=self.config['paths']['logs'],
            experiment_name=f"dqn_vs_random_{self.config['environment']['seed']}"
        )
        
        # Setup evaluator
        self.evaluator = Evaluator(self.env)
        
        # Training counters
        self.episode = 0
        self.total_steps = 0
        
    def _process_state(self, state):
        """Process state for the agent."""
        return self.agent._process_state(state)
    
    def train_episode(self):
        """Train for one episode."""
        state, player_id = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_losses = []
        
        agents = [self.agent, self.opponent]
        
        while not self.env.is_over():
            # Get current agent
            current_agent = agents[player_id]
            
            # Store previous state for learning (only for DQN agent)
            if player_id == 0:  # DQN agent's turn
                prev_state = self._process_state(state)
            
            # Take action
            action = current_agent.use_raw(state)
            next_state, next_player_id = self.env.step(action)
            episode_length += 1
            self.total_steps += 1
            
            # Calculate reward and store experience (only for DQN agent)
            if player_id == 0:  # DQN agent's turn
                reward = calculate_reward(self.env, player_id, action, next_state)
                episode_reward += reward
                
                # Store experience
                next_state_processed = self._process_state(next_state) if not self.env.is_over() else prev_state
                self.replay_buffer.add(
                    prev_state, action, reward, next_state_processed, self.env.is_over()
                )
                
                # Learn from experience
                if self.replay_buffer.is_ready():
                    loss = self.agent.learn(self.replay_buffer)
                    if loss is not None:
                        episode_losses.append(loss)
            
            # Update state and player
            state = next_state
            player_id = next_player_id
        
        # Final reward for DQN agent
        if not self.env.is_over():  # Safety check
            payoffs = self.env.get_payoffs()
            final_reward = 1.0 if payoffs[0] > 0 else -1.0
            episode_reward += final_reward
        
        # Log episode data
        episode_data = {
            'reward': episode_reward,
            'length': episode_length,
            'epsilon': self.agent.epsilon,
            'total_steps': self.total_steps
        }
        
        if episode_losses:
            episode_data['avg_loss'] = np.mean(episode_losses)
            self.logger.log_training_step({'loss': np.mean(episode_losses)})
        
        return episode_data
    
    def evaluate_agent(self, num_games=1000):
        """
        Evaluate the agent against random opponent.
        
        Args:
            num_games (int): Number of games to evaluate
            
        Returns:
            dict: Evaluation results
        """
        # Set agent to evaluation mode
        self.agent.eval()
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0  # No exploration during evaluation
        
        # Run evaluation
        agents = [self.agent, self.opponent]
        results = self.evaluator.evaluate_agents(agents, num_games, verbose=False)
        
        # Restore training mode
        self.agent.train()
        self.agent.epsilon = original_epsilon
        
        # Calculate DQN-specific metrics
        dqn_win_rate = results['win_rates'][0]
        
        eval_data = {
            'win_rate': dqn_win_rate,
            'avg_game_length': results['avg_game_length'],
            'total_games': num_games,
            'wins': results['wins'][0],
            'losses': results['wins'][1]
        }
        
        return eval_data
    
    def save_model(self, filepath=None):
        """Save the trained model."""
        if filepath is None:
            os.makedirs(self.config['paths']['models'], exist_ok=True)
            filepath = os.path.join(
                self.config['paths']['models'],
                f"dqn_episode_{self.episode}.pth"
            )
        
        self.agent.save(filepath)
        self.logger.save_checkpoint(self.episode, filepath)
        return filepath
    
    def train(self):
        """Main training loop."""
        self.logger.start_training(self.config)
        
        num_episodes = self.config['training']['episodes']
        eval_freq = self.config['evaluation']['eval_freq']
        save_freq = self.config['logging']['save_freq']
        
        try:
            for episode in tqdm(range(num_episodes), desc="Training DQN"):
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
                    self.logger.log_evaluation(episode, eval_data)
                
                # Save model periodically
                if episode % save_freq == 0 and episode > 0:
                    self.save_model()
                
                # Early stopping if target win rate achieved
                if episode % eval_freq == 0 and episode > 0:
                    if eval_data['win_rate'] >= 0.6:  # 60% win rate target
                        self.logger.logger.info(f"Target win rate achieved! Stopping training.")
                        break
        
        except KeyboardInterrupt:
            self.logger.logger.info("Training interrupted by user")
        
        finally:
            # Final evaluation and save
            self.logger.logger.info("Performing final evaluation...")
            final_eval = self.evaluate_agent(
                num_games=self.config['evaluation']['eval_episodes']
            )
            self.logger.log_evaluation(self.episode, final_eval)
            
            # Save final model
            final_model_path = self.save_model(
                os.path.join(self.config['paths']['models'], "dqn_final.pth")
            )
            
            self.logger.end_training()
            
            return final_eval, final_model_path


def main():
    """Main function to start training."""
    print("🎮 UNO DQN Training")
    print("=" * 50)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create trainer and start training
    trainer = DQNTrainer()
    trainer.agent.set_device(device)
    
    print(f"Training DQN agent...")
    print(f"State size: {trainer.state_size}")
    print(f"Action size: {trainer.env.num_actions}")
    print(f"Episodes: {trainer.config['training']['episodes']}")
    print(f"Target win rate: 60%")
    print()
    
    # Start training
    final_eval, model_path = trainer.train()
    
    # Print final results
    print("\n🏆 TRAINING COMPLETED!")
    print("=" * 50)
    print(f"Final win rate: {final_eval['win_rate']:.1%}")
    print(f"Final model saved to: {model_path}")
    print(f"Logs saved to: {trainer.logger.log_dir}")


if __name__ == "__main__":
    main()