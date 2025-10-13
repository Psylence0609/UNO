"""
Training utilities and logger for UNO RL project.
"""

import os
import time
import json
import logging
from datetime import datetime
from collections import defaultdict, deque
import numpy as np
import matplotlib.pyplot as plt

class TrainingLogger:
    """Logger for training metrics and visualization."""
    
    def __init__(self, log_dir="logs", experiment_name=None):
        """
        Initialize the training logger.
        
        Args:
            log_dir (str): Directory to save logs
            experiment_name (str): Name of the experiment
        """
        if experiment_name is None:
            experiment_name = f"uno_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_name = experiment_name
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(f"UNO_RL_{experiment_name}")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(os.path.join(self.log_dir, "training.log"))
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        # Metrics storage
        self.metrics = defaultdict(list)
        self.episode_metrics = defaultdict(list)
        self.evaluation_metrics = defaultdict(list)
        
        # Training info
        self.start_time = None
        self.episode_start_time = None
        
        self.logger.info(f"Training logger initialized for experiment: {experiment_name}")
        self.logger.info(f"Log directory: {self.log_dir}")
    
    def start_training(self, config):
        """Log the start of training."""
        self.start_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info("STARTING TRAINING")
        self.logger.info("=" * 60)
        
        # Save config
        config_path = os.path.join(self.log_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Configuration saved to: {config_path}")
        self.logger.info(f"Training episodes: {config.get('training', {}).get('episodes', 'N/A')}")
        self.logger.info(f"Learning rate: {config.get('training', {}).get('learning_rate', 'N/A')}")
        self.logger.info(f"Batch size: {config.get('training', {}).get('batch_size', 'N/A')}")
    
    def start_episode(self, episode):
        """Log the start of an episode."""
        self.episode_start_time = time.time()
        if episode % 1000 == 0:
            self.logger.info(f"Starting episode {episode}")
    
    def end_episode(self, episode, episode_data):
        """
        Log the end of an episode.
        
        Args:
            episode (int): Episode number
            episode_data (dict): Episode data (reward, length, etc.)
        """
        if self.episode_start_time:
            episode_time = time.time() - self.episode_start_time
            episode_data['episode_time'] = episode_time
        
        # Store episode metrics
        for key, value in episode_data.items():
            self.episode_metrics[key].append(value)
        
        # Log every 100 episodes
        if episode % 100 == 0:
            recent_rewards = self.episode_metrics['reward'][-100:]
            recent_lengths = self.episode_metrics['length'][-100:]
            
            self.logger.info(
                f"Episode {episode:6d} | "
                f"Avg Reward: {np.mean(recent_rewards):6.2f} | "
                f"Avg Length: {np.mean(recent_lengths):6.1f} | "
                f"Epsilon: {episode_data.get('epsilon', 0):.3f}"
            )
    
    def log_training_step(self, step_data):
        """
        Log training step data.
        
        Args:
            step_data (dict): Training step data (loss, etc.)
        """
        for key, value in step_data.items():
            self.metrics[key].append(value)
    
    def log_evaluation(self, episode, eval_data):
        """
        Log evaluation results.
        
        Args:
            episode (int): Episode number
            eval_data (dict): Evaluation data
        """
        eval_data['episode'] = episode
        
        for key, value in eval_data.items():
            self.evaluation_metrics[key].append(value)
        
        self.logger.info("=" * 50)
        self.logger.info(f"EVALUATION AT EPISODE {episode}")
        self.logger.info("-" * 50)
        
        for key, value in eval_data.items():
            if key != 'episode':
                if isinstance(value, (list, np.ndarray)):
                    self.logger.info(f"{key}: {value}")
                else:
                    self.logger.info(f"{key}: {value:.4f}")
        
        self.logger.info("=" * 50)
    
    def save_checkpoint(self, episode, model_path):
        """Log model checkpoint saving."""
        self.logger.info(f"Model checkpoint saved at episode {episode}: {model_path}")
    
    def plot_training_metrics(self, save_path=None, show_plot=True):
        """
        Plot training metrics.
        
        Args:
            save_path (str): Path to save the plot
            show_plot (bool): Whether to display the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Training Metrics - {self.experiment_name}', fontsize=16)
        
        # Episode rewards
        if 'reward' in self.episode_metrics:
            rewards = self.episode_metrics['reward']
            episodes = range(len(rewards))
            
            axes[0, 0].plot(episodes, rewards, alpha=0.6, linewidth=0.5)
            # Moving average
            if len(rewards) > 100:
                window_size = min(100, len(rewards) // 10)
                moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                axes[0, 0].plot(range(window_size-1, len(rewards)), moving_avg, 'r-', linewidth=2, label=f'MA({window_size})')
            
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Episode lengths
        if 'length' in self.episode_metrics:
            lengths = self.episode_metrics['length']
            episodes = range(len(lengths))
            
            axes[0, 1].plot(episodes, lengths, alpha=0.6, linewidth=0.5)
            # Moving average
            if len(lengths) > 100:
                window_size = min(100, len(lengths) // 10)
                moving_avg = np.convolve(lengths, np.ones(window_size)/window_size, mode='valid')
                axes[0, 1].plot(range(window_size-1, len(lengths)), moving_avg, 'g-', linewidth=2, label=f'MA({window_size})')
            
            axes[0, 1].set_title('Episode Lengths')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Length')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Training loss
        if 'loss' in self.metrics:
            losses = self.metrics['loss']
            steps = range(len(losses))
            
            axes[0, 2].plot(steps, losses, alpha=0.6, linewidth=0.5)
            # Moving average
            if len(losses) > 100:
                window_size = min(100, len(losses) // 10)
                moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
                axes[0, 2].plot(range(window_size-1, len(losses)), moving_avg, 'orange', linewidth=2, label=f'MA({window_size})')
            
            axes[0, 2].set_title('Training Loss')
            axes[0, 2].set_xlabel('Training Step')
            axes[0, 2].set_ylabel('Loss')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Epsilon decay
        if 'epsilon' in self.episode_metrics:
            epsilons = self.episode_metrics['epsilon']
            episodes = range(len(epsilons))
            
            axes[1, 0].plot(episodes, epsilons, 'purple', linewidth=2)
            axes[1, 0].set_title('Epsilon Decay')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Epsilon')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Evaluation win rates
        if 'win_rate' in self.evaluation_metrics:
            eval_episodes = self.evaluation_metrics['episode']
            win_rates = self.evaluation_metrics['win_rate']
            
            axes[1, 1].plot(eval_episodes, win_rates, 'ro-', linewidth=2, markersize=6)
            axes[1, 1].set_title('Evaluation Win Rate')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Win Rate')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].grid(True, alpha=0.3)
        
        # Q-values (if available)
        if 'avg_q_value' in self.metrics:
            q_values = self.metrics['avg_q_value']
            steps = range(len(q_values))
            
            axes[1, 2].plot(steps, q_values, 'teal', linewidth=1)
            axes[1, 2].set_title('Average Q-Values')
            axes[1, 2].set_xlabel('Training Step')
            axes[1, 2].set_ylabel('Avg Q-Value')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training plots saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def save_metrics(self):
        """Save all metrics to files."""
        # Save episode metrics
        episode_metrics_path = os.path.join(self.log_dir, "episode_metrics.json")
        with open(episode_metrics_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_metrics = {}
            for key, values in self.episode_metrics.items():
                if isinstance(values, np.ndarray):
                    serializable_metrics[key] = values.tolist()
                else:
                    serializable_metrics[key] = values
            json.dump(serializable_metrics, f, indent=2)
        
        # Save training metrics
        training_metrics_path = os.path.join(self.log_dir, "training_metrics.json")
        with open(training_metrics_path, 'w') as f:
            serializable_metrics = {}
            for key, values in self.metrics.items():
                if isinstance(values, np.ndarray):
                    serializable_metrics[key] = values.tolist()
                else:
                    serializable_metrics[key] = values
            json.dump(serializable_metrics, f, indent=2)
        
        # Save evaluation metrics
        eval_metrics_path = os.path.join(self.log_dir, "evaluation_metrics.json")
        with open(eval_metrics_path, 'w') as f:
            serializable_metrics = {}
            for key, values in self.evaluation_metrics.items():
                if isinstance(values, np.ndarray):
                    serializable_metrics[key] = values.tolist()
                else:
                    serializable_metrics[key] = values
            json.dump(serializable_metrics, f, indent=2)
        
        self.logger.info(f"Metrics saved to: {self.log_dir}")
    
    def end_training(self):
        """Log the end of training."""
        if self.start_time:
            total_time = time.time() - self.start_time
            
            self.logger.info("=" * 60)
            self.logger.info("TRAINING COMPLETED")
            self.logger.info("=" * 60)
            self.logger.info(f"Total training time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
            self.logger.info(f"Total episodes: {len(self.episode_metrics.get('reward', []))}")
            
            if 'reward' in self.episode_metrics:
                rewards = self.episode_metrics['reward']
                self.logger.info(f"Final 100-episode average reward: {np.mean(rewards[-100:]):.3f}")
            
            # Save final metrics and plots
            self.save_metrics()
            plot_path = os.path.join(self.log_dir, "training_plots.png")
            self.plot_training_metrics(save_path=plot_path, show_plot=False)
            
            self.logger.info(f"All logs and plots saved to: {self.log_dir}")


def calculate_reward(env, player_id, action, next_state, payoffs=None):
    """
    Calculate reward for a given action.
    
    Args:
        env: UNO environment
        player_id (int): Current player ID
        action (int): Action taken
        next_state: Resulting state
        payoffs: Final payoffs (if game ended)
        
    Returns:
        float: Calculated reward
    """
    # Basic sparse reward function
    if env.is_over():
        # Game ended - give win/loss reward
        if payoffs is not None:
            return 1.0 if payoffs[player_id] > 0 else -1.0
        else:
            # Fallback: check if this player won
            final_payoffs = env.get_payoffs()
            return 1.0 if final_payoffs[player_id] > 0 else -1.0
    else:
        # Game continues - no intermediate reward in basic version
        return 0.0


def get_reward_shaping(env, state, action, next_state, player_id):
    """
    Advanced reward shaping (placeholder for future MCTS implementation).
    
    Args:
        env: UNO environment
        state: Current state
        action: Action taken
        next_state: Resulting state
        player_id: Current player ID
        
    Returns:
        float: Shaped reward
    """
    # For now, return basic reward
    # TODO: Implement MCTS-based reward shaping
    return calculate_reward(env, player_id, action, next_state)