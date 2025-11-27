"""
Advanced Training Script for DMC Agent with Sophisticated Opponent Modeling.
Optimized for MPS GPU with early stopping and progress monitoring.
"""

import os
import sys
import yaml
import torch
import numpy as np
from tqdm import tqdm
import time
from collections import deque

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.environments.uno_env import UnoEnvironment
from src.agents.dmc_agent_advanced_opponent import AdvancedDMCAgentWithOpponent
from src.agents.dmc_agent import DMCAgent
from src.agents.random_agent import RandomAgent
from src.agents.heuristic_agent import HeuristicAgent
from src.mcts.proper_mcts import ProperMCTS
from src.training.logger import TrainingLogger
from src.evaluation.evaluator import Evaluator


def calculate_mcts_reward(env, state, action, next_state, player_id, base_reward, mcts_shaper):
    """
    Calculate MCTS-shaped reward.
    Helper function for reward shaping.
    """
    if env.is_over():
        return base_reward  # No shaping if game is over
    
    # Perform MCTS from the next state to estimate value
    try:
        mcts_shaper.set_root_state(env.env.get_raw_state(), next_state.get('legal_actions', {}))
        mcts_shaper.search()
        
        # Get the value of the next state from MCTS
        next_state_value = mcts_shaper.get_root_value()
        
        # Reward shaping: R_shaped = R_base + gamma * V(S')
        mcts_reward_weight = mcts_shaper.config.get('intermediate_reward_weight', 0.3)
        shaped_reward = base_reward + mcts_reward_weight * next_state_value
        
        return shaped_reward
    except Exception as e:
        # If MCTS fails, return base reward
        return base_reward


class EarlyStoppingMonitor:
    """
    Early stopping monitor that tracks training progress and decides when to stop.
    """
    
    def __init__(
        self,
        patience: int = 2000,
        min_improvement: float = 0.01,
        baseline_win_rate: float = 0.50,
        target_win_rate: float = 0.55,
        check_interval: int = 500,
        min_episodes: int = 5000,  # Increased minimum episodes
        smoothing_window: int = 5  # Number of recent evaluations to average
    ):
        """
        Initialize early stopping monitor.
        
        Args:
            patience: Number of episodes without improvement before stopping
            min_improvement: Minimum improvement to consider progress (absolute)
            baseline_win_rate: Baseline win rate to compare against
            target_win_rate: Target win rate (informational only - training continues beyond this to maximize performance)
            check_interval: Episodes between progress checks
            min_episodes: Minimum episodes before early stopping can trigger
            smoothing_window: Number of recent evaluations to average for smoother progress tracking
        """
        self.patience = patience
        self.min_improvement = min_improvement
        self.baseline_win_rate = baseline_win_rate
        self.target_win_rate = target_win_rate
        self.check_interval = check_interval
        self.min_episodes = min_episodes
        self.smoothing_window = smoothing_window
        
        # Track progress
        self.best_win_rate = baseline_win_rate
        self.best_smoothed_win_rate = baseline_win_rate  # Track smoothed win rate
        self.episodes_since_improvement = 0
        self.evaluations_since_improvement = 0  # Track evaluations for frame-based training
        self.win_rate_history = deque(maxlen=50)  # Last 50 evaluations for smoothing
        self.episode_history = deque(maxlen=50)
        
        # Statistics
        self.total_checks = 0
        self.improvements = 0
        
    def check_progress(
        self,
        current_episode: int,
        current_win_rate: float,
        verbose: bool = True
    ) -> dict:
        """
        Check training progress and decide if training should continue.
        
        Args:
            current_episode: Current episode number
            current_win_rate: Current win rate from evaluation
            verbose: Whether to print progress
            
        Returns:
            Dictionary with progress information and recommendation
        """
        self.total_checks += 1
        self.win_rate_history.append(current_win_rate)
        self.episode_history.append(current_episode)
        
        # Calculate smoothed win rate (moving average of recent evaluations)
        recent_win_rates = list(self.win_rate_history)
        if len(recent_win_rates) >= self.smoothing_window:
            smoothed_win_rate = np.mean(recent_win_rates[-self.smoothing_window:])
        else:
            # Use all available data if we don't have enough for smoothing window
            smoothed_win_rate = np.mean(recent_win_rates) if recent_win_rates else current_win_rate
        
        # Calculate statistics
        avg_recent = np.mean(recent_win_rates) if recent_win_rates else current_win_rate
        std_recent = np.std(recent_win_rates) if len(recent_win_rates) > 1 else 0.0
        
        # Check for improvement using SMOOTHED win rate to reduce noise
        improvement_raw = current_win_rate - self.best_win_rate
        improvement_smoothed = smoothed_win_rate - self.best_smoothed_win_rate
        
        # Consider it an improvement if smoothed rate improves OR if raw rate is significantly better
        # VERY LENIENT: Accept even small improvements
        is_improvement = (
            improvement_smoothed >= self.min_improvement * 0.1 or  # Very lenient for smoothed (10% of min_improvement)
            improvement_raw >= self.min_improvement * 0.5 or  # More lenient for raw (50% of min_improvement)
            current_win_rate > self.best_win_rate  # Any improvement counts
        )
        
        if is_improvement:
            # Update best if we see improvement
            if smoothed_win_rate > self.best_smoothed_win_rate:
                self.best_smoothed_win_rate = smoothed_win_rate
            if current_win_rate > self.best_win_rate:
                self.best_win_rate = current_win_rate
            self.episodes_since_improvement = 0
            self.evaluations_since_improvement = 0
            self.improvements += 1
        else:
            # Increment both counters (for compatibility)
            self.episodes_since_improvement += self.check_interval
            self.evaluations_since_improvement += 1  # Track evaluations for frame-based training
        
        # Determine if training is learning (more lenient - use smoothed rate)
        is_learning = (
            smoothed_win_rate > self.baseline_win_rate - 0.05 or  # Allow some variance below baseline
            avg_recent > self.baseline_win_rate - 0.03 or
            improvement_smoothed > -0.02  # Allow small fluctuations
        )
        
        # Determine if training should continue
        # VERY LENIENT: Only stop in extreme cases
        should_continue = True
        reason = ""
        
        # Use evaluation count for frame-based training, episode count as fallback
        evaluation_count = self.total_checks
        min_evaluations = max(20, self.min_episodes // self.check_interval)  # Convert episodes to evaluations
        
        if evaluation_count < min_evaluations:
            should_continue = True
            reason = f"Below minimum evaluations ({min_evaluations}) - too early to assess (evaluation {evaluation_count})"
        elif evaluation_count < min_evaluations * 2:
            # Very early in training - be very lenient
            should_continue = True
            reason = f"Early training phase (evaluation {evaluation_count}/{min_evaluations * 2}) - allowing exploration"
        elif not is_learning and evaluation_count > min_evaluations * 5 and smoothed_win_rate < self.baseline_win_rate - 0.10:
            # Only stop if clearly not learning AND well past minimum AND significantly below baseline (10% below)
            should_continue = False
            reason = f"Not learning (smoothed win rate {smoothed_win_rate:.1%} significantly below baseline {self.baseline_win_rate:.1%})"
        elif self.evaluations_since_improvement >= self.patience // self.check_interval and evaluation_count >= min_evaluations * 3:
            # Stop if no improvement for many evaluations AND we're well into training
            # patience // check_interval converts episode patience to evaluation patience
            should_continue = False
            reason = f"No improvement for {self.evaluations_since_improvement} evaluations (patience: {self.patience // self.check_interval}). Best smoothed: {self.best_smoothed_win_rate:.1%}, Best raw: {self.best_win_rate:.1%}"
        elif std_recent < 0.002 and evaluation_count > min_evaluations * 8 and smoothed_win_rate < self.baseline_win_rate - 0.05:
            # Very stable but low performance - only stop if well above minimum and clearly plateaued
            should_continue = False
            reason = f"Performance plateaued at low level (std: {std_recent:.4f}). Smoothed: {smoothed_win_rate:.1%}, Best: {self.best_win_rate:.1%}"
        else:
            # Continue training to maximize performance (even if above target)
            should_continue = True
            if smoothed_win_rate >= self.target_win_rate:
                reason = f"Above target ({self.target_win_rate:.1%}) - continuing to maximize. Smoothed: {smoothed_win_rate:.1%}, Best: {self.best_win_rate:.1%}"
            else:
                reason = f"Learning in progress. Smoothed: {smoothed_win_rate:.1%}, Best: {self.best_win_rate:.1%}, Current: {current_win_rate:.1%}"
        
        result = {
            'should_continue': should_continue,
            'reason': reason,
            'current_win_rate': current_win_rate,
            'smoothed_win_rate': smoothed_win_rate,
            'best_win_rate': self.best_win_rate,
            'best_smoothed_win_rate': self.best_smoothed_win_rate,
            'improvement_raw': improvement_raw,
            'improvement_smoothed': improvement_smoothed,
            'is_improvement': is_improvement,
            'episodes_since_improvement': self.episodes_since_improvement,
            'evaluations_since_improvement': self.evaluations_since_improvement,
            'avg_recent': avg_recent,
            'std_recent': std_recent,
            'is_learning': is_learning,
            'meets_target': smoothed_win_rate >= self.target_win_rate,
            'meets_baseline': smoothed_win_rate > self.baseline_win_rate
        }
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"📊 PROGRESS CHECK (Episode {current_episode})")
            print(f"{'='*80}")
            print(f"Current Win Rate: {current_win_rate:.1%} (raw)")
            print(f"Smoothed Win Rate: {smoothed_win_rate:.1%} ({self.smoothing_window}-eval average)")
            print(f"Best Win Rate: {self.best_win_rate:.1%} (raw)")
            print(f"Best Smoothed: {self.best_smoothed_win_rate:.1%} (smoothed)")
            print(f"Improvement (raw): {improvement_raw:+.1%}")
            print(f"Improvement (smoothed): {improvement_smoothed:+.1%}")
            print(f"Episodes Since Improvement: {self.episodes_since_improvement}/{self.patience}")
            print(f"Average Recent Win Rate: {avg_recent:.1%} ± {std_recent:.3f}")
            print(f"Is Learning: {'✅ Yes' if is_learning else '❌ No'}")
            print(f"Meets Target ({self.target_win_rate:.1%}): {'✅ Yes' if result['meets_target'] else '❌ No'}")
            print(f"Meets Baseline ({self.baseline_win_rate:.1%}): {'✅ Yes' if result['meets_baseline'] else '❌ No'}")
            print(f"\nRecommendation: {'✅ CONTINUE' if should_continue else '⛔ STOP'}")
            print(f"Reason: {reason}")
            print(f"{'='*80}\n")
        
        return result


class AdvancedDMCTrainerWithOpponent:
    """Advanced trainer for DMC agent with sophisticated opponent modeling."""
    
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
        
        # Setup advanced DMC agent with opponent modeling
        self.dmc_agent = AdvancedDMCAgentWithOpponent(
            state_size=self.state_size,
            action_size=self.env.num_actions,
            config=self.config
        )
        
        # Reset opponent tracking
        self.dmc_agent.reset_opponent_tracking(num_players=self.env.num_players)
        
        # Setup MCTS reward shaper (optional)
        mcts_config = self.config.get('mcts', {})
        if mcts_config.get('use_full_mcts', False) or mcts_config.get('enabled', False):
            try:
                self.mcts_shaper = ProperMCTS(
                    env=self.env,
                    config=mcts_config
                )
                self.use_mcts_reward = True
                print("✅ MCTS reward shaping enabled")
            except Exception as e:
                print(f"⚠️  MCTS reward shaping failed to initialize: {e}")
                self.mcts_shaper = None
                self.use_mcts_reward = False
        else:
            self.mcts_shaper = None
            self.use_mcts_reward = False
            print("⚠️  MCTS reward shaping disabled")
        
        # Setup opponents
        self.random_opponent = RandomAgent(self.env.num_actions)
        self.heuristic_opponent = HeuristicAgent(self.env.num_actions)
        
        # Setup logger
        self.logger = TrainingLogger(
            log_dir=self.config['paths']['logs'],
            experiment_name=f"dmc_advanced_opponent_{self.config['environment']['seed']}"
        )
        
        # Setup evaluator
        self.evaluator = Evaluator(self.env)
        
        # Self-play configuration
        self.self_play_config = self.config.get('self_play', {})
        self.use_self_play = self.self_play_config.get('enabled', False)
        
        if self.use_self_play:
            print("✅ Self-play enabled")
            self.opponent_pool = deque(maxlen=self.self_play_config.get('pool_size', 10))
            self.pool_save_interval = self.self_play_config.get('save_interval', 2000)
            self.opponent_probs = self.self_play_config.get('opponent_probs', {
                'random': 0.2,
                'heuristic': 0.2,
                'current': 0.3,
                'past': 0.3
            })
            # Initialize pool directory
            self.pool_dir = os.path.join(self.config['paths']['models'], "opponent_pool")
            os.makedirs(self.pool_dir, exist_ok=True)
            
            # Temporary agent for loading past opponents
            self.past_opponent_agent = AdvancedDMCAgentWithOpponent(
                state_size=self.state_size,
                action_size=self.env.num_actions,
                config=self.config
            )
        else:
            print("⚠️  Self-play disabled")
            self.opponent_pool = None
            
        # Training counters
        self.episode = 0
        self.total_steps = 0
        
        # Best model tracking
        self.best_win_rate = 0.0
        self.best_model_path = None
        
        # Early stopping monitor
        early_stop_config = self.config.get('early_stopping', {})
        self.early_stopping = EarlyStoppingMonitor(
            patience=early_stop_config.get('patience', 3000),
            min_improvement=early_stop_config.get('min_improvement', 0.01),
            baseline_win_rate=early_stop_config.get('baseline_win_rate', 0.50),
            target_win_rate=early_stop_config.get('target_win_rate', 0.55),
            check_interval=early_stop_config.get('check_interval', 500),
            min_episodes=early_stop_config.get('min_episodes', 5000),
            smoothing_window=early_stop_config.get('smoothing_window', 5)
        )
        
        # Progress tracking
        self.progress_check_interval = early_stop_config.get('check_interval', 500)
        # Use more evaluation games to reduce variance (at least 1000 for reliable estimates)
        self.eval_episodes = max(self.config['evaluation'].get('eval_episodes', 1000), 1000)
        
    def train_episode(self):
        """Train for one episode with opponent tracking."""
        state, player_id = self.env.reset()
        
        # Reset opponent tracking
        self.dmc_agent.reset_opponent_tracking(num_players=self.env.num_players)
        
        episode_reward = 0
        episode_length = 0
        episode_reward = 0
        episode_length = 0
        
        # Select opponent
        opponent_agent = self.random_opponent
        opponent_type = "random"
        
        if self.use_self_play:
            rand = np.random.random()
            probs = self.opponent_probs
            
            if rand < probs['random']:
                opponent_agent = self.random_opponent
                opponent_type = "random"
            elif rand < probs['random'] + probs.get('heuristic', 0.0):
                opponent_agent = self.heuristic_opponent
                opponent_type = "heuristic"
            elif rand < probs['random'] + probs.get('heuristic', 0.0) + probs['current']:
                # Play against current self (copy weights)
                # Note: We use the same agent instance but in eval mode for opponent
                # This works because we process turns sequentially
                opponent_agent = self.dmc_agent
                opponent_type = "current"
            else:
                # Play against past self
                if len(self.opponent_pool) > 0:
                    past_model_path = np.random.choice(self.opponent_pool)
                    try:
                        self.past_opponent_agent.load(past_model_path)
                        self.past_opponent_agent.eval()
                        opponent_agent = self.past_opponent_agent
                        opponent_type = "past"
                    except Exception as e:
                        print(f"⚠️  Failed to load past opponent: {e}")
                        opponent_agent = self.random_opponent
                        opponent_type = "random (fallback)"
                else:
                    # Fallback to random if pool empty
                    opponent_agent = self.random_opponent
                    opponent_type = "random (empty pool)"
        
        agents = [self.dmc_agent, opponent_agent]
        
        while not self.env.is_over():
            current_agent = agents[player_id]
            
            # Extract opponent features before action (for our agent)
            if player_id == 0:
                opponent_features = self.dmc_agent.extract_opponent_features(state)
                prev_state = state
            
            # Take action
            if current_agent == self.dmc_agent:
                # Our agent (training)
                action = current_agent.use_raw(state)
            elif current_agent == self.past_opponent_agent:
                # Past opponent (eval)
                # Need to extract features for opponent
                # Note: For simplicity, past opponent uses its own internal tracking
                # We need to ensure it has the correct opponent_id (0)
                # But wait, the opponent sees US as opponent_id=1 relative to them?
                # Actually, in 2-player:
                # Player 0 (Us): Opponent is Player 1
                # Player 1 (Opponent): Opponent is Player 0
                
                # For the past agent, we need to make sure it tracks US
                # But extract_opponent_features uses self.current_opponent_id which defaults to 1
                # We need to temporarily swap it or handle it
                
                # Simplification: Past agent just uses raw state without advanced features for now
                # Or we can properly implement it, but it requires careful state management
                # Let's use use_raw which handles feature extraction internally if implemented
                
                # IMPORTANT: We need to set the opponent agent to eval mode
                current_agent.eval()
                with torch.no_grad():
                    action = current_agent.use_raw(state)
            else:
                # Random agent or current agent (as opponent)
                if current_agent == self.dmc_agent:
                    current_agent.eval()
                    with torch.no_grad():
                        action = current_agent.use_raw(state)
                    current_agent.train()
                else:
                    action = current_agent.use_raw(state)
            next_state, next_player_id = self.env.step(action)
            episode_length += 1
            self.total_steps += 1
            
            # Update opponent history after opponent's action
            if player_id == 1:
                opponent_hand_size = None
                if 'raw_obs' in next_state and 'num_cards' in next_state['raw_obs']:
                    num_cards = next_state['raw_obs']['num_cards']
                    if isinstance(num_cards, (list, np.ndarray)):
                        if 1 < len(num_cards):
                            opponent_hand_size = num_cards[1]
                    elif isinstance(num_cards, dict):
                        opponent_hand_size = num_cards.get(1)
                
                self.dmc_agent.update_opponent_history(
                    action=action,
                    opponent_id=1,
                    hand_size=opponent_hand_size
                )
            
            # Calculate reward and store experience (only for DMC agent)
            if player_id == 0:
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
                next_opponent_features = (
                    self.dmc_agent.extract_opponent_features(next_state) 
                    if not self.env.is_over() else opponent_features
                )
                
                # Store transition
                self.dmc_agent.store_transition(
                    state=prev_state,
                    action=action,
                    reward=shaped_reward,
                    next_state=next_state if not self.env.is_over() else prev_state,
                    done=self.env.is_over(),
                    opponent_features=opponent_features
                )
            
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
        episode_data.update(learning_stats)
        
        return episode_data
    
    def evaluate_agent(self, num_games: int = 200) -> dict:
        """Evaluate the agent with opponent modeling."""
        results = {}
        
        # Set to evaluation mode
        self.dmc_agent.eval()
        original_epsilon = self.dmc_agent.epsilon
        self.dmc_agent.epsilon = 0.0
        
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
        
        # Evaluate against past self (if pool is available)
        if self.use_self_play and self.opponent_pool and len(self.opponent_pool) > 0:
            try:
                # Select a random past model
                past_model_path = np.random.choice(self.opponent_pool)
                self.past_opponent_agent.load(past_model_path)
                self.past_opponent_agent.eval()
                
                # Evaluate
                agents = [self.dmc_agent, self.past_opponent_agent]
                past_results = self.evaluator.evaluate_agents(agents, num_games, verbose=False)
                
                results['vs_past'] = {
                    'win_rate': past_results['win_rates'][0],
                    'avg_game_length': past_results['avg_game_length'],
                    'wins': past_results['wins'][0],
                    'losses': past_results['wins'][1]
                }
            except Exception as e:
                print(f"⚠️  Failed to evaluate against past opponent: {e}")
                results['vs_past'] = {'win_rate': 0.0, 'avg_game_length': 0.0}
        
        return results
    
    def save_model(self, filepath=None, include_training_state=False, suffix=None):
        """
        Save the trained model with optional training state.
        
        Args:
            filepath: Path to save the model. If None, uses default naming.
            include_training_state: If True, includes training state (episode, best_win_rate, etc.)
            suffix: Optional suffix to append to the filename (e.g., timestamp or version)
        """
        if filepath is None:
            os.makedirs(os.path.join(self.config['paths']['models'], "custom"), exist_ok=True)
            filename = f"dmc_advanced_opponent_episode_{self.episode}"
            if suffix:
                filename += f"_{suffix}"
            filename += ".pth"
            
            filepath = os.path.join(
                self.config['paths']['models'],
                "custom",
                filename
            )
        
        # Save model weights (existing functionality)
        self.dmc_agent.save(filepath)
        
        # Add training state if requested
        if include_training_state:
            checkpoint = torch.load(filepath, map_location=self.dmc_agent.device, weights_only=False)
            checkpoint.update({
                'episode': self.episode,
                'best_win_rate': self.best_win_rate,
                'best_smoothed_win_rate': getattr(self.early_stopping, 'best_smoothed_win_rate', self.best_win_rate),
                'episodes_since_improvement': self.early_stopping.episodes_since_improvement,
                'total_steps': self.total_steps
            })
            torch.save(checkpoint, filepath)
        
        self.logger.save_checkpoint(self.episode, filepath)
        return filepath
    
    def load_checkpoint(self, checkpoint_path: str, resume_training: bool = True):
        """
        Load checkpoint and optionally resume training.
        
        Args:
            checkpoint_path: Path to checkpoint file
            resume_training: If True, resume from checkpoint episode; if False, load weights only
            
        Returns:
            dict: Checkpoint data
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.dmc_agent.device, weights_only=False)
        
        # Load model weights
        self.dmc_agent.load(checkpoint_path)
        
        # Load training state if resuming
        if resume_training:
            # Handle old checkpoints that don't have training state
            # Checkpoint episode number is the last completed episode, so resume from next episode
            last_completed_episode = checkpoint.get('episode', 0)
            start_episode = last_completed_episode + 1  # Start from next episode after last completed
            self.episode = start_episode
            
            # Load best win rate and early stopping state (if available)
            if 'best_win_rate' in checkpoint:
                self.best_win_rate = checkpoint['best_win_rate']
                self.early_stopping.best_win_rate = checkpoint['best_win_rate']
            else:
                # Old checkpoint - initialize from current evaluation
                print("⚠️  Old checkpoint format detected (no training state).")
                print("   Running initial evaluation to set baseline...")
                initial_eval = self.evaluate_agent(num_games=self.eval_episodes)
                initial_win_rate = initial_eval['vs_random']['win_rate']
                self.best_win_rate = initial_win_rate
                self.early_stopping.best_win_rate = initial_win_rate
                self.early_stopping.best_smoothed_win_rate = initial_win_rate
                self.early_stopping.baseline_win_rate = initial_win_rate
                print(f"   Initial win rate: {initial_win_rate:.1%}")
            
            if 'best_smoothed_win_rate' in checkpoint:
                self.early_stopping.best_smoothed_win_rate = checkpoint['best_smoothed_win_rate']
            elif 'best_win_rate' in checkpoint:
                self.early_stopping.best_smoothed_win_rate = checkpoint['best_win_rate']
            else:
                self.early_stopping.best_smoothed_win_rate = self.best_win_rate
                
            if 'episodes_since_improvement' in checkpoint:
                self.early_stopping.episodes_since_improvement = checkpoint['episodes_since_improvement']
            if 'total_steps' in checkpoint:
                self.total_steps = checkpoint['total_steps']
            
            print(f"✅ Resuming from episode {start_episode} (last completed: {last_completed_episode})")
            print(f"   Best win rate: {self.best_win_rate:.1%}")
            print(f"   Best smoothed win rate: {self.early_stopping.best_smoothed_win_rate:.1%}")
        else:
            print(f"✅ Model weights loaded (not resuming training)")
        
        return checkpoint
    
    def train(self):
        """Main training loop with early stopping and progress monitoring."""
        self.logger.start_training(self.config)
        
        num_episodes = self.config['training']['episodes']
        save_freq = self.config['logging']['save_freq']
        
        print(f"\n🚀 ADVANCED DMC + OPPONENT MODELING TRAINING")
        print(f"{'='*80}")
        print(f"Device: {self.dmc_agent.device}")
        print(f"State Size: {self.state_size}")
        print(f"Opponent Feature Size: {self.dmc_agent.opponent_feature_size}")
        print(f"Strategy Dimension: {self.dmc_agent.strategy_dim}")
        print(f"Episodes: {num_episodes}")
        print(f"Progress Check Interval: {self.progress_check_interval} episodes")
        print(f"Evaluation Episodes: {self.eval_episodes}")
        print(f"Early Stopping Patience: {self.early_stopping.patience} episodes (no improvement)")
        print(f"Target Win Rate: {self.early_stopping.target_win_rate:.1%} (informational - training continues beyond this)")
        print(f"\n📊 Training Configuration (Matching RLCard DMC):")
        optimizer_type = type(self.dmc_agent.optimizer).__name__
        print(f"   Optimizer: {optimizer_type}")
        print(f"   Learning Rate: {self.config['training']['learning_rate']}")
        print(f"   Gradient Clipping: {self.dmc_agent.max_grad_norm}")
        print(f"   Epsilon End: {self.config['training']['epsilon_end']}")
        print(f"   Dropout: {self.config['network']['dropout']}")
        print(f"   Network LR: {self.dmc_agent.network_lr}")
        print(f"   Opponent LR: {self.dmc_agent.opponent_lr}")
        print(f"\n⚠️  Note: Training will NOT stop just because 55% is reached - it will continue to maximize performance!")
        print(f"{'='*80}\n")
        
        # Initial evaluation (only if not resuming)
        if self.episode == 0:
            print("📊 Initial Evaluation...")
            initial_eval = self.evaluate_agent(num_games=self.eval_episodes)
            initial_win_rate = initial_eval['vs_random']['win_rate']
            print(f"Initial Win Rate: {initial_win_rate:.1%}\n")
            
            self.early_stopping.best_win_rate = initial_win_rate
            self.early_stopping.best_smoothed_win_rate = initial_win_rate
            self.early_stopping.baseline_win_rate = initial_win_rate
            self.best_win_rate = initial_win_rate
        else:
            print(f"📊 Resuming training from episode {self.episode}")
            print(f"   Current best win rate: {self.best_win_rate:.1%}\n")
        
        # Track episodes completed (accessible in finally block)
        episodes_completed_in_loop = 0
        
        try:
            # Adjust episode range if resuming
            start_episode = self.episode
            end_episode = num_episodes
            episodes_to_run = end_episode - start_episode
            
            print(f"🔍 DEBUG: Starting training loop")
            print(f"   Start episode: {start_episode}")
            print(f"   End episode: {end_episode}")
            print(f"   Episodes to run: {episodes_to_run}")
            print(f"   Range: range({start_episode}, {end_episode})")
            
            if episodes_to_run <= 0:
                print(f"⚠️  WARNING: No episodes to run! start_episode={start_episode}, num_episodes={num_episodes}")
                print(f"   This means training is already complete!")
                return
            
            pbar = tqdm(range(start_episode, end_episode), desc="Training", unit="episode", initial=start_episode, total=end_episode)
            
            for episode in pbar:
                try:
                    self.episode = episode
                    episodes_completed_in_loop += 1
                    
                    # Train one episode
                    self.logger.start_episode(episode)
                    episode_data = self.train_episode()
                    self.logger.end_episode(episode, episode_data)
                except Exception as e:
                    print(f"❌ ERROR during episode {episode}: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
                
                # Update progress bar
                pbar.set_postfix({
                    'win_rate': f"{self.early_stopping.best_win_rate:.1%}",
                    'epsilon': f"{self.dmc_agent.epsilon:.3f}",
                    'loss': f"{episode_data.get('total_loss', 0):.4f}"
                })
                
                # Progress check and early stopping
                if (episode + 1) % self.progress_check_interval == 0 and episode > 0:
                    # Evaluate agent
                    eval_results = self.evaluate_agent(num_games=self.eval_episodes)
                    current_win_rate = eval_results['vs_random']['win_rate']
                    
                    # Check progress
                    progress_info = self.early_stopping.check_progress(
                        current_episode=episode + 1,
                        current_win_rate=current_win_rate,
                        verbose=True
                    )
                    
                    # Store last evaluation for error recovery
                    self._last_eval_results = eval_results
                    
                    # Check if this is a new best model and save it
                    if current_win_rate > self.best_win_rate:
                        self.best_win_rate = current_win_rate
                        self.best_win_rate = current_win_rate
                        
                        # Save with unique name to avoid overwriting previous bests
                        timestamp = int(time.time())
                        best_model_filename = f"dmc_advanced_opponent_best_v2_{timestamp}.pth"
                        best_model_path = os.path.join(
                            self.config['paths']['models'],
                            "custom",
                            best_model_filename
                        )
                        self.save_model(best_model_path, include_training_state=True)
                        self.best_model_path = best_model_path
                        print(f"✅ New best model saved: {current_win_rate:.1%} win rate (episode {episode + 1})")
                        print(f"   Saved as: {best_model_filename}")
                    
                    # Log evaluation
                    eval_data = {
                        'random_win_rate': current_win_rate,
                        'random_avg_length': eval_results['vs_random']['avg_game_length'],
                        'best_win_rate': progress_info['best_win_rate'],
                        'best_smoothed_win_rate': progress_info.get('best_smoothed_win_rate', progress_info['best_win_rate']),
                        'smoothed_win_rate': progress_info.get('smoothed_win_rate', current_win_rate),
                        'improvement_raw': progress_info.get('improvement_raw', progress_info.get('improvement', 0)),
                        'improvement_smoothed': progress_info.get('improvement_smoothed', 0),
                        'episodes_since_improvement': progress_info['episodes_since_improvement'],
                        'episodes_since_improvement': progress_info['episodes_since_improvement'],
                        'is_learning': progress_info['is_learning']
                    }
                    
                    # Add vs_past stats if available
                    if 'vs_past' in eval_results:
                        eval_data['past_win_rate'] = eval_results['vs_past']['win_rate']
                        print(f"Vs Past Self: {eval_results['vs_past']['win_rate']:.1%} win rate")
                        
                    self.logger.log_evaluation(episode + 1, eval_data)
                    
                    # Update learning rate scheduler
                    self.dmc_agent.scheduler.step(current_win_rate)
                    
                    # Early stopping decision
                    if not progress_info['should_continue']:
                        print(f"\n⛔ EARLY STOPPING TRIGGERED")
                        print(f"Reason: {progress_info['reason']}")
                        print(f"Best Win Rate Achieved: {progress_info['best_win_rate']:.1%}")
                        print(f"Current Win Rate: {current_win_rate:.1%}")
                        if progress_info['best_win_rate'] >= self.early_stopping.target_win_rate:
                            print(f"✅ Target win rate ({self.early_stopping.target_win_rate:.1%}) exceeded!")
                        break
                
                # Save model periodically (with training state)
                if (episode + 1) % save_freq == 0 and episode > 0:
                    self.save_model(include_training_state=True)
                    
                # Save to opponent pool
                if self.use_self_play and (episode + 1) % self.pool_save_interval == 0:
                    pool_model_path = os.path.join(
                        self.pool_dir,
                        f"opponent_episode_{episode + 1}.pth"
                    )
                    self.save_model(pool_model_path, include_training_state=False)
                    self.opponent_pool.append(pool_model_path)
                    # print(f"💾 Added model to opponent pool: {pool_model_path}")
        
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
        except Exception as e:
            print(f"\n❌ ERROR in training loop: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            print(f"🔍 DEBUG: Training loop exited. Episodes completed in loop: {episodes_completed_in_loop}")
        
            # Final evaluation (with error handling)
            print("\n📊 Final Evaluation...")
            try:
                # Use more games for final evaluation to get reliable estimate
                final_eval_games = max(self.eval_episodes * 2, 2000)
                final_eval = self.evaluate_agent(num_games=final_eval_games)
                final_win_rate = final_eval['vs_random']['win_rate']
                final_avg_length = final_eval['vs_random']['avg_game_length']
            except Exception as e:
                print(f"⚠️  Error during final evaluation: {e}")
                print("   Using last evaluation results if available...")
                # Try to get last evaluation if available
                if hasattr(self, '_last_eval_results'):
                    final_eval = self._last_eval_results
                    final_win_rate = final_eval.get('vs_random', {}).get('win_rate', 0.0)
                    final_avg_length = final_eval.get('vs_random', {}).get('avg_game_length', 0.0)
                else:
                    final_win_rate = self.early_stopping.best_win_rate
                    final_avg_length = 0.0
                    final_eval = {'vs_random': {'win_rate': final_win_rate, 'avg_game_length': final_avg_length}}
            
            # Log final results
            try:
                final_eval_data = {
                    'random_win_rate': final_win_rate,
                    'random_avg_length': final_avg_length
                }
                self.logger.log_evaluation(self.episode, final_eval_data)
            except Exception as e:
                print(f"⚠️  Error logging final evaluation: {e}")
            
            # Save final model (with training state)
            try:
                os.makedirs(os.path.join(self.config['paths']['models'], "custom"), exist_ok=True)
                final_model_path = self.save_model(
                    os.path.join(self.config['paths']['models'], "custom", "dmc_advanced_opponent_final.pth"),
                    include_training_state=True
                )
            except Exception as e:
                print(f"⚠️  Error saving final model: {e}")
                final_model_path = None
            
            try:
                self.logger.end_training()
            except Exception as e:
                print(f"⚠️  Error ending training log: {e}")
            
            # Print summary
            print(f"\n{'='*80}")
            print(f"🏆 TRAINING SUMMARY")
            print(f"{'='*80}")
            print(f"Total Episodes: {self.episode + 1}")
            print(f"Final Win Rate: {final_win_rate:.1%}")
            if hasattr(self.early_stopping, 'best_smoothed_win_rate'):
                print(f"Best Smoothed Win Rate: {self.early_stopping.best_smoothed_win_rate:.1%}")
            print(f"Best Win Rate: {self.best_win_rate:.1%}")
            print(f"Improvements: {self.early_stopping.improvements}")
            print(f"Progress Checks: {self.early_stopping.total_checks}")
            if self.best_model_path:
                print(f"Best Model: {self.best_model_path}")
            if final_model_path:
                print(f"Final Model: {final_model_path}")
            print(f"{'='*80}\n")
            
            return final_eval, final_model_path


def main():
    """Main function to start training."""
    import argparse
    parser = argparse.ArgumentParser(description='Train DMC Agent with Advanced Opponent Modeling')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume from')
    parser.add_argument('--resume-best', action='store_true',
                      help='Resume from best model checkpoint')
    args = parser.parse_args()
    
    # Create trainer
    trainer = AdvancedDMCTrainerWithOpponent(config_path="config.yaml")
    
    # Load checkpoint if specified
    if args.resume_best:
        best_checkpoint = os.path.join(
            trainer.config['paths']['models'],
            "custom",
            "dmc_advanced_opponent_best.pth"
        )
        if os.path.exists(best_checkpoint):
            trainer.load_checkpoint(best_checkpoint, resume_training=True)
        else:
            # Fallback to final checkpoint if best doesn't exist
            final_checkpoint = os.path.join(
                trainer.config['paths']['models'],
                "custom",
                "dmc_advanced_opponent_final.pth"
            )
            if os.path.exists(final_checkpoint):
                print(f"⚠️  Best checkpoint not found. Using final checkpoint: {final_checkpoint}")
                trainer.load_checkpoint(final_checkpoint, resume_training=True)
            else:
                print(f"⚠️  No checkpoint found. Starting fresh training...")
    elif args.resume:
        trainer.load_checkpoint(args.resume, resume_training=True)
    
    # Train
    final_eval, model_path = trainer.train()
    
    # Print final results
    print("\n🎉 TRAINING COMPLETED!")
    print("=" * 80)
    print(f"Final Win Rate vs Random: {final_eval['vs_random']['win_rate']:.1%}")
    print(f"Final Model: {model_path}")
    if trainer.best_model_path:
        print(f"Best Model: {trainer.best_model_path}")
    print(f"Logs: {trainer.logger.log_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

