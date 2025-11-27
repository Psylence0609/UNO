"""
Actor-Learner Training Script for DMC Agent with Opponent Modeling.
Implements parallel actor-learner architecture with unroll-based learning.
"""

import os
import sys
import yaml
import torch
import numpy as np
import multiprocessing as mp
import time
from tqdm import tqdm
from typing import Dict, Optional
import signal

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.environments.uno_env import UnoEnvironment
from src.agents.dmc_agent_advanced_opponent import AdvancedDMCAgentWithOpponent
from src.agents.random_agent import RandomAgent
from src.training.logger import TrainingLogger
from src.evaluation.evaluator import Evaluator
from src.training.actor_process import ActorManager
from src.training.learner_thread import LearnerThread
from src.utils.shared_experience_buffer import SharedExperienceBuffer
from src.training.train_dmc_advanced_opponent import EarlyStoppingMonitor


# Set multiprocessing start method
if __name__ == '__main__':
    # Use 'spawn' for better compatibility (especially on macOS)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set


class ActorLearnerTrainer:
    """
    Actor-learner trainer for DMC with opponent modeling.
    Orchestrates parallel actors, shared buffers, and learner threads.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the actor-learner trainer."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Actor-learner configuration
        actor_learner_config = self.config.get('actor_learner', {})
        self.num_actors = actor_learner_config.get('num_actors', 3)
        self.num_buffers = actor_learner_config.get('num_buffers', 10)
        self.num_learner_threads = actor_learner_config.get('num_learner_threads', 2)
        self.unroll_length = actor_learner_config.get('unroll_length', 100)
        self.batch_size = actor_learner_config.get('batch_size', 32)
        self.model_sync_frequency = actor_learner_config.get('model_sync_frequency', 100)  # Updates between syncs
        
        # Setup environment for state size calculation
        self.env = UnoEnvironment(seed=self.config['environment']['seed'])
        
        # Calculate state size
        sample_state, _ = self.env.reset()
        features = []
        if 'obs' in sample_state:
            features.extend(sample_state['obs'].flatten())
        features.extend(np.zeros(self.env.num_actions))
        self.state_size = len(features)
        
        # Create main agent (learner will use this)
        self.agent = AdvancedDMCAgentWithOpponent(
            state_size=self.state_size,
            action_size=self.env.num_actions,
            config=self.config
        )
        
        # Create shared experience buffer
        self.shared_buffer = SharedExperienceBuffer(
            num_buffers=self.num_buffers,
            buffer_capacity=actor_learner_config.get('buffer_capacity', 100),
            unroll_length=self.unroll_length
        )
        
        # Create queues for communication
        self.model_state_queue = mp.Queue(maxsize=10)
        self.actor_stats_queue = mp.Queue()
        self.learner_stats_queue = mp.Queue()
        
        # Create actor manager
        self.actor_manager = ActorManager(
            num_actors=self.num_actors,
            config=self.config,
            shared_buffer=self.shared_buffer,
            model_state_queue=self.model_state_queue,
            stats_queue=self.actor_stats_queue,
            unroll_length=self.unroll_length
        )
        
        # Create a lock for synchronizing model access across learner threads
        # This prevents in-place operation errors when multiple threads access the same model
        import threading
        self.model_lock = threading.Lock()
        
        # Create learner threads
        self.learner_threads = []
        for i in range(self.num_learner_threads):
            learner = LearnerThread(
                agent=self.agent,
                shared_buffer=self.shared_buffer,
                batch_size=self.batch_size,
                gamma=self.config['training']['gamma'],
                max_grad_norm=self.config['training'].get('max_grad_norm', 40.0),
                update_frequency=0.01,  # Minimal throttling for better update frequency
                stats_queue=self.learner_stats_queue,
                learner_id=f"thread_{i}",  # Assign unique ID to each learner
                model_lock=self.model_lock  # Share lock across all learners
            )
            self.learner_threads.append(learner)
        
        # Setup logger
        self.logger = TrainingLogger(
            log_dir=self.config['paths']['logs'],
            experiment_name=f"dmc_actor_learner_{self.config['environment']['seed']}"
        )
        
        # Setup evaluator
        self.evaluator = Evaluator(self.env)
        
        # Training counters
        self.total_frames = 0
        self.total_updates = 0
        self.episode = 0  # Evaluation counter
        self.total_game_episodes = 0  # Actual game episodes from actors
        self.training_start_time = None  # Track training start time for diagnostics
        self.learner_update_counts = {}  # Track last update count per learner to calculate deltas
        
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
        
        # Evaluation configuration
        self.eval_episodes = max(self.config['evaluation'].get('eval_episodes', 1000), 1000)
        self.eval_freq = self.config['evaluation'].get('eval_freq', 2000)  # Evaluate every N frames
        
        # Model synchronization counter
        self.updates_since_sync = 0
        
        # Signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.shutdown_requested = False
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print("\n⚠️  Shutdown signal received, stopping training...")
        self.shutdown_requested = True
    
    def evaluate_agent(self, num_games: int = None) -> Dict:
        """Evaluate the agent."""
        if num_games is None:
            num_games = self.eval_episodes
        
        # Set to evaluation mode
        self.agent.eval()
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0
        
        # Evaluate against random agent
        agents = [self.agent, RandomAgent(self.env.num_actions)]
        random_results = self.evaluator.evaluate_agents(agents, num_games, verbose=False)
        
        # Restore training mode
        self.agent.train()
        self.agent.epsilon = original_epsilon
        
        return {
            'vs_random': {
                'win_rate': random_results['win_rates'][0],
                'avg_game_length': random_results['avg_game_length'],
                'wins': random_results['wins'][0],
                'losses': random_results['wins'][1]
            }
        }
    
    def save_model(self, filepath: Optional[str] = None, include_training_state: bool = False):
        """Save the trained model."""
        if filepath is None:
            os.makedirs(os.path.join(self.config['paths']['models'], "custom"), exist_ok=True)
            filepath = os.path.join(
                self.config['paths']['models'],
                "custom",
                f"dmc_actor_learner_episode_{self.episode}.pth"
            )
        
        # Save model weights
        self.agent.save(filepath)
        
        # Add training state if requested
        if include_training_state:
            checkpoint = torch.load(filepath, map_location=self.agent.device, weights_only=False)
            checkpoint.update({
                'episode': self.episode,
                'total_frames': self.total_frames,
                'total_updates': self.total_updates,
                'total_game_episodes': self.total_game_episodes,
                'best_win_rate': self.best_win_rate,
                'best_smoothed_win_rate': getattr(self.early_stopping, 'best_smoothed_win_rate', self.best_win_rate),
                'episodes_since_improvement': self.early_stopping.episodes_since_improvement
            })
            torch.save(checkpoint, filepath)
        
        self.logger.save_checkpoint(self.episode, filepath)
        return filepath
    
    def load_model(self, filepath: str):
        """
        Load model from checkpoint.
        
        Args:
            filepath: Path to checkpoint file
        """
        if not os.path.exists(filepath):
            print(f"⚠️  Model file not found: {filepath}")
            return False
        
        try:
            checkpoint = torch.load(filepath, map_location=self.agent.device, weights_only=False)
            
            # Load model weights
            if 'network_state_dict' in checkpoint:
                self.agent.network.load_state_dict(checkpoint['network_state_dict'])
            if 'opponent_model_state_dict' in checkpoint:
                self.agent.opponent_model.load_state_dict(checkpoint['opponent_model_state_dict'])
            
            # Load training state if available
            if 'total_frames' in checkpoint:
                self.total_frames = checkpoint['total_frames']
            if 'total_updates' in checkpoint:
                self.total_updates = checkpoint['total_updates']
            if 'episode' in checkpoint:
                self.episode = checkpoint['episode']
            if 'total_game_episodes' in checkpoint:
                self.total_game_episodes = checkpoint['total_game_episodes']
            if 'best_win_rate' in checkpoint:
                self.best_win_rate = checkpoint['best_win_rate']
            if 'epsilon' in checkpoint:
                self.agent.epsilon = checkpoint['epsilon']
            
            print(f"✅ Model loaded from: {filepath}")
            print(f"   Frames: {self.total_frames:,}")
            print(f"   Updates: {self.total_updates:,}")
            print(f"   Best Win Rate: {self.best_win_rate:.1%}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def find_best_model(self) -> Optional[str]:
        """Find the best saved model path."""
        model_dir = os.path.join(self.config['paths']['models'], "custom")
        best_model_path = os.path.join(model_dir, "dmc_actor_learner_best.pth")
        
        if os.path.exists(best_model_path):
            return best_model_path
        
        # Fallback: look for any actor_learner model
        if os.path.exists(model_dir):
            import glob
            models = glob.glob(os.path.join(model_dir, "dmc_actor_learner*.pth"))
            if models:
                # Return most recent
                return max(models, key=os.path.getmtime)
        
        return None
    
    def train(self, total_frames: Optional[int] = None, resume_from_best: bool = True):
        """
        Main training loop with actor-learner architecture.
        
        Args:
            total_frames: Total number of frames to train (None = use config)
        """
        if total_frames is None:
            total_frames = self.config['training'].get('total_frames', 10_000_000)
        
        # Load best model if resuming
        if resume_from_best:
            best_model = self.find_best_model()
            if best_model:
                print(f"\n📂 Resuming from best model: {best_model}")
                self.load_model(best_model)
                self.best_model_path = best_model
            else:
                print("\n📂 No existing model found, starting from scratch")
        
        self.logger.start_training(self.config)
        self.training_start_time = time.time()  # Track training start
        
        print(f"\n🚀 ACTOR-LEARNER DMC TRAINING")
        print(f"{'='*80}")
        print(f"Device: {self.agent.device}")
        print(f"State Size: {self.state_size}")
        print(f"Opponent Feature Size: {self.agent.opponent_feature_size}")
        print(f"Strategy Dimension: {self.agent.strategy_dim}")
        print(f"Total Frames: {total_frames:,}")
        print(f"Unroll Length: {self.unroll_length}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Number of Actors: {self.num_actors}")
        print(f"Number of Learner Threads: {self.num_learner_threads}")
        print(f"Number of Buffers: {self.num_buffers}")
        print(f"Evaluation Frequency: Every {self.eval_freq:,} frames")
        print(f"Evaluation Episodes: {self.eval_episodes}")
        print(f"{'='*80}\n")
        
        # Start actors and learners
        print("🎬 Starting actors and learners...")
        try:
            self.actor_manager.start()
            # Wait a moment to ensure actors start
            time.sleep(2.0)
            
            # Check if actors are alive
            if not self.actor_manager.is_alive():
                print("⚠️  WARNING: Actors failed to start!")
                raise RuntimeError("Actor processes failed to start")
            
            for learner in self.learner_threads:
                learner.start()
            
            # Wait a moment for learners to start
            time.sleep(1.0)
            
            print("✅ All actors and learners started")
            print(f"   Actors alive: {self.actor_manager.is_alive()}")
            print(f"   Learners alive: {all(lt.is_alive() for lt in self.learner_threads)}\n")
        except Exception as e:
            print(f"❌ ERROR starting actors/learners: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Initial evaluation
        print("📊 Initial Evaluation...")
        initial_eval = self.evaluate_agent(num_games=self.eval_episodes)
        initial_win_rate = initial_eval['vs_random']['win_rate']
        print(f"Initial Win Rate: {initial_win_rate:.1%}\n")
        
        self.early_stopping.best_win_rate = initial_win_rate
        self.early_stopping.best_smoothed_win_rate = initial_win_rate
        self.early_stopping.baseline_win_rate = initial_win_rate
        self.best_win_rate = initial_win_rate
        
        # Save initial model
        self.best_model_path = self.save_model(
            os.path.join(
                self.config['paths']['models'],
                "custom",
                "dmc_actor_learner_best.pth"
            ),
            include_training_state=True
        )
        
        # Training loop
        last_eval_time = time.time()
        last_stats_time = time.time()
        stats_interval = 10.0  # Print stats every 10 seconds
        
        # Track steps per actor for frame counting
        actor_step_counts = {}
        actor_episode_counts = {}  # Track episodes per actor to avoid double counting
        
        try:
            # Wait for initial statistics to ensure actors are working
            print("⏳ Waiting for actors to start producing data...")
            initial_wait_start = time.time()
            initial_wait_timeout = 30.0  # 30 seconds timeout
            
            while time.time() - initial_wait_start < initial_wait_timeout:
                if not self.actor_stats_queue.empty():
                    break
                if not self.actor_manager.is_alive():
                    print("❌ ERROR: Actors died during startup!")
                    raise RuntimeError("Actor processes died")
                time.sleep(0.5)
            
            if self.actor_stats_queue.empty():
                print("⚠️  WARNING: No actor statistics received after 30 seconds")
                print("   This might indicate actors are not running properly")
                print("   Continuing anyway, but training may not progress...")
            
            with tqdm(total=total_frames, desc="Training", unit="frame") as pbar:
                iteration_count = 0
                last_frame_count = 0
                stuck_check_time = time.time()
                
                while self.total_frames < total_frames and not self.shutdown_requested:
                    iteration_count += 1
                    
                    # Check if actors are still alive
                    if not self.actor_manager.is_alive():
                        print("\n❌ ERROR: Actor processes died!")
                        print("   Stopping training...")
                        break
                    
                    # Check for stuck training (no frames collected for 60 seconds)
                    if time.time() - stuck_check_time > 60.0:
                        if self.total_frames == last_frame_count:
                            print(f"\n⚠️  WARNING: No frames collected in last 60 seconds")
                            print(f"   Current frames: {self.total_frames:,}")
                            print(f"   Buffer stats: {self.shared_buffer.get_stats()}")
                            print(f"   Actors alive: {self.actor_manager.is_alive()}")
                            print(f"   Continuing to wait...")
                        last_frame_count = self.total_frames
                        stuck_check_time = time.time()
                    
                    # Check for actor statistics
                    actor_stats = []
                    while not self.actor_stats_queue.empty():
                        try:
                            stats = self.actor_stats_queue.get_nowait()
                            actor_stats.append(stats)
                            
                            # Check for errors
                            if 'error' in stats:
                                print(f"\n❌ ERROR from Actor {stats.get('actor_id', 'unknown')}: {stats.get('error', 'Unknown error')}")
                                print(f"   Error type: {stats.get('error_type', 'Unknown')}")
                            
                            if 'steps_collected' in stats:
                                actor_id = stats.get('actor_id', 0)
                                new_steps = stats.get('steps_collected', 0)
                                old_steps = actor_step_counts.get(actor_id, 0)
                                if new_steps > old_steps:
                                    self.total_frames += (new_steps - old_steps)
                                    actor_step_counts[actor_id] = new_steps
                            
                            # Track actual game episodes from actors (avoid double counting)
                            if 'episodes_completed' in stats:
                                actor_id = stats.get('actor_id', 0)
                                new_episodes = stats.get('episodes_completed', 0)
                                old_episodes = actor_episode_counts.get(actor_id, 0)
                                if new_episodes > old_episodes:
                                    self.total_game_episodes += (new_episodes - old_episodes)
                                    actor_episode_counts[actor_id] = new_episodes
                        except Exception as e:
                            print(f"\n⚠️  Error processing actor stats: {e}")
                            break
                    
                    # Check for learner statistics
                    learner_stats = []
                    while not self.learner_stats_queue.empty():
                        try:
                            stats = self.learner_stats_queue.get_nowait()
                            learner_stats.append(stats)
                        except:
                            break
                    
                    # Calculate total updates by directly summing all learner update counts
                    # This is simpler and more reliable than delta tracking
                    total_learner_updates = 0
                    for i, learner in enumerate(self.learner_threads):
                        stats = learner.get_stats()
                        total_learner_updates += stats.get('update_count', 0)
                    
                    # Update total_updates to match the sum (this is the ground truth)
                    self.total_updates = total_learner_updates
                    
                    # Synchronize model to actors periodically
                    if self.updates_since_sync >= self.model_sync_frequency:
                        model_state = {
                            'network_state_dict': self.agent.network.state_dict(),
                            'opponent_model_state_dict': self.agent.opponent_model.state_dict(),
                            'epsilon': self.agent.epsilon
                        }
                        self.actor_manager.broadcast_model_update(model_state)
                        self.updates_since_sync = 0
                    
                    # Update sync counter from learner stats
                    if learner_stats:
                        self.updates_since_sync = max(
                            self.updates_since_sync,
                            max(s.get('update_count', 0) for s in learner_stats) % self.model_sync_frequency
                        )
                    
                    # Update progress bar
                    buffer_stats = self.shared_buffer.get_stats()
                    pbar.set_postfix({
                        'frames': f"{self.total_frames:,}",
                        'updates': f"{self.total_updates:,}",
                        'episodes': f"{self.total_game_episodes:,}",
                        'win_rate': f"{self.best_win_rate:.1%}",
                        'buffer': f"{buffer_stats['current_size']}"
                    })
                    pbar.update(min(1000, total_frames - self.total_frames))
                    
                    # Print statistics periodically
                    current_time = time.time()
                    if current_time - last_stats_time >= stats_interval:
                        buffer_stats = self.shared_buffer.get_stats()
                        learner_stats_summary = {}
                        for learner in self.learner_threads:
                            stats = learner.get_stats()
                            for key, value in stats.items():
                                if key not in learner_stats_summary:
                                    learner_stats_summary[key] = []
                                learner_stats_summary[key].append(value)
                        
                        # Also get stats from learner_stats queue
                        temp_learner_stats = []
                        while not self.learner_stats_queue.empty() and len(temp_learner_stats) < 10:
                            try:
                                temp_learner_stats.append(self.learner_stats_queue.get_nowait())
                            except:
                                break
                        
                        # Add to summary
                        for stats in temp_learner_stats:
                            for key, value in stats.items():
                                if key not in learner_stats_summary:
                                    learner_stats_summary[key] = []
                                if isinstance(value, (int, float)):
                                    learner_stats_summary[key].append(value)
                        
                        # Calculate update frequency metrics
                        frames_per_update = self.total_frames / max(1, self.total_updates)
                        # Calculate updates per second (use training start time)
                        if self.training_start_time:
                            training_duration = current_time - self.training_start_time
                            updates_per_second = self.total_updates / max(1.0, training_duration)
                        else:
                            updates_per_second = 0.0
                        
                        print(f"\n📊 Statistics (Frames: {self.total_frames:,}, Updates: {self.total_updates:,})")
                        print(f"   Game Episodes: {self.total_game_episodes:,}")
                        print(f"   Update Rate: {frames_per_update:.0f} frames/update (target: <2000)")
                        print(f"   Updates/sec: {updates_per_second:.2f}")
                        print(f"   Buffer: {buffer_stats['current_size']} sequences, {buffer_stats['total_sequences']} total, {buffer_stats['total_dropped']} dropped")
                        if buffer_stats['total_dropped'] > 0:
                            print(f"   ⚠️  WARNING: {buffer_stats['total_dropped']} sequences dropped (buffer may be too small)")
                        if buffer_stats['current_size'] == 0:
                            print(f"   ⚠️  WARNING: Buffer is empty - actors are too slow! Need more actors or shorter unroll_length")
                        
                        # Calculate actor production rate
                        if self.total_frames > 0:
                            sequences_per_frame = buffer_stats['total_sequences'] / max(1, self.total_frames)
                            frames_per_sequence = 1.0 / max(0.0001, sequences_per_frame)
                            print(f"   Actor Rate: {sequences_per_frame*1000:.2f} sequences/1000 frames, {frames_per_sequence:.0f} frames/sequence")
                        
                        # Learner-specific stats
                        if learner_stats_summary:
                            avg_loss = np.mean(learner_stats_summary.get('avg_loss', [0])) if learner_stats_summary.get('avg_loss') else 0.0
                            total_sequences = sum(learner_stats_summary.get('sequences_processed', [0])) if learner_stats_summary.get('sequences_processed') else 0
                            update_counts = learner_stats_summary.get('update_count', [])
                            print(f"   Avg Loss: {avg_loss:.4f}")
                            print(f"   Sequences Processed: {total_sequences:,}")
                            if update_counts:
                                print(f"   Updates per Learner: {update_counts}")
                        
                        # Warning if update rate is too high
                        if frames_per_update > 5000:
                            print(f"   ⚠️  WARNING: Update rate too high ({frames_per_update:.0f} frames/update). Learners may be too slow!")
                        elif frames_per_update < 1000:
                            print(f"   ✅ Update rate good ({frames_per_update:.0f} frames/update)")
                        print()
                        last_stats_time = current_time
                    
                    # Evaluation
                    if self.total_frames >= self.eval_freq and (self.total_frames % self.eval_freq == 0 or current_time - last_eval_time >= 300):
                        frames_per_update = self.total_frames / max(1, self.total_updates)
                        print(f"\n📊 Evaluation at {self.total_frames:,} frames...")
                        print(f"   Game Episodes: {self.total_game_episodes:,}")
                        print(f"   Updates: {self.total_updates:,} (avg {frames_per_update:.0f} frames/update)")
                        if frames_per_update > 5000:
                            print(f"   ⚠️  WARNING: Update frequency too low! Only {self.total_updates} updates for {self.total_frames:,} frames")
                        eval_results = self.evaluate_agent(num_games=self.eval_episodes)
                        win_rate = eval_results['vs_random']['win_rate']
                        
                        # Check progress using actual game episodes
                        progress_result = self.early_stopping.check_progress(
                            current_episode=self.total_game_episodes,  # Use actual game episodes
                            current_win_rate=win_rate,
                            verbose=True
                        )
                        
                        # Update best model
                        if win_rate > self.best_win_rate:
                            self.best_win_rate = win_rate
                            self.best_model_path = self.save_model(
                                os.path.join(
                                    self.config['paths']['models'],
                                    "custom",
                                    "dmc_actor_learner_best.pth"
                                ),
                                include_training_state=True
                            )
                            print(f"✅ New best model saved: {self.best_win_rate:.1%}")
                        
                        # Check early stopping
                        if not progress_result['should_continue']:
                            print(f"\n⛔ Early stopping triggered: {progress_result['reason']}")
                            break
                        
                        self.episode += 1
                        last_eval_time = current_time
                    
                    # Small sleep to prevent busy waiting
                    time.sleep(0.01)
        
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            import traceback
            traceback.print_exc()
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
        finally:
            # Stop actors and learners
            print("\n🛑 Stopping actors and learners...")
            self.actor_manager.stop()
            for learner in self.learner_threads:
                learner.stop()
            print("✅ All stopped")
            
            # Save final model
            print("\n💾 Saving final model...")
            final_model_path = self.save_model(
                os.path.join(
                    self.config['paths']['models'],
                    "custom",
                    "dmc_actor_learner_final.pth"
                ),
                include_training_state=True
            )
            print(f"✅ Final model saved: {final_model_path}")
            
            # Final evaluation
            print("\n📊 Final Evaluation...")
            final_eval = self.evaluate_agent(num_games=self.eval_episodes)
            final_win_rate = final_eval['vs_random']['win_rate']
            print(f"Final Win Rate: {final_win_rate:.1%}")
            print(f"Best Win Rate: {self.best_win_rate:.1%}")
            
            self.logger.end_training()
            
            print(f"\n🏆 TRAINING SUMMARY")
            print(f"{'='*80}")
            print(f"Total Frames: {self.total_frames:,}")
            print(f"Total Updates: {self.total_updates:,}")
            print(f"Best Win Rate: {self.best_win_rate:.1%}")
            print(f"Final Win Rate: {final_win_rate:.1%}")
            print(f"{'='*80}\n")


def main():
    """Main function to start training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DMC with Actor-Learner Architecture')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--total-frames', type=int, default=None, help='Total frames to train')
    args = parser.parse_args()
    
    trainer = ActorLearnerTrainer(config_path=args.config)
    trainer.train(total_frames=args.total_frames, resume_from_best=True)
    
    print("\n🎉 TRAINING COMPLETED!")


if __name__ == '__main__':
    main()

