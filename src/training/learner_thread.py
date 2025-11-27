"""
Learner thread for actor-learner architecture.
Consumes sequences from shared buffer and updates network weights.
"""

import threading
import time
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
from collections import deque

from src.utils.shared_experience_buffer import SharedExperienceBuffer, UnrollSequence
from src.agents.dmc_agent_advanced_opponent import AdvancedDMCAgentWithOpponent


class LearnerThread(threading.Thread):
    """
    Learner thread that processes sequences and updates the network.
    """
    
    def __init__(
        self,
        agent: AdvancedDMCAgentWithOpponent,
        shared_buffer: SharedExperienceBuffer,
        batch_size: int = 32,
        gamma: float = 0.99,
        max_grad_norm: float = 40.0,
        update_frequency: float = 0.01,  # Minimal throttling (reduced from 0.1s)
        stats_queue: Optional[any] = None,
        learner_id: Optional[str] = None,
        model_lock: Optional[any] = None
    ):
        """
        Initialize learner thread.
        
        Args:
            agent: DMC agent with opponent modeling
            shared_buffer: Shared experience buffer
            batch_size: Number of sequences per batch
            gamma: Discount factor for returns
            max_grad_norm: Maximum gradient norm for clipping
            update_frequency: Minimum time between updates (seconds)
            stats_queue: Queue for sending statistics
        """
        super().__init__(daemon=True)
        self.agent = agent
        self.shared_buffer = shared_buffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.update_frequency = update_frequency
        self.stats_queue = stats_queue
        
        self.stop_event = threading.Event()
        self.learner_id = learner_id or f"learner_{id(self)}"
        self.update_count = 0
        self.sequences_processed = 0
        self.last_update_time = time.time()
        self.model_lock = model_lock  # Lock for synchronizing model access
        
        # Statistics
        self.loss_history = deque(maxlen=100)
        self.value_loss_history = deque(maxlen=100)
        self.policy_loss_history = deque(maxlen=100)
        self.mc_loss_history = deque(maxlen=100)
    
    def run(self):
        """Main learning loop."""
        while not self.stop_event.is_set():
            try:
                # Use sample_sequences with very short timeout - process whatever we get quickly
                # This balances between processing frequently and having reasonable batch sizes
                sequences = self.shared_buffer.sample_sequences(
                    batch_size=self.batch_size,
                    timeout=0.02  # Very short timeout (20ms) - process quickly
                )
                
                if len(sequences) > 0:
                    # Process batch immediately (even if smaller than batch_size)
                    self._learn_from_sequences(sequences)
                    self.sequences_processed += len(sequences)
                    self.last_update_time = time.time()
                else:
                    # No sequences available - very brief sleep to avoid busy-waiting
                    time.sleep(0.0005)  # 0.5ms sleep
                    
            except Exception as e:
                # Log error but continue
                print(f"Learner thread error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.5)
    
    def _learn_from_sequences(self, sequences: List[UnrollSequence]):
        """
        Learn from a batch of sequences.
        
        Args:
            sequences: List of UnrollSequence objects
        """
        # Flatten sequences into individual transitions
        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_dones = []
        all_opponent_features = []
        
        # Calculate returns for each sequence
        all_returns = []
        
        for sequence in sequences:
            # Calculate Monte Carlo returns for this sequence
            returns = []
            G = 0
            for i in range(len(sequence.rewards) - 1, -1, -1):
                if sequence.dones[i]:
                    G = sequence.rewards[i]
                else:
                    G = sequence.rewards[i] + self.gamma * G
                returns.insert(0, G)
            
            # Add to batch
            all_states.extend(sequence.states)
            all_actions.extend(sequence.actions)
            all_rewards.extend(sequence.rewards)
            all_next_states.extend(sequence.next_states)
            all_dones.extend(sequence.dones)
            all_opponent_features.extend(sequence.opponent_features)
            all_returns.extend(returns)
        
        if len(all_states) == 0:
            return
        
        # Convert to tensors
        states = torch.stack([torch.FloatTensor(s) for s in all_states]).to(self.agent.device)
        actions = torch.LongTensor(all_actions).to(self.agent.device)
        returns_tensor = torch.FloatTensor(all_returns).to(self.agent.device)
        opponent_features = torch.stack([torch.FloatTensor(of) for of in all_opponent_features]).to(self.agent.device)
        
        # Ensure returns_tensor has correct shape
        if returns_tensor.dim() == 0:
            returns_tensor = returns_tensor.unsqueeze(0)
        if returns_tensor.dim() == 1 and len(returns_tensor) == 1 and states.shape[0] > 1:
            returns_tensor = returns_tensor.expand(states.shape[0])
        
        # With single learner thread, no lock needed
        # Lock only used if multiple threads exist (but we use 1 thread now)
        lock = self.model_lock if self.model_lock else None
        
        # Only lock if multiple threads (shouldn't happen with config, but defensive)
        if lock:
            lock.acquire()
        
        try:
            # Get opponent strategy representations (batch processing)
            self.agent.opponent_model.train()
            opponent_strategies, hand_size_preds, action_preds = self.agent.opponent_model(
                opponent_features, sequence_features=None
            )
            
            # Forward pass through main network
            self.agent.network.train()
            values, policy_logits, mc_values = self.agent.network(states, opponent_strategies)
            
            # Ensure values have correct shape
            if values.dim() > 1:
                values = values.squeeze()
            if values.dim() == 0:
                values = values.unsqueeze(0)
            
            # Calculate losses
            # Value loss
            value_loss = F.mse_loss(values, returns_tensor)
            
            # Policy loss (REINFORCE with baseline)
            advantages = returns_tensor - values.detach()
            log_probs = F.log_softmax(policy_logits, dim=1)
            selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
            
            # Ensure shapes match
            if selected_log_probs.dim() == 0:
                selected_log_probs = selected_log_probs.unsqueeze(0)
            if advantages.dim() == 0:
                advantages = advantages.unsqueeze(0)
            
            policy_loss = -(selected_log_probs * advantages).mean()
            
            # Monte Carlo value loss
            selected_mc_values = mc_values.gather(1, actions.unsqueeze(1)).squeeze()
            
            # Ensure shapes match
            if selected_mc_values.dim() == 0:
                selected_mc_values = selected_mc_values.unsqueeze(0)
            
            mc_loss = F.mse_loss(selected_mc_values, returns_tensor)
            
            # Combined loss
            total_loss = (
                self.agent.value_weight * value_loss + 
                self.agent.policy_weight * policy_loss + 
                self.agent.monte_carlo_weight * mc_loss
            )
            
            # Backward pass
            self.agent.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.agent.network.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.agent.opponent_model.parameters(), self.max_grad_norm)
            
            # Update weights (all in one atomic operation with single thread)
            try:
                self.agent.optimizer.step()
            except (KeyError, RuntimeError) as e:
                # RMSProp optimizer state might not be initialized yet
                # This can happen on first step - reinitialize optimizer if needed
                error_str = str(e).lower()
                if 'square_avg' in error_str or 'exp_avg' in error_str or 'state' in error_str:
                    # Reinitialize optimizer with same parameters from existing optimizer
                    # Extract parameters from existing optimizer's param_groups
                    if isinstance(self.agent.optimizer, torch.optim.RMSprop):
                        optimizer_params = self.agent.optimizer.param_groups[0]
                        lr = optimizer_params['lr']
                        alpha = optimizer_params.get('alpha', 0.99)
                        momentum = optimizer_params.get('momentum', 0.0)
                        eps = optimizer_params.get('eps', 1e-5)
                        weight_decay = optimizer_params.get('weight_decay', 0.0)
                        
                        # Get all parameters (preserve param groups with different LRs)
                        all_params = [
                            {'params': self.agent.network.parameters(), 'lr': self.agent.network_lr},
                            {'params': self.agent.opponent_model.parameters(), 'lr': self.agent.opponent_lr}
                        ]
                    
                        # Recreate optimizer with same parameters
                        self.agent.optimizer = torch.optim.RMSprop(
                            all_params,
                            lr=lr,  # Base LR (param groups override)
                            alpha=alpha,
                            momentum=momentum,
                            eps=eps,
                            weight_decay=weight_decay
                        )
                        # Try step again
                        self.agent.optimizer.step()
                    elif isinstance(self.agent.optimizer, torch.optim.AdamW):
                        optimizer_params = self.agent.optimizer.param_groups[0]
                        lr = optimizer_params['lr']
                        betas = optimizer_params.get('betas', (0.9, 0.999))
                        eps = optimizer_params.get('eps', 1e-8)
                        weight_decay = optimizer_params.get('weight_decay', 0.01)
                        
                        # Get all parameters (preserve param groups with different LRs)
                        all_params = [
                            {'params': self.agent.network.parameters(), 'lr': self.agent.network_lr},
                            {'params': self.agent.opponent_model.parameters(), 'lr': self.agent.opponent_lr}
                        ]
                        
                        # Recreate optimizer with same parameters
                        self.agent.optimizer = torch.optim.AdamW(
                            all_params,
                            lr=lr,  # Base LR (param groups override)
                            betas=betas,
                            eps=eps,
                            weight_decay=weight_decay
                        )
                        # Try step again
                        self.agent.optimizer.step()
                    else:
                        # Unknown optimizer type - just re-raise
                        raise
                else:
                    raise  # Re-raise if different error
        finally:
            # Release lock if we acquired it
            if lock:
                lock.release()
        
        # Update learning rate scheduler if configured
        if hasattr(self.agent, 'scheduler') and self.agent.scheduler is not None:
            # Use mean return as metric for scheduler
            mean_return = returns_tensor.mean().item()
            self.agent.scheduler.step(mean_return)
        
        # Update epsilon
        if self.agent.epsilon > self.agent.epsilon_min:
            self.agent.epsilon *= self.agent.epsilon_decay
        
        # Store statistics
        self.loss_history.append(total_loss.item())
        self.value_loss_history.append(value_loss.item())
        self.policy_loss_history.append(policy_loss.item())
        self.mc_loss_history.append(mc_loss.item())
        
        self.update_count += 1
        
        # Send statistics if queue available
        if self.stats_queue is not None:
            try:
                self.stats_queue.put_nowait({
                    'learner_id': self.learner_id,
                    'update_count': self.update_count,
                    'sequences_processed': self.sequences_processed,
                    'total_loss': total_loss.item(),
                    'value_loss': value_loss.item(),
                    'policy_loss': policy_loss.item(),
                    'mc_loss': mc_loss.item(),
                    'mean_return': returns_tensor.mean().item(),
                    'mean_advantage': advantages.mean().item(),
                    'mean_value': values.mean().item()
                })
            except:
                pass
    
    def stop(self):
        """Stop the learner thread."""
        self.stop_event.set()
        self.join(timeout=5.0)
    
    def get_stats(self) -> Dict:
        """Get learner statistics."""
        return {
            'update_count': self.update_count,
            'sequences_processed': self.sequences_processed,
            'avg_loss': np.mean(self.loss_history) if self.loss_history else 0.0,
            'avg_value_loss': np.mean(self.value_loss_history) if self.value_loss_history else 0.0,
            'avg_policy_loss': np.mean(self.policy_loss_history) if self.policy_loss_history else 0.0,
            'avg_mc_loss': np.mean(self.mc_loss_history) if self.mc_loss_history else 0.0
        }
    
    def get_model_state_dict(self) -> Dict:
        """Get current model state dictionary for broadcasting to actors."""
        return {
            'network_state_dict': self.agent.network.state_dict(),
            'opponent_model_state_dict': self.agent.opponent_model.state_dict(),
            'epsilon': self.agent.epsilon
        }

