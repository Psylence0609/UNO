"""
Advanced DMC Agent with Sophisticated Opponent Modeling.
Uses attention mechanisms, gated fusion, and optimized for MPS GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Optional, List, Tuple

from .dmc_agent import DMCAgent
from .opponent_modeling_advanced import (
    AdvancedOpponentModelingNetwork,
    AdvancedDMCNetworkWithOpponent
)
from ..features.opponent_features import OpponentFeatureExtractor


class AdvancedDMCAgentWithOpponent(DMCAgent):
    """
    Advanced DMC Agent with sophisticated opponent modeling.
    Optimized for MPS GPU with batch processing and parallel evaluation.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        config: dict,
        opponent_feature_size: Optional[int] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize advanced DMC agent with opponent modeling.
        
        Args:
            state_size: Dimension of input state
            action_size: Dimension of action space
            config: Configuration parameters
            opponent_feature_size: Size of opponent feature vector (auto-detected if None)
            device: PyTorch device (CPU/GPU/MPS)
        """
        # Initialize opponent feature extractor
        opponent_config = config.get('opponent_modeling', {})
        history_size = opponent_config.get('history_size', 20)
        
        env_config = config.get('environment', {})
        num_players = env_config.get('num_players', 2)
        
        self.opponent_extractor = OpponentFeatureExtractor(
            history_size=history_size,
            num_actions=action_size
        )
        
        # Initialize extractor for all players
        self.opponent_extractor.reset(num_players=num_players)
        
        # Determine opponent feature size
        if opponent_feature_size is None:
            opponent_feature_size = self.opponent_extractor.get_feature_size()
        
        # Advanced opponent modeling configuration
        opponent_hidden_layers = opponent_config.get('hidden_layers', [256, 128, 64])
        strategy_dim = opponent_config.get('strategy_dim', 64)
        num_attention_heads = opponent_config.get('num_attention_heads', 4)
        use_attention = opponent_config.get('use_attention', True)
        use_temporal = opponent_config.get('use_temporal', True)
        opponent_dropout = opponent_config.get('dropout', 0.1)
        use_gated_fusion = opponent_config.get('use_gated_fusion', True)
        
        # Store config
        self.opponent_feature_size = opponent_feature_size
        self.strategy_dim = strategy_dim
        self.sequence_feature_size = self.opponent_extractor.get_sequence_feature_size()
        
        # Initialize base attributes (without calling super().__init__ to avoid creating base network)
        self.state_size = state_size
        self.action_size = action_size
        
        # Get device (MPS, CUDA, or CPU)
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        
        print(f"Advanced DMC with Opponent Modeling using device: {self.device}")
        
        # Hyperparameters
        training_config = config.get('training', {})
        network_config = config.get('network', {})
        dmc_config = config.get('dmc', {})
        
        self.learning_rate = training_config.get('learning_rate', 0.0005)
        self.gamma = training_config.get('gamma', 0.99)
        self.epsilon = training_config.get('epsilon_start', 0.8)
        self.epsilon_min = training_config.get('epsilon_end', 0.05)
        self.epsilon_decay = training_config.get('epsilon_decay', 0.9995)
        
        # DMC parameters
        self.monte_carlo_rollouts = dmc_config.get('monte_carlo_rollouts', 10)
        self.temperature = dmc_config.get('temperature', 1.0)
        self.monte_carlo_weight = dmc_config.get('mc_weight', 0.5)
        self.policy_weight = dmc_config.get('policy_weight', 0.3)
        self.value_weight = dmc_config.get('value_weight', 0.2)
        
        # Network configuration
        hidden_layers = network_config.get('hidden_layers', [512, 256, 128])
        activation = network_config.get('activation', 'relu')
        dropout = network_config.get('dropout', 0.1)
        
        # Create advanced network
        self.network = AdvancedDMCNetworkWithOpponent(
            state_size=state_size,
            opponent_strategy_dim=strategy_dim,
            action_size=action_size,
            hidden_layers=hidden_layers,
            activation=activation,
            dropout=dropout,
            use_gated_fusion=use_gated_fusion
        ).to(self.device)
        
        # Initialize advanced opponent modeling network
        self.opponent_model = AdvancedOpponentModelingNetwork(
            feature_size=opponent_feature_size,
            hidden_layers=opponent_hidden_layers,
            strategy_dim=strategy_dim,
            num_attention_heads=num_attention_heads,
            use_attention=use_attention,
            use_temporal=use_temporal,
            dropout=opponent_dropout,
            sequence_feature_size=self.sequence_feature_size
        ).to(self.device)
        
        # Optimizer configuration (matching RLCard DMC)
        optimizer_config = config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'rmsprop').lower()
        
        # Separate learning rates for network and opponent model
        self.network_lr = self.learning_rate
        self.opponent_lr = self.learning_rate * 0.5  # Lower LR for opponent model
        
        all_params = [
            {'params': self.network.parameters(), 'lr': self.network_lr},
            {'params': self.opponent_model.parameters(), 'lr': self.opponent_lr}
        ]
        
        # Use RMSProp optimizer (matching RLCard DMC)
        if optimizer_type == 'rmsprop':
            # RMSProp with per-parameter-group learning rates (matching RLCard DMC)
            # Convert config values to appropriate types (YAML may read scientific notation as string)
            alpha = float(optimizer_config.get('alpha', 0.99))
            momentum = float(optimizer_config.get('momentum', 0.0))
            eps = float(optimizer_config.get('eps', 1e-5))
            weight_decay = float(optimizer_config.get('weight_decay', 0.0))
            
            self.optimizer = torch.optim.RMSprop(
                all_params,
                lr=self.learning_rate,  # Base LR - param groups will override with their own lr
                alpha=alpha,
                momentum=momentum,
                eps=eps,
                weight_decay=weight_decay
            )
        else:
            # Fallback to AdamW if specified
            self.optimizer = torch.optim.AdamW(
                all_params,
                weight_decay=optimizer_config.get('weight_decay', 1e-5),
                eps=optimizer_config.get('eps', 1e-8)
            )
        
        # Learning rate scheduler (configured from config file)
        scheduler_config = config.get('learning_rate_scheduler', {})
        if scheduler_config.get('enabled', True):
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=scheduler_config.get('mode', 'max'),
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 500),
                min_lr=scheduler_config.get('min_lr', 1e-6)
            )
        else:
            # Dummy scheduler that does nothing
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=1.0, patience=1e9, min_lr=1e-10
            )
        
        # Episode data
        self.mc_returns = defaultdict(list)
        self.episode_data = []
        self.t_step = 0
        self.accumulation_step = 0  # Track gradient accumulation
        
        # Opponent tracking
        self.current_opponent_id = 1
        
        # Batch processing for efficiency
        self.batch_size = training_config.get('batch_size', 64)
        self.gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', 1)
        
        # Gradient clipping (matching RLCard DMC - 40.0 instead of 1.0)
        self.max_grad_norm = training_config.get('max_grad_norm', 40.0)
        
        # Initialize optimizer with zero gradients
        self.optimizer.zero_grad()
    
    def reset_opponent_tracking(self, num_players: int = 2):
        """Reset opponent tracking for a new game."""
        self.opponent_extractor.reset(num_players)
        self.current_opponent_id = 1
    
    def update_opponent_history(self, action: int, opponent_id: int, hand_size: Optional[int] = None):
        """Update opponent history with new action."""
        self.opponent_extractor.update_history(action, opponent_id, hand_size)
    
    def extract_opponent_features(self, state: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Extract opponent features from current state."""
        static = self.opponent_extractor.extract_features(state, self.current_opponent_id)
        sequence = self.opponent_extractor.extract_sequence_features(self.current_opponent_id)
        return static, sequence
    
    def _process_state(self, state):
        """Process state for the network."""
        if isinstance(state, dict):
            features = []
            if 'obs' in state:
                features.extend(state['obs'].flatten())
            
            if 'legal_actions' in state:
                legal_mask = np.zeros(self.action_size)
                if hasattr(state['legal_actions'], 'keys'):
                    legal_indices = list(state['legal_actions'].keys())
                else:
                    legal_indices = state['legal_actions']
                legal_mask[legal_indices] = 1
                features.extend(legal_mask)
            else:
                features.extend(np.ones(self.action_size))
            
            return np.array(features, dtype=np.float32)
        else:
            return np.array(state, dtype=np.float32)
    
    def act(self, state, legal_actions: list, epsilon: Optional[float] = None) -> int:
        """Choose action using epsilon-greedy policy with opponent modeling."""
        if epsilon is None:
            epsilon = self.epsilon
        
        # Process state and opponent features
        # Process state and opponent features
        if isinstance(state, dict):
            state_features = self._process_state(state)
            opponent_features, sequence_features = self.extract_opponent_features(state)
        else:
            state_features = state
            opponent_features = np.zeros(self.opponent_feature_size)
            sequence_features = np.zeros((self.opponent_extractor.history_size, self.sequence_feature_size))
        
        # Convert to tensors
        state_tensor = torch.FloatTensor(state_features).unsqueeze(0).to(self.device)
        opponent_tensor = torch.FloatTensor(opponent_features).unsqueeze(0).to(self.device)
        sequence_tensor = torch.FloatTensor(sequence_features).unsqueeze(0).to(self.device)
        
        # Get opponent strategy representation
        self.opponent_model.eval()
        with torch.no_grad():
            opponent_strategy, _, _ = self.opponent_model(opponent_tensor, sequence_features=sequence_tensor)
        
        # Epsilon-greedy action selection
        if np.random.random() > epsilon:
            self.network.eval()
            with torch.no_grad():
                value, policy_logits, mc_values = self.network(state_tensor, opponent_strategy)
                
                # Combine policy and MC values
                policy_probs = F.softmax(policy_logits / self.temperature, dim=1)
                combined_values = (
                    self.policy_weight * policy_probs.squeeze() + 
                    self.monte_carlo_weight * mc_values.squeeze() +
                    self.value_weight * value.squeeze()
                )
                
                # Mask illegal actions
                masked_values = combined_values.clone()
                mask = torch.full((self.action_size,), float('-inf'), device=self.device)
                mask[legal_actions] = 0
                masked_values += mask
                
                action = masked_values.argmax().item()
            self.network.train()
        else:
            action = np.random.choice(legal_actions)
        
        return action
    
    def use_raw(self, state: Dict) -> int:
        """Interface for RLCard compatibility."""
        if isinstance(state, dict) and 'legal_actions' in state:
            legal_actions = list(state['legal_actions'].keys())
        else:
            legal_actions = list(range(self.action_size))
        return self.act(state, legal_actions)
    
    def store_transition(
        self,
        state: Dict,
        action: int,
        reward: float,
        next_state: Dict,
        done: bool,
        opponent_features: Tuple[np.ndarray, np.ndarray]
    ):
        """Store transition for episode-based learning."""
        state_features = self._process_state(state)
        static_features, sequence_features = opponent_features
        self.episode_data.append({
            'state': state_features,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'opponent_features': static_features,
            'sequence_features': sequence_features
        })
    
    def learn_episode(self) -> dict:
        """Learn from episode with batch processing and gradient accumulation."""
        if not self.episode_data:
            return {}
        
        # Calculate Monte Carlo returns
        returns = []
        G = 0
        for transition in reversed(self.episode_data):
            G = transition['reward'] + self.gamma * G
            returns.insert(0, G)
        
        # Convert to tensors (batch processing) - use torch operations for MPS optimization
        states_list = [t['state'] for t in self.episode_data]
        actions_list = [t['action'] for t in self.episode_data]
        returns_list = returns
        opponent_features_list = [t['opponent_features'] for t in self.episode_data]
        
        # Stack into tensors (more efficient for MPS)
        states = torch.stack([torch.FloatTensor(s) for s in states_list]).to(self.device)
        actions = torch.LongTensor(actions_list).to(self.device)
        returns_tensor = torch.FloatTensor(returns_list).to(self.device)
        opponent_features = torch.stack([torch.FloatTensor(of) for of in opponent_features_list]).to(self.device)
        
        sequence_features_list = [t['sequence_features'] for t in self.episode_data]
        sequence_features = torch.stack([torch.FloatTensor(sf) for sf in sequence_features_list]).to(self.device)
        
        # Get opponent strategy representations (batch processing)
        self.opponent_model.train()
        opponent_strategies, hand_size_preds, action_preds = self.opponent_model(
            opponent_features, sequence_features=sequence_features
        )
        
        # Forward pass through main network
        self.network.train()
        values, policy_logits, mc_values = self.network(states, opponent_strategies)
        
        # Calculate losses
        # Value loss
        value_loss = F.mse_loss(values.squeeze(), returns_tensor)
        
        # Policy loss (REINFORCE with baseline)
        advantages = returns_tensor - values.squeeze().detach()
        log_probs = F.log_softmax(policy_logits, dim=1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        policy_loss = -(selected_log_probs * advantages).mean()
        
        # Monte Carlo value loss
        selected_mc_values = mc_values.gather(1, actions.unsqueeze(1)).squeeze()
        mc_loss = F.mse_loss(selected_mc_values, returns_tensor)
        
        # Combined loss
        total_loss = (
            self.value_weight * value_loss + 
            self.policy_weight * policy_loss + 
            self.monte_carlo_weight * mc_loss
        )
        
        # Backward pass with gradient accumulation
        # Scale loss by accumulation steps
        scaled_loss = total_loss / self.gradient_accumulation_steps
        scaled_loss.backward()
        
        self.accumulation_step += 1
        
        # Update weights when we've accumulated enough gradients
        if self.accumulation_step >= self.gradient_accumulation_steps:
            # Gradient clipping for stability (matching RLCard DMC - 40.0 instead of 1.0)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.opponent_model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.accumulation_step = 0
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Clear episode data
        effective_loss = total_loss.item() * self.gradient_accumulation_steps if self.gradient_accumulation_steps > 1 else total_loss.item()
        stats = {
            'total_loss': effective_loss,
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item(),
            'mc_loss': mc_loss.item(),
            'mean_return': returns_tensor.mean().item(),
            'mean_advantage': advantages.mean().item()
        }
        self.episode_data = []
        self.t_step += 1
        
        return stats
    
    def save(self, filepath: str):
        """Save the model with all components."""
        import os
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'opponent_model_state_dict': self.opponent_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            't_step': self.t_step,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'opponent_feature_size': self.opponent_feature_size,
            'sequence_feature_size': self.sequence_feature_size,
            'strategy_dim': self.strategy_dim
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
    
    def load(self, filepath: str):
        """Load the model with all components."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.opponent_model.load_state_dict(checkpoint['opponent_model_state_dict'])
        
        # Try to load optimizer state (may fail if optimizer type changed, e.g., AdamW -> RMSProp)
        optimizer_loaded = False
        try:
            # Check if the saved optimizer type matches current optimizer type
            saved_state = checkpoint['optimizer_state_dict']
            
            # Try to detect optimizer type mismatch by checking state structure
            # RMSprop uses 'square_avg', Adam/AdamW use 'exp_avg' and 'exp_avg_sq'
            is_rmsprop_current = isinstance(self.optimizer, torch.optim.RMSprop)
            
            # Check saved state to detect type
            if 'state' in saved_state and saved_state['state']:
                # Get first parameter state to check keys
                first_state = next(iter(saved_state['state'].values()))
                has_square_avg = 'square_avg' in first_state
                has_exp_avg = 'exp_avg' in first_state
                
                # Check compatibility
                if is_rmsprop_current and not has_square_avg:
                    raise ValueError("Optimizer type mismatch: checkpoint has Adam/AdamW state but current is RMSprop")
                elif not is_rmsprop_current and has_square_avg:
                    raise ValueError("Optimizer type mismatch: checkpoint has RMSprop state but current is Adam/AdamW")
            
            # If we got here, types match - load the state
            self.optimizer.load_state_dict(saved_state)
            optimizer_loaded = True
            print(" Successfully loaded optimizer state")
        except (KeyError, ValueError, RuntimeError) as e:
            print(f"  Warning: Could not load optimizer state: {e}")
            print("   Initializing fresh optimizer state (this is normal when changing optimizer types)")
        
        # If optimizer wasn't loaded or there was an error, ensure clean state
        if not optimizer_loaded:
            # Reset optimizer state completely to avoid partial state corruption
            self.optimizer.state = defaultdict(dict)
            self.optimizer.zero_grad()
        
        # Try to load scheduler state
        if 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except (KeyError, ValueError, RuntimeError) as e:
                print(f"  Warning: Could not load scheduler state: {e}")
                print("   Initializing fresh scheduler state")
        
        self.epsilon = checkpoint.get('epsilon', 0.0)
        self.t_step = checkpoint.get('t_step', 0)
        self.network.to(self.device)
        self.opponent_model.to(self.device)
    
    def load_state_dict(self, state_dict: Dict):
        """
        Load model state dictionary for synchronization (used in actor-learner architecture).
        
        Args:
            state_dict: Dictionary containing 'network_state_dict' and 'opponent_model_state_dict'
        """
        if 'network_state_dict' in state_dict:
            self.network.load_state_dict(state_dict['network_state_dict'])
        if 'opponent_model_state_dict' in state_dict:
            self.opponent_model.load_state_dict(state_dict['opponent_model_state_dict'])
        if 'epsilon' in state_dict:
            self.epsilon = state_dict['epsilon']
        
        # Ensure models are on correct device
        self.network.to(self.device)
        self.opponent_model.to(self.device)

