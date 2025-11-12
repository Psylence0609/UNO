"""
Deep Monte Carlo (DMC) Agent with Opponent Modeling for UNO.
Extends DMC agent to include opponent modeling capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from collections import defaultdict
from typing import Dict, Optional

from .dmc_agent import DMCAgent, DMCNetwork
from .opponent_modeling import OpponentModelingNetwork
from ..features.opponent_features import OpponentFeatureExtractor


class DMCNetworkWithOpponent(nn.Module):
    """
    Deep Monte Carlo Network with opponent modeling.
    Extends DMCNetwork to accept opponent features and fuse them with state features.
    """
    
    def __init__(
        self,
        state_size: int,
        opponent_feature_size: int,
        action_size: int,
        hidden_layers: list = [256, 128],
        activation: str = 'relu',
        dropout: float = 0.1
    ):
        """
        Initialize DMC network with opponent modeling.
        
        Args:
            state_size: Dimension of input state features
            opponent_feature_size: Dimension of opponent strategy representation
            action_size: Dimension of action space
            hidden_layers: List of hidden layer sizes
            activation: Activation function
            dropout: Dropout probability
        """
        super(DMCNetworkWithOpponent, self).__init__()
        
        self.state_size = state_size
        self.opponent_feature_size = opponent_feature_size
        self.action_size = action_size
        self.combined_size = state_size + opponent_feature_size
        
        # Choose activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Fusion layer: combine state and opponent features
        # Use first hidden layer size for fusion
        fusion_size = hidden_layers[0] if len(hidden_layers) > 0 else 256
        self.fusion_layer = nn.Linear(self.combined_size, fusion_size)
        
        # Build network layers (starting from fusion output)
        layers = []
        prev_size = fusion_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        self.feature_layers = nn.ModuleList(layers)
        
        # Output layers for different components (same as original DMC)
        self.value_head = nn.Linear(prev_size, 1)  # State value
        self.policy_head = nn.Linear(prev_size, action_size)  # Action probabilities
        self.monte_carlo_head = nn.Linear(prev_size, action_size)  # MC value estimates
    
    def forward(self, state_features: torch.Tensor, opponent_features: torch.Tensor) -> tuple:
        """
        Forward pass through the network.
        
        Args:
            state_features: State features tensor [batch_size, state_size]
            opponent_features: Opponent strategy features tensor [batch_size, opponent_feature_size]
            
        Returns:
            Tuple of (value, policy_logits, mc_values)
        """
        # Concatenate state and opponent features
        combined_features = torch.cat([state_features, opponent_features], dim=1)
        
        # Fusion layer
        x = self.fusion_layer(combined_features)
        x = self.activation(x)
        
        # Pass through feature layers
        for i in range(0, len(self.feature_layers), 2):
            x = self.feature_layers[i](x)  # Linear layer
            x = self.activation(x)
            if i + 1 < len(self.feature_layers):
                x = self.feature_layers[i + 1](x)  # Dropout layer
        
        # Compute outputs
        value = self.value_head(x)
        policy_logits = self.policy_head(x)
        mc_values = self.monte_carlo_head(x)
        
        return value, policy_logits, mc_values


class DMCAgentWithOpponentModeling(DMCAgent):
    """
    DMC Agent with opponent modeling capabilities.
    Extends DMCAgent to include opponent feature extraction and modeling.
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
        Initialize DMC agent with opponent modeling.
        
        Args:
            state_size: Dimension of input state
            action_size: Dimension of action space
            config: Configuration parameters
            opponent_feature_size: Size of opponent feature vector (auto-detected if None)
            device: PyTorch device (CPU/GPU)
        """
        # Initialize opponent feature extractor
        opponent_config = config.get('opponent_modeling', {})
        history_size = opponent_config.get('history_size', 20)
        
        self.opponent_extractor = OpponentFeatureExtractor(
            history_size=history_size,
            num_actions=action_size
        )
        
        # Determine opponent feature size
        if opponent_feature_size is None:
            opponent_feature_size = self.opponent_extractor.get_feature_size()
        
        # Initialize opponent modeling network
        opponent_hidden_layers = opponent_config.get('hidden_layers', [128, 64])
        strategy_dim = opponent_config.get('strategy_dim', 32)
        opponent_activation = opponent_config.get('activation', 'relu')
        opponent_dropout = opponent_config.get('dropout', 0.1)
        
        # Store opponent modeling config for network initialization
        self.opponent_feature_size = opponent_feature_size
        self.strategy_dim = strategy_dim
        
        # Initialize base attributes without calling super().__init__()
        # (to avoid creating the base network)
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
            
        print(f"DMC with Opponent Modeling using device: {self.device}")
        
        # Hyperparameters - get from training section first, then fallback
        training_config = config.get('training', {})
        network_config = config.get('network', {})
        dmc_config = config.get('dmc', {})
        
        self.learning_rate = training_config.get('learning_rate', 0.0005)
        self.gamma = training_config.get('gamma', 0.99)
        self.epsilon = training_config.get('epsilon_start', 0.8)
        self.epsilon_min = training_config.get('epsilon_end', 0.05)
        self.epsilon_decay = training_config.get('epsilon_decay', 0.9995)
        
        # DMC specific parameters
        self.monte_carlo_rollouts = dmc_config.get('monte_carlo_rollouts', 10)
        self.temperature = dmc_config.get('temperature', 1.0)
        self.monte_carlo_weight = dmc_config.get('mc_weight', 0.5)
        self.policy_weight = dmc_config.get('policy_weight', 0.3)
        self.value_weight = dmc_config.get('value_weight', 0.2)
        
        # Network configuration
        hidden_layers = network_config.get('hidden_layers', [256, 128])
        activation = network_config.get('activation', 'relu')
        dropout = network_config.get('dropout', 0.1)
        
        # Create opponent-aware network
        self.network = DMCNetworkWithOpponent(
            state_size=state_size,
            opponent_feature_size=strategy_dim,  # Use strategy dim, not raw feature size
            action_size=action_size,
            hidden_layers=hidden_layers,
            activation=activation,
            dropout=dropout
        ).to(self.device)
        
        # Initialize opponent modeling network
        self.opponent_model = OpponentModelingNetwork(
            feature_size=opponent_feature_size,
            hidden_layers=opponent_hidden_layers,
            strategy_dim=strategy_dim,
            activation=opponent_activation,
            dropout=opponent_dropout
        ).to(self.device)
        
        # Optimizer for both networks
        all_params = list(self.network.parameters()) + list(self.opponent_model.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=self.learning_rate)
        
        # Monte Carlo statistics
        self.mc_returns = defaultdict(list)
        self.episode_data = []
        
        # Training step counter
        self.t_step = 0
        
        # Track current opponent ID (for 2-player games, this is always 1)
        self.current_opponent_id = 1
        
    def reset_opponent_tracking(self, num_players: int = 2):
        """
        Reset opponent tracking for a new game.
        
        Args:
            num_players: Number of players in the game
        """
        self.opponent_extractor.reset(num_players)
        self.current_opponent_id = 1  # In 2-player game, opponent is always player 1
    
    def update_opponent_history(self, action: int, opponent_id: int, hand_size: Optional[int] = None):
        """
        Update opponent history with new action.
        
        Args:
            action: Action taken by opponent
            opponent_id: ID of the opponent player
            hand_size: Current hand size of opponent (if available)
        """
        self.opponent_extractor.update_history(action, opponent_id, hand_size)
    
    def extract_opponent_features(self, state: Dict) -> np.ndarray:
        """
        Extract opponent features from current state.
        
        Args:
            state: Current game state (RLCard format)
            
        Returns:
            Opponent feature vector
        """
        return self.opponent_extractor.extract_features(state, self.current_opponent_id)
    
    def _process_state(self, state):
        """
        Process state for the network (same as base DMC).
        
        Args:
            state: Current game state
            
        Returns:
            Processed state vector
        """
        if isinstance(state, dict):
            # Extract features from RLCard state
            features = []
            if 'obs' in state:
                features.extend(state['obs'].flatten())
            
            # Add legal actions mask
            if 'legal_actions' in state:
                legal_mask = np.zeros(self.action_size)
                # Handle legal_actions format (could be list, array, or OrderedDict)
                if hasattr(state['legal_actions'], 'keys'):
                    # OrderedDict format - use keys as indices
                    legal_indices = list(state['legal_actions'].keys())
                else:
                    # List or array format
                    legal_indices = state['legal_actions']
                legal_mask[legal_indices] = 1
                features.extend(legal_mask)
            else:
                features.extend(np.ones(self.action_size))
            
            return np.array(features, dtype=np.float32)
        else:
            return np.array(state, dtype=np.float32)
    
    def _process_state_with_opponent(self, state: Dict) -> tuple:
        """
        Process state and extract opponent features.
        
        Args:
            state: Current game state
            
        Returns:
            Tuple of (state_features, opponent_features)
        """
        # Process state features (same as base DMC)
        state_features = self._process_state(state)
        
        # Extract opponent features
        opponent_features = self.extract_opponent_features(state)
        
        return state_features, opponent_features
    
    def act(self, state, legal_actions: list, epsilon: Optional[float] = None) -> int:
        """
        Choose action using epsilon-greedy policy with opponent modeling.
        
        Args:
            state: Current game state (dict or processed state vector)
            legal_actions: List of legal actions
            epsilon: Exploration rate (uses self.epsilon if None)
            
        Returns:
            Selected action
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        # Process state and opponent features
        if isinstance(state, dict):
            state_features, opponent_features = self._process_state_with_opponent(state)
        else:
            # If state is already processed, we need opponent features from state dict
            # For now, use zero features if state is already processed
            state_features = state
            opponent_features = np.zeros(self.opponent_feature_size)
        
        # Convert to tensors
        state_tensor = torch.FloatTensor(state_features).unsqueeze(0).to(self.device)
        opponent_tensor = torch.FloatTensor(opponent_features).unsqueeze(0).to(self.device)
        
        # Get opponent strategy representation
        self.opponent_model.eval()
        with torch.no_grad():
            opponent_strategy = self.opponent_model(opponent_tensor)
        
        # Epsilon-greedy action selection
        if np.random.random() > epsilon:
            # Exploitation: use DMC policy with opponent awareness
            self.network.eval()
            with torch.no_grad():
                value, policy_logits, mc_values = self.network(state_tensor, opponent_strategy)
                
                # Combine policy and MC values (same as base DMC)
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
            # Exploration: random legal action
            action = np.random.choice(legal_actions)
        
        return action
    
    def use_raw(self, state: Dict) -> int:
        """
        Interface for RLCard compatibility with opponent modeling.
        
        Args:
            state: RLCard state dictionary
            
        Returns:
            Selected action
        """
        # Extract legal actions
        if isinstance(state, dict) and 'legal_actions' in state:
            legal_actions = list(state['legal_actions'].keys())
        else:
            legal_actions = list(range(self.action_size))
        
        # Update opponent hand size if available
        if 'raw_obs' in state and 'num_cards' in state['raw_obs']:
            opponent_hand_size = state['raw_obs']['num_cards'].get(self.current_opponent_id)
            if opponent_hand_size is not None:
                # Store for later update (after we see opponent's action)
                pass
        
        return self.act(state, legal_actions)
    
    def store_transition(self, state: Dict, action: int, reward: float, next_state: Dict, done: bool, opponent_features: np.ndarray):
        """
        Store transition for episode-based learning with opponent features.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            opponent_features: Opponent features at this step
        """
        # Process state to get state features
        state_features = self._process_state(state)
        
        self.episode_data.append({
            'state': state_features,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'opponent_features': opponent_features
        })
    
    def learn_episode(self):
        """
        Learn from a complete episode using Monte Carlo returns with opponent modeling.
        
        Returns:
            dict: Learning statistics
        """
        if not self.episode_data:
            return {}
        
        # Calculate Monte Carlo returns
        returns = []
        G = 0
        for transition in reversed(self.episode_data):
            G = transition['reward'] + self.gamma * G
            returns.insert(0, G)
        
        # Convert to tensors
        states = torch.FloatTensor([t['state'] for t in self.episode_data]).to(self.device)
        actions = torch.LongTensor([t['action'] for t in self.episode_data]).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        opponent_features = torch.FloatTensor([t['opponent_features'] for t in self.episode_data]).to(self.device)
        
        # Get opponent strategy representations (train opponent model too)
        self.opponent_model.train()
        opponent_strategies = self.opponent_model(opponent_features)
        
        # Forward pass
        self.network.train()
        values, policy_logits, mc_values = self.network(states, opponent_strategies)
        
        # Calculate losses (same as base DMC)
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
        
        # Combined loss (same weighting as base DMC)
        total_loss = (self.value_weight * value_loss + 
                     self.policy_weight * policy_loss + 
                     self.monte_carlo_weight * mc_loss)
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.opponent_model.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Clear episode data
        stats = {
            'total_loss': total_loss.item(),
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
        """
        Save the model with opponent modeling components.
        
        Args:
            filepath: Path to save the model
        """
        import os
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'opponent_model_state_dict': self.opponent_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'opponent_feature_size': self.opponent_feature_size,
            'strategy_dim': self.strategy_dim
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
    
    def load(self, filepath: str):
        """
        Load the model with opponent modeling components.
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.opponent_model.load_state_dict(checkpoint['opponent_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', 0.0)
        
        # Move to correct device
        self.network.to(self.device)
        self.opponent_model.to(self.device)

