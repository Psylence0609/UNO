"""
Deep Monte Carlo (DMC) Agent for UNO.
Combines Monte Carlo methods with deep neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from collections import defaultdict

class DMCNetwork(nn.Module):
    """Deep Monte Carlo Network architecture."""
    
    def __init__(self, state_size, action_size, hidden_layers=[512, 256], 
                 activation='relu', dropout=0.1):
        """
        Initialize the DMC network.
        
        Args:
            state_size (int): Dimension of input state
            action_size (int): Dimension of action space
            hidden_layers (list): List of hidden layer sizes
            activation (str): Activation function
            dropout (float): Dropout probability
        """
        super(DMCNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Choose activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network layers
        layers = []
        prev_size = state_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        self.feature_layers = nn.ModuleList(layers)
        
        # Output layers for different components
        self.value_head = nn.Linear(prev_size, 1)  # State value
        self.policy_head = nn.Linear(prev_size, action_size)  # Action probabilities
        self.monte_carlo_head = nn.Linear(prev_size, action_size)  # MC value estimates
    
    def forward(self, x):
        """Forward pass through the network."""
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


class DMCAgent:
    """Deep Monte Carlo Agent for UNO."""
    
    def __init__(self, state_size, action_size, config, device=None):
        """
        Initialize the DMC agent.
        
        Args:
            state_size (int): Dimension of input state
            action_size (int): Dimension of action space
            config (dict): Configuration parameters
            device: PyTorch device (CPU/GPU)
        """
        self.state_size = state_size
        self.action_size = action_size
        
        # Get device (MPS, CUDA, or CPU)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        print(f"DMC using device: {self.device}")
        
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
        
        # Network - use consistent architecture
        hidden_layers = network_config.get('hidden_layers', [256, 128])
        activation = network_config.get('activation', 'relu')
        dropout = network_config.get('dropout', 0.1)
        
        self.network = DMCNetwork(
            state_size, action_size,
            hidden_layers=hidden_layers,
            activation=activation,
            dropout=dropout
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        
        # Monte Carlo statistics
        self.mc_returns = defaultdict(list)
        self.episode_data = []
        
        # Training step counter
        self.t_step = 0
    
    def act(self, state, legal_actions, epsilon=None):
        """
        Choose an action using DMC policy.
        
        Args:
            state: Current state
            legal_actions: List of legal action indices
            epsilon: Exploration probability (if None, use current epsilon)
            
        Returns:
            int: Selected action
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Epsilon-greedy with DMC policy
        if np.random.random() > epsilon:
            # Exploitation: use DMC policy
            self.network.eval()
            with torch.no_grad():
                value, policy_logits, mc_values = self.network(state_tensor)
                
                # Combine policy and MC values
                policy_probs = F.softmax(policy_logits / self.temperature, dim=1)
                combined_values = (self.policy_weight * policy_probs.squeeze() + 
                                 self.monte_carlo_weight * mc_values.squeeze() +
                                 self.value_weight * value.squeeze())
                
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
    
    def use_raw(self, state):
        """
        Interface for RLCard compatibility.
        
        Args:
            state: RLCard state dictionary
            
        Returns:
            int: Selected action
        """
        # Extract legal actions
        if isinstance(state, dict) and 'legal_actions' in state:
            legal_actions = list(state['legal_actions'].keys())
        else:
            legal_actions = list(range(self.action_size))
        
        # Convert state to our format
        state_vector = self._process_state(state)
        
        return self.act(state_vector, legal_actions)
    
    def eval_step(self, state):
        """
        Evaluation step for RLCard compatibility.
        
        Args:
            state: RLCard state dictionary
            
        Returns:
            tuple: (action, probs)
        """
        # Extract legal actions
        if isinstance(state, dict) and 'legal_actions' in state:
            legal_actions = list(state['legal_actions'].keys())
        else:
            legal_actions = list(range(self.action_size))
        
        # Convert state to our format
        state_vector = self._process_state(state)
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
        
        # Get network outputs
        self.network.eval()
        with torch.no_grad():
            value, policy_logits, mc_values = self.network(state_tensor)
            
            # Get action probabilities
            policy_probs = F.softmax(policy_logits, dim=1).squeeze().cpu().numpy()
            
            # Mask illegal actions
            masked_probs = np.zeros(self.action_size)
            for action in legal_actions:
                masked_probs[action] = policy_probs[action]
            
            # Normalize
            if masked_probs.sum() > 0:
                masked_probs = masked_probs / masked_probs.sum()
            else:
                # Fallback to uniform
                for action in legal_actions:
                    masked_probs[action] = 1.0 / len(legal_actions)
            
            # Select action
            action = np.random.choice(range(self.action_size), p=masked_probs)
        
        self.network.train()
        return action, masked_probs
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store transition for episode-based learning.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.episode_data.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
    
    def learn_episode(self):
        """
        Learn from a complete episode using Monte Carlo returns.
        
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
        
        # Forward pass
        values, policy_logits, mc_values = self.network(states)
        
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
        total_loss = (self.value_weight * value_loss + 
                     self.policy_weight * policy_loss + 
                     self.monte_carlo_weight * mc_loss)
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Clear episode data
        self.episode_data = []
        self.t_step += 1
        
        return {
            'total_loss': total_loss.item(),
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item(),
            'mc_loss': mc_loss.item(),
            'mean_return': returns_tensor.mean().item(),
            'mean_advantage': advantages.mean().item()
        }
    
    def _process_state(self, state):
        """
        Process RLCard state into a feature vector.
        
        Args:
            state: RLCard state dictionary
            
        Returns:
            np.array: Processed state vector
        """
        if isinstance(state, dict):
            features = []
            
            # Add observation tensor (flattened)
            if 'obs' in state:
                obs = state['obs']
                features.extend(obs.flatten())
            
            # Add legal actions mask
            legal_mask = np.zeros(self.action_size)
            if 'legal_actions' in state:
                legal_actions = state['legal_actions']
                if hasattr(legal_actions, 'keys'):
                    for action in legal_actions.keys():
                        legal_mask[action] = 1
                
            features.extend(legal_mask)
            
            return np.array(features, dtype=np.float32)
        
        else:
            # If state is already a vector
            return np.array(state, dtype=np.float32)
    
    def save(self, filepath):
        """Save the model."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            't_step': self.t_step
        }, filepath)
    
    def load(self, filepath):
        """Load the model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.t_step = checkpoint['t_step']
    
    def set_device(self, device):
        """Set device for compatibility."""
        self.device = device
        self.network.to(device)
    
    def train(self):
        """Set to training mode."""
        self.network.train()
    
    def eval(self):
        """Set to evaluation mode."""
        self.network.eval()