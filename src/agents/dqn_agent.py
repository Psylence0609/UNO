"""
Deep Q-Network (DQN) implementation for UNO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):
    """Deep Q-Network architecture."""
    
    def __init__(self, state_size, action_size, hidden_layers=[512, 256, 128], 
                 activation='relu', dropout=0.1, dueling=True):
        """
        Initialize the DQN network.
        
        Args:
            state_size (int): Dimension of input state
            action_size (int): Dimension of action space
            hidden_layers (list): List of hidden layer sizes
            activation (str): Activation function ('relu', 'tanh', 'leaky_relu')
            dropout (float): Dropout probability
            dueling (bool): Whether to use dueling architecture
        """
        super(DQN, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.dueling = dueling
        
        # Choose activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build the network layers
        layers = []
        prev_size = state_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        self.feature_layers = nn.ModuleList(layers)
        
        if dueling:
            # Dueling DQN architecture
            self.value_stream = nn.Linear(prev_size, 1)
            self.advantage_stream = nn.Linear(prev_size, action_size)
        else:
            # Standard DQN
            self.output_layer = nn.Linear(prev_size, action_size)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Pass through feature layers
        for i in range(0, len(self.feature_layers), 2):
            x = self.feature_layers[i](x)  # Linear layer
            x = self.activation(x)
            if i + 1 < len(self.feature_layers):
                x = self.feature_layers[i + 1](x)  # Dropout layer
        
        if self.dueling:
            # Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s))
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            
            # Subtract mean advantage for identifiability
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            # Standard DQN
            q_values = self.output_layer(x)
        
        return q_values


class DQNAgent:
    """DQN Agent for UNO."""
    
    def __init__(self, state_size, action_size, config, device=None):
        """
        Initialize the DQN agent.
        
        Args:
            state_size (int): Dimension of input state
            action_size (int): Dimension of action space
            config (dict): Configuration parameters
            device: PyTorch device (CPU/GPU)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.learning_rate = config.get('learning_rate', 0.001)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.target_update_freq = config.get('target_update_freq', 1000)
        self.double_dqn = config.get('double_dqn', True)
        
        # Networks
        network_config = config.get('dqn', {})
        self.q_network = DQN(
            state_size, action_size,
            hidden_layers=network_config.get('hidden_layers', [512, 256, 128]),
            activation=network_config.get('activation', 'relu'),
            dropout=network_config.get('dropout', 0.1),
            dueling=network_config.get('dueling', True)
        ).to(self.device)
        
        self.target_network = DQN(
            state_size, action_size,
            hidden_layers=network_config.get('hidden_layers', [512, 256, 128]),
            activation=network_config.get('activation', 'relu'),
            dropout=network_config.get('dropout', 0.1),
            dueling=network_config.get('dueling', True)
        ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Training step counter
        self.t_step = 0
    
    def act(self, state, legal_actions, epsilon=None):
        """
        Choose an action using epsilon-greedy policy.
        
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
        
        # Epsilon-greedy action selection
        if np.random.random() > epsilon:
            # Exploitation: choose best action
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                
                # Mask illegal actions with very negative values
                masked_q_values = q_values.clone()
                mask = torch.full((self.action_size,), float('-inf'), device=self.device)
                mask[legal_actions] = 0
                masked_q_values += mask
                
                action = masked_q_values.argmax().item()
            self.q_network.train()
        else:
            # Exploration: choose random legal action
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
        action = self.use_raw(state)
        
        # Create probability distribution (not exact, but for compatibility)
        probs = np.zeros(self.action_size)
        if isinstance(state, dict) and 'legal_actions' in state:
            legal_actions = list(state['legal_actions'].keys())
            for a in legal_actions:
                probs[a] = 1.0 / len(legal_actions)
        
        return action, probs
    
    def learn(self, replay_buffer):
        """
        Learn from a batch of experiences.
        
        Args:
            replay_buffer: Experience replay buffer
        """
        if not replay_buffer.is_ready():
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = replay_buffer.sample()
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        if self.double_dqn:
            # Double DQN: use main network to select action, target network to evaluate
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions).detach()
        else:
            # Standard DQN
            next_q_values = self.target_network(next_states).max(1)[0].detach().unsqueeze(1)
        
        # Target Q values
        target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * (~dones).unsqueeze(1))
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()
        
        # Update target network
        self.t_step += 1
        if self.t_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
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
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            't_step': self.t_step
        }, filepath)
    
    def load(self, filepath):
        """Load the model."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.t_step = checkpoint['t_step']
    
    def set_device(self, device):
        """Set device for compatibility."""
        self.device = device
        self.q_network.to(device)
        self.target_network.to(device)
    
    def train(self):
        """Set to training mode."""
        self.q_network.train()
        self.target_network.train()
    
    def eval(self):
        """Set to evaluation mode."""
        self.q_network.eval()
        self.target_network.eval()