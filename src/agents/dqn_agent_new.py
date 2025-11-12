"""
DQN Agent with consistent architecture for fair comparison with DMC+MCTS.
Uses MPS GPU acceleration and matches the original trained model architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import os

# Experience tuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQN(nn.Module):
    """Deep Q-Network with dueling architecture."""
    
    def __init__(self, state_size, action_size, hidden_layers=[256, 128], 
                 activation='relu', dropout=0.1, dueling=True):
        """Initialize DQN network."""
        super(DQN, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.dueling = dueling
        
        # Activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        
        # Feature layers
        layers = []
        prev_size = state_size
        
        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            if i < len(hidden_layers) - 1:  # No dropout after last layer
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        self.feature_layers = nn.ModuleList(layers)
        
        if dueling:
            # Dueling architecture
            self.value_stream = nn.Linear(prev_size, 1)
            self.advantage_stream = nn.Linear(prev_size, action_size)
        else:
            # Standard DQN
            self.output_layer = nn.Linear(prev_size, action_size)
    
    def forward(self, x):
        """Forward pass."""
        # Feature extraction
        for i in range(0, len(self.feature_layers), 2):
            x = self.activation(self.feature_layers[i](x))
            if i + 1 < len(self.feature_layers):
                x = self.feature_layers[i + 1](x)  # Dropout
        
        if self.dueling:
            # Dueling DQN
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            
            # Combine value and advantage
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
            return q_values
        else:
            # Standard DQN
            return self.output_layer(x)


class ReplayBuffer:
    """Experience replay buffer."""
    
    def __init__(self, capacity, device):
        """Initialize replay buffer."""
        self.buffer = deque(maxlen=capacity)
        self.device = device
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Sample batch of experiences."""
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return buffer size."""
        return len(self.buffer)


class DQNAgent:
    """DQN Agent with consistent architecture."""
    
    def __init__(self, state_size, action_size, config):
        """Initialize DQN agent."""
        self.state_size = state_size
        self.action_size = action_size
        
        # Get device (MPS, CUDA, or CPU)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        print(f"DQN using device: {self.device}")
        
        # Hyperparameters from config
        training_config = config.get('training', {})
        network_config = config.get('network', {})
        dqn_config = config.get('dqn', {})
        
        self.learning_rate = training_config.get('learning_rate', 0.0005)
        self.gamma = training_config.get('gamma', 0.99)
        self.epsilon = training_config.get('epsilon_start', 1.0)
        self.epsilon_min = training_config.get('epsilon_end', 0.05)
        self.epsilon_decay = training_config.get('epsilon_decay', 0.9995)
        self.batch_size = training_config.get('batch_size', 64)
        self.memory_size = training_config.get('memory_size', 20000)
        self.target_update_freq = training_config.get('target_update_freq', 1000)
        
        # Network architecture
        hidden_layers = network_config.get('hidden_layers', [256, 128])
        activation = network_config.get('activation', 'relu')
        dropout = network_config.get('dropout', 0.1)
        dueling = dqn_config.get('dueling', True)
        
        # Create networks
        self.q_network = DQN(
            state_size, action_size, hidden_layers, activation, dropout, dueling
        ).to(self.device)
        
        self.target_network = DQN(
            state_size, action_size, hidden_layers, activation, dropout, dueling
        ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Experience replay
        self.memory = ReplayBuffer(self.memory_size, self.device)
        
        # Training step counter
        self.t_step = 0
        
    def act(self, state, legal_actions=None):
        """Choose action using epsilon-greedy policy."""
        # Convert state to tensor
        if isinstance(state, dict):
            # Process state to get state vector
            state_tensor = self._process_state(state)
            
            # Convert to tensor if it's numpy array
            if isinstance(state_tensor, np.ndarray):
                state_tensor = torch.FloatTensor(state_tensor).unsqueeze(0).to(self.device)
            elif not isinstance(state_tensor, torch.Tensor):
                state_tensor = torch.FloatTensor([state_tensor]).unsqueeze(0).to(self.device)
            elif state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0).to(self.device)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            # Greedy action
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                
                # Apply legal action mask if provided
                if legal_actions is not None:
                    # Create mask for illegal actions
                    mask = torch.full((self.action_size,), float('-inf')).to(self.device)
                    mask[legal_actions] = 0
                    q_values = q_values + mask.unsqueeze(0)
                
                action = q_values.argmax().item()
            self.q_network.train()
        else:
            # Random action
            if legal_actions is not None:
                action = random.choice(legal_actions)
            else:
                action = random.randrange(self.action_size)
        
        return action
    
    def use_raw(self, state):
        """Interface for RLCard compatibility."""
        # Get legal actions if available
        legal_actions = None
        if hasattr(state, 'get') and 'legal_actions' in state:
            legal_actions_raw = state['legal_actions']
            if hasattr(legal_actions_raw, 'keys'):
                # OrderedDict format - use keys as indices
                legal_actions = list(legal_actions_raw.keys())
            else:
                # List or array format
                legal_actions = legal_actions_raw
        
        return self.act(state, legal_actions)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience and learn."""
        # Process states
        processed_state = self._process_state(state)
        processed_next_state = self._process_state(next_state)
        
        # Save experience
        self.memory.push(processed_state, action, reward, processed_next_state, done)
        
        # Learn if enough experiences
        if len(self.memory) >= self.batch_size:
            self.learn()
        
        # Update target network
        self.t_step += 1
        if self.t_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def learn(self):
        """Update Q-network using batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            # Double DQN: use main network to select action, target network to evaluate
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * (~dones).unsqueeze(1))
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def _process_state(self, state):
        """Process state for network input."""
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
    
    def save(self, filepath):
        """Save the model."""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            't_step': self.t_step
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
    
    def load(self, filepath):
        """Load the model."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.t_step = checkpoint['t_step']
        
        # Move to correct device
        self.q_network.to(self.device)
        self.target_network.to(self.device)
    
    def eval(self):
        """Set to evaluation mode."""
        self.q_network.eval()
        self.target_network.eval()
    
    def train(self):
        """Set to training mode."""
        self.q_network.train()
        self.target_network.train()
    
    def set_device(self, device):
        """Move model to device."""
        self.device = device
        self.q_network.to(device)
        self.target_network.to(device)