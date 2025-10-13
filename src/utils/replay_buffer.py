"""
Replay buffer for storing and sampling experiences.
"""

import random
import numpy as np
from collections import deque, namedtuple

# Experience tuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    
    def __init__(self, buffer_size=10000, batch_size=32, seed=42):
        """
        Initialize a ReplayBuffer object.
        
        Args:
            buffer_size (int): Maximum size of buffer
            batch_size (int): Size of each training batch
            seed (int): Random seed
        """
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.np_random = np.random.RandomState(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.buffer, k=self.batch_size)
        
        states = np.array([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None])
        rewards = np.array([e.reward for e in experiences if e is not None])
        next_states = np.array([e.next_state for e in experiences if e is not None])
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)
    
    def is_ready(self):
        """Check if buffer has enough samples for a batch."""
        return len(self.buffer) >= self.batch_size