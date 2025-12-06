"""
Shared memory buffer for actor-learner architecture.
Supports unroll sequences with opponent features for parallel training.
"""

import multiprocessing as mp
import time
from collections import deque
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch


class UnrollSequence:
    """
    Represents a sequence of unroll_length steps for learning.
    """
    
    def __init__(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        next_states: List[np.ndarray],
        dones: List[bool],
        opponent_features: List[np.ndarray],
        sequence_id: int
    ):
        """
        Initialize unroll sequence.
        
        Args:
            states: List of state vectors (length = unroll_length)
            actions: List of action indices (length = unroll_length)
            rewards: List of rewards (length = unroll_length)
            next_states: List of next state vectors (length = unroll_length)
            dones: List of done flags (length = unroll_length)
            opponent_features: List of opponent feature vectors (length = unroll_length)
            sequence_id: Unique identifier for this sequence
        """
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones
        self.opponent_features = opponent_features
        self.sequence_id = sequence_id
        self.length = len(states)
    
    def to_dict(self) -> Dict:
        """Convert sequence to dictionary for serialization."""
        return {
            'states': self.states,
            'actions': self.actions,
            'rewards': self.rewards,
            'next_states': self.next_states,
            'dones': self.dones,
            'opponent_features': self.opponent_features,
            'sequence_id': self.sequence_id,
            'length': self.length
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UnrollSequence':
        """Create sequence from dictionary."""
        return cls(
            states=data['states'],
            actions=data['actions'],
            rewards=data['rewards'],
            next_states=data['next_states'],
            dones=data['dones'],
            opponent_features=data['opponent_features'],
            sequence_id=data['sequence_id']
        )


class SharedExperienceBuffer:
    """
    Thread-safe shared memory buffer for storing unroll sequences.
    Uses multiprocessing queues for communication between actors and learner.
    """
    
    def __init__(
        self,
        num_buffers: int = 10,
        buffer_capacity: int = 100,
        unroll_length: int = 100
    ):
        """
        Initialize shared experience buffer.
        
        Args:
            num_buffers: Number of parallel buffers (for load balancing)
            buffer_capacity: Maximum sequences per buffer
            unroll_length: Length of each unroll sequence
        """
        self.num_buffers = num_buffers
        self.buffer_capacity = buffer_capacity
        self.unroll_length = unroll_length
        
        # Create multiple queues for load balancing
        # Each actor can push to any queue, learner consumes from all
        self.queues = [mp.Queue(maxsize=buffer_capacity) for _ in range(num_buffers)]
        
        # Statistics
        self.total_sequences = mp.Value('i', 0)  # Shared counter
        self.total_dropped = mp.Value('i', 0)  # Dropped sequences when full
        
        # Manual queue size tracking (qsize() not available on macOS with spawn)
        self.queue_sizes = [mp.Value('i', 0) for _ in range(num_buffers)]
        
        # Lock for statistics updates
        self.stats_lock = mp.Lock()
    
    def push_sequence(
        self,
        sequence: UnrollSequence,
        buffer_idx: Optional[int] = None
    ) -> bool:
        """
        Push a sequence to one of the buffers.
        
        Args:
            sequence: UnrollSequence to push
            buffer_idx: Specific buffer index (None = round-robin)
            
        Returns:
            True if successfully pushed, False if buffer was full
        """
        # Round-robin if no specific buffer specified
        if buffer_idx is None:
            buffer_idx = self.total_sequences.value % self.num_buffers
        
        try:
            # Try to put sequence in queue (non-blocking)
            self.queues[buffer_idx].put_nowait(sequence)
            
            with self.stats_lock:
                self.total_sequences.value += 1
                self.queue_sizes[buffer_idx].value += 1
            
            return True
        except:
            # Queue is full, drop sequence
            with self.stats_lock:
                self.total_dropped.value += 1
            return False
    
    def sample_sequences(
        self,
        batch_size: int,
        timeout: float = 1.0
    ) -> List[UnrollSequence]:
        """
        Sample a batch of sequences from all buffers.
        Aggressively collects sequences from all queues without waiting.
        
        Args:
            batch_size: Target number of sequences to sample (may return fewer)
            timeout: Maximum time to wait for sequences (seconds)
            
        Returns:
            List of UnrollSequence objects (may be less than batch_size if timeout)
        """
        sequences = []
        start_time = time.time()
        
        # Aggressively collect from all queues - don't wait for full batches
        # This allows faster updates even with partial batches
        max_iterations = batch_size * 2  # Try many times to get sequences
        iteration = 0
        
        while len(sequences) < batch_size and iteration < max_iterations:
            # Check timeout
            if time.time() - start_time > timeout:
                break
            
            # Round-robin through all queues
            for buffer_idx, queue in enumerate(self.queues):
                if len(sequences) >= batch_size:
                    break
                if time.time() - start_time > timeout:
                    break
                
                # Try to get a sequence (non-blocking first, then with short timeout)
                try:
                    sequence = queue.get_nowait()
                    sequences.append(sequence)
                    
                    # Update queue size counter
                    with self.stats_lock:
                        if self.queue_sizes[buffer_idx].value > 0:
                            self.queue_sizes[buffer_idx].value -= 1
                except:
                    # Queue empty - try next queue
                    pass
            
            iteration += 1
            
            # If we got some sequences but not enough, try one more quick pass
            if len(sequences) > 0 and len(sequences) < batch_size:
                # Very short sleep to let actors produce more
                time.sleep(0.001)
        
        return sequences
    
    def get_queue_size(self, buffer_idx: int) -> int:
        """
        Get current size of a specific buffer queue.
        Uses manual counter since qsize() is not available on macOS with spawn method.
        """
        try:
            # Try qsize() first (works on Linux/Windows)
            return self.queues[buffer_idx].qsize()
        except (NotImplementedError, AttributeError):
            # Fall back to manual counter (required on macOS)
            return self.queue_sizes[buffer_idx].value
    
    def get_total_size(self) -> int:
        """Get total number of sequences across all buffers."""
        return sum(self.get_queue_size(i) for i in range(self.num_buffers))
    
    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        with self.stats_lock:
            return {
                'total_sequences': self.total_sequences.value,
                'total_dropped': self.total_dropped.value,
                'current_size': self.get_total_size(),
                'buffer_sizes': [self.get_queue_size(i) for i in range(self.num_buffers)]
            }
    
    def clear(self):
        """Clear all buffers (for testing/reset)."""
        for buffer_idx, queue in enumerate(self.queues):
            while not queue.empty():
                try:
                    queue.get_nowait()
                    # Update counter
                    with self.stats_lock:
                        if self.queue_sizes[buffer_idx].value > 0:
                            self.queue_sizes[buffer_idx].value -= 1
                except:
                    pass
        
        with self.stats_lock:
            self.total_sequences.value = 0
            self.total_dropped.value = 0
            for size_counter in self.queue_sizes:
                size_counter.value = 0

