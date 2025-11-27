"""
Actor process for parallel experience collection in actor-learner architecture.
Each actor runs games independently and collects unroll sequences.
"""

import multiprocessing as mp
import numpy as np
import torch
import time
from typing import Dict, List, Optional
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.environments.uno_env import UnoEnvironment
from src.agents.dmc_agent_advanced_opponent import AdvancedDMCAgentWithOpponent
from src.agents.random_agent import RandomAgent
from src.utils.shared_experience_buffer import SharedExperienceBuffer, UnrollSequence


def actor_worker(
    actor_id: int,
    config: Dict,
    shared_buffer: SharedExperienceBuffer,
    model_state_queue: mp.Queue,
    stats_queue: mp.Queue,
    stop_event: mp.Event,
    unroll_length: int = 100,
    seed_offset: int = 0
):
    """
    Worker function for actor process.
    
    Args:
        actor_id: Unique identifier for this actor
        config: Training configuration dictionary
        shared_buffer: Shared experience buffer for pushing sequences
        model_state_queue: Queue for receiving model updates
        stats_queue: Queue for sending statistics
        stop_event: Event to signal when to stop
        unroll_length: Length of unroll sequences
        seed_offset: Offset for environment seed (for diversity)
    """
    try:
        # Set process name for debugging
        mp.current_process().name = f'Actor-{actor_id}'
        
        # Create environment with unique seed
        env_seed = config.get('environment', {}).get('seed', 42) + actor_id + seed_offset
        env = UnoEnvironment(seed=env_seed)
        
        # Calculate state size
        sample_state, _ = env.reset()
        features = []
        if 'obs' in sample_state:
            features.extend(sample_state['obs'].flatten())
        features.extend(np.zeros(env.num_actions))
        state_size = len(features)
        
        # Create agent (will be updated with model weights from learner)
        agent = AdvancedDMCAgentWithOpponent(
            state_size=state_size,
            action_size=env.num_actions,
            config=config
        )
        
        # Create opponent
        opponent = RandomAgent(env.num_actions)
        
        # Statistics
        sequences_collected = 0
        steps_collected = 0
        episodes_completed = 0
        
        # Current unroll sequence being collected
        current_sequence = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'opponent_features': []
        }
        
        # Game state
        state, player_id = env.reset()
        agent.reset_opponent_tracking(num_players=env.num_players)
        episode_reward = 0
        episode_length = 0
        
        # Main loop
        while not stop_event.is_set():
            # Check for model updates (non-blocking)
            try:
                while not model_state_queue.empty():
                    model_state = model_state_queue.get_nowait()
                    # Update agent with new model weights
                    agent.load_state_dict(model_state)
            except:
                pass
            
            # Collect unroll sequence
            sequence_steps = 0
            while sequence_steps < unroll_length and not stop_event.is_set():
                # Check if game is over, reset if needed
                if env.is_over():
                    # Calculate final reward
                    payoffs = env.get_payoffs()
                    final_reward = 1.0 if payoffs[0] > 0 else -1.0
                    
                    # Store final transition if we have a state
                    if current_sequence['states']:
                        # Use last state as next state for final transition
                        last_state = current_sequence['states'][-1] if current_sequence['states'] else None
                        if last_state is not None:
                            current_sequence['rewards'][-1] = final_reward
                            current_sequence['dones'][-1] = True
                    
                    # Reset environment
                    state, player_id = env.reset()
                    agent.reset_opponent_tracking(num_players=env.num_players)
                    episode_reward = 0
                    episode_length = 0
                    episodes_completed += 1
                    
                    # If we have a partial sequence, we can either:
                    # 1. Discard it (simpler)
                    # 2. Pad it and mark as done (more complex)
                    # For now, we'll discard partial sequences at episode boundaries
                    if len(current_sequence['states']) > 0 and len(current_sequence['states']) < unroll_length:
                        # Reset sequence if it's too short
                        current_sequence = {
                            'states': [],
                            'actions': [],
                            'rewards': [],
                            'next_states': [],
                            'dones': [],
                            'opponent_features': []
                        }
                    continue
                
                # Get current agent (DMC agent or opponent)
                current_agent = agent if player_id == 0 else opponent
                
                # Process state and extract opponent features (for DMC agent)
                if player_id == 0:
                    state_features = agent._process_state(state)
                    opponent_features = agent.extract_opponent_features(state)
                else:
                    # For opponent turns, we still need to track
                    state_features = agent._process_state(state) if hasattr(agent, '_process_state') else None
                    opponent_features = agent.extract_opponent_features(state) if hasattr(agent, 'extract_opponent_features') else np.zeros(agent.opponent_feature_size)
                
                # Take action
                action = current_agent.use_raw(state)
                next_state, next_player_id = env.step(action)
                
                # Update opponent tracking
                if player_id == 0:
                    # DMC agent's turn - track opponent's response
                    if not env.is_over():
                        # Get opponent's hand size if available
                        perfect_info = env.get_perfect_information() if hasattr(env, 'get_perfect_information') else {}
                        opponent_hand_size = perfect_info.get('hand_sizes', [0, 0])[1] if 'hand_sizes' in perfect_info else 0
                        agent.opponent_extractor.update_history(action, 1, opponent_hand_size)
                else:
                    # Opponent's turn - track their action
                    perfect_info = env.get_perfect_information() if hasattr(env, 'get_perfect_information') else {}
                    opponent_hand_size = perfect_info.get('hand_sizes', [0, 0])[1] if 'hand_sizes' in perfect_info else 0
                    agent.opponent_extractor.update_history(action, 1, opponent_hand_size)
                
                # Calculate reward (only for DMC agent's actions)
                reward = 0.0
                if player_id == 0:
                    if env.is_over():
                        payoffs = env.get_payoffs()
                        reward = 1.0 if payoffs[0] > 0 else -1.0
                    # Could add intermediate rewards here if needed
                
                # Store transition (only for DMC agent's turns)
                if player_id == 0:
                    next_state_features = agent._process_state(next_state) if not env.is_over() else state_features
                    
                    current_sequence['states'].append(state_features)
                    current_sequence['actions'].append(action)
                    current_sequence['rewards'].append(reward)
                    current_sequence['next_states'].append(next_state_features)
                    current_sequence['dones'].append(env.is_over())
                    current_sequence['opponent_features'].append(opponent_features)
                    
                    sequence_steps += 1
                    steps_collected += 1
                    episode_reward += reward
                    episode_length += 1
                
                # Update state
                state = next_state
                player_id = next_player_id
                
                # Check if we've collected a full sequence
                if len(current_sequence['states']) >= unroll_length:
                    # Create and push sequence
                    sequence = UnrollSequence(
                        states=current_sequence['states'][:unroll_length],
                        actions=current_sequence['actions'][:unroll_length],
                        rewards=current_sequence['rewards'][:unroll_length],
                        next_states=current_sequence['next_states'][:unroll_length],
                        dones=current_sequence['dones'][:unroll_length],
                        opponent_features=current_sequence['opponent_features'][:unroll_length],
                        sequence_id=sequences_collected
                    )
                    
                    # Push to shared buffer
                    success = shared_buffer.push_sequence(sequence, buffer_idx=actor_id % shared_buffer.num_buffers)
                    
                    if success:
                        sequences_collected += 1
                    
                    # Reset sequence (keep any overflow for next sequence)
                    if len(current_sequence['states']) > unroll_length:
                        # Keep overflow
                        overflow = len(current_sequence['states']) - unroll_length
                        current_sequence = {
                            'states': current_sequence['states'][unroll_length:],
                            'actions': current_sequence['actions'][unroll_length:],
                            'rewards': current_sequence['rewards'][unroll_length:],
                            'next_states': current_sequence['next_states'][unroll_length:],
                            'dones': current_sequence['dones'][unroll_length:],
                            'opponent_features': current_sequence['opponent_features'][unroll_length:]
                        }
                    else:
                        current_sequence = {
                            'states': [],
                            'actions': [],
                            'rewards': [],
                            'next_states': [],
                            'dones': [],
                            'opponent_features': []
                        }
            
            # Send statistics periodically (every sequence to ensure trainer gets updates)
            if sequences_collected % 1 == 0:  # Send every sequence for better tracking
                try:
                    stats_queue.put_nowait({
                        'actor_id': actor_id,
                        'sequences_collected': sequences_collected,
                        'steps_collected': steps_collected,
                        'episodes_completed': episodes_completed,
                        'episode_reward': episode_reward,
                        'episode_length': episode_length
                    })
                except:
                    # Queue might be full, try again next time
                    pass
        
        # Send final statistics
        try:
            stats_queue.put_nowait({
                'actor_id': actor_id,
                'sequences_collected': sequences_collected,
                'steps_collected': steps_collected,
                'episodes_completed': episodes_completed,
                'final': True
            })
        except:
            pass
            
    except Exception as e:
        # Send error to stats queue
        try:
            stats_queue.put_nowait({
                'actor_id': actor_id,
                'error': str(e),
                'error_type': type(e).__name__
            })
        except:
            pass
        raise


class ActorManager:
    """
    Manager for multiple actor processes.
    """
    
    def __init__(
        self,
        num_actors: int,
        config: Dict,
        shared_buffer: SharedExperienceBuffer,
        model_state_queue: mp.Queue,
        stats_queue: mp.Queue,
        unroll_length: int = 100
    ):
        """
        Initialize actor manager.
        
        Args:
            num_actors: Number of parallel actors
            config: Training configuration
            shared_buffer: Shared experience buffer
            model_state_queue: Queue for model updates
            stats_queue: Queue for statistics
            unroll_length: Length of unroll sequences
        """
        self.num_actors = num_actors
        self.config = config
        self.shared_buffer = shared_buffer
        self.model_state_queue = model_state_queue
        self.stats_queue = stats_queue
        self.unroll_length = unroll_length
        
        self.processes = []
        self.stop_event = mp.Event()
    
    def start(self):
        """Start all actor processes."""
        for actor_id in range(self.num_actors):
            process = mp.Process(
                target=actor_worker,
                args=(
                    actor_id,
                    self.config,
                    self.shared_buffer,
                    self.model_state_queue,
                    self.stats_queue,
                    self.stop_event,
                    self.unroll_length,
                    actor_id * 1000  # Seed offset for diversity
                )
            )
            process.start()
            self.processes.append(process)
    
    def stop(self):
        """Stop all actor processes."""
        self.stop_event.set()
        for process in self.processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join()
    
    def broadcast_model_update(self, model_state_dict: Dict):
        """
        Broadcast model update to all actors.
        
        Args:
            model_state_dict: State dictionary of the model
        """
        # Put model state in queue (all actors will receive it)
        try:
            self.model_state_queue.put_nowait(model_state_dict)
        except:
            pass  # Queue might be full, actors will get it on next check
    
    def is_alive(self) -> bool:
        """Check if any actors are still alive."""
        return any(p.is_alive() for p in self.processes)

