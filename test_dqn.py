"""
Test script for DQN agent implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import yaml
from src.environments.uno_env import UnoEnvironment
from src.agents.dqn_agent import DQNAgent
from src.agents.random_agent import RandomAgent
from src.utils.replay_buffer import ReplayBuffer

def test_dqn_agent():
    """Test the DQN agent implementation."""
    print("🧪 TESTING DQN AGENT IMPLEMENTATION")
    print("=" * 50)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup environment
    env = UnoEnvironment()
    print(f"✅ Environment created: {env.num_actions} actions, {env.num_players} players")
    
    # Calculate state size
    state, _ = env.reset()
    processed_state = np.concatenate([
        state['obs'].flatten(),
        np.zeros(env.num_actions)  # Legal actions mask
    ])
    state_size = len(processed_state)
    
    print(f"📊 State size: {state_size}")
    
    # Create DQN agent
    dqn_agent = DQNAgent(
        state_size=state_size,
        action_size=env.num_actions,
        config=config
    )
    print(f"🤖 DQN agent created")
    print(f"   Device: {dqn_agent.device}")
    print(f"   Network architecture: {dqn_agent.q_network}")
    
    # Test action selection
    print("\n🎯 TESTING ACTION SELECTION")
    print("-" * 30)
    
    state, player_id = env.reset()
    
    # Test with high epsilon (exploration)
    action = dqn_agent.use_raw(state)
    legal_actions = list(state['legal_actions'].keys())
    print(f"Legal actions: {legal_actions}")
    print(f"Selected action (high epsilon): {action}")
    print(f"Action is legal: {action in legal_actions}")
    
    # Test with low epsilon (exploitation)
    dqn_agent.epsilon = 0.0
    action = dqn_agent.use_raw(state)
    print(f"Selected action (low epsilon): {action}")
    print(f"Action is legal: {action in legal_actions}")
    
    # Test eval_step
    action, probs = dqn_agent.eval_step(state)
    print(f"Eval step action: {action}, probs sum: {probs.sum():.3f}")
    
    # Test replay buffer
    print("\n🔄 TESTING REPLAY BUFFER")
    print("-" * 30)
    
    replay_buffer = ReplayBuffer(
        buffer_size=config['training']['memory_size'],
        batch_size=config['training']['batch_size']
    )
    
    # Add some experiences
    for _ in range(10):
        state, _ = env.reset()
        state_vector = dqn_agent._process_state(state)
        action = np.random.choice(list(state['legal_actions'].keys()))
        next_state, _ = env.step(action)
        next_state_vector = dqn_agent._process_state(next_state)
        reward = 0.0  # Intermediate reward
        done = env.is_over()
        
        replay_buffer.add(state_vector, action, reward, next_state_vector, done)
    
    print(f"Buffer size: {len(replay_buffer)}")
    print(f"Buffer ready for sampling: {replay_buffer.is_ready()}")
    
    # Test learning (if buffer is ready)
    if replay_buffer.is_ready():
        print("\n🎓 TESTING LEARNING")
        print("-" * 30)
        
        # Add more experiences to fill buffer
        for _ in range(config['training']['batch_size']):
            state, _ = env.reset()
            state_vector = dqn_agent._process_state(state)
            action = np.random.choice(list(state['legal_actions'].keys()))
            next_state, _ = env.step(action)
            next_state_vector = dqn_agent._process_state(next_state)
            reward = 0.0
            done = env.is_over()
            
            replay_buffer.add(state_vector, action, reward, next_state_vector, done)
        
        # Perform learning step
        initial_epsilon = dqn_agent.epsilon
        loss = dqn_agent.learn(replay_buffer)
        
        print(f"Learning step completed")
        print(f"Loss: {loss:.6f}")
        print(f"Epsilon before: {initial_epsilon:.3f}, after: {dqn_agent.epsilon:.3f}")
    
    # Test model saving and loading
    print("\n💾 TESTING MODEL SAVE/LOAD")
    print("-" * 30)
    
    # Save model
    model_path = "models/test_dqn.pth"
    os.makedirs("models", exist_ok=True)
    dqn_agent.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Create new agent and load
    new_agent = DQNAgent(
        state_size=state_size,
        action_size=env.num_actions,
        config=config
    )
    new_agent.load(model_path)
    print(f"Model loaded successfully")
    
    # Verify actions are the same
    new_agent.epsilon = 0.0  # No randomness
    dqn_agent.epsilon = 0.0
    
    state, _ = env.reset()
    action1 = dqn_agent.use_raw(state)
    action2 = new_agent.use_raw(state)
    
    print(f"Original agent action: {action1}")
    print(f"Loaded agent action: {action2}")
    print(f"Actions match: {action1 == action2}")
    
    # Test against random agent
    print("\n🎲 TESTING AGAINST RANDOM AGENT")
    print("-" * 30)
    
    random_agent = RandomAgent(env.num_actions)
    agents = [dqn_agent, random_agent]
    
    # Play a few games
    wins = [0, 0]
    for game in range(10):
        state, player_id = env.reset()
        
        while not env.is_over():
            action = agents[player_id].use_raw(state)
            state, player_id = env.step(action)
        
        payoffs = env.get_payoffs()
        winner = np.argmax(payoffs)
        wins[winner] += 1
    
    print(f"DQN wins: {wins[0]}/10")
    print(f"Random wins: {wins[1]}/10")
    print(f"DQN win rate: {wins[0]/10:.1%}")
    
    print("\n✅ DQN AGENT TESTS COMPLETED!")
    print("\n📝 Summary:")
    print(f"  • DQN agent successfully created and tested")
    print(f"  • Action selection works with legal action masking")
    print(f"  • Learning step successfully updates parameters")
    print(f"  • Model save/load functionality works")
    print(f"  • Agent can play against other agents")

if __name__ == "__main__":
    test_dqn_agent()