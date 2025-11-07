#!/usr/bin/env python3
"""
Evaluate RLCard DMC Model against Random Agent
Similar evaluation to evaluate_all_models.py but specifically for RLCard's trained DMC model.
"""

import os
import sys
import torch
import numpy as np
import rlcard

# Add project root to path
sys.path.append('.')

from src.environments.uno_env import UnoEnvironment
from src.agents.random_agent import RandomAgent
from src.evaluation.evaluator import Evaluator
from rlcard.agents.dmc_agent.model import DMCAgent


class RLCardDMCAgentWrapper:
    """
    Wrapper to make RLCard's DMCAgent compatible with our evaluation framework.
    RLCard's agent uses different interface (step/eval_step) than our custom agents.
    """
    
    def __init__(self, rlcard_agent):
        """
        Initialize wrapper.
        
        Args:
            rlcard_agent: RLCard DMCAgent instance
        """
        self.agent = rlcard_agent
        self.agent.eval()  # Set to evaluation mode
        self.agent.exp_epsilon = 0.0  # No exploration during evaluation
    
    def use_raw(self, state):
        """
        Interface for RLCard compatibility.
        Uses RLCard's eval_step method for evaluation.
        
        Args:
            state: RLCard state dictionary
            
        Returns:
            int: Selected action
        """
        # Ensure state has raw_legal_actions if not present
        if 'raw_legal_actions' not in state and 'legal_actions' in state:
            # Convert legal_actions to raw_legal_actions format
            legal_actions = state['legal_actions']
            if hasattr(legal_actions, 'keys'):
                state['raw_legal_actions'] = list(legal_actions.keys())
            else:
                state['raw_legal_actions'] = list(legal_actions)
        
        try:
            action, _ = self.agent.eval_step(state)
            return action
        except Exception as e:
            # Fallback: use step method if eval_step fails
            print(f"⚠️  eval_step failed, using step: {e}")
            return self.agent.step(state)
    
    def eval_step(self, state):
        """
        Evaluation step (same as use_raw for compatibility).
        
        Args:
            state: RLCard state dictionary
            
        Returns:
            tuple: (action, info)
        """
        # Ensure state has raw_legal_actions if not present
        if 'raw_legal_actions' not in state and 'legal_actions' in state:
            legal_actions = state['legal_actions']
            if hasattr(legal_actions, 'keys'):
                state['raw_legal_actions'] = list(legal_actions.keys())
            else:
                state['raw_legal_actions'] = list(legal_actions)
        
        return self.agent.eval_step(state)


def load_rlcard_dmc_model(model_path, state_shape, action_shape, device="cpu"):
    """
    Load RLCard DMC model from checkpoint.
    
    Args:
        model_path: Path to model.tar checkpoint file
        state_shape: State shape tuple
        action_shape: Action shape tuple
        device: Device string ("0" for CUDA, "cpu" for CPU)
        
    Returns:
        RLCardDMCAgentWrapper: Wrapped agent ready for evaluation
    """
    # Determine map_location for loading
    if device == "cpu":
        map_location = "cpu"
    else:
        map_location = f"cuda:{device}"
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=map_location)
    
    # Create RLCard DMCAgent
    # Note: RLCard uses default MLP layers [512,512,512,512,512]
    # You may need to adjust if you trained with different architecture
    agent = DMCAgent(
        state_shape=state_shape,
        action_shape=action_shape,
        mlp_layers=[512, 512, 512, 512, 512],  # RLCard default
        exp_epsilon=0.0,  # No exploration
        device=device
    )
    
    # Load model state dict
    # RLCard saves models per player, we use player 0
    if "model_state_dict" in checkpoint:
        agent.load_state_dict(checkpoint["model_state_dict"][0])
    else:
        # If checkpoint format is different, try direct load
        agent.load_state_dict(checkpoint)
    
    # Wrap agent for compatibility
    wrapped_agent = RLCardDMCAgentWrapper(agent)
    
    return wrapped_agent


def evaluate_vs_random(evaluator, agent, num_games=500):
    """
    Evaluate agent against random baseline.
    
    Args:
        evaluator: Evaluator instance
        agent: Agent to evaluate
        num_games: Number of games to play
        
    Returns:
        tuple: (win_rate, avg_turns)
    """
    random_agent = RandomAgent(evaluator.env.num_actions)
    
    # Agent as player 0, random as player 1
    results = evaluator.evaluate_agents([agent, random_agent], num_games=num_games, verbose=False)
    
    agent_wins = results['wins'][0]
    win_rate = agent_wins / num_games
    avg_turns = np.mean(results['game_lengths'])
    
    return win_rate, avg_turns


def main():
    """Main evaluation function."""
    print("=" * 80)
    print("🏆 RLCARD DMC MODEL EVALUATION")
    print("=" * 80)
    
    # Default model path (adjust if your model is in a different location)
    default_model_path = "experiments/rlcard_dmc_uno/uno_rlcard_dmc/model.tar"
    
    # Allow command line argument for model path
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = default_model_path
    
    print(f"\n📁 Model path: {model_path}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found at: {model_path}")
        print("\n💡 Usage:")
        print(f"   python evaluate_rlcard_dmc.py [model_path]")
        print(f"\n   Default path: {default_model_path}")
        print("\n   Example:")
        print(f"   python evaluate_rlcard_dmc.py experiments/rlcard_dmc_uno/uno_rlcard_dmc/model.tar")
        return
    
    # Initialize environment
    print("\n📦 Initializing environment...")
    env = UnoEnvironment(seed=42)
    evaluator = Evaluator(env)
    
    # Get state and action shapes for RLCard
    sample_state, _ = env.reset()
    state_shape = sample_state['obs'].shape
    action_shape = (env.num_actions,)
    
    print(f"✅ Environment created:")
    print(f"   State shape: {state_shape}")
    print(f"   Action shape: {action_shape}")
    print(f"   Number of actions: {env.num_actions}")
    
    # Check device
    # RLCard uses "0" for CUDA GPU, "cpu" for CPU
    if torch.cuda.is_available():
        device = "0"
        device_str = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "cpu"  # RLCard doesn't support MPS
        device_str = "cpu"
        print(f"\n⚠️  MPS detected but RLCard doesn't support it, using CPU")
    else:
        device = "cpu"
        device_str = "cpu"
    
    print(f"   Device: {device_str}")
    
    # Load RLCard DMC model
    print(f"\n📦 Loading RLCard DMC model...")
    try:
        agent = load_rlcard_dmc_model(
            model_path=model_path,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device if device != "cpu" else "cpu"
        )
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print(f"\n💡 Troubleshooting:")
        print(f"   1. Check if model.tar exists at the specified path")
        print(f"   2. Verify the model was trained with RLCard's DMCTrainer")
        print(f"   3. Check if state/action shapes match")
        import traceback
        traceback.print_exc()
        return
    
    # Evaluate vs Random
    print("\n" + "=" * 80)
    print("📊 PERFORMANCE VS RANDOM BASELINE")
    print("=" * 80)
    
    num_games = 500
    print(f"\n🎮 Evaluating RLCard DMC vs Random... ({num_games} games)")
    
    try:
        win_rate, avg_turns = evaluate_vs_random(evaluator, agent, num_games=num_games)
        
        print(f"\n✅ Evaluation completed!")
        print(f"\n📊 Results:")
        print(f"   Win Rate: {win_rate:.1%}")
        print(f"   Average Turns: {avg_turns:.1f}")
        print(f"   Wins: {int(win_rate * num_games)}/{num_games}")
        print(f"   Losses: {int((1 - win_rate) * num_games)}/{num_games}")
        
        # Comparison with baseline
        print(f"\n📈 Analysis:")
        if win_rate > 0.50:
            improvement = (win_rate - 0.50) * 100
            print(f"   ✅ Performs {improvement:.1f}% better than random baseline")
        elif win_rate < 0.50:
            deficit = (0.50 - win_rate) * 100
            print(f"   ⚠️  Performs {deficit:.1f}% worse than random baseline")
        else:
            print(f"   ➡️  Performs at random baseline level")
        
        # Performance level
        if win_rate >= 0.60:
            level = "Excellent"
        elif win_rate >= 0.55:
            level = "Good"
        elif win_rate >= 0.52:
            level = "Above Average"
        elif win_rate >= 0.48:
            level = "Average"
        else:
            level = "Below Average"
        
        print(f"   Performance Level: {level}")
        
    except Exception as e:
        print(f"\n❌ Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Summary
    print("\n" + "=" * 80)
    print("🎯 SUMMARY")
    print("=" * 80)
    print(f"\n🏆 RLCard DMC Model Performance:")
    print(f"   Win Rate vs Random: {win_rate:.1%}")
    print(f"   Average Game Length: {avg_turns:.1f} turns")
    print(f"\n💡 Note: Compare this with your custom DMC models:")
    print(f"   - DMC Ep10k: 52.6% (best performer)")
    print(f"   - DMC Old: 50.8%")
    print(f"   - DQN+MCTS: 50.0%")
    print(f"   - DQN Original: 48.6%")
    print("=" * 80)


if __name__ == "__main__":
    main()

