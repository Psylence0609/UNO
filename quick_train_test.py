#!/usr/bin/env python3
"""Quick training test for DQN with MCTS."""

import sys
sys.path.append('.')

from src.training.train_dqn_new import DQNTrainer

def main():
    """Run a quick training test."""
    print("🚀 Quick DQN+MCTS Training Test")
    print("=" * 50)
    
    trainer = DQNTrainer()
    
    # Train for just 100 episodes
    print("Training for 100 episodes...")
    
    try:
        for i in range(100):
            episode_data = trainer.train_episode()
            
            if (i + 1) % 25 == 0:
                print(f"Episode {i+1}: reward={episode_data.get('reward', 0):.3f}, "
                      f"length={episode_data.get('length', 0)}, "
                      f"epsilon={episode_data.get('epsilon', 0):.3f}")
        
        print("\n✅ Training completed successfully!")
        
        # Quick evaluation
        print("Running quick evaluation...")
        eval_result = trainer.evaluate_agent(50)  # 50 games
        print(f"Win rate: {eval_result['win_rate']:.3f} ({eval_result['win_rate']*100:.1f}%)")
        
        # Save model
        import os
        os.makedirs('models', exist_ok=True)
        trainer.save_model('models/dqn_quick_test.pth')
        print("Model saved to: models/dqn_quick_test.pth")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()