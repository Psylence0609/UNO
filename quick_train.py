"""
Quick training run to validate the setup.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.train_dqn import DQNTrainer

def quick_train():
    """Run a quick training session."""
    print("🚀 Starting quick training validation...")
    
    # Create trainer with quick config
    trainer = DQNTrainer('quick_config.yaml')
    
    print(f"Training setup:")
    print(f"  Episodes: {trainer.config['training']['episodes']}")
    print(f"  Evaluation frequency: {trainer.config['evaluation']['eval_freq']}")
    print(f"  State size: {trainer.state_size}")
    print()
    
    # Start training
    final_eval, model_path = trainer.train()
    
    print(f"\n🎉 Quick training completed!")
    print(f"Final win rate: {final_eval['win_rate']:.1%}")
    print(f"Model saved: {model_path}")

if __name__ == "__main__":
    quick_train()