#!/usr/bin/env python3
"""
Resume training for DMC with Opponent Modeling.
Loads best checkpoint (or final checkpoint if best doesn't exist) and continues training with improved hyperparameters.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.training.train_dmc_advanced_opponent import AdvancedDMCTrainerWithOpponent

def main():
    """Resume training from best available checkpoint."""
    trainer = AdvancedDMCTrainerWithOpponent(config_path="config.yaml")
    
    # Try to load best checkpoint
    best_checkpoint = os.path.join(
        trainer.config['paths']['models'],
        "custom",
        "dmc_advanced_opponent_best.pth"
    )
    
    # Fallback to final checkpoint if best doesn't exist
    final_checkpoint = os.path.join(
        trainer.config['paths']['models'],
        "custom",
        "dmc_advanced_opponent_final.pth"
    )
    
    if os.path.exists(best_checkpoint):
        print(f"📂 Loading best checkpoint: {best_checkpoint}")
        trainer.load_checkpoint(best_checkpoint, resume_training=True)
    elif os.path.exists(final_checkpoint):
        print(f"📂 Loading final checkpoint: {final_checkpoint}")
        print("   (Best checkpoint not found, using final checkpoint instead)")
        trainer.load_checkpoint(final_checkpoint, resume_training=True)
    else:
        print("⚠️  No checkpoint found. Starting fresh training...")
        print("   (This will train from scratch with improved hyperparameters)")
    
    # Train with improved hyperparameters (matching RLCard DMC)
    print("\n🚀 Starting training with improved hyperparameters (matching RLCard DMC):")
    print("   - Optimizer: RMSProp (was AdamW)")
    print("   - Gradient Clipping: 40.0 (was 1.0)")
    print("   - Learning rate: 0.0001 (reduced from 0.0005)")
    print("   - Dropout: 0.1 (reduced from 0.2)")
    print("   - Epsilon end: 0.01 (reduced from 0.1)")
    print("   - Better learning rate scheduling")
    print("=" * 80 + "\n")
    
    trainer.train()

if __name__ == "__main__":
    main()

