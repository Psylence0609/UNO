"""
RLCard Built-in DMC Trainer Script
Uses RLCard's native DMCTrainer for training on UNO.
Reference: https://rlcard.org/rlcard.agents.html#rlcard.agents.dmc_agent.trainer
"""

import os
import sys
import rlcard
import torch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Import RLCard's DMC Trainer
from rlcard.agents.dmc_agent.trainer import DMCTrainer


def main():
    """Main function to start RLCard's built-in DMC training."""
    print(" RLCard Built-in DMC Training")
    print("=" * 50)
    
    # Initialize UNO environment
    print(" Initializing UNO environment...")
    env = rlcard.make('uno', config={'seed': 42})
    print(f" Environment created: {env.num_actions} actions, {env.num_players} players")
    print()
    
    # Check device availability
    # Note: RLCard's DMCTrainer only supports CUDA or CPU, not MPS
    if torch.cuda.is_available():
        device = "0"  # RLCard uses string "0" for first GPU
        print(f" Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = ""  # Fall back to CPU - RLCard doesn't support MPS
        print(f"  MPS (Apple Silicon) detected but RLCard's DMCTrainer doesn't support it")
        print(f"   Falling back to CPU (training will be slower)")
    else:
        device = ""  # Empty string for CPU in RLCard
        print(f"  Using CPU (training will be slower)")
    
    print(f"Device for RLCard: {device if device else 'CPU'}")
    print()
    
    # Training configuration
    # Note: RLCard's DMCTrainer uses frames, not episodes
    TOTAL_FRAMES = 100_000_000  # 10M frames (adjust based on your needs)
    SAVE_INTERVAL = 30  # minutes between checkpoints
    
    print(" Training Configuration:")
    print(f"   Total Frames: {TOTAL_FRAMES:,}")
    print(f"   Save Interval: {SAVE_INTERVAL} minutes")
    print(f"   Batch Size: 32")
    print(f"   Learning Rate: 0.0001")
    print(f"   Unroll Length: 100")
    print(f"   Number of Actors: 5")
    print(f"   Exploration Epsilon: 0.01")
    print()
    
    # Create RLCard's DMCTrainer
    print(" Creating RLCard DMCTrainer...")
    trainer = DMCTrainer(
        env=env,
        cuda=device,  # Device string for RLCard
        is_pettingzoo_env=False,
        load_model=False,  # Set to True to load existing model
        xpid='uno_rlcard_dmc',  # Experiment ID
        save_interval=SAVE_INTERVAL,
        num_actor_devices=1,  # Number of devices for simulation
        num_actors=5,  # Parallel games for faster experience gathering
        training_device=device,  # Device for training
        savedir=os.path.join(project_root, 'experiments/rlcard_dmc_uno_100M'),
        total_frames=TOTAL_FRAMES,
        exp_epsilon=0.01,  # Exploration probability
        batch_size=32,
        unroll_length=100,  # Steps before update
        num_buffers=50,  # Shared-memory buffers
        num_threads=4,  # Learner threads
        max_grad_norm=40,
        learning_rate=0.0001,
        alpha=0.99,  # RMSProp smoothing
        momentum=0,  # RMSProp momentum
        epsilon=1e-05,  # RMSProp epsilon
    )
    
    print(" Trainer created successfully")
    print()
    print(" Starting training...")
    print("=" * 50)
    print("Note: RLCard's DMCTrainer uses an actor-learner architecture")
    print("      with parallel actors for efficient training.")
    print("      Training will continue until total_frames is reached.")
    print("=" * 50)
    print()
    
    # Start training
    try:
        trainer.start()
    except KeyboardInterrupt:
        print("\n  Training interrupted by user")
        print(" Checkpoint should be saved in experiments/rlcard_dmc_uno/")
    except Exception as e:
        print(f"\n Training error: {e}")
        raise
    
    print("\n" + "=" * 50)
    print(" TRAINING COMPLETED!")
    print("=" * 50)
    print(f" Results saved to: {os.path.join(project_root, 'experiments/rlcard_dmc_uno')}")
    print("=" * 50)


if __name__ == "__main__":
    main()

