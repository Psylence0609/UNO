"""
Quick test of the training infrastructure.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from src.training.train_dqn import DQNTrainer

def test_training():
    """Test the training infrastructure with a short run."""
    print("🧪 TESTING TRAINING INFRASTRUCTURE")
    print("=" * 50)
    
    # Create a test config
    test_config = {
        'training': {
            'episodes': 100,  # Short test
            'batch_size': 32,
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'memory_size': 1000,  # Smaller memory
            'target_update_freq': 100
        },
        'dqn': {
            'hidden_layers': [128, 64],  # Smaller network
            'activation': 'relu',
            'dropout': 0.1,
            'dueling': True,
            'double_dqn': True
        },
        'environment': {
            'seed': 42
        },
        'evaluation': {
            'eval_episodes': 100,  # Fewer eval episodes
            'eval_freq': 50  # Evaluate more frequently
        },
        'logging': {
            'save_freq': 50
        },
        'paths': {
            'models': './models',
            'logs': './logs'
        }
    }
    
    # Save test config
    with open('test_config.yaml', 'w') as f:
        yaml.dump(test_config, f)
    
    print("✅ Test configuration created")
    
    # Create trainer
    trainer = DQNTrainer('test_config.yaml')
    
    print(f"✅ Trainer created")
    print(f"   State size: {trainer.state_size}")
    print(f"   Action size: {trainer.env.num_actions}")
    print(f"   Episodes: {trainer.config['training']['episodes']}")
    
    # Test single episode
    print("\n🎯 Testing single episode...")
    episode_data = trainer.train_episode()
    print(f"Episode completed:")
    print(f"  Reward: {episode_data['reward']:.2f}")
    print(f"  Length: {episode_data['length']}")
    print(f"  Epsilon: {episode_data['epsilon']:.3f}")
    
    # Test evaluation
    print("\n📊 Testing evaluation...")
    eval_data = trainer.evaluate_agent(num_games=50)
    print(f"Evaluation completed:")
    print(f"  Win rate: {eval_data['win_rate']:.1%}")
    print(f"  Avg game length: {eval_data['avg_game_length']:.1f}")
    
    # Test model saving
    print("\n💾 Testing model saving...")
    model_path = trainer.save_model("models/test_model.pth")
    print(f"Model saved to: {model_path}")
    
    print("\n✅ ALL TESTS PASSED!")
    print("\n📝 Ready for full training run")

if __name__ == "__main__":
    test_training()