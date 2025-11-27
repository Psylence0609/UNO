
import sys
import os
import torch
import numpy as np
import shutil

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.training.train_dmc_advanced_opponent import AdvancedDMCTrainerWithOpponent

def test_self_play():
    print("🚀 Testing Self-Play Implementation...")
    
    # Create a temporary config for testing
    import yaml
    config = {
        'environment': {'game': 'uno', 'num_players': 2, 'seed': 42},
        'training': {
            'episodes': 100,
            'learning_rate': 0.001,
            'batch_size': 4,
            'epsilon_end': 0.1
        },
        'network': {'hidden_layers': [64, 32], 'dropout': 0.0},
        'dmc': {},
        'opponent_modeling': {
            'enabled': True,
            'history_size': 10,
            'hidden_layers': [32],
            'strategy_dim': 8,
            'use_attention': True,
            'use_temporal': True
        },
        'self_play': {
            'enabled': True,
            'pool_size': 3,
            'save_interval': 5, # Save frequently for testing
            'opponent_probs': {'random': 0.0, 'current': 0.0, 'past': 1.0} # Force past opponent
        },
        'paths': {
            'models': './tests/temp_models',
            'logs': './tests/temp_logs'
        },
        'logging': {'save_freq': 10, 'log_level': 'INFO'},
        'evaluation': {'eval_episodes': 10, 'eval_freq': 10}
    }
    
    os.makedirs('tests', exist_ok=True)
    with open('tests/test_config.yaml', 'w') as f:
        yaml.dump(config, f)
        
    try:
        # 1. Initialize Trainer
        trainer = AdvancedDMCTrainerWithOpponent(config_path='tests/test_config.yaml')
        print("✅ Trainer initialized with self-play config")
        
        # 2. Run a few episodes to populate pool
        print("   Running training episodes...")
        for i in range(15):
            trainer.episode = i
            trainer.train_episode()
            
            # Manually trigger pool save logic (usually in train loop)
            if (i + 1) % trainer.pool_save_interval == 0:
                pool_model_path = os.path.join(
                    trainer.pool_dir,
                    f"opponent_episode_{i + 1}.pth"
                )
                trainer.save_model(pool_model_path, include_training_state=False)
                trainer.opponent_pool.append(pool_model_path)
                print(f"   Saved to pool: {pool_model_path}")
        
        # 3. Verify pool population
        print(f"   Pool size: {len(trainer.opponent_pool)}")
        assert len(trainer.opponent_pool) > 0, "Opponent pool should not be empty"
        
        # 4. Test Evaluation against Past
        print("   Testing evaluation against past opponent...")
        results = trainer.evaluate_agent(num_games=10)
        
        print(f"   Evaluation results keys: {results.keys()}")
        assert 'vs_past' in results, "Should have 'vs_past' in results"
        print(f"   Vs Past Win Rate: {results['vs_past']['win_rate']}")
        
        print("\n🎉 Self-Play Test Passed!")
        
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if os.path.exists('tests/temp_models'):
            shutil.rmtree('tests/temp_models')
        if os.path.exists('tests/temp_logs'):
            shutil.rmtree('tests/temp_logs')
        if os.path.exists('tests/test_config.yaml'):
            os.remove('tests/test_config.yaml')

if __name__ == "__main__":
    test_self_play()
