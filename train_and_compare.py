"""
Comprehensive training and comparison of DQN vs DMC+MCTS agents.
Both use the same architecture and MCTS reward shaping for fair comparison.
"""

import os
import sys
import yaml
import torch
import time
from concurrent.futures import ThreadPoolExecutor
import subprocess

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_training(script_name, model_name):
    """Run training script and return results."""
    print(f"\n🚀 Starting {model_name} training...")
    start_time = time.time()
    
    try:
        # Run the training script
        result = subprocess.run([
            "python", script_name
        ], capture_output=True, text=True, cwd="/Users/praneetsurabhi/Desktop/SEM_3/RL/UNO")
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {model_name} training completed in {duration:.1f} seconds")
            return {
                'model': model_name,
                'success': True,
                'duration': duration,
                'output': result.stdout,
                'error': result.stderr
            }
        else:
            print(f"❌ {model_name} training failed")
            return {
                'model': model_name,
                'success': False,
                'duration': duration,
                'output': result.stdout,
                'error': result.stderr
            }
            
    except Exception as e:
        print(f"❌ Error running {model_name}: {e}")
        return {
            'model': model_name,
            'success': False,
            'duration': 0,
            'error': str(e)
        }

def main():
    """Main comparison function."""
    print("🎯 UNO RL Agents Comprehensive Training & Comparison")
    print("=" * 60)
    
    # Check device availability
    if torch.backends.mps.is_available():
        device = "MPS (Apple Silicon GPU)"
    elif torch.cuda.is_available():
        device = "CUDA GPU"
    else:
        device = "CPU"
    print(f"Using device: {device}")
    
    # Load config to show training parameters
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\nTraining Configuration:")
    print(f"  Episodes: {config['training']['episodes']}")
    print(f"  Architecture: {config['network']['hidden_layers']}")
    print(f"  MCTS Simulations: {config['mcts']['num_simulations']}")
    print(f"  Batch Size: {config['training']['batch_size']}")
    print(f"  Learning Rate: {config['training']['learning_rate']}")
    
    # Training scripts to run
    training_jobs = [
        ("src/training/train_dqn_new.py", "DQN+MCTS"),
        ("src/training/train_dmc.py", "DMC+MCTS")
    ]
    
    print(f"\n📋 Training Plan:")
    print(f"  1. DQN with MCTS reward shaping")
    print(f"  2. DMC with MCTS reward shaping")
    print(f"  3. Comprehensive evaluation")
    
    print("\n🚀 Starting training automatically...")
    
    # Train models sequentially (to avoid resource conflicts)
    results = []
    for script, model_name in training_jobs:
        result = run_training(script, model_name)
        results.append(result)
        
        # Brief pause between trainings
        time.sleep(2)
    
    # Print training summary
    print(f"\n📊 TRAINING SUMMARY")
    print("=" * 40)
    
    for result in results:
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        duration = f"{result['duration']:.1f}s" if result['success'] else "N/A"
        print(f"{result['model']:12} | {status} | {duration}")
        
        if not result['success'] and 'error' in result:
            print(f"  Error: {result['error'][:100]}...")
    
    # Check which models were successfully trained
    successful_models = [r['model'] for r in results if r['success']]
    
    if len(successful_models) >= 2:
        print(f"\n🎉 Both models trained successfully!")
        print(f"✅ Ready for comprehensive evaluation")
        
        # Run evaluation
        print(f"\n🔍 Starting comprehensive evaluation...")
        try:
            subprocess.run([
                "python", "src/evaluation/comprehensive_eval.py"
            ], cwd="/Users/praneetsurabhi/Desktop/SEM_3/RL/UNO")
            print(f"✅ Evaluation completed")
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            
    elif len(successful_models) == 1:
        print(f"\n⚠️  Only {successful_models[0]} trained successfully")
        print(f"   You can still evaluate against random baseline")
        
    else:
        print(f"\n❌ No models trained successfully")
        print(f"   Check the error messages above for debugging")
    
    print(f"\n📁 Generated Files:")
    print(f"  Models: models/")
    print(f"  Logs: logs/")
    print(f"  Results: results/")
    
    print(f"\n🏁 Training and evaluation process completed!")

if __name__ == "__main__":
    main()