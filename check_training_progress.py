#!/usr/bin/env python3
"""Check the progress of ongoing training."""

import os
import glob

print("🔍 Training Progress Check")
print("=" * 50)

# Check if training is running
import subprocess
try:
    result = subprocess.run(['pgrep', '-f', 'train_and_compare.py'], 
                          capture_output=True, text=True)
    if result.stdout.strip():
        print("✅ Training is currently running")
        print(f"   Process IDs: {result.stdout.strip()}")
    else:
        print("⚠️  Training process not found")
except:
    print("⚠️  Could not check process status")

# Check log files
print("\n📋 Recent Log Files:")
log_files = glob.glob("logs/*/training.log")
for log_file in sorted(log_files)[-5:]:
    size = os.path.getsize(log_file) / 1024  # KB
    print(f"   {log_file}: {size:.1f} KB")

# Check models
print("\n💾 Saved Models:")
model_files = glob.glob("models/*.pth")
for model_file in sorted(model_files):
    size = os.path.getsize(model_file) / 1024  # KB
    mtime = os.path.getmtime(model_file)
    from datetime import datetime
    mod_time = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
    print(f"   {model_file}: {size:.1f} KB (modified: {mod_time})")

# Check training output
print("\n📄 Training Output (last 20 lines):")
if os.path.exists("training_output.log"):
    with open("training_output.log", 'r') as f:
        lines = f.readlines()
        for line in lines[-20:]:
            print(f"   {line.rstrip()}")
else:
    print("   training_output.log not found")

print("\n" + "=" * 50)
print("Run this script periodically to check progress!")
