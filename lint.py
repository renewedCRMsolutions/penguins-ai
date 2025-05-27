# File: penguins_ai/lint.py
import subprocess
import sys
import os

def run_lint():
    """Run linting on key files only"""
    
    # Define files to lint
    files_to_check = [
        "api/main.py",
        "train/train_nhl_optimized.py",
        "models/*.py",
        "config/*.py"
    ]
    
    print("🔍 Running Black formatter...")
    subprocess.run([sys.executable, "-m", "black"] + files_to_check + ["--check"])
    
    print("\n🔍 Running Flake8...")
    subprocess.run([sys.executable, "-m", "flake8"] + files_to_check + ["--max-line-length=120"])

if __name__ == "__main__":
    run_lint()