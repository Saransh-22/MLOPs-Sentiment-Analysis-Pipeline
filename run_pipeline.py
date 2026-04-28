"""
MLOps Pipeline Orchestrator
Executes: preprocessing -> training -> saves artifacts
"""

import subprocess
import sys
from pathlib import Path


def run_command(command, step_name):
    """
    Run a shell command and handle errors.
    
    Args:
        command (str): Command to run
        step_name (str): Name of the step for logging
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"[*] {step_name}...")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            cwd=Path(__file__).parent,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
        print(f"[*] {step_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[!] {step_name} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"[!] {step_name} failed with error: {str(e)}")
        return False


def main():
    """
    Main pipeline orchestration.
    """
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS - MLOps PIPELINE")
    print("="*60)
    
    # Step 1: Preprocessing
    success = run_command(
        f'"{sys.executable}" "src/preprocess.py"',
        "Preprocessing"
    )
    
    if not success:
        print("\n[!] Pipeline failed at preprocessing step")
        sys.exit(1)
    
    # Step 2: Training with MLflow
    success = run_command(
        f'"{sys.executable}" "src/train_mlflow.py"',
        "Training"
    )
    
    if not success:
        print("\n[!] Pipeline failed at training step")
        sys.exit(1)
    
    # Success message
    print("\n" + "="*60)
    print("[*] Done")
    print("="*60)
    print("\nPipeline completed successfully!")
    print("\nNext steps:")
    print("  1. View MLflow: mlflow ui")
    print("  2. Start API: uvicorn app.main:app --reload")
    print("  3. Or run with Docker: docker build -t sentiment-api . && docker run -p 8000:8000 sentiment-api")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
