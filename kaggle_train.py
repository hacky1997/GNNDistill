#!/usr/bin/env python3
"""
EGAV Training Script for Kaggle

Usage on Kaggle:
1. Create a new Kaggle notebook with GPU enabled
2. Add this repo as a dataset OR clone it (see below)
3. Run this script

To push results back to GitHub, you need to set up GitHub credentials as Kaggle secrets:
- GITHUB_TOKEN: Personal access token with repo scope
- GITHUB_USERNAME: Your GitHub username (e.g., hacky1997)
"""

import os
import subprocess
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================
REPO_URL = "https://github.com/hacky1997/GNNDistill.git"
REPO_NAME = "GNNDistill"
GITHUB_USERNAME = "hacky1997"

# Training config
LANGUAGES = "en,de,es,ar,hi,vi,zh"  # All MLQA languages
SEED = 42
NUM_EPOCHS = 200  # Full training

# =============================================================================
# STEP 1: Setup environment
# =============================================================================
def setup_environment():
    """Install dependencies and clone repo."""
    print("=" * 60)
    print("STEP 1: Setting up environment")
    print("=" * 60)
    
    # Check if running on Kaggle
    is_kaggle = os.path.exists("/kaggle")
    work_dir = "/kaggle/working" if is_kaggle else "."
    os.chdir(work_dir)
    
    # Clone repo if not exists
    if not os.path.exists(REPO_NAME):
        print(f"Cloning {REPO_URL}...")
        subprocess.run(["git", "clone", REPO_URL], check=True)
    else:
        print(f"Repo already exists, pulling latest...")
        os.chdir(REPO_NAME)
        subprocess.run(["git", "pull"], check=True)
        os.chdir(work_dir)
    
    # Install requirements
    print("Installing requirements...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "-r", f"{REPO_NAME}/requirements.txt"
    ], check=True)
    
    # Add repo to path
    repo_path = os.path.join(work_dir, REPO_NAME)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    
    return repo_path


# =============================================================================
# STEP 2: Train baseline QA model
# =============================================================================
def train_baseline(repo_path):
    """Train the baseline XLM-R QA model."""
    print("=" * 60)
    print("STEP 2: Training baseline QA model")
    print("=" * 60)
    
    os.chdir(repo_path)
    
    # Run training
    cmd = [
        sys.executable, "-m", "egav.qa_baseline",
        "--languages", LANGUAGES,
        "--seed", str(SEED),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    print("Baseline training complete!")
    return os.path.join(repo_path, "runs", "baseline", f"seed_{SEED}")


# =============================================================================
# STEP 3: Generate candidates (Top-K spans)
# =============================================================================
def generate_candidates(repo_path, model_path):
    """Generate top-K candidate spans for verification."""
    print("=" * 60)
    print("STEP 3: Generating candidate spans")
    print("=" * 60)
    
    os.chdir(repo_path)
    
    output_path = os.path.join(repo_path, "runs", "baseline", "candidates_dev.jsonl")
    
    cmd = [
        sys.executable, "-m", "egav.candidates",
        "--model", model_path,
        "--output", output_path,
        "--split", "validation",
        "--languages", LANGUAGES,
    ]
    print(f"Running: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("Candidate generation not implemented yet or failed. Skipping...")
        return None
    
    return output_path


# =============================================================================
# STEP 4: Train verifier (MLP)
# =============================================================================
def train_verifier(repo_path, candidates_path):
    """Train the MLP verifier."""
    print("=" * 60)
    print("STEP 4: Training MLP verifier")
    print("=" * 60)
    
    if candidates_path is None:
        print("No candidates available. Skipping verifier training.")
        return None
    
    os.chdir(repo_path)
    
    output_path = os.path.join(repo_path, "runs", "verifier", f"seed_{SEED}")
    
    cmd = [
        sys.executable, "-m", "egav.train_verifier",
        "--candidates", candidates_path,
        "--output", output_path,
        "--lang", LANGUAGES,
    ]
    print(f"Running: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("Verifier training not implemented yet or failed. Skipping...")
        return None
    
    return output_path


# =============================================================================
# STEP 5: Push results to GitHub
# =============================================================================
def push_to_github(repo_path):
    """Push trained models and results to GitHub."""
    print("=" * 60)
    print("STEP 5: Pushing results to GitHub")
    print("=" * 60)
    
    os.chdir(repo_path)
    
    # Try to get GitHub token from Kaggle secrets
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        github_token = secrets.get_secret("GITHUB_TOKEN")
        github_username = secrets.get_secret("GITHUB_USERNAME") or GITHUB_USERNAME
    except Exception as e:
        print(f"Could not get Kaggle secrets: {e}")
        print("To push to GitHub, add GITHUB_TOKEN as a Kaggle secret.")
        print("Skipping GitHub push.")
        return False
    
    if not github_token:
        print("GITHUB_TOKEN not found in Kaggle secrets. Skipping push.")
        return False
    
    # Configure git
    subprocess.run(["git", "config", "user.email", "kaggle@example.com"], check=True)
    subprocess.run(["git", "config", "user.name", "Kaggle Training"], check=True)
    
    # Update remote URL with token
    remote_url = f"https://{github_username}:{github_token}@github.com/{github_username}/{REPO_NAME}.git"
    subprocess.run(["git", "remote", "set-url", "origin", remote_url], check=True)
    
    # Add results (but not large model files - those should go to HuggingFace Hub)
    # Only add metrics and small artifacts
    files_to_add = [
        "runs/baseline/*/eval_metrics.json",
        "runs/baseline/*/predictions_dev.json",
        "runs/results/*.json",
    ]
    
    for pattern in files_to_add:
        subprocess.run(["git", "add", "-f", pattern], check=False)
    
    # Check if there are changes
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if not result.stdout.strip():
        print("No changes to commit.")
        return True
    
    # Commit and push
    subprocess.run(["git", "commit", "-m", f"Add training results (seed={SEED})"], check=True)
    subprocess.run(["git", "push"], check=True)
    
    print("Results pushed to GitHub!")
    return True


# =============================================================================
# STEP 6: Save model to Kaggle output (for download)
# =============================================================================
def save_to_kaggle_output(repo_path, model_path):
    """Copy model files to Kaggle output directory for easy download."""
    print("=" * 60)
    print("STEP 6: Saving model to Kaggle output")
    print("=" * 60)
    
    import shutil
    
    output_dir = "/kaggle/working/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy the trained model
    if model_path and os.path.exists(model_path):
        dest = os.path.join(output_dir, "baseline_model")
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(model_path, dest)
        print(f"Model saved to {dest}")
    
    # List output files
    print("\nOutput files:")
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  {fpath} ({size:.2f} MB)")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("EGAV Training Pipeline")
    print("=" * 60)
    
    # Step 1: Setup
    repo_path = setup_environment()
    
    # Step 2: Train baseline
    model_path = train_baseline(repo_path)
    
    # Step 3: Generate candidates (optional, may not be fully implemented)
    candidates_path = generate_candidates(repo_path, model_path)
    
    # Step 4: Train verifier (optional)
    verifier_path = train_verifier(repo_path, candidates_path)
    
    # Step 5: Push to GitHub
    push_to_github(repo_path)
    
    # Step 6: Save to Kaggle output
    if os.path.exists("/kaggle"):
        save_to_kaggle_output(repo_path, model_path)
    
    print("=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
