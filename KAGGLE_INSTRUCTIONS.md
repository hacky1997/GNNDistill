# EGAV Training on Kaggle

## Quick Start (Copy-Paste to Kaggle Notebook)

### Option 1: One-liner setup and train

```python
# Run this cell in a Kaggle notebook with GPU enabled
!git clone https://github.com/hacky1997/GNNDistill.git
!pip install -q -r GNNDistill/requirements.txt
!cd GNNDistill && python -m egav.qa_baseline --languages en --seed 42
```

### Option 2: Use the training script

```python
!git clone https://github.com/hacky1997/GNNDistill.git
!python GNNDistill/kaggle_train.py
```

---

## Pushing Results Back to GitHub

### Step 1: Create a GitHub Personal Access Token

1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Give it a name like "Kaggle Training"
4. Select scopes: `repo` (full control of private repositories)
5. Click "Generate token"
6. **Copy the token immediately** (you won't see it again)

### Step 2: Add Token to Kaggle Secrets

1. In your Kaggle notebook, click "Add-ons" → "Secrets"
2. Add a new secret:
   - **Label**: `GITHUB_TOKEN`
   - **Value**: (paste your token)
3. Enable the secret for this notebook

### Step 3: Push from Kaggle

```python
import os
import subprocess

# Get token from Kaggle secrets
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
token = secrets.get_secret("GITHUB_TOKEN")

# Configure git
os.chdir("/kaggle/working/GNNDistill")
!git config user.email "kaggle@example.com"
!git config user.name "Kaggle Training"

# Add results (metrics only - models are too large for git)
!git add -f runs/baseline/*/eval_metrics.json
!git add -f runs/baseline/*/predictions_dev.json

# Commit
!git commit -m "Add Kaggle training results"

# Push with token
!git remote set-url origin https://hacky1997:{token}@github.com/hacky1997/GNNDistill.git
!git push
```

---

## Saving Large Models

For large model files (checkpoints), use one of these options:

### Option A: Save as Kaggle Dataset

```python
# Models will be in /kaggle/working/GNNDistill/runs/baseline/seed_42/
# After training, save the notebook - outputs are automatically saved
```

### Option B: Upload to Hugging Face Hub

```python
!pip install huggingface_hub
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="/kaggle/working/GNNDistill/runs/baseline/seed_42",
    repo_id="hacky1997/egav-baseline",
    repo_type="model",
)
```

---

## Full Training Pipeline

```python
# Cell 1: Clone and install
!git clone https://github.com/hacky1997/GNNDistill.git
%cd GNNDistill
!pip install -q -r requirements.txt

# Cell 2: Train baseline QA model (trains on SQuAD, evals on MLQA)
!python -m egav.qa_baseline --languages en --seed 42

# Cell 3: Generate candidate spans
!python -m egav.candidates \
    --model runs/baseline/seed_42 \
    --output runs/baseline/candidates_dev.jsonl \
    --split validation \
    --languages en

# Cell 4: Train MLP verifier
!python -m egav.train_verifier \
    --candidates runs/baseline/candidates_dev.jsonl \
    --output runs/verifier/seed_42 \
    --lang en

# Cell 5: Run inference with reranking + abstention
!python -m egav.inference \
    --candidates runs/baseline/candidates_dev.jsonl \
    --verifier runs/verifier/seed_42/verifier_mlp.pt \
    --output runs/results/preds_dev.jsonl \
    --gamma 0.5 --tau_correct 0.5 --tau_margin 0.1

# Cell 6: Push results to GitHub (see above for token setup)
```

---

## Troubleshooting

### "Dataset scripts are no longer supported"

This is handled automatically. The code downloads MLQA JSON files directly from Facebook.

### Out of memory

Reduce batch size in `egav/config.py`:
```python
per_device_train_batch_size: int = 4  # Reduce from 8
```

### Training too slow

Use Kaggle's GPU accelerator (P100 or T4).
