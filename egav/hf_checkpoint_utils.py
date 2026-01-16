# egav/hf_checkpoint_utils.py
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
import json

def download_latest_checkpoint(repo_id: str, local_dir: str, token: str = None) -> str:
    """
    Download the latest checkpoint from HF Hub.
    
    Returns:
        Path to the checkpoint folder or None if no checkpoint exists.
    """
    try:
        api = HfApi(token=token)
        
        # List all files in repo
        files = api.list_repo_files(repo_id, token=token)
        
        # Find all checkpoint folders
        checkpoints = [f for f in files if f.startswith("checkpoint-") and "/" in f]
        
        if not checkpoints:
            print("ℹ️ No checkpoints found in HF repo. Starting from scratch.")
            return None
        
        # Extract checkpoint numbers
        checkpoint_nums = []
        for cp in checkpoints:
            try:
                num = int(cp.split("checkpoint-")[1].split("/")[0])
                checkpoint_nums.append(num)
            except:
                continue
        
        if not checkpoint_nums:
            return None
        
        # Get latest checkpoint
        latest_step = max(checkpoint_nums)
        latest_checkpoint = f"checkpoint-{latest_step}"
        
        print(f"\n📥 Downloading latest checkpoint: {latest_checkpoint}")
        
        # Download entire repo
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            token=token,
            allow_patterns=[f"{latest_checkpoint}/*", "trainer_state.json"],
        )
        
        checkpoint_path = Path(local_dir) / latest_checkpoint
        
        if checkpoint_path.exists():
            print(f"✅ Downloaded to: {checkpoint_path}")
            return str(checkpoint_path)
        else:
            print("⚠️ Checkpoint download incomplete")
            return None
            
    except Exception as e:
        print(f"⚠️ Download failed: {e}")
        print("   Starting training from scratch...")
        return None