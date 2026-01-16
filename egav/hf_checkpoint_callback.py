# egav/hf_checkpoint_callback.py
import os
from pathlib import Path
from transformers import TrainerCallback
from huggingface_hub import HfApi, create_repo

class HFCheckpointCallback(TrainerCallback):
    """
    Callback to automatically upload checkpoints to Hugging Face Hub.
    
    Usage:
        callback = HFCheckpointCallback(
            repo_id="your-username/model-name",
            token="hf_...",
            upload_every_n_steps=500
        )
    """
    
    def __init__(self, repo_id: str, token: str = None, upload_every_n_steps: int = 500):
        self.repo_id = repo_id
        self.token = token or os.environ.get("HF_TOKEN")
        self.upload_every_n_steps = upload_every_n_steps
        self.api = HfApi(token=self.token)
        
        # Create repo if it doesn't exist
        try:
            create_repo(repo_id, token=self.token, exist_ok=True, private=True)
            print(f"✅ Repo ready: https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"⚠️ Repo creation warning: {e}")
    
    def on_save(self, args, state, control, **kwargs):
        """Called when a checkpoint is saved"""
        checkpoint_folder = f"checkpoint-{state.global_step}"
        checkpoint_path = Path(args.output_dir) / checkpoint_folder
        
        if checkpoint_path.exists():
            try:
                print(f"\n📤 Uploading {checkpoint_folder} to HF Hub...")
                
                # Upload the checkpoint folder
                self.api.upload_folder(
                    folder_path=str(checkpoint_path),
                    repo_id=self.repo_id,
                    path_in_repo=checkpoint_folder,
                    token=self.token,
                    commit_message=f"Checkpoint at step {state.global_step}"
                )
                
                # Also upload training state
                trainer_state_file = Path(args.output_dir) / "trainer_state.json"
                if trainer_state_file.exists():
                    self.api.upload_file(
                        path_or_fileobj=str(trainer_state_file),
                        path_in_repo="trainer_state.json",
                        repo_id=self.repo_id,
                        token=self.token,
                    )
                
                print(f"✅ Uploaded to: https://huggingface.co/{self.repo_id}")
                
            except Exception as e:
                print(f"❌ Upload failed: {e}")
                # Don't crash training if upload fails