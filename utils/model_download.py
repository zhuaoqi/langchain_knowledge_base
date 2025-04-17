from huggingface_hub import hf_hub_download
import os

def download_model(model_name, save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)
    
    # Download model files
    hf_hub_download(
        repo_id=model_name,
        filename="pytorch_model.bin",
        local_dir=save_dir
    )
    
    hf_hub_download(
        repo_id=model_name,
        filename="config.json",
        local_dir=save_dir
    )