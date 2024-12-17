import os
import huggingface_hub
from tqdm import tqdm

def download_checkpoint():
    """
    Downloads the RealVisXL V4.0 checkpoint from Hugging Face
    """
    # Define the model info
    repo_id = "frankjoshua/realvisxlV40_v40Bakedvae"
    filename = "realvisxlV40_v40Bakedvae.safetensors"
    
    # Set up the destination directory (ComfyUI models path)
    models_dir = "ComfyUI/models/checkpoints"
    os.makedirs(models_dir, exist_ok=True)
    
    destination_path = os.path.join(models_dir, filename)
    
    # Skip if file already exists
    if os.path.exists(destination_path):
        print(f"Checkpoint already exists at {destination_path}")
        return destination_path
        
    print(f"Downloading {filename} from {repo_id}...")
    
    try:
        # Download the file from Hugging Face
        huggingface_hub.hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=models_dir,
            local_dir_use_symlinks=False,
            token=os.getenv("HF_TOKEN")  # Set this if the model requires authentication
        )
        
        print(f"Successfully downloaded checkpoint to {destination_path}")
        return destination_path
        
    except Exception as e:
        print(f"Error downloading checkpoint: {str(e)}")
        raise

def setup_requirements():
    """
    Ensures required packages are installed
    """
    try:
        import huggingface_hub
    except ImportError:
        print("Installing huggingface_hub...")
        os.system("pip install --quiet huggingface_hub")

if __name__ == "__main__":
    setup_requirements()
    download_checkpoint()