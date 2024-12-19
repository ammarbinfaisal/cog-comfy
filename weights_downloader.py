import subprocess
import time
import os
from weights_manifest import WeightsManifest

class WeightsDownloader:
    supported_filetypes = [
        ".ckpt",
        ".safetensors",
        ".pt",
        ".pth",
        ".bin",
        ".onnx",
        ".torchscript",
        ".engine",
        ".patch",
    ]

    def __init__(self):
        self.weights_manifest = WeightsManifest()
        self.weights_map = self.weights_manifest.weights_map

    def get_weights_by_type(self, type):
        return self.weights_manifest.get_weights_by_type(type)

    def download_weights(self, weight_str):
        if weight_str in self.weights_map:
            if self.weights_manifest.is_non_commercial_only(weight_str):
                print(
                    f"⚠️  {weight_str} is for non-commercial use only. Unless you have obtained a commercial license.\nDetails: https://github.com/fofr/cog-comfyui/blob/main/weights_licenses.md"
                )

            if isinstance(self.weights_map[weight_str], list):
                for weight in self.weights_map[weight_str]:
                    self.download_if_not_exists(
                        weight_str, weight["url"], weight["dest"]
                    )
            else:
                self.download_if_not_exists(
                    weight_str,
                    self.weights_map[weight_str]["url"],
                    self.weights_map[weight_str]["dest"],
                )
        else:
            raise ValueError(
                f"{weight_str} unavailable. View the list of available weights: https://github.com/fofr/cog-comfyui/blob/main/supported_weights.md"
            )

    def check_if_file_exists(self, weight_str, dest):
        if dest.endswith(weight_str):
            path_string = dest
        else:
            path_string = os.path.join(dest, weight_str)
        return os.path.exists(path_string)

    def download_if_not_exists(self, weight_str, url, dest):
        if self.check_if_file_exists(weight_str, dest):
            print(f"✅ {weight_str} exists in {dest}")
            return
        
        if weight_str == "realvisxlV40_v40Bakedvae.safetensors":
            self.download_realvis_xl_v40(dest)
        else:
            self.download(weight_str, url, dest)

    def download_realvis_xl_v40(self, dest):
        """
        Special handling for realvisxlV40_v40Bakedvae.safetensors from Hugging Face
        using huggingface_hub library
        """
        from huggingface_hub import hf_hub_download
        import shutil
        
        weight_str = "realvisxlV40_v40Bakedvae.safetensors"
        repo_id = "frankjoshua/realvisxlV40_v40Bakedvae"
        filename = "realvisxlV40_v40Bakedvae.safetensors"
        
        # Define paths
        final_path = os.path.join(dest, weight_str)
        
        # Check if file already exists with correct size
        if os.path.exists(final_path):
            expected_size = 7_456_800_000  # ~6.94GB
            if os.path.getsize(final_path) >= expected_size:
                print(f"✅ {weight_str} already exists in {dest}")
                return
        
        print(f"⏳ Downloading {weight_str} from Hugging Face")
        
        try:
            start = time.time()
            
            # Download directly to final location
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                local_dir=dest,
                force_download=False  # Ensure we get a fresh download
            )
            
            elapsed_time = time.time() - start
            
            file_size_bytes = os.path.getsize(downloaded_path)
            file_size_gigabytes = file_size_bytes / (1024 * 1024 * 1024)
            print(
                f"✅ {weight_str} downloaded to {dest} in {elapsed_time:.2f}s, size: {file_size_gigabytes:.2f}GB"
            )
                    
        except Exception as e:
            print(f"❌ Failed to download {weight_str} from Hugging Face: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise

    @staticmethod
    def download(weight_str, url, dest):
        if "/" in weight_str:
            subfolder = weight_str.rsplit("/", 1)[0]
            dest = os.path.join(dest, subfolder)
            os.makedirs(dest, exist_ok=True)

        print(f"⏳ Downloading {weight_str} to {dest}")
        start = time.time()
        subprocess.check_call(
            ["pget", "--log-level", "warn", "-xf", url, dest], close_fds=False
        )
        elapsed_time = time.time() - start
        try:
            file_size_bytes = os.path.getsize(
                os.path.join(dest, os.path.basename(weight_str))
            )
            file_size_megabytes = file_size_bytes / (1024 * 1024)
            print(
                f"✅ {weight_str} downloaded to {dest} in {elapsed_time:.2f}s, size: {file_size_megabytes:.2f}MB"
            )
        except FileNotFoundError:
            print(f"✅ {weight_str} downloaded to {dest} in {elapsed_time:.2f}s")

    def delete_weights(self, weight_str):
        if weight_str in self.weights_map:
            weight_path = os.path.join(self.weights_map[weight_str]["dest"], weight_str)
            if os.path.exists(weight_path):
                os.remove(weight_path)
                print(f"Deleted {weight_path}")