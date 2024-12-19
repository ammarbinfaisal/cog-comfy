import os
import mimetypes
import json
import shutil
from typing import List, Optional
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images
from cog_model_helpers import seed as seed_helper

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

# Ensure proper MIME type handling
mimetypes.add_type("image/webp", ".webp")

class Predictor(BasePredictor):
    def __init__(self):
        """Initialize ComfyUI server and download required model weights"""
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        # Load workflow to analyze required weights
        workflow = self._load_workflow()

        # Required weights based on the workflow
        required_weights = [
            "realvisxlV40_v40Bakedvae.safetensors"
        ]
        
        self.comfyUI.handle_weights(workflow, weights_to_download=required_weights)

    def _load_workflow(self) -> dict:
        """Load workflow from JSON file"""
        try:
            with open("workflow_api.json", "r") as file:
                return json.load(file)
        except FileNotFoundError:
            raise RuntimeError("workflow_api.json not found")
        except json.JSONDecodeError:
            raise RuntimeError("Invalid JSON in workflow_api.json")

    def _handle_input_file(self, input_file: Path, prefix: str) -> str:
        """
        Copy input file to input directory with preserved extension
        Returns: filename in input directory
        """
        if not input_file.exists():
            raise ValueError(f"Input file does not exist: {input_file}")
            
        extension = os.path.splitext(input_file.name)[1]
        filename = f"{prefix}{extension}"
        input_path = os.path.join(INPUT_DIR, filename)
        
        shutil.copy(input_file, input_path)
        return filename

    def _update_workflow(self, workflow: dict, **kwargs) -> None:
        """Update workflow nodes based on input parameters"""
        updates = {
            "prompt": (["4", "inputs", "text"], kwargs.get("prompt")),
            "negative_prompt": (["5", "inputs", "text"], kwargs.get("negative_prompt")),
            "image_filename": (
                [("242", "inputs", "image"), ("155", "inputs", "image")],
                kwargs.get("image_filename")
            ),
            "image_2": (["155", "inputs", "image"], kwargs.get("image_filename")),
            "seed": (["81", "inputs", "seed"], kwargs.get("seed"))
        }

        for key, (paths, value) in updates.items():
            if value is not None:
                if isinstance(paths[0], tuple):  # Multiple paths to update
                    for path in paths:
                        self._set_nested_value(workflow, path, value)
                else:  # Single path
                    self._set_nested_value(workflow, paths, value)

    def _set_nested_value(self, obj: dict, path: List[str], value: any) -> None:
        """Helper method to set nested dictionary value"""
        for key in path[:-1]:
            obj = obj[key]
        obj[path[-1]] = value

    def predict(
        self,
        prompt: str = Input(
            description="Main prompt describing what you want to see in the image",
            default="",
        ),
        negative_prompt: str = Input(
            description="Things you do not want to see in your image",
            default="cgi, render, blured, semi-realistic, digital, unrealistic, ugly, low-quality, bad quality",
        ),
        image: Path = Input(
            description="Input image to be processed",
            default=None,
        ),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
        seed: int = seed_helper.predict_seed(),
    ) -> List[Path]:
        """Run prediction on the model"""
        try:
            if image is None:
                raise ValueError("An input image is required for this workflow")
                
            # Add image format validation
            valid_formats = ['.jpg', '.jpeg', '.png', '.webp']
            if not any(str(image).lower().endswith(ext) for ext in valid_formats):
                raise ValueError(f"Image must be one of these formats: {', '.join(valid_formats)}")

            # Try to validate image dimensions
            from PIL import Image
            with Image.open(image) as img:
                if img.mode not in ['RGB', 'RGBA']:
                    img = img.convert('RGB')

            # Clean up previous runs
            self.comfyUI.cleanup(ALL_DIRECTORIES)

            # Handle input image and seed
            image_filename = self._handle_input_file(image, "image")
            actual_seed = seed_helper.generate(seed)

            # Load and update workflow
            workflow = self._load_workflow()
            self._update_workflow(
                workflow,
                prompt=prompt,
                negative_prompt=negative_prompt,
                image_filename=image_filename,
                seed=actual_seed,
            )

            # Execute workflow
            self.comfyUI.connect()
            wf = self.comfyUI.load_workflow(workflow)
            self.comfyUI.run_workflow(wf)

            # Get and optimize output files
            output_files = self.comfyUI.get_files(OUTPUT_DIR)
            if not output_files:
                raise RuntimeError("No output files generated")

            return optimise_images.optimise_image_files(
                output_format,
                output_quality,
                output_files
            )

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
