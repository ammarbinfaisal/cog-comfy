from pathlib import Path
from predict import Predictor

def main():
    # Initialize the predictor
    predictor = Predictor()
    
    # Setup the ComfyUI server and required models
    print("Setting up predictor...")
    predictor.setup()
    
    # Prepare inputs
    input_image = Path("image.jpg")  # Replace with your image path
    
    # Parameters for the prediction
    params = {
        "prompt": "A beautiful landscape with vibrant colors",
        "negative_prompt": "cgi, render, blurred, semi-realistic, digital, unrealistic, ugly, low-quality, bad quality",
        "image": input_image,
        "output_format": "png",  # Supported formats: jpg, jpeg, png, webp
        "output_quality": 95,    # 1-100
        "seed": 42              # Optional: specific seed for reproducibility
    }
    
    try:
        # Run the prediction
        print("Running prediction...")
        output_files = predictor.predict(**params)
        
        # Print the paths of generated files
        print("\nGenerated files:")
        for output_file in output_files:
            print(f"- {output_file}")
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        
if __name__ == "__main__":
    main()