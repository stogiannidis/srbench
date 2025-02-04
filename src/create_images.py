import re
import random
import json
from typing import Any, Dict
from diffusers import FluxPipeline, StableDiffusion3Pipeline, StableDiffusionPipeline

# -------------------------------
# Function to create the appropriate pipeline based on model ID.
# -------------------------------
def create_pipeline(model_id: str) -> Any:
    if model_id == "stabilityai/stable-diffusion-3.5-large":
        return StableDiffusion3Pipeline.from_pretrained(
            model_id,
            # torch_dtype=torch.bfloat16,
            cache_dir="bin/models",
            device_map="balanced",
        )
    elif model_id == "black-forest-labs/FLUX.1-dev":
        return FluxPipeline.from_pretrained(
            model_id,
            # torch_dtype=torch.bfloat16,
            cache_dir="bin/models",
            device_map="balanced",
        )
    else:
        return StableDiffusionPipeline.from_pretrained(
            model_id,
            # torch_dtype=torch.float16,
            cache_dir="bin/models",
            device_map="balanced",
        )

# -------------------------------
# Function to generate an image using a given prompt and model.
# Returns the path of the saved image.
# -------------------------------
def generate_image(prompt: str, model_id: str, output_filename: str, steps: int = 30, scale: float = 7.5) -> str:
    pipe = create_pipeline(model_id)
    image = pipe(prompt, num_inference_steps=steps, guidance_scale=scale).images[0]
    image_file_path = "output/images/" + output_filename
    image.save(image_file_path)
    print(f"Image saved to {image_file_path}")
    return image_file_path

# -------------------------------
# Function to load the prompt (and metadata) from a JSON file.
# -------------------------------
def load_metadata_from_json(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

# -------------------------------
# Post-Processing Function: Removes text between <think> and </think> tags.
# -------------------------------
def post_process_output(output_text: str) -> str:
    cleaned_text: str = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL)
    return cleaned_text.strip()

# -------------------------------
# Main Routine
# -------------------------------
def main() -> None:
    # JSON file containing the generated prompt and metadata.
    json_file = "llm_scene_description.json"
    metadata: Dict[str, Any] = load_metadata_from_json(json_file)
    
    # Extract the prompt from the metadata.
    prompt: str = metadata.get("generated_scene_description", "").strip()
    if not prompt:
        print("No prompt found in the JSON file.")
        return

    # Optionally, post-process the prompt in case it contains any extraneous text.
    prompt = post_process_output(prompt)
    
    # Print the loaded prompt for verification.
    print("Loaded prompt from JSON:")
    print(prompt)
    
    # Define the diffusion model to be used and output image file name.
    diffusion_model: str = "black-forest-labs/FLUX.1-dev"  # Change to desired model ID.
    output_image_filename: str = "generated_scene.png"
    
    # Generate the image using the prompt from the JSON.
    image_file_path = generate_image(prompt, diffusion_model, output_image_filename, steps=50, scale=7.5)
    
    # Update metadata with diffusion model name and generated image file path.
    metadata["diffusion_model"] = diffusion_model
    metadata["output_image_file"] = image_file_path
    
    # Save the updated metadata (with image details) to a new JSON file.
    output_metadata_file = "llm_scene_description_with_image.json"
    with open(output_metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)
    
    print(f"\nUpdated metadata saved to '{output_metadata_file}'.")

if __name__ == "__main__":
    main()
