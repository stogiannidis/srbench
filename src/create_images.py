import os
import json
import torch
import logging
from typing import Any, Dict, List, Literal
from diffusers import FluxPipeline, StableDiffusion3Pipeline, StableDiffusionPipeline

# -------------------------------
# Logging configuration
# -------------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)  # Create logs directory if it doesn't exist.
LOG_FILE = os.path.join(LOG_DIR, "app.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        # logging.StreamHandler()  # Optional: log to console as well.
    ]
)

# -------------------------------
# Type Aliases
# -------------------------------
Pipeline = Literal[StableDiffusionPipeline, StableDiffusion3Pipeline, FluxPipeline]

# -------------------------------
# Function to create the appropriate pipeline based on model ID.
# -------------------------------
def create_pipeline(model_id: str) -> Any:
    logging.info(f"Creating pipeline for model: {model_id}")
    if model_id == "stabilityai/stable-diffusion-3.5-large":
        return StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
        )
    elif model_id == "black-forest-labs/FLUX.1-dev":
        return FluxPipeline.from_pretrained(
            "bin/models/diffusers/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            device_map="balanced",
        )
    else:
        return StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
        )

# -------------------------------
# Function to generate an image using a given prompt and pipeline.
# Returns the file path where the image was saved.
# -------------------------------
def generate_image(prompt: str, pipe: Pipeline, output_filename: str, output_dir: str, steps: int = 40, scale: float = 4.5) -> str:
    logging.info(f"Generating image with prompt: {prompt[:30]}... using {output_filename}, steps={steps}, scale={scale}")
    image = pipe(prompt, num_inference_steps=steps, guidance_scale=scale).images[0]
    os.makedirs(output_dir, exist_ok=True)
    image_file_path = os.path.join(output_dir, output_filename)
    image.save(image_file_path)
    logging.info(f"Image saved to {image_file_path}")
    return image_file_path

# -------------------------------
# Function to load the prompt (and metadata) from a JSON Lines file.
# -------------------------------
def load_metadata_from_json(json_path: str) -> List[Dict[str, Any]]:
    logging.info(f"Loading metadata from {json_path}")
    with open(json_path, 'r') as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]
    logging.info(f"Loaded {len(data)} metadata items.")
    return data

# -------------------------------
# Main Routine
# -------------------------------
def main() -> None:
    
    logging.info("\n\nStarting the image generation process.")
    
    try:
        # Define the list of diffusion models to use.
        diffusion_models = [
            {
                "model_id": "black-forest-labs/FLUX.1-dev",
                "steps": 50,
                "scale": 4.5,
            },
            {
                "model_id": "stabilityai/stable-diffusion-3.5-large",
                "steps": 50,
                "scale": 5.5,
            },
        ]
        
        # Load all the metadata once.
        json_file = "output/prompts/all-llm-prompts-trial_v1.jsonl"
        metadata_list: List[Dict[str, Any]] = load_metadata_from_json(json_file)
        if not metadata_list:
            logging.error("No metadata found in the JSON file.")
            return

        # Prepare a list to collect output metadata.
        output_metadata = []

        # For each diffusion model, instantiate the pipeline once and process all prompts.
        for diff_model in diffusion_models:
            diffusion_model_id = diff_model["model_id"]
            steps = diff_model.get("steps", 40)
            scale = diff_model.get("scale", 4.5)

            # Create a safe folder name for the model.
            safe_model_id = diffusion_model_id.replace("/", "_")
            output_dir = os.path.join("output", "images", safe_model_id)
            logging.info(f"Initializing pipeline for diffusion model '{diffusion_model_id}'")
            try:
                pipeline = create_pipeline(diffusion_model_id)
            except Exception as e:
                logging.error(f"Failed to create pipeline for model {diffusion_model_id}: {e}")
                continue

            # Process each prompt using the current pipeline.
            for idx, item in enumerate(metadata_list[40:50]):
                prompt = item.get("generated_scene_description")
                prompt_model = item.get("model", "unknown")  # Model that generated the prompt.
                task_type = item.get("task_type", "unknown")   # Task type, if available.
                
                if not prompt:
                    logging.warning(f"Skipping metadata item at index {idx} due to missing prompt.")
                    continue
                
                output_image_filename = f"image_{idx:03d}_{task_type}_{prompt_model}.png"
                logging.info(f"Processing prompt index {idx} with diffusion model '{diffusion_model_id}'")
                try:
                    image_file_path = generate_image(
                        prompt,
                        pipeline,
                        output_image_filename,
                        output_dir,
                        steps=steps,
                        scale=scale
                    )
                except Exception as e:
                    logging.error(f"Error generating image for prompt index {idx} with model {diffusion_model_id}: {e}")
                    continue

                # Record combined metadata.
                record = {
                    "prompt_model": prompt_model,
                    "prompt": prompt,
                    "task_type": task_type,
                    "diffusion_model": diffusion_model_id,
                    "image_path": image_file_path,
                    "generation_details": {
                        "steps": steps,
                        "scale": scale,
                    }
                }
                output_metadata.append(record)
                logging.info(f"Generated image for prompt index {idx} with diffusion model '{diffusion_model_id}'.")

        # Save the combined metadata to a new JSON Lines file.
        output_metadata_file = "all-llm-prompts-trial_v1_metadata.jsonl"
        with open(output_metadata_file, "w") as f:
            for record in output_metadata:
                f.write(json.dumps(record) + "\n")
        logging.info(f"Updated metadata saved to '{output_metadata_file}'.")

    except Exception as err:
        logging.exception(f"An unexpected error occurred: {err}")

if __name__ == "__main__":
    main()
