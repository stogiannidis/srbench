import os
import json
import torch
import logging
import argparse
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
        logging.StreamHandler(),  # Log to console as well.
    ],
)

# -------------------------------
# Type Aliases
# -------------------------------
Pipeline = Literal[StableDiffusionPipeline, StableDiffusion3Pipeline, FluxPipeline]


# -------------------------------
# Function to parse command-line arguments.
# -------------------------------
def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments using argparse.

    Returns:
        argparse.Namespace: Parsed command-line arguments including:
            - input: Path to the input JSON Lines file containing prompt metadata.
            - output: Path to the output JSON Lines file for saving metadata with image paths.
            - model: Diffusion model name or ID to use for image generation.
            - steps: Number of inference steps (default is 40).
            - scale: Guidance scale (default is 4.5).
    """
    parser = argparse.ArgumentParser(
        description="Generate images using a diffusion model based on input metadata."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to the input JSON Lines file containing prompt metadata.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to the output JSON Lines file for saving metadata with image paths.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Diffusion model name or ID to use for image generation.",
    )
    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        default=40,
        help="Number of inference steps (default: 40).",
    )
    parser.add_argument(
        "--scale", "-c", type=float, default=4.5, help="Guidance scale (default: 4.5)."
    )
    return parser.parse_args()


# -------------------------------
# Function to create the appropriate pipeline based on model ID.
# -------------------------------
def create_pipeline(model_id: str) -> Any:
    """
    Instantiate and return a diffusion model pipeline based on the provided model ID.

    Args:
        model_id (str): The identifier or name of the diffusion model.

    Returns:
        Any: An instantiated pipeline object for the specified model.

    Raises:
        Exception: If the pipeline instantiation fails.
    """
    logging.info(f"Creating pipeline for model: {model_id}")
    if model_id == "stabilityai/stable-diffusion-3.5-large":
        return StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
        )
    elif model_id == "black-forest-labs/FLUX.1-dev":
        return FluxPipeline.from_pretrained(
            "bin/models/diffusers/black-forest-labs_FLUX.1-dev",
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
# -------------------------------
def generate_image(
    prompt: str,
    pipe: Pipeline,
    output_filename: str,
    output_dir: str,
    steps: int = 40,
    scale: float = 4.5,
) -> str:
    """
    Generate an image from a prompt using the specified diffusion pipeline and save it.

    Args:
        prompt (str): The text prompt for image generation.
        pipe (Pipeline): The instantiated diffusion pipeline.
        output_filename (str): The filename for the generated image.
        output_dir (str): The directory where the image will be saved.
        steps (int, optional): Number of inference steps. Defaults to 40.
        scale (float, optional): Guidance scale. Defaults to 4.5.

    Returns:
        str: The file path where the generated image is saved.
    """
    logging.info(
        f"Generating image with prompt (first 30 chars): {prompt[:30]}... using {output_filename}, steps={steps}, scale={scale}"
    )
    image = pipe(prompt, num_inference_steps=steps, guidance_scale=scale).images[0]
    os.makedirs(output_dir, exist_ok=True)
    image_file_path = os.path.join(output_dir, output_filename)
    image.save(image_file_path)
    logging.info(f"Image saved to {image_file_path}")
    return image_file_path


# -------------------------------
# Function to load metadata from a JSON Lines file.
# -------------------------------
def load_metadata_from_json(json_path: str) -> List[Dict[str, Any]]:
    """
    Load metadata from a JSON Lines file.

    Args:
        json_path (str): The path to the JSON Lines file containing metadata.

    Returns:
        List[Dict[str, Any]]: A list of metadata dictionaries.
    """
    logging.info(f"Loading metadata from {json_path}")
    with open(json_path, "r") as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]
    logging.info(f"Loaded {len(data)} metadata items.")
    return data


# -------------------------------
# Main Routine
# -------------------------------
def main(args: argparse.Namespace) -> None:
    """
    Main routine to generate images based on input metadata and update metadata with image paths.

    Args:
        args (argparse.Namespace): Command-line arguments containing input file, output file,
                                     model ID, steps, and scale.
    """
    try:
        # Retrieve arguments.
        input_file = args.input
        output_metadata_file = args.output
        diffusion_model_id = args.model
        steps = args.steps
        scale = args.scale

        # Load metadata from the input file.
        metadata_list: List[Dict[str, Any]] = load_metadata_from_json(input_file)
        if not metadata_list:
            logging.error("No metadata found in the input file.")
            return

        # Instantiate the pipeline for the provided model.
        logging.info(
            f"Initializing pipeline for diffusion model '{diffusion_model_id}'"
        )
        try:
            pipeline = create_pipeline(diffusion_model_id)
        except Exception as e:
            logging.error(
                f"Failed to create pipeline for model {diffusion_model_id}: {e}"
            )
            return

        # Prepare a safe directory name for images.
        safe_model_id = diffusion_model_id.replace("/", "_")
        output_image_dir = os.path.join("output", "images", safe_model_id)

        # Prepare a list to collect updated metadata.
        output_metadata = []

        # Process each prompt from the metadata.
        for idx, item in enumerate(metadata_list):
            prompt = item.get("generated_scene_description")
            prompt_model = item.get(
                "model", "unknown"
            )  # Model that generated the prompt.
            task_type = item.get("task_type", "unknown")  # Task type, if available.

            if not prompt:
                logging.warning(
                    f"Skipping metadata item at index {idx} due to missing prompt."
                )
                continue

            # Create a unique filename for the generated image.
            output_image_filename = f"generated_scene_{idx}_{safe_model_id}.png"
            logging.info(
                f"Processing prompt index {idx} with diffusion model '{diffusion_model_id}'"
            )
            try:
                image_file_path = generate_image(
                    prompt,
                    pipeline,
                    output_image_filename,
                    output_image_dir,
                    steps=steps,
                    scale=scale,
                )
            except Exception as e:
                logging.error(
                    f"Error generating image for prompt index {idx} with model {diffusion_model_id}: {e}"
                )
                continue

            # Record the combined metadata.
            record = {
                "prompt_model": prompt_model,
                "prompt": prompt,
                "task_type": task_type,
                "diffusion_model": diffusion_model_id,
                "image_path": image_file_path,
                "generation_details": {
                    "steps": steps,
                    "scale": scale,
                },
            }
            output_metadata.append(record)
            logging.info(
                f"Generated image for prompt index {idx} with diffusion model '{diffusion_model_id}'."
            )

        # Save the combined metadata to the output metadata file.
        with open(output_metadata_file, "w") as f:
            for record in output_metadata:
                f.write(json.dumps(record) + "\n")
        logging.info(f"Updated metadata saved to '{output_metadata_file}'.")

    except Exception as err:
        logging.exception(f"An unexpected error occurred: {err}")


# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    args = parse_arguments()
    main(args)
