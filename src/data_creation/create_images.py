import os
import json
import torch
import logging
import random
import numpy as np
from diffusers import FluxPipeline, StableDiffusion3Pipeline, DiffusionPipeline
import gc
from tqdm import tqdm

"""
Image generation module for creating synthetic images using diffusion models.

This module provides a wrapper for diffusion models and utilities for batch processing
image generation tasks. It supports Stable Diffusion 3.5 and FLUX.1-dev models with
optimized memory management and error handling.
"""

# Configure logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "image_creation.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE)],
)

def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed (int): The random seed to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to {seed} for reproducibility")


class DiffusionPipelineWrapper:
    """
    A wrapper class for diffusion model pipelines with optimized memory management.
    
    This class provides a unified interface for different diffusion models, handles
    lazy loading/unloading of models to optimize GPU memory usage, and provides
    consistent image generation capabilities.
    
    Attributes:
        model_id (str): Identifier for the diffusion model to use
        steps (int): Number of inference steps for image generation
        scale (float): Guidance scale for controlling adherence to prompt
        pipeline: The loaded diffusion pipeline (None when unloaded)
    
    Supported Models:
        - stabilityai/stable-diffusion-3.5-large
        - black-forest-labs/FLUX.1-dev
    """
    
    def __init__(self, model_id: str, steps: int = 40, scale: float = 4.5):
        """
        Initialize the diffusion pipeline wrapper.
        
        Args:
            model_id (str): The model identifier for the diffusion model
            steps (int, optional): Number of inference steps. Defaults to 40.
            scale (float, optional): Guidance scale value. Defaults to 4.5.
        
        Raises:
            ValueError: If an unsupported model_id is provided
        """
        self.model_id = model_id
        self.steps = steps
        self.scale = scale
        self.pipeline = None
        logging.info(f"Initializing wrapper for {model_id}")
        
    def _create_pipeline(self, model_id: str):
        """
        Create and configure the appropriate diffusion pipeline based on model ID.
        
        Args:
            model_id (str): The model identifier
            
        Returns:
            Union[StableDiffusion3Pipeline, FluxPipeline]: Configured pipeline instance
            
        Raises:
            ValueError: If the model_id is not supported
        """
        if model_id == "stabilityai/stable-diffusion-3.5-large":
            pipe = StableDiffusion3Pipeline.from_pretrained(
                "bin/models/diffusers/stable-diffusion-3.5-large",
                device_map="balanced",
            )
        elif model_id == "black-forest-labs/FLUX.1-dev":
            pipe = FluxPipeline.from_pretrained(
                "bin/models/diffusers/FLUX.1-dev",
                torch_dtype=torch.bfloat16,
                device_map="balanced",
            )
        else:
            raise ValueError(
                "Unsupported model id. Use 'stabilityai/stable-diffusion-3.5-large' or 'black-forest-labs/FLUX.1-dev'."
            )
        return pipe
        
    def load_pipeline(self):
        """
        Lazy load the diffusion pipeline to optimize memory usage.
        
        This method implements lazy loading, only creating the pipeline when needed.
        Subsequent calls return the already loaded pipeline without recreating it.
        
        Returns:
            Union[StableDiffusion3Pipeline, FluxPipeline]: The loaded pipeline
        """
        if self.pipeline is None:
            self.pipeline = self._create_pipeline(self.model_id)
        return self.pipeline
    
    def unload_pipeline(self):
        """
        Unload the pipeline and free GPU memory.
        
        This method properly deallocates the pipeline from memory, clears CUDA cache,
        and forces garbage collection to maximize memory availability for other models.
        Essential for processing multiple models sequentially on limited GPU memory.
        """
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            torch.cuda.empty_cache()
            gc.collect()
            logging.info(f"Pipeline for {self.model_id} unloaded from memory")

    def _call_pipeline(self, prompt: str):
        """
        Execute the pipeline with appropriate parameters based on model type.
        
        Args:
            prompt (str): The text prompt for image generation
            
        Returns:
            GenerationOutput: The pipeline output containing generated images
            
        Raises:
            ValueError: If the pipeline type is not supported
            
        Note:
            Clips the prompt to 300 characters for CLIP compatibility and uses
            the full prompt for secondary prompt parameters when available.
        """
        clip_prompt = prompt[:300]
        if isinstance(self.pipeline, StableDiffusion3Pipeline):
            return self.pipeline(
                prompt=clip_prompt,
                prompt_3=prompt,
                negative_prompt="bad anatomy, poorly drawn face, low resolution, blurry, artifacts, bad lighting, bad composition, cartoonish",
                num_inference_steps=self.steps,
                guidance_scale=self.scale,
            )
        elif isinstance(self.pipeline, FluxPipeline):
            return self.pipeline(
                prompt=clip_prompt,
                prompt_2=prompt,
                negative_prompt="bad anatomy, poorly drawn face, low resolution, blurry, artifacts, bad lighting, bad composition, cartoonish",
                num_inference_steps=self.steps,
                guidance_scale=self.scale,
            )
        else:
            raise ValueError("Unsupported pipeline type.")

    def generate_image(self, prompt: str, output_filename: str, output_dir: str) -> str:
        """
        Generate an image from a text prompt and save it to disk.
        
        Args:
            prompt (str): Text description for image generation
            output_filename (str): Filename for the saved image
            output_dir (str): Directory path where the image will be saved
            
        Returns:
            str: Full path to the saved image file
            
        Raises:
            Exception: Re-raises any exception that occurs during generation or saving
            
        Note:
            Creates the output directory if it doesn't exist and uses torch.no_grad()
            context for memory efficiency during inference.
        """
        try:
            self.load_pipeline()
            with torch.no_grad():
                result = self._call_pipeline(prompt)
                image = result.images[0]
            os.makedirs(output_dir, exist_ok=True)
            image_path = os.path.join(output_dir, output_filename)
            image.save(image_path)
            logging.info(f"Image saved to {image_path}")
            return image_path
        except Exception as e:
            logging.error(f"Error generating image {output_filename}: {e}")
            raise

    def __call__(self, prompt: str, output_filename: str, output_dir: str) -> str:
        """
        Make the wrapper callable, delegating to generate_image method.
        
        Args:
            prompt (str): Text description for image generation
            output_filename (str): Filename for the saved image
            output_dir (str): Directory path where the image will be saved
            
        Returns:
            str: Full path to the saved image file
        """
        return self.generate_image(prompt, output_filename, output_dir)

    @staticmethod
    def load_metadata_from_json(json_path: str):
        """
        Load metadata from a JSONL file containing prompt information.
        
        Args:
            json_path (str): Path to the JSONL file containing metadata
            
        Returns:
            list: List of dictionaries containing prompt metadata
            
        Note:
            Expects JSONL format where each line is a valid JSON object.
            Empty lines are automatically skipped.
        """
        with open(json_path, "r") as f:
            data = [json.loads(line.strip()) for line in f if line.strip()]
        return data


def process_model_batch(model_config, metadata_list, base_output_dir, output_metadata):
    """
    Process all prompts for a single diffusion model in batch.
    
    This function handles the complete workflow for generating images with one model:
    initializing the wrapper, processing all prompts with progress tracking,
    collecting metadata, and proper cleanup.
    
    Args:
        model_config (dict): Configuration dictionary containing:
            - model_id (str): The model identifier
            - steps (int, optional): Number of inference steps
            - scale (float, optional): Guidance scale
        metadata_list (list): List of prompt metadata dictionaries
        base_output_dir (str): Base directory for saving images
        output_metadata (list): List to append generation metadata to
        
    Note:
        Automatically handles errors for individual prompts and continues processing.
        Ensures proper cleanup of GPU memory regardless of success or failure.
    """
    model_id = model_config["model_id"]
    steps = model_config.get("steps", 40)
    scale = model_config.get("scale", 4.5)
    safe_model_id = model_id.replace("/", "_")
    output_dir = os.path.join(base_output_dir, safe_model_id)
    
    logging.info(f"Processing {len(metadata_list)} prompts for model '{model_id}'")
    
    wrapper = None
    try:
        wrapper = DiffusionPipelineWrapper(model_id, steps, scale)
        
        # Process prompts with progress bar
        for idx, item in enumerate(tqdm(metadata_list, desc=f"Generating with {safe_model_id}")):
            prompt = item.get("generated_scene_description")
            if not prompt:
                logging.warning(f"Skipping prompt index {idx} due to missing description.")
                continue
                
            output_filename = f"image_{idx:03d}.png"
            try:
                image_path = wrapper.generate_image(prompt, output_filename, output_dir)
                record = {
                    "model_id": model_id,
                    "prompt": prompt,
                    "image_path": image_path,
                    "steps": steps,
                    "scale": scale,
                    "prompt_index": idx
                }
                output_metadata.append(record)
                
            except Exception as e:
                logging.error(f"Error generating image for prompt index {idx}: {e}")
                continue
                
    except Exception as e:
        logging.error(f"Failed to initialize wrapper for model {model_id}: {e}")
        return
    finally:
        if wrapper:
            wrapper.unload_pipeline()


def save_metadata_batch(output_metadata, output_file):
    """
    Save generation metadata to a JSONL file in batch.
    
    Args:
        output_metadata (list): List of metadata dictionaries to save
        output_file (str): Path to the output JSONL file
        
    Note:
        Overwrites the output file if it exists. Each metadata record is written
        as a separate JSON line for easy parsing and streaming.
    """
    with open(output_file, "w") as f:
        for record in output_metadata:
            f.write(json.dumps(record) + "\n")
    logging.info(f"Saved {len(output_metadata)} records to '{output_file}'")


def main():
    """
    Main execution function for the image generation pipeline.
    
    This function orchestrates the complete image generation workflow:
    1. Loads prompt metadata from JSONL file
    2. Processes each configured diffusion model sequentially
    3. Generates images for all prompts with each model
    4. Saves comprehensive metadata about the generation process
    5. Handles memory optimization between models
    
    The function is designed to be memory-efficient for multi-GPU setups and
    provides comprehensive logging for monitoring long-running generation tasks.
    
    Configuration:
        - Supports multiple diffusion models processed sequentially
        - Automatic memory cleanup between models
        - Progress tracking with tqdm
        - Comprehensive error handling and logging
        
    Raises:
        Exception: Logs and handles any unexpected errors during execution
    """
    
    set_seed(42)  # Set a fixed seed for reproducibility
    
    logging.info("\n\nStarting image generation process using DiffusionPipelineWrapper.")
    try:
        diffusion_models = [
            {"model_id": "black-forest-labs/FLUX.1-dev", "steps": 50, "scale": 7.5},
            {"model_id": "stabilityai/stable-diffusion-3.5-large", "steps": 50, "scale": 7.5},
        ]
        json_file = "output/prompts/claude3.7-prompt.jsonl"
        base_output_dir = "output/images_" + json_file.split("/")[-1].split("-")[0]
        
        metadata_list = DiffusionPipelineWrapper.load_metadata_from_json(json_file)
        if not metadata_list:
            logging.error("No metadata found in the JSON file.")
            return

        output_metadata = []
        
        # Process each model sequentially to optimize memory usage
        for model_config in diffusion_models:
            process_model_batch(model_config, metadata_list, base_output_dir, output_metadata)
            
            # Force garbage collection between models
            torch.cuda.empty_cache()
            gc.collect()

        # Save all metadata at once
        output_metadata_file = f"metadata_{json_file.split('/')[-1].split('-')[0]}.jsonl"
        save_metadata_batch(output_metadata, output_metadata_file)

    except Exception as err:
        logging.exception(f"An unexpected error occurred: {err}")


if __name__ == "__main__":
    main()
