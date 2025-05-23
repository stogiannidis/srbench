import os
import json
import torch
import gc
import logging
from contextlib import contextmanager
from typing import Dict, List, Any, Optional
from diffusers import FluxPipeline, StableDiffusion3Pipeline
import psutil
from tqdm import tqdm

# Configure logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "image_creation.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@contextmanager
def memory_cleanup():
    """Context manager for aggressive memory cleanup."""
    try:
        yield
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class OptimizedDiffusionPipelineWrapper:
    """Optimized diffusion pipeline with memory management."""
    
    def __init__(self, model_id: str, steps: int = 40, scale: float = 4.5):
        self.model_id = model_id
        self.steps = steps
        self.scale = scale
        self.pipeline = None
        
        logger.info(f"Initializing optimized wrapper for {model_id}")
        
        # Check available memory
        if torch.cuda.is_available():
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU memory available: {memory_gb:.1f}GB")
        else:
            memory_gb = psutil.virtual_memory().total / (1024**3)
            logger.info(f"System memory available: {memory_gb:.1f}GB")
        
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initialize pipeline with optimizations."""
        try:
            with memory_cleanup():
                if self.model_id == "stabilityai/stable-diffusion-3.5-large":
                    self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                        "bin/models/diffusers/stable-diffusion-3.5-large",
                        device_map="balanced",
                        torch_dtype=torch.bfloat16,
                        use_safetensors=True,
                        variant="fp16" if torch.cuda.is_available() else None
                    )
                    
                elif self.model_id == "black-forest-labs/FLUX.1-dev":
                    self.pipeline = FluxPipeline.from_pretrained(
                        "bin/models/diffusers/FLUX.1-dev",
                        torch_dtype=torch.bfloat16,
                        device_map="balanced",
                        use_safetensors=True,
                        variant="fp16" if torch.cuda.is_available() else None
                    )
                    
                else:
                    raise ValueError(
                        f"Unsupported model_id: {self.model_id}. "
                        "Use 'stabilityai/stable-diffusion-3.5-large' or 'black-forest-labs/FLUX.1-dev'."
                    )
                
                # Enable memory efficient attention
                if hasattr(self.pipeline, 'enable_attention_slicing'):
                    self.pipeline.enable_attention_slicing()
                
                if hasattr(self.pipeline, 'enable_model_cpu_offload'):
                    self.pipeline.enable_model_cpu_offload()
                
                logger.info(f"Pipeline initialized successfully for {self.model_id}")
                
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise

    def _generate_with_pipeline(self, prompt: str) -> Any:
        """Generate image with appropriate pipeline parameters."""
        # Clip prompt to prevent token limit issues
        clip_prompt = prompt[:300]
        
        # Common parameters
        common_params = {
            "num_inference_steps": self.steps,
            "guidance_scale": self.scale,
            "negative_prompt": (
                "bad anatomy, poorly drawn face, low resolution, blurry, artifacts, "
                "bad lighting, bad composition, cartoonish"
            )
        }
        
        try:
            if isinstance(self.pipeline, StableDiffusion3Pipeline):
                return self.pipeline(
                    prompt=clip_prompt,
                    prompt_3=prompt,
                    **common_params
                )
            elif isinstance(self.pipeline, FluxPipeline):
                return self.pipeline(
                    prompt=clip_prompt,
                    prompt_2=prompt,
                    **common_params
                )
            else:
                raise ValueError(f"Unsupported pipeline type: {type(self.pipeline)}")
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def generate_image(self, prompt: str, output_filename: str, output_dir: str) -> str:
        """Generate image with memory management and error handling."""
        try:
            with memory_cleanup():
                with torch.inference_mode():
                    result = self._generate_with_pipeline(prompt)
                    image = result.images[0]
                
                # Ensure output directory exists
                os.makedirs(output_dir, exist_ok=True)
                
                # Save image
                image_path = os.path.join(output_dir, output_filename)
                image.save(image_path, format='PNG', optimize=True)
                
                logger.info(f"Image saved to {image_path}")
                return image_path
                
        except Exception as e:
            logger.error(f"Failed to generate image for prompt '{prompt[:50]}...': {e}")
            raise

    def __call__(self, prompt: str, output_filename: str, output_dir: str) -> str:
        """Callable interface for image generation."""
        return self.generate_image(prompt, output_filename, output_dir)

    @staticmethod
    def load_metadata_from_json(json_path: str) -> List[Dict[str, Any]]:
        """Load metadata from JSONL file with error handling."""
        try:
            with open(json_path, "r", encoding='utf-8') as f:
                data = [json.loads(line.strip()) for line in f if line.strip()]
            logger.info(f"Loaded {len(data)} items from {json_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load metadata from {json_path}: {e}")
            raise

    def __del__(self):
        """Cleanup resources on deletion."""
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            del self.pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def validate_config(diffusion_models: List[Dict[str, Any]], json_file: str) -> None:
    """Validate configuration parameters."""
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON file not found: {json_file}")
    
    for model_config in diffusion_models:
        if "model_id" not in model_config:
            raise ValueError("model_id is required in diffusion model config")
        
        steps = model_config.get("steps", 40)
        scale = model_config.get("scale", 4.5)
        
        if not (10 <= steps <= 100):
            raise ValueError(f"steps must be between 10 and 100, got {steps}")
        
        if not (1.0 <= scale <= 20.0):
            raise ValueError(f"scale must be between 1.0 and 20.0, got {scale}")


def process_model_config(
    model_config: Dict[str, Any],
    metadata_list: List[Dict[str, Any]],
    base_output_dir: str,
    json_file: str
) -> List[Dict[str, Any]]:
    """Process a single model configuration."""
    model_id = model_config["model_id"]
    steps = model_config.get("steps", 40)
    scale = model_config.get("scale", 4.5)
    
    safe_model_id = model_id.replace("/", "_")
    output_dir = os.path.join(base_output_dir, safe_model_id)
    
    logger.info(f"Processing model '{model_id}' with steps={steps}, scale={scale}")
    
    output_metadata = []
    
    try:
        # Initialize wrapper
        wrapper = OptimizedDiffusionPipelineWrapper(model_id, steps, scale)
        
        # Process each prompt with progress bar
        for idx, item in enumerate(tqdm(
            metadata_list,
            desc=f"Generating {safe_model_id}",
            unit="image",
            colour="green"
        )):
            prompt = item.get("generated_scene_description")
            
            if not prompt:
                logger.warning(f"Skipping item {idx} due to missing description")
                continue
            
            output_filename = f"image_{idx:03d}.png"
            
            try:
                with memory_cleanup():
                    image_path = wrapper(prompt, output_filename, output_dir)
                
                # Create metadata record
                record = {
                    "model_id": model_id,
                    "prompt": prompt,
                    "image_path": image_path,
                    "steps": steps,
                    "scale": scale,
                    "index": idx
                }
                output_metadata.append(record)
                
            except Exception as e:
                logger.error(f"Error generating image {idx} with model {model_id}: {e}")
                continue
        
        # Cleanup wrapper
        del wrapper
        gc.collect()
        
        logger.info(f"Completed processing {model_id}: {len(output_metadata)} images generated")
        
    except Exception as e:
        logger.error(f"Failed to initialize wrapper for model {model_id}: {e}")
    
    return output_metadata


def main():
    """Optimized main function with comprehensive error handling."""
    logger.info("Starting optimized image generation process")
    
    try:
        # Configuration
        diffusion_models = [
            {
                "model_id": "black-forest-labs/FLUX.1-dev",
                "steps": 50,
                "scale": 7.5
            },
            {
                "model_id": "stabilityai/stable-diffusion-3.5-large",
                "steps": 50,
                "scale": 7.5
            },
        ]
        
        json_file = "output/prompts/claude3.7-prompt.jsonl"
        base_output_dir = "output/images_" + json_file.split("/")[-1].split("-")[0]
        
        # Validate configuration
        validate_config(diffusion_models, json_file)
        
        # Load metadata
        metadata_list = OptimizedDiffusionPipelineWrapper.load_metadata_from_json(json_file)
        
        if not metadata_list:
            logger.error("No metadata found in the JSON file")
            return
        
        logger.info(f"Processing {len(metadata_list)} prompts with {len(diffusion_models)} models")
        
        # Process each model
        all_output_metadata = []
        
        for model_config in diffusion_models:
            try:
                model_metadata = process_model_config(
                    model_config, metadata_list, base_output_dir, json_file
                )
                all_output_metadata.extend(model_metadata)
                
                # Force cleanup between models
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
            except Exception as e:
                logger.error(f"Failed to process model {model_config.get('model_id', 'unknown')}: {e}")
                continue
        
        # Save consolidated metadata
        if all_output_metadata:
            output_metadata_file = f"metadata_{json_file.split('/')[-1].split('-')[0]}.jsonl"
            
            with open(output_metadata_file, "w", encoding='utf-8') as f:
                for record in all_output_metadata:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            logger.info(f"Consolidated metadata saved to '{output_metadata_file}'")
            logger.info(f"Total images generated: {len(all_output_metadata)}")
        else:
            logger.warning("No images were generated successfully")

    except Exception as e:
        logger.error(f"Image generation process failed: {e}")
        raise


if __name__ == "__main__":
    main()
