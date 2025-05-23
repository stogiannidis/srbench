import argparse
import os
import torch
import torchvision.transforms as T
import gc
import logging
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import psutil
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("intern_eval.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------
# Constants and Configuration
# ---------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def build_transform(input_size: int) -> T.Compose:
    """Creates an optimized transform pipeline for images."""
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: set,
    width: int,
    height: int,
    image_size: int
) -> Tuple[int, int]:
    """Optimized aspect ratio matching."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    area_threshold = 0.5 * image_size * image_size
    
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > area_threshold * ratio[0] * ratio[1]:
            best_ratio = ratio
            
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False
) -> List[Image.Image]:
    """Optimized dynamic preprocessing with caching."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Pre-compute target ratios (cached)
    target_ratios = {
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Efficient resizing
    resized_img = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    # Process blocks efficiently
    processed_images = []
    cols = target_width // image_size
    
    for i in range(blocks):
        col = i % cols
        row = i // cols
        box = (
            col * image_size,
            row * image_size,
            (col + 1) * image_size,
            (row + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        processed_images.append(thumbnail_img)
        
    return processed_images


def load_image_optimized(
    image_input: Any,
    input_size: int = 448,
    max_num: int = 12
) -> torch.Tensor:
    """
    Optimized image loading with memory management.
    """
    try:
        # Handle different input types
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, dict) and "path" in image_input:
            image = Image.open(image_input["path"]).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Optimize for memory - reduce max_num for large images
        if max(image.size) > 2048:
            max_num = min(max_num, 8)
        elif max(image.size) > 1024:
            max_num = min(max_num, 10)

        transform = build_transform(input_size)
        images = dynamic_preprocess(
            image, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        
        # Efficient tensor creation
        pixel_values = torch.stack([transform(img) for img in images])
        
        return pixel_values
        
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        # Return a dummy tensor to prevent batch failure
        dummy_transform = build_transform(input_size)
        dummy_image = Image.new("RGB", (input_size, input_size), color='white')
        return dummy_transform(dummy_image).unsqueeze(0)


def optimized_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Optimized collate function with memory management."""
    all_pixel_values = []
    questions = []
    num_patches_list = []
    original_batch = []
    
    for sample in batch:
        try:
            question = sample["question"]
            image_input = sample["image"]
            
            with memory_cleanup():
                pv = load_image_optimized(image_input, max_num=12)
                
            all_pixel_values.append(pv)
            num_patches_list.append(pv.size(0))
            questions.append(question)
            original_batch.append(sample)
            
        except Exception as e:
            logger.error(f"Error in collate_fn: {e}")
            # Add dummy data to maintain batch consistency
            dummy_pv = torch.zeros((1, 3, 448, 448))
            all_pixel_values.append(dummy_pv)
            num_patches_list.append(1)
            questions.append("Error loading question")
            original_batch.append({
                "question": "Error",
                "answer": "Error",
                "split": None
            })
    
    # Efficient concatenation
    all_pixel_values = torch.cat(all_pixel_values, dim=0)
    
    return {
        "pixel_values": all_pixel_values,
        "questions": questions,
        "num_patches_list": num_patches_list,
        "original_batch": original_batch
    }


@torch.inference_mode()
def run_inference_optimized(
    dataloader: DataLoader,
    model: Any,
    tokenizer: Any,
    generation_config: Dict[str, Any],
    dataset: Any,
    output_dir: str,
    m_name: str
) -> None:
    """Optimized inference with memory management and error handling."""
    results = []
    total_processed = 0
    
    with tqdm(dataloader, desc=f"Evaluating {m_name}", unit="batch", colour="magenta") as pbar:
        for batch_idx, batch in enumerate(pbar):
            try:
                with memory_cleanup():
                    pixel_values = batch["pixel_values"].to(torch.bfloat16).cuda()
                    questions = batch["questions"]
                    num_patches_list = batch["num_patches_list"]
                    original_batch = batch["original_batch"]
                    
                    # Run batch inference with error handling
                    try:
                        responses = model.batch_chat(
                            tokenizer,
                            pixel_values,
                            num_patches_list=num_patches_list,
                            questions=questions,
                            generation_config=generation_config,
                        )
                    except Exception as inference_error:
                        logger.error(f"Batch inference failed: {inference_error}")
                        responses = [f"ERROR: {str(inference_error)}"] * len(questions)
                    
                    # Process results
                    for question, response, original in zip(questions, responses, original_batch):
                        results.append({
                            "question": question,
                            "response": response,
                            "answer": original.get("answer", ""),
                            "split": original.get("split", "")
                        })
                    
                    total_processed += len(questions)
                    
                    # Clear GPU memory
                    del pixel_values
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                # Add error entries to maintain sample count
                batch_size = len(batch.get("questions", []))
                for i in range(batch_size):
                    results.append({
                        "question": f"Error in batch {batch_idx}",
                        "response": f"ERROR: {str(e)}",
                        "answer": "",
                        "split": ""
                    })
                total_processed += batch_size
            
            # Update progress
            pbar.set_postfix({
                'processed': total_processed,
                'batch': f"{batch_idx + 1}"
            })
            
            # Periodic cleanup
            if batch_idx % 5 == 0:
                gc.collect()
    
    # Create and save results
    results_df = pd.DataFrame(results)
    
    # Ensure we have the right number of results
    expected_samples = len(dataset)
    if len(results_df) != expected_samples:
        logger.warning(f"Result count mismatch: expected {expected_samples}, got {len(results_df)}")
    
    # Save results
    output_path = os.path.join(output_dir, f"{m_name}.csv")
    results_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")


def get_optimal_batch_size(model_name: str, memory_gb: float) -> int:
    """Determine optimal batch size based on model and available memory."""
    # Heuristics based on model size and GPU memory
    if "26B" in model_name or "22B" in model_name:
        if memory_gb > 40:
            return 8
        elif memory_gb > 24:
            return 4
        else:
            return 2
    elif "8B" in model_name or "7B" in model_name:
        if memory_gb > 24:
            return 16
        elif memory_gb > 16:
            return 12
        else:
            return 8
    else:
        if memory_gb > 16:
            return 24
        else:
            return 16


def initialize_components(
    model_name: str,
    dataset_name: str,
    batch_size: Optional[int] = None
) -> Tuple[Any, Any, Dict[str, Any], Any, DataLoader, str, str]:
    """
    Optimized component initialization with memory management.
    """
    logger.info(f"Initializing components for {model_name}")
    
    # Get memory info
    if torch.cuda.is_available():
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    else:
        memory_gb = psutil.virtual_memory().total / (1024**3)
    
    logger.info(f"Available memory: {memory_gb:.1f}GB")
    
    # Determine optimal batch size
    if batch_size is None:
        batch_size = get_optimal_batch_size(model_name, memory_gb)
    
    logger.info(f"Using batch size: {batch_size}")
    
    try:
        with memory_cleanup():
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=False
            )
            
            # Load model with optimizations
            model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,  # Enable flash attention
                trust_remote_code=True,
                device_map="auto",
            ).eval()
            
        logger.info("Model and tokenizer loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    # Generation config
    generation_config = {
        "max_new_tokens": 64,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "temperature": 0.1
    }
    
    # Load dataset
    try:
        dataset = load_dataset(dataset_name, split="train")
        logger.info(f"Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise
    
    # Create output directory
    d_name = dataset_name.split("/")[-1]
    m_name = model_name.split("/")[-1]
    output_dir = f"output/evaluations/{d_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create optimized dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=optimized_collate_fn,
        num_workers=0,  # Keep at 0 for stability with vision models
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    return model, tokenizer, generation_config, dataset, dataloader, output_dir, m_name


def set_seed(seed: int = 42) -> None:
    """Set seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Seed set to {seed}")


def main():
    """Optimized main function with comprehensive error handling."""
    parser = argparse.ArgumentParser(
        description="Run optimized batched inference on a dataset with InternVL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Name or path of the model (e.g., OpenGVLab/InternVL2_5-1B-MPO)"
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True,
        help="Name or id of the dataset (e.g., stogian/mrt_pf_mix)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Batch size for inference (auto-determined if not specified)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    logger.info("Starting InternVL evaluation")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Dataset: {args.dataset_name}")
    
    try:
        # Initialize components
        model, tokenizer, generation_config, dataset, dataloader, output_dir, m_name = initialize_components(
            args.model_name, args.dataset_name, args.batch_size
        )
        
        # Run inference
        run_inference_optimized(
            dataloader, model, tokenizer, generation_config, dataset, output_dir, m_name
        )
        
        logger.info("Evaluation completed successfully")
        
        # Final cleanup
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
