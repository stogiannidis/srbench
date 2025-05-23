import argparse
import os
import gc
import psutil
import pandas as pd
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from datasets import load_dataset, Dataset
import torch
from torch.utils.data import DataLoader
from contextlib import contextmanager
import random
import numpy as np

from utils import VLMWrapper as VLM

# Configure logger with rotation and better formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("eval.log", mode='a'),
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


class OptimizedDataLoader:
    """Optimized data loader with memory-efficient batching."""
    
    def __init__(self, dataset: Dataset, batch_size: int, num_workers: int = 0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch = self.dataset[i:i + self.batch_size]
            yield self._process_batch(batch)
    
    def _process_batch(self, batch: Dict[str, List]) -> Dict[str, Any]:
        """Process batch with memory optimization."""
        return {
            "questions": batch["question"],
            "images": batch["image"],
            "answers": batch["answer"],
            "splits": batch.get("split", [None] * len(batch["question"]))
        }


def get_optimal_batch_size(model_id: str, base_batch_size: int) -> int:
    """Dynamically determine optimal batch size based on available memory."""
    if not torch.cuda.is_available():
        return min(base_batch_size, 4)
    
    # Get GPU memory info
    memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    # Heuristic for batch size based on model size and GPU memory
    if "3B" in model_id or "7B" in model_id:
        return min(base_batch_size, max(1, int(memory_gb // 8)))
    elif "11B" in model_id or "13B" in model_id:
        return min(base_batch_size, max(1, int(memory_gb // 12)))
    elif "26B" in model_id or "90B" in model_id:
        return min(base_batch_size, max(1, int(memory_gb // 20)))
    
    return min(base_batch_size, max(1, int(memory_gb // 4)))


def create_optimized_messages(questions: List[str], images: List[Any]) -> List[List[Dict]]:
    """Create message format optimized for batch processing."""
    return [
        [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }]
        for question, image in zip(questions, images)
    ]


def evaluate(model_id: str, dataset_id: str, batch_size: int = 16, 
             max_new_tokens: int = 64, num_workers: int = 0) -> None:
    """
    Optimized evaluation with memory management and error handling.
    """
    short_name = model_id.split("/")[-1]
    dataset_name = dataset_id.split("/")[-1]
    
    logger.info(f"Starting optimized evaluation: {model_id} on {dataset_id}")
    logger.info(f"System memory: {psutil.virtual_memory().total / (1024**3):.1f}GB")
    
    # Optimize batch size based on model and hardware
    optimal_batch_size = get_optimal_batch_size(model_id, batch_size)
    logger.info(f"Using optimized batch size: {optimal_batch_size}")
    
    # Load and prepare dataset
    try:
        dataset = load_dataset(dataset_id, split="train")
        logger.info(f"Loaded dataset with {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_id}: {e}")
        raise
    
    # Initialize model with error handling
    try:
        with memory_cleanup():
            vlm = VLM(model_id, "auto")
        logger.info(f"Model {short_name} loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model {model_id}: {e}")
        raise
    
    # Create optimized data loader
    data_loader = OptimizedDataLoader(dataset, optimal_batch_size, num_workers)
    
    results = []
    total_batches = (len(dataset) + optimal_batch_size - 1) // optimal_batch_size
    
    # Process batches with memory management
    with tqdm(total=total_batches, desc=f"Evaluating {short_name}", 
              colour="green", unit="batch") as pbar:
        
        for batch_idx, batch in enumerate(data_loader):
            try:
                with memory_cleanup():
                    # Create messages efficiently
                    messages = create_optimized_messages(batch["questions"], batch["images"])
                    
                    # Preprocess inputs
                    inputs = vlm.preprocess(conversation=messages, image_input=batch["images"])
                    
                    # Generate with optimized parameters
                    with torch.inference_mode():
                        generated_ids = vlm(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            pad_token_id=getattr(vlm.processor, 'pad_token_id', None)
                        )
                    
                    # Decode outputs
                    output_texts = vlm.decode(generated_ids)
                    
                    # Process results efficiently
                    batch_results = [
                        {
                            "response": output.strip(),
                            "answer": answer,
                            "question": question,
                            "split": split_val
                        }
                        for output, answer, question, split_val in zip(
                            output_texts, batch["answers"], 
                            batch["questions"], batch["splits"]
                        )
                    ]
                    results.extend(batch_results)
                    
                    # Clear intermediate variables
                    del inputs, generated_ids, output_texts, messages
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                # Add dummy results to maintain alignment
                batch_results = [
                    {
                        "response": "ERROR",
                        "answer": answer,
                        "question": question,
                        "split": split_val
                    }
                    for answer, question, split_val in zip(
                        batch["answers"], batch["questions"], batch["splits"]
                    )
                ]
                results.extend(batch_results)
            
            pbar.update(1)
            
            # Periodic memory cleanup
            if batch_idx % 10 == 0:
                gc.collect()
    
    # Save results efficiently
    output_dir = f"output/evaluations/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(output_dir, f"{short_name}.csv")
    
    # Use efficient CSV writing
    results_df.to_csv(results_csv_path, index=False, chunksize=1000)
    logger.info(f"Results saved to {results_csv_path}")
    
    # Final cleanup
    del vlm, dataset, results_df
    gc.collect()


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


def parse_args() -> argparse.Namespace:
    """Enhanced argument parsing with validation."""
    parser = argparse.ArgumentParser(
        description="Optimized evaluation of vision-language models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-m", "--model", type=str, required=True,
        help="Model identifier (e.g., 'meta-llama/Llama-3.2-11B-Vision-Instruct')"
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True,
        help="Dataset identifier (e.g., 'stogian/srbench')"
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=16,
        help="Maximum batch size (will be optimized based on hardware)"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=64,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("Max new tokens must be positive")
    
    return args


if __name__ == "__main__":
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    try:
        evaluate(
            model_id=args.model,
            dataset_id=args.dataset,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            num_workers=args.num_workers
        )
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
