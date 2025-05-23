import torch
import gc
import logging
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import os
import psutil
from contextlib import contextmanager
from typing import Dict, Any, List, Optional
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("minicpm_eval.log", mode='a'),
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


class OptimizedMiniCPMDataset(torch.utils.data.Dataset):
    """Optimized dataset with memory-efficient loading."""
    
    def __init__(self, dataset, preprocess_images: bool = True):
        self.dataset = dataset
        self.preprocess_images = preprocess_images
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            item = self.dataset[idx]
            
            # Efficient image handling
            image = item["image"]
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif hasattr(image, 'convert'):
                image = image.convert("RGB")
            
            # Optimize image if needed
            if self.preprocess_images and max(image.size) > 1024:
                # Resize large images to reduce memory usage
                ratio = 1024 / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            return {
                "id": idx,
                "image": image,
                "question": item["question"],
                "answer": item.get("answer", None),
                "split": item.get("split", None)
            }
            
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            # Return a dummy item to prevent batch failure
            return {
                "id": idx,
                "image": Image.new("RGB", (224, 224), color='white'),
                "question": "Error loading question",
                "answer": "Error",
                "split": None
            }


def optimized_collate_fn(batch: List[Dict[str, Any]]) -> tuple:
    """Optimized collate function with efficient message creation."""
    msgs_batch = []
    original_batch = []
    
    for sample in batch:
        try:
            # Convert PIL image to numpy array with efficient memory layout
            np_img = np.asarray(sample["image"], dtype=np.uint8)
            
            # Ensure correct channel order (C x H x W)
            if np_img.ndim == 3 and np_img.shape[-1] == 3:
                np_img = np_img.transpose(2, 0, 1)
            
            # Create conversation with optimized format
            msgs = [{
                "role": "user", 
                "content": [np_img, sample["question"]]
            }]
            
            msgs_batch.append(msgs)
            original_batch.append(sample)
            
        except Exception as e:
            logger.error(f"Error in collate_fn for sample {sample.get('id', 'unknown')}: {e}")
            # Add dummy data to maintain batch consistency
            dummy_img = np.zeros((3, 224, 224), dtype=np.uint8)
            msgs_batch.append([{"role": "user", "content": [dummy_img, "Error"]}])
            original_batch.append({
                "id": sample.get("id", -1),
                "question": "Error",
                "answer": "Error",
                "split": None
            })
    
    return msgs_batch, original_batch


def get_optimal_batch_size(available_memory_gb: float) -> int:
    """Determine optimal batch size based on available memory."""
    if available_memory_gb > 32:
        return 32
    elif available_memory_gb > 16:
        return 16
    elif available_memory_gb > 8:
        return 8
    else:
        return 4


class MiniCPMEvaluator:
    """Optimized MiniCPM evaluator with memory management."""
    
    def __init__(self, model_path: str = "openbmb/MiniCPM-V-2_6"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load model with optimizations."""
        logger.info(f"Loading MiniCPM model: {self.model_path}")
        
        try:
            with memory_cleanup():
                self.model = AutoModel.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                ).eval()
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    use_fast=True
                )
                
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def evaluate_batch(self, msgs_batch: List, max_tokens: int = 128) -> List[str]:
        """Evaluate a batch of messages with error handling."""
        responses = []
        
        try:
            with memory_cleanup():
                with torch.inference_mode():
                    # Try batch inference first
                    try:
                        batch_responses = self.model.chat(
                            image=None,
                            msgs=msgs_batch,
                            tokenizer=self.tokenizer,
                            max_tokens=max_tokens,
                            do_sample=False,
                            temperature=0.1
                        )
                        
                        # Handle different response formats
                        if isinstance(batch_responses, list):
                            responses.extend(batch_responses)
                        else:
                            responses.append(str(batch_responses))
                            
                    except Exception as batch_error:
                        logger.warning(f"Batch inference failed: {batch_error}. Falling back to individual inference.")
                        
                        # Fallback to individual inference
                        for msgs in msgs_batch:
                            try:
                                response = self.model.chat(
                                    image=None,
                                    msgs=msgs,
                                    tokenizer=self.tokenizer,
                                    max_tokens=max_tokens,
                                    do_sample=False
                                )
                                responses.append(str(response))
                            except Exception as e:
                                logger.error(f"Individual inference failed: {e}")
                                responses.append("ERROR: Inference failed")
                                
        except Exception as e:
            logger.error(f"Batch evaluation failed: {e}")
            responses = ["ERROR: Batch evaluation failed"] * len(msgs_batch)
        
        return responses
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
        description="Evaluate MiniCPM model on a dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True,
        help="Name of the Hugging Face dataset"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Batch size (auto-determined if not specified)"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=128,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--model_path", type=str, default="openbmb/MiniCPM-V-2_6",
        help="Model path or identifier"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output/evaluations",
        help="Output directory for results"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    logger.info(f"Starting MiniCPM evaluation")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {args.model_path}")
    
    # Get system info
    memory_gb = psutil.virtual_memory().total / (1024**3)
    logger.info(f"System memory: {memory_gb:.1f}GB")
    
    # Determine optimal batch size
    if args.batch_size is None:
        batch_size = get_optimal_batch_size(memory_gb)
        logger.info(f"Auto-determined batch size: {batch_size}")
    else:
        batch_size = args.batch_size
        logger.info(f"Using specified batch size: {batch_size}")
    
    try:
        # Load dataset
        logger.info(f"Loading dataset: {args.dataset}")
        hf_dataset = load_dataset(args.dataset, split="train")
        logger.info(f"Dataset loaded with {len(hf_dataset)} samples")
        
        # Create optimized dataset and dataloader
        dataset_obj = OptimizedMiniCPMDataset(hf_dataset, preprocess_images=True)
        data_loader = DataLoader(
            dataset_obj,
            batch_size=batch_size,
            collate_fn=optimized_collate_fn,
            num_workers=0,  # Keep at 0 for image processing stability
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )
        
        # Initialize evaluator
        evaluator = MiniCPMEvaluator(args.model_path)
        
        # Run evaluation
        results = []
        total_batches = len(data_loader)
        
        with tqdm(
            data_loader,
            desc="MiniCPM Evaluation",
            total=total_batches,
            unit="batch",
            colour="blue"
        ) as pbar:
            
            for batch_idx, (msgs_batch, original_batch) in enumerate(pbar):
                try:
                    # Generate responses
                    responses = evaluator.evaluate_batch(msgs_batch, args.max_tokens)
                    
                    # Process results
                    for sample, response in zip(original_batch, responses):
                        result_entry = {
                            "id": sample["id"],
                            "question": sample["question"],
                            "response": response,
                            "answer": sample["answer"],
                            "split": sample["split"]
                        }
                        results.append(result_entry)
                    
                    # Update progress
                    pbar.set_postfix({
                        'batch': f"{batch_idx + 1}/{total_batches}",
                        'samples': len(results)
                    })
                    
                    # Periodic cleanup
                    if batch_idx % 10 == 0:
                        gc.collect()
                        
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    # Add error entries to maintain sample count
                    for sample in original_batch:
                        results.append({
                            "id": sample["id"],
                            "question": sample["question"],
                            "response": f"ERROR: {str(e)}",
                            "answer": sample["answer"],
                            "split": sample["split"]
                        })
        
        # Save results
        df = pd.DataFrame(results)
        
        # Create output directory
        short_dataset_name = args.dataset.split("/")[-1]
        output_dir = os.path.join(args.output_dir, short_dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save CSV
        output_path = os.path.join(output_dir, "MiniCPM-V-2_6.csv")
        df.to_csv(output_path, index=False)
        
        logger.info(f"Evaluation completed successfully")
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"Processed {len(results)} samples")
        
        # Final cleanup
        del evaluator
        gc.collect()
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()

