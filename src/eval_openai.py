import argparse
import base64
import logging
import os
import time
import asyncio
from functools import wraps
from io import BytesIO
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import BoundedSemaphore

import dotenv
import pandas as pd
from datasets import load_dataset
from openai import AzureOpenAI
from PIL import Image
from tqdm import tqdm
import psutil

# Load the environment variables
dotenv.load_dotenv()

# Configure logging with better formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("openai_eval.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def exponential_backoff_retry(max_retries: int = 5, initial_delay: float = 1.0):
    """Enhanced retry decorator with exponential backoff and jitter."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    if "429" in error_str or "rate" in error_str:
                        if attempt < max_retries - 1:
                            # Add jitter to prevent thundering herd
                            jitter = delay * 0.1 * (0.5 - abs(hash(str(args)) % 100) / 100)
                            sleep_time = delay + jitter
                            logger.warning(f"Rate limit hit, retrying in {sleep_time:.2f}s (attempt {attempt + 1})")
                            await asyncio.sleep(sleep_time)
                            delay *= 2  # Exponential backoff
                        else:
                            logger.error(f"Max retries exceeded for rate limiting")
                            raise
                    elif "502" in error_str or "503" in error_str or "504" in error_str:
                        if attempt < max_retries - 1:
                            logger.warning(f"Server error, retrying in {delay}s")
                            await asyncio.sleep(delay)
                            delay *= 1.5
                        else:
                            raise
                    else:
                        logger.error(f"Non-retryable error: {e}")
                        raise
            raise Exception(f"Max retries ({max_retries}) exceeded")
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    if "429" in error_str or "rate" in error_str:
                        if attempt < max_retries - 1:
                            jitter = delay * 0.1 * (0.5 - abs(hash(str(args)) % 100) / 100)
                            sleep_time = delay + jitter
                            logger.warning(f"Rate limit hit, retrying in {sleep_time:.2f}s (attempt {attempt + 1})")
                            time.sleep(sleep_time)
                            delay *= 2
                        else:
                            logger.error(f"Max retries exceeded for rate limiting")
                            raise
                    elif "502" in error_str or "503" in error_str or "504" in error_str:
                        if attempt < max_retries - 1:
                            logger.warning(f"Server error, retrying in {delay}s")
                            time.sleep(delay)
                            delay *= 1.5
                        else:
                            raise
                    else:
                        logger.error(f"Non-retryable error: {e}")
                        raise
            raise Exception(f"Max retries ({max_retries}) exceeded")
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def optimize_image_for_api(image: Image.Image, max_size: int = 1024, quality: int = 85) -> str:
    """
    Optimize image size and quality for API transmission.
    """
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize if too large
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    # Compress to base64
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=quality, optimize=True)
    img_str = base64.b64encode(buffered.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{img_str}"


class OptimizedAzureOpenAIClient:
    """Optimized Azure OpenAI client with connection pooling and rate limiting."""
    def __init__(self, model: str, max_concurrent: int = 5):
        self.model = model
        self.max_concurrent = max_concurrent
        self.semaphore = BoundedSemaphore(max_concurrent)
        
        # Validate environment variables
        # Validate environment variables
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = (
            os.getenv("AZURE_OPENAI_ENDPOINT") 
            if model == "gpt-4o" 
            else os.getenv("O1_ENDPOINT")
        )
        
        if not self.api_key or not self.endpoint:
            raise EnvironmentError(
                "Azure OpenAI API key or endpoint not set in environment variables."
            )
        
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version="2024-07-01-preview",
            azure_endpoint=self.endpoint,
        )
    @exponential_backoff_retry(max_retries=3, initial_delay=1.0)
    def infer_single(self, prompt: str, image: Image.Image) -> str:
        """Infer response for a single prompt-image pair."""
        with self.semaphore:
            image_content = optimize_image_for_api(image)
            
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_content}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                seed=69,
                temperature=0.5,
                max_tokens=64,
                timeout=30,  # Add timeout
            )
            
            return response.choices[0].message.content.strip()


def process_batch_concurrent(
    client: OptimizedAzureOpenAIClient,
    prompts: List[str], 
    images: List[Image.Image],
    max_workers: int = 5
) -> List[str]:
    """Process a batch of prompts and images concurrently."""
    results = [None] * len(prompts)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(client.infer_single, prompt, image): idx
            for idx, (prompt, image) in enumerate(zip(prompts, images))
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"Failed to process item {idx}: {e}")
                results[idx] = f"ERROR: {str(e)}"
    
    return results


def validate_inputs(dataset_name: str, model: str, batch_size: int) -> None:
    """Validate input parameters."""
    if not dataset_name or not model:
        raise ValueError("Dataset name and model must be provided")
    
    if batch_size <= 0 or batch_size > 100:
        raise ValueError("Batch size must be between 1 and 100")
    
    # Check available memory
    memory_gb = psutil.virtual_memory().total / (1024**3)
    if memory_gb < 4:
        logger.warning("Low system memory detected, consider reducing batch size")


def main():
    """
    Optimized main function with better error handling and resource management.
    """
    parser = argparse.ArgumentParser(
        description="Process dataset and perform inference using Azure OpenAI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Dataset name in Hugging Face format (e.g., stogian/mrt_pf_mix)"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model name (e.g., gpt-4o)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--max_workers", type=int, default=5,
        help="Maximum number of concurrent workers"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output/evaluations",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    validate_inputs(args.dataset, args.model, args.batch_size)
    
    dataset_name = args.dataset
    short_name = dataset_name.split("/")[-1]
    model = args.model
    model_name = model.split("/")[-1]
    
    logger.info(f"Starting evaluation: {model} on {dataset_name}")
    logger.info(f"Batch size: {args.batch_size}, Max workers: {args.max_workers}")
    
    try:
        # Load dataset
        dataset = load_dataset(dataset_name, split="train")
        logger.info(f"Loaded dataset with {len(dataset)} samples")
        
        # Initialize optimized client
        client = OptimizedAzureOpenAIClient(model, max_concurrent=args.max_workers)
        
        all_responses = []
        
        # Process in batches with progress bar
        for i in tqdm(
            range(0, len(dataset), args.batch_size),
            desc=f"Processing {model_name}",
            unit="batch",
            colour="cyan"
        ):
            batch = dataset[i : i + args.batch_size]
            prompts = batch["question"]
            images = batch["image"]
            
            # Process batch concurrently
            responses = process_batch_concurrent(
                client, prompts, images, args.max_workers
            )
            all_responses.extend(responses)
            
            # Small delay to respect rate limits
            time.sleep(0.1)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            "question": dataset["question"],
            "response": all_responses,
            "answer": dataset["answer"],
            "split": dataset["split"],
        })
        
        # Save results
        results_dir = os.path.join(args.output_dir, short_name)
        os.makedirs(results_dir, exist_ok=True)
        
        results_path = os.path.join(results_dir, f"{model_name}.csv")
        results_df.to_csv(results_path, index=False)
        
        logger.info(f"Results saved to {results_path}")
        logger.info(f"Processed {len(all_responses)} samples successfully")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
