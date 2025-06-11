import argparse
import os
import gc
import pandas as pd
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from datasets import load_dataset, Dataset
import torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from utils.vlm_helpers import VLMWrapper as VLM

# Configure optimized logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("eval.log"),
        # logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)

class OptimizedInference:
    """Optimized inference class with memory management and error handling."""
    
    def __init__(self, model_id: str, device_map: str = "auto", use_cot: bool = False, one_shot_example: Optional[Dict] = None):
        """Initialize the inference engine."""
        self.model_id = model_id
        self.short_name = self._extract_model_name(model_id)
        self.vlm = None
        self.device_map = device_map
        self.use_cot = use_cot
        self.one_shot_example = one_shot_example
        logger.info(f"Initialized inference engine for model: {model_id}")
    
    def _extract_model_name(self, model_id: str) -> str:
        """Extract a clean model name for file naming."""
        return model_id.split("/")[-1].replace("-", "_")
    
    def _lazy_load_model(self):
        """Lazy load the VLM model to save memory."""
        if self.vlm is None:
            logger.info(f"Loading model: {self.model_id}")
            self.vlm = VLM(self.model_id, self.device_map)
            logger.info("Model loaded successfully")
    
    def _prepare_messages(self, questions: List[str], images: List[Any]) -> List[List[Dict]]:
        """Prepare messages in the required format with optional CoT and one-shot examples."""
        messages = []
        
        for question, image in zip(questions, images):
            conversation = []
            
            # Add one-shot example if provided
            if self.one_shot_example:
                # Add the example question with its image
                conversation.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": self.one_shot_example["image"]},
                        {"type": "text", "text": self._format_question_with_cot(self.one_shot_example["question"])},
                    ],
                })
                
                # Add the example response (with CoT reasoning if available)
                example_response = self.one_shot_example.get("reasoning", "")
                if example_response and self.use_cot:
                    example_response += f"\n\nTherefore, the answer is: {self.one_shot_example['answer']}"
                else:
                    example_response = self.one_shot_example["answer"]
                
                conversation.append({
                    "role": "assistant",
                    "content": example_response
                })
            
            # Add the actual question
            conversation.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self._format_question_with_cot(question)},
                ],
            })
            
            messages.append(conversation)
        
        return messages
    
    def _format_question_with_cot(self, question: str) -> str:
        """Format question with Chain-of-Thought prompting if enabled."""
        if not self.use_cot:
            return question.strip()
        
        cot_prompt = (
            "Please think step by step and explain your reasoning before providing the final answer.\n\n"
            f"{question.strip()}\n\n"
            "Let me think through this step by step:"
        )
        
        return cot_prompt
    
    def _trim_response(self, response: str, original_question: str) -> str:
        """
        Remove the original prompt/question from the response if it appears.
        
        Args:
            response: The full model response
            original_question: The original question to remove
            
        Returns:
            Cleaned response without the original prompt
        """
        # Clean the response
        response = response.strip()
        
        # Remove common prefixes that models might add
        prefixes_to_remove = [
            "Answer:",
            "Response:",
            "The answer is:",
            "A:",
            "Question:",
            "Q:",
        ]
        
        for prefix in prefixes_to_remove:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()
        
        # Remove the original question if it appears at the beginning
        question_clean = original_question.strip().lower()
        response_clean = response.lower()
        
        if response_clean.startswith(question_clean):
            response = response[len(original_question):].strip()
        
        # Remove any leading colons or spaces
        response = response.lstrip(": \n\t")
        
        return response
    
    def _process_batch(self, batch: Dict[str, List], batch_idx: int, total_batches: int) -> List[Dict[str, str]]:
        """
        Process a single batch of examples.
        
        Args:
            batch: Dictionary containing batch data
            batch_idx: Current batch index
            total_batches: Total number of batches
            
        Returns:
            List of results for this batch
        """
        try:
            batch_size = len(batch["question"])
            logger.debug(f"Processing batch {batch_idx + 1}/{total_batches} with {batch_size} examples")
            
            # Prepare inputs
            messages = self._prepare_messages(batch["question"], batch["image"])
            
            # Use memory efficient mode
            with self.vlm.memory_efficient_mode():
                # Preprocess inputs
                inputs = self.vlm.preprocess(conversation=messages, image_input=batch["image"])
                
                # Generate responses with fixed configuration
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    generated_ids = self.vlm(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=False,
                        pad_token_id=self.vlm.processor.tokenizer.eos_token_id if hasattr(self.vlm.processor, 'tokenizer') else None,
                    )
                
                # Calculate input length for proper decoding
                input_length = inputs.get("input_ids", torch.tensor([])).shape[-1] if "input_ids" in inputs else 0
                
                # Decode responses
                if self.vlm.model_type == "molmo":
                    output_texts = self.vlm.decode(generated_ids, extra=input_length)
                else:
                    # For most models, we need to trim the input tokens
                    if hasattr(generated_ids, 'shape') and len(generated_ids.shape) > 1:
                        if self.vlm.model_type not in ["minicpm"]:
                            generated_ids = generated_ids[:, input_length:]
                    output_texts = self.vlm.decode(generated_ids)
            
            # Clean up GPU memory
            del inputs, generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Process results
            results = []
            for idx, (raw_prediction, question, ground_truth) in enumerate(
                zip(output_texts, batch["question"], batch["answer"])
            ):
                # Trim the response to remove prompt repetition
                cleaned_response = self._trim_response(raw_prediction, question)
                
                results.append({
                    "response": cleaned_response,
                    "answer": ground_truth,
                    "question": question,
                    "raw_response": raw_prediction,  # Keep original for debugging
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx + 1}: {e}")
            # Return empty results for failed batch
            return [{
                "response": f"ERROR: {str(e)}",
                "answer": answer,
                "question": question,
                "raw_response": f"ERROR: {str(e)}",
            } for question, answer in zip(batch["question"], batch["answer"])]
    
    def evaluate(self, dataset_id: str, batch_size: int = 16, max_samples: Optional[int] = None) -> str:
        """
        Evaluate the model on the specified dataset.
        
        Args:
            dataset_id: HuggingFace dataset identifier
            batch_size: Number of examples per batch
            max_samples: Maximum number of samples to process (for testing)
            
        Returns:
            Path to the saved results CSV file
        """
        logger.info(f"Starting evaluation on dataset: {dataset_id}")
        if self.use_cot:
            logger.info("Chain-of-Thought prompting enabled")
        if self.one_shot_example:
            logger.info("One-shot example provided")
        
        # Load model lazily
        self._lazy_load_model()
        
        # Load dataset
        try:
            data = load_dataset(dataset_id, split="train")
            if max_samples:
                data = data.select(range(min(max_samples, len(data))))
            logger.info(f"Loaded dataset with {len(data)} samples")
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_id}: {e}")
            raise
        
        # Validate dataset format
        required_columns = ["question", "answer", "image"]
        missing_columns = [col for col in required_columns if col not in data.column_names]
        if missing_columns:
            raise ValueError(f"Dataset missing required columns: {missing_columns}")
        
        all_results = []
        total_batches = (len(data) + batch_size - 1) // batch_size
        
        # Process dataset in batches with progress bar
        with tqdm(total=total_batches, desc="Processing batches", colour="green") as pbar:
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                batch_results = self._process_batch(batch, i // batch_size, total_batches)
                all_results.extend(batch_results)
                
                pbar.update(1)
                pbar.set_postfix({
                    "processed": len(all_results),
                    "memory": f"{torch.cuda.memory_allocated() / 1e9:.1f}GB" if torch.cuda.is_available() else "N/A"
                })
                
                # Periodic garbage collection
                if (i // batch_size) % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        # Save results
        output_path = self._save_results(all_results, dataset_id)
        logger.info(f"Evaluation completed. Results saved to: {output_path}")
        
        return output_path
    
    def _save_results(self, results: List[Dict[str, str]], dataset_id: str) -> str:
        """Save results to CSV file with naming based on prompting strategy."""
        dataset_name = dataset_id.split("/")[-1]
        
        # Create directory structure with prompting strategy
        strategy_dir = self._get_strategy_directory_name()
        output_dir = Path("output/evaluations") / dataset_name / strategy_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_df = pd.DataFrame(results)
        # Use just the model name without prompting suffix for filename
        output_path = output_dir / f"{self.short_name}.csv"
        
        # Save with error handling
        try:
            results_df.to_csv(output_path, index=False)
            logger.info(f"Successfully saved {len(results)} results to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
        
        # Save summary statistics with detailed prompting info
        summary_path = output_dir / f"{self.short_name}_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"Model: {self.model_id}\n")
            f.write(f"Dataset: {dataset_id}\n")
            f.write(f"Prompting Strategy: {self._get_strategy_description()}\n")
            f.write(f"Total samples: {len(results)}\n")
            f.write(f"Successful responses: {sum(1 for r in results if not r['response'].startswith('ERROR'))}\n")
            f.write(f"Average response length: {sum(len(r['response']) for r in results) / len(results):.1f} chars\n")
            
            # Add prompting-specific metadata
            if self.use_cot:
                f.write(f"Chain-of-Thought: Enabled\n")
            if self.one_shot_example:
                f.write(f"One-shot example: {self.one_shot_example.get('question', 'N/A')[:50]}...\n")
        
        return str(output_path)

    def _get_strategy_directory_name(self) -> str:
        """Generate directory name based on prompting strategy."""
        if self.use_cot and self.one_shot_example:
            return "cot_oneshot"
        elif self.use_cot:
            return "cot"
        elif self.one_shot_example:
            return "oneshot"
        else:
            return "baseline"

    def _get_filename_suffix(self) -> str:
        """Generate filename suffix based on prompting strategy."""
        suffix_parts = []
        
        if self.use_cot and self.one_shot_example:
            suffix_parts.append("cot_oneshot")
        elif self.use_cot:
            suffix_parts.append("cot")
        elif self.one_shot_example:
            suffix_parts.append("oneshot")
        else:
            suffix_parts.append("baseline")
        
        return "_" + "_".join(suffix_parts) if suffix_parts else ""

    def _get_strategy_description(self) -> str:
        """Get human-readable description of prompting strategy."""
        if self.use_cot and self.one_shot_example:
            return "Chain-of-Thought + One-shot Example"
        elif self.use_cot:
            return "Chain-of-Thought"
        elif self.one_shot_example:
            return "One-shot Example"
        else:
            return "Baseline (No special prompting)"

def evaluate(model_id: str, dataset_id: str, batch_size: int = 16, max_samples: Optional[int] = None) -> str:
    """
    Main evaluation function with improved interface.
    
    Args:
        model_id: Model identifier
        dataset_id: Dataset identifier  
        batch_size: Batch size for processing
        max_samples: Maximum samples to process (for testing)
        
    Returns:
        Path to saved results
    """
    inference_engine = OptimizedInference(model_id)
    return inference_engine.evaluate(dataset_id, batch_size, max_samples)


def parse_args():
    """Parse command-line arguments with additional options."""
    parser = argparse.ArgumentParser(
        description="Optimized evaluation of vision-language models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-m", "--model",
        type=str, required=True,
        help="Model identifier (e.g., 'meta-llama/Llama-3.2-90B-Vision-Instruct')"
    )
    parser.add_argument(
        "-d", "--dataset", 
        type=str, required=True,
        help="Dataset identifier (e.g., 'stogian/sr_test')"
    )
    parser.add_argument(
        "-b", "--batch_size",
        type=int, default=16,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--max_samples",
        type=int, default=None,
        help="Maximum samples to process (for testing)"
    )
    parser.add_argument(
        "--device_map",
        type=str, default="auto",
        help="Device mapping strategy"
    )
    parser.add_argument(
        "--cot", "--chain-of-thought",
        action="store_true",
        help="Enable Chain-of-Thought prompting"
    )
    parser.add_argument(
        "--one-shot",
        type=str, default=None,
        help="Path to one-shot example JSON file"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()

def load_one_shot_example(json_path: str) -> Optional[Dict]:
    """Load one-shot example from JSON file with image loading."""
    if not json_path:
        return None
    
    # Handle relative paths for JSON file
    if not os.path.isabs(json_path):
        json_path = os.path.join('/users/stogian/srbench', json_path)
    
    if not os.path.exists(json_path):
        logger.warning(f"One-shot JSON file not found: {json_path}")
        return None
    
    try:
        import json
        from PIL import Image
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Handle image path with multiple fallbacks
        image_path = data.get('image_path', '')
        if not image_path:
            logger.warning("No image_path found in one-shot example")
            return None
        
        # Try multiple path variations
        possible_paths = [
            image_path,  # Use path as provided
            os.path.join(os.path.dirname(json_path), os.path.basename(image_path)),  # Same dir as JSON
            os.path.join('/users/stogian/srbench', image_path.lstrip('./')),  # Relative to project root
            os.path.join('/users/stogian/srbench/example', os.path.basename(image_path)),  # In example dir
        ]
        
        image_loaded = False
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    data['image'] = Image.open(path).convert('RGB')
                    logger.info(f"Successfully loaded one-shot image from: {path}")
                    image_loaded = True
                    break
                except Exception as e:
                    logger.warning(f"Failed to load image from {path}: {e}")
                    continue
        
        if not image_loaded:
            logger.error(f"Could not load image from any of these paths: {possible_paths}")
            return None
        
        # Validate required keys
        required_keys = ['question', 'answer']
        if not all(key in data for key in required_keys):
            logger.warning(f"One-shot example missing required keys: {required_keys}")
            return None
        
        logger.info(f"Loaded one-shot example: {data['question'][:50]}...")
        return data
        
    except Exception as e:
        logger.error(f"Failed to load one-shot example: {e}")
        return None

if __name__ == "__main__":
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load one-shot example if provided
    one_shot_example = load_one_shot_example(args.one_shot) if args.one_shot else None
    
    try:
        # Create inference engine with CoT and one-shot support
        inference_engine = OptimizedInference(
            model_id=args.model,
            device_map=args.device_map,
            use_cot=args.cot,
            one_shot_example=one_shot_example
        )
        
        output_path = inference_engine.evaluate(
            dataset_id=args.dataset,
            batch_size=args.batch_size,
            max_samples=args.max_samples
        )
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📁 Results saved to: {output_path}")
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        print("\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"\n❌ Evaluation failed: {e}")
        raise
