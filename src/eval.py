import argparse
import os
import gc
import json
import random
import hashlib
import pandas as pd
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from datasets import load_dataset, Dataset
import torch
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image

from utils.vlm_wrapper import VLMWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EvaluationEngine:
    """Evaluation engine for all VLM types with enhanced features."""
    
    def __init__(self, model_id: str, device_map: str = "auto", seed: int = 42, 
                 use_cot: bool = False, one_shot_example: Optional[Dict] = None):
        """
        Initialize the evaluation engine.
        
        Args:
            model_id: HuggingFace model identifier
            device_map: Device mapping strategy
            seed: Random seed for reproducibility
            use_cot: Enable Chain-of-Thought prompting
            one_shot_example: One-shot example dictionary
        """
        self.model_id = model_id
        self.short_name = self._extract_model_name(model_id)
        self.device_map = device_map
        self.seed = seed
        self.use_cot = use_cot
        self.one_shot_example = one_shot_example
        
        # Initialize reproducibility
        self._set_seed(seed)
        
        # Lazy initialization
        self.vlm = None
        
        # Evaluation metadata
        self.eval_metadata = {
            "model_id": model_id,
            "seed": seed,
            "use_cot": use_cot,
            "has_one_shot": one_shot_example is not None,
            "timestamp": datetime.now().isoformat(),
            "torch_version": torch.__version__,
        }
        
        logger.info(f"Initialized EvaluationEngine for model: {model_id}")
        logger.info(f"Seed: {seed}, CoT: {use_cot}, One-shot: {one_shot_example is not None}")
    
    def _set_seed(self, seed: int):
        """Set all random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Make cudnn deterministic
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Set environment variables for reproducibility
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'  # For deterministic CUDA operations
        
        logger.info(f"Set all seeds to {seed} for reproducibility")
    
    def _extract_model_name(self, model_id: str) -> str:
        """Extract a clean model name for file naming."""
        return model_id.split("/")[-1].replace("-", "_")
    
    def _compute_dataset_hash(self, dataset_id: str, max_samples: Optional[int] = None) -> str:
        """Compute a hash of the dataset configuration for reproducibility tracking."""
        config_str = f"{dataset_id}_{max_samples}_{self.seed}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _load_model(self):
        """Load the VLM model lazily to save memory."""
        if self.vlm is None:
            logger.info(f"Loading model: {self.model_id}")
            self.vlm = VLMWrapper(self.model_id, self.device_map)
            logger.info("Model loaded successfully")
            
            # Add model-specific metadata
            self.eval_metadata.update({
                "model_type": self.vlm.model_type,
                "inference_type": self.vlm.config.inference_type,
                "device_map": self.device_map,
                "dtype": str(self.vlm.dtype),
            })
    
    def _prepare_messages(self, questions: List[str], images: List[Image]) -> List[List[Dict]]:
        """Prepare messages in the required format with optional CoT and one-shot examples."""
        messages = []
        
        for question, image in zip(questions, images):
            conversation = []
            
            # Add system message with model and strategy info
            system_message = {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": ("You are a spatial reasoning AI assistant specialized in analyzing, "
                            "understanding, and solving problems involving spatial relationships, "
                            "geometric transformations, and visual-spatial concepts."
                            )
                }]
            }
            conversation.append(system_message)
            
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
    
    def _process_batch(self, batch: Dict[str, List], batch_idx: int, total_batches: int) -> List[Dict[str, str]]:
        """
        Process a single batch of examples with unified interface.
        
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
            
            # Unified batch processing leveraging unified preprocessing in VLMWrapper
            results = self._process_batch_standard(messages, batch, batch_idx)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx + 1}: {e}", exc_info=True)
            # Return empty results for failed batch
            return [{
                "response": f"ERROR: {str(e)}",
                "answer": answer,
                "question": question,
                "raw_response": f"ERROR: {str(e)}",
                "batch_idx": batch_idx,
                "example_idx": idx,
            } for idx, (question, answer) in enumerate(zip(batch["question"], batch["answer"]))]
    
    def _process_batch_standard(self, messages: List[List[Dict]], batch: Dict[str, List], batch_idx: int) -> List[Dict[str, str]]:
        """Process batch for standard VLM models."""
        results = []
        
        with self.vlm.memory_efficient_mode():
            # Preprocess the entire batch
            inputs = self.vlm.preprocess(conversation=messages, image_input=batch["image"])
            

            generated_ids = self.vlm.generate(
                inputs,
                max_new_tokens=1024,
                do_sample=False
            )
            
            output_texts = self.vlm.decode(generated_ids)


            # Process results for each example in the batch
            for idx, (raw_prediction, question, ground_truth) in enumerate(zip(output_texts, batch["question"], batch["answer"])):
                results.append({
                    "question": question,
                    "response": raw_prediction,
                    "gold answer": ground_truth,
                    "batch_idx": batch_idx,
                    "example_idx": idx,
                })

        return results
    
    def evaluate(self, dataset_id: str, batch_size: int = 16, max_samples: Optional[int] = None, 
                sample_strategy: str = "first") -> str:
        """
        Evaluate the model on the specified dataset with reproducible sampling.
        
        Args:
            dataset_id: HuggingFace dataset identifier
            batch_size: Number of examples per batch
            max_samples: Maximum number of samples to process
            sample_strategy: Sampling strategy ("first", "random", "stratified")
            
        Returns:
            Path to the saved results CSV file
        """
        # Load model lazily
        self._load_model()
        
        # Load and sample dataset
        try:
            data = load_dataset(dataset_id, split="train")
            original_size = len(data)
            
            # Apply sampling strategy
            if max_samples and max_samples < original_size:
                data = self._sample_dataset(data, max_samples, sample_strategy)
            
            logger.info(f"Loaded dataset: {original_size} total, {len(data)} selected samples")
            
            # Add dataset hash to metadata
            self.eval_metadata.update({
                "dataset_id": dataset_id,
                "dataset_hash": self._compute_dataset_hash(dataset_id, max_samples),
                "original_size": original_size,
                "sampled_size": len(data),
                "sample_strategy": sample_strategy,
                "max_samples": max_samples,
            })
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_id}: {e}")
            raise
        
        # Validate dataset format
        required_columns = ["question", "answer", "image"]
        missing_columns = [col for col in required_columns if col not in data.column_names]
        if missing_columns:
            raise ValueError(f"Dataset missing required columns: {missing_columns}")
        
        # Process dataset in batches
        all_results = []
        total_batches = (len(data) + batch_size - 1) // batch_size
    
        
        with tqdm(total=total_batches, desc="Processing batches", colour="green") as pbar:
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                batch_results = self._process_batch(batch, i // batch_size, total_batches)
                all_results.extend(batch_results)
                
                pbar.update(1)
                pbar.set_postfix({
                    "processed": len(all_results),
                    "memory": f"{torch.cuda.memory_allocated() / 1e9:.1f}GB" if torch.cuda.is_available() else "N/A",
                    "success_rate": f"{sum(1 for r in all_results if not r['response'].startswith('ERROR')) / len(all_results) * 100:.1f}%"
                })
                
                # Periodic garbage collection
                if (i // batch_size) % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        # Save results with metadata
        output_path = self._save_results_with_metadata(all_results, dataset_id)
        logger.info(f"Evaluation completed. Results saved to: {output_path}")
        
        return output_path
    
    def _sample_dataset(self, dataset: Dataset, max_samples: int, strategy: str) -> Dataset:
        """Sample dataset using specified strategy for reproducibility."""
        if strategy == "first":
            return dataset.select(range(max_samples))
        elif strategy == "random":
            indices = list(range(len(dataset)))
            random.shuffle(indices)  # Uses the set seed
            selected_indices = indices[:max_samples]
            return dataset.select(selected_indices)
        elif strategy == "stratified":
            # Try to stratify by answer if possible, otherwise fall back to random
            try:
                answers = dataset["answer"]
                unique_answers = list(set(answers))
                samples_per_answer = max_samples // len(unique_answers)
                
                selected_indices = []
                for answer in unique_answers:
                    answer_indices = [i for i, a in enumerate(answers) if a == answer]
                    random.shuffle(answer_indices)
                    selected_indices.extend(answer_indices[:samples_per_answer])
                
                # Fill remaining slots randomly
                remaining = max_samples - len(selected_indices)
                if remaining > 0:
                    all_indices = set(range(len(dataset)))
                    available_indices = list(all_indices - set(selected_indices))
                    random.shuffle(available_indices)
                    selected_indices.extend(available_indices[:remaining])
                
                return dataset.select(selected_indices[:max_samples])
            except Exception as e:
                logger.warning(f"Stratified sampling failed, falling back to random: {e}")
                return self._sample_dataset(dataset, max_samples, "random")
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    def _save_results_with_metadata(self, results: List[Dict[str, str]], dataset_id: str) -> str:
        """Save results with comprehensive metadata for reproducibility."""
        dataset_name = dataset_id.split("/")[-1]
        
        # Create directory structure with prompting strategy
        strategy_dir = self._get_strategy_directory_name()
        output_dir = Path("output/evaluations") / dataset_name / strategy_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        results_df = pd.DataFrame(results)
        output_path = output_dir / f"{self.short_name}.csv"
        
        try:
            results_df.to_csv(output_path, index=False)
            logger.info(f"Successfully saved {len(results)} results to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
        
        # Save detailed metadata for reproducibility
        metadata_path = output_dir / f"{self.short_name}_metadata.json"
        
        # Add evaluation statistics to metadata
        successful_responses = sum(1 for r in results if not r['response'].startswith('ERROR'))
        self.eval_metadata.update({
            "total_samples": len(results),
            "successful_responses": successful_responses,
            "success_rate": successful_responses / len(results) * 100,
            "average_response_length": sum(len(r['response']) for r in results) / len(results),
            "output_path": str(output_path),
            "evaluation_completed_at": datetime.now().isoformat(),
        })
        
        with open(metadata_path, 'w') as f:
            json.dump(self.eval_metadata, f, indent=2, default=str)
        
        logger.info(f"Saved metadata to {metadata_path}")
        
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
            return "_baseline"


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
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Handle image path with multiple fallbacks
        image_path = data.get('image_path', '')
        if not image_path:
            logger.warning("No image_path found in one-shot example")
            return None
        
        # Try multiple path variations
        possible_paths = [
            image_path,
            os.path.join(os.path.dirname(json_path), os.path.basename(image_path)),
            os.path.join('/users/stogian/srbench', image_path.lstrip('./')),
            os.path.join('/users/stogian/srbench/example', os.path.basename(image_path)),
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


def parse_args():
    """Parse command-line arguments with comprehensive options."""
    parser = argparse.ArgumentParser(
        description="Reproducible evaluation of vision-language models",
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
        "--sample_strategy",
        type=str, default="first", choices=["first", "random", "stratified"],
        help="Sampling strategy for subset selection"
    )
    parser.add_argument(
        "--device_map",
        type=str, default="auto",
        help="Device mapping strategy"
    )
    parser.add_argument(
        "--seed",
        type=int, default=42,
        help="Random seed for reproducibility"
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


def main():
    """Main function with comprehensive error handling."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load one-shot example if provided
    one_shot_example = load_one_shot_example(args.one_shot) if args.one_shot else None
    
    try:
        # Create reproducible evaluation engine
        eval_engine = EvaluationEngine(
            model_id=args.model,
            device_map=args.device_map,
            seed=args.seed,
            use_cot=args.cot,
            one_shot_example=one_shot_example
        )
        
        output_path = eval_engine.evaluate(
            dataset_id=args.dataset,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            sample_strategy=args.sample_strategy
        )
        
        print(f"\n✅ Reproducible evaluation completed successfully!")
        print(f"📁 Results saved to: {output_path}")
        print(f"🔄 Evaluation can be reproduced using seed: {args.seed}")
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        print("\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        print(f"\n❌ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()