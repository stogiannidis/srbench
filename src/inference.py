import argparse
import os
import re
import pandas as pd
import logging
from tqdm import tqdm
from datasets import load_dataset
import torch

from utils import VLMWrapper as VLM

# Configure logger
logging.basicConfig(
    filename="eval.log",
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.propagate = False

def evaluate(model_id: str, dataset_id: str, batch_size: int = 16):
    """
    Evaluate a given vision-language model on the specified dataset.

    Args:
        model_id (str): The identifier of the model to evaluate.
        dataset_id (str): The identifier of the dataset to use for evaluation.
        batch_size (int): Number of examples processed per batch.
    """
    # Compute short name for the model
    short_name = model_id.split("/")[-1]
    logger.info(f"\n\nStarting evaluation script for model: {model_id}")

    dataset_name = dataset_id.split("/")[-1]

    # Instantiate VLM with the specified model_id.
    vlm = VLM(model_id, "auto")

    # Load dataset from Hugging Face Datasets.
    data = load_dataset(dataset_id, split="train")

    results = []

    # Loop through the dataset in batches.
    for i in tqdm(
        range(0, len(data), batch_size), desc="Processing batches", colour="red"
    ):
        batch = data[i : i + batch_size]

        messages = []
        ground_truths = batch["answer"]
        images = batch["image"]
        questions = batch["question"]

        for q, image in zip(questions, images):
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {
                                "type": "text",
                                "text": q,
                            },
                        ],
                    }
                ]
            )

        logger.info("Messages: %s", messages)

        inputs = vlm.preprocess(conversation=messages, image_input=images)
        logger.info("Preprocessed inputs: %s", inputs)

        # Run generation using the callable vlm interface.
        generated_ids = vlm(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )
        logger.info("Generated output: %s", generated_ids)

        # Decode the generated ids
        output_texts = vlm.decode(generated_ids)
        
        del generated_ids, inputs

        # Process the outputs
        for idx, raw_prediction in enumerate(output_texts):
            pred = raw_prediction.strip()
            ground_truth = ground_truths[idx]  # Get corresponding ground truth
            question = questions[idx]
            results.append(
                {
                    "response": pred,
                    "answer": ground_truth,
                    "question": question,
                }
            )

    # Create output directory and save results to a CSV file.
    output_dir = f"output/evaluations/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(output_dir, f"{short_name}.csv")
    results_df.to_csv(results_csv_path, index=False)
    logger.info("Results saved to %s", results_csv_path)


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        Namespace: Parsed arguments including model_id and dataset_id.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a vision-language model on a specified dataset."
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Identifier of the model to evaluate, e.g., 'meta-llama/Llama-3.2-90B-Vision-Instruct'.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="Identifier of the dataset to use for evaluation, e.g., 'stogian/sr_test'.",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for processing the dataset.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(model_id=args.model, dataset_id=args.dataset, batch_size=args.batch_size)
