import argparse
import base64
import logging
import os
import time
from functools import wraps
from io import BytesIO
from typing import List

import dotenv
import pandas as pd
from datasets import load_dataset
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

# Load the environment variables
dotenv.load_dotenv()

# Set up logging with both file and console output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("eval_openai.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def retry_with_exponential_backoff(max_retries=5, initial_delay=5):
    """
    Retry decorator that retry a function call with exponential backoff.

    Args:
        max_retries (int): Maximum number of retries.
        initial_delay (int): Initial delay in seconds.

    Returns:
        Callable: Decorated function with retry logic.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "429" in str(e):
                        logger.warning(f"Rate limit exceeded (attempt {attempt + 1}/{max_retries}). Retrying in {delay} seconds...")
                        time.sleep(delay)
                        delay *= 2  # exponential backoff
                    else:
                        logger.error(f"Error during API call: {e}")
                        raise e
            logger.error("Max retries exceeded")
            raise Exception("Max retries exceeded")

        return wrapper

    return decorator


def image_to_base64(image: Image.Image) -> str:
    """
    Convert a PIL Image to a base64 encoded string.

    Args:
        image (Image.Image): The image to convert.

    Returns:
        str: A base64 encoded string of the image.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("ascii")
    return img_str


@retry_with_exponential_backoff()
def infer(prompts: List[str], images: List[Image.Image], model: str) -> List[str]:
    """
    Infer responses using OpenRouter from provided prompts and images.

    Args:
        prompts (List[str]): A list of prompts/questions.
        images (List[Image.Image]): A list of images corresponding to the prompts.
        model (str): The model identifier to use for inference.

    Returns:
        List[str]: A list of responses from the model.
    """
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("OpenRouter API key not set in environment variables.")
        raise EnvironmentError(
            "OpenRouter API key not set in environment variables."
        )

    logger.debug(f"Initializing OpenRouter client for model: {model}")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    contents = []
    for idx, (prompt, image) in enumerate(zip(prompts, images)):
        logger.debug(f"Processing item {idx + 1}/{len(prompts)}")
        if isinstance(image, Image.Image):
            image_content = image_to_base64(image)
            image_content = f"data:image/png;base64,{image_content}"
        else:
            image_content = str(image)


        # Merge the image and text into one message
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

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            seed=69,
            temperature=0.5,
            max_tokens=64,
        )
        content = response.choices[0].message.content.strip()
        contents.append(content)

    return contents


def main():
    """
    Main function to process the dataset and perform inference based on
    provided command-line arguments for dataset name and model.
    """
    parser = argparse.ArgumentParser(
        description="Process dataset and perform inference using OpenRouter."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name in Hugging Face format (e.g., stogian/mrt_pf_mix)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., gpt-4o)",
        default="x-ai/grok-4.1-fast:free",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for processing abstracts"
    )
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("Starting evaluation")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("=" * 50)

    dataset_name = args.dataset
    short_name = dataset_name.split("/")[-1]

    model = args.model
    model_name = model.split("/")[-1]

    # Load the specified dataset
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="test")
    logger.info(f"Dataset loaded successfully. Total samples: {len(dataset)}")

    all_responses = []
    total_batches = (len(dataset) + args.batch_size - 1) // args.batch_size
    logger.info(f"Starting inference with {total_batches} batches")

    for i in tqdm(
        range(0, len(dataset), args.batch_size),
        desc="Processing batches",
        unit="batch",
        leave=False,
        colour="magenta",
    ):
        batch_num = i // args.batch_size + 1
        logger.info(f"Processing batch {batch_num}/{total_batches}")
        batch = dataset[i : i + args.batch_size]
        prompts = batch["question"]
        images = batch["image"]

        responses = infer(prompts, images, model)
        all_responses.extend(responses)
        logger.info(f"Batch {batch_num} completed. Total responses so far: {len(all_responses)}")

    logger.info(f"Inference completed. Total responses: {len(all_responses)}")

    results_df = pd.DataFrame(
        {
            "question": dataset["question"],
            "response": all_responses,
            "answer": dataset["answer"],
            "split": dataset["split"],
        }
    )
    results_dir = f"output/evaluations/{short_name}/"
    os.makedirs(results_dir, exist_ok=True)

    output_path = os.path.join(results_dir, f"{model_name}.csv")
    results_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
    logger.info("Evaluation completed successfully")


if __name__ == "__main__":
    main()
