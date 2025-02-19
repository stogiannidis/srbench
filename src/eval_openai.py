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
from openai import AzureOpenAI
from PIL import Image
from tqdm import tqdm

# Load the environment variables
dotenv.load_dotenv()

# Set up logging
logging.basicConfig(filename="example.log", encoding="utf-8", level=logging.INFO)

dataset = load_dataset("stogian/sr_test", split="train")
model = "o1"

def retry_with_exponential_backoff(max_retries=5, initial_delay=5):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for _ in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "429" in str(e):
                        print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                        time.sleep(delay)
                        delay *= 2  # exponential backoff
                    else:
                        raise e
            raise Exception("Max retries exceeded")

        return wrapper

    return decorator


def image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


# Function to create a mapping based on abstracts
@retry_with_exponential_backoff()
def infer(prompts: List[str], images: List[Image.Image]) -> List[str]:
    if not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_ENDPOINT"):
        raise EnvironmentError(
            "Azure OpenAI API key or endpoint not set in environment variables."
        )

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-07-01-preview",
        azure_endpoint=os.getenv("O1_ENDPOINT"),
    )

    contents = []
    for prompt, image in zip(prompts, images):
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
                "content": f"{prompt}\nImage data: {image_content}"
            },
        ]

        global model # Use the global model variable
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            seed=69,
            temperature=1,
            max_tokens=3,
        )
        content = response.choices[0].message.content.strip()
        contents.append(content)

    return contents


# Example usage
if __name__ == "__main__":
    # Process abstracts in batches
    batch_size = 10
    all_responses = []

    for i in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
        batch = dataset[i : i + batch_size]
        prompts = batch["question"]
        images = batch["image"]

        responses = infer(prompts, images)
        all_responses.extend(responses)

    results_df = pd.DataFrame({
         "question": dataset["question"],
         "response": all_responses,
         "answer": dataset["answer"]
    })
    results_df.to_csv(f"results_{model}.csv", index=False)