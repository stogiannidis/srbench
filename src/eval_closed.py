import argparse
import base64
import logging
import os
import asyncio
from io import BytesIO

import dotenv
import pandas as pd
from datasets import load_dataset
from openai import AsyncOpenAI, DefaultAioHttpClient
from PIL import Image
from tqdm.asyncio import tqdm

# Load environment
dotenv.load_dotenv()

# Set up logging
logging.basicConfig(filename="example.log", encoding="utf-8", level=logging.INFO)

COT_PROMPT = "Let's think step by step. Provide a detailed reasoning process before giving the final answer within curly brackets, e.g. {{yes}}."

def image_to_base64(image: Image.Image) -> str:
    """
    Optimised for speed: Converts to JPEG (smaller payload than PNG) 
    and ensures RGB mode.
    """
    buffered = BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    # JPEG at 85 quality is significantly faster to upload than PNG
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

async def infer_task(sem, client, prompt, image, model, use_cot):
    """
    Wraps the inference in a semaphore to control concurrency.
    """
    async with sem:
        try:
            # Prepare content
            img_b64 = image_to_base64(image)
            full_prompt = f"{prompt}\n\n{COT_PROMPT}" if use_cot else prompt
            
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{img_b64}"},
                            {"type": "text", "text": full_prompt},
                        ],
                    },
                ],
                max_tokens=4096,
                temperature=0,
            )
            message = response.choices[0].message
            reasoning = getattr(message, "reasoning", None)
            content = message.content
            return content, reasoning
        except Exception as e:
            logging.error(f"Inference error: {e}")
            return f"ERROR: {str(e)}", None

async def run_inference(dataset, model, use_cot, max_concurrency):
    # Initialise the client with the aiohttp backend
    async with AsyncOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        http_client=DefaultAioHttpClient(),
    ) as client:
        
        sem = asyncio.Semaphore(max_concurrency)
        tasks = []
        
        for row in dataset:
            tasks.append(
                infer_task(sem, client, row["question"], row["image"], model, use_cot)
            )
        
        # tqdm.gather provides a nice progress bar for async tasks
        return await tqdm.gather(*tasks, desc="Processing Images")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--max_concurrency", type=int, default=100)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, split="test")
    
    responses = await run_inference(
        dataset, 
        args.model, 
        args.cot, 
        args.max_concurrency
    )

    contents, reasonings = zip(*responses)
    # Save results
    results_df = pd.DataFrame({
        "question": dataset["question"],
        "response": contents,
        "reasoning": reasonings,
        "answer": dataset["answer"],
    })
    
    output_path = f"results_{args.model.split('/')[-1]}.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Done. Saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())