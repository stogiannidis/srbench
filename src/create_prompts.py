#!/usr/bin/env python
"""
This script demonstrates how to:
1. Define a list of simple objects with attributes and a set of spatial test tasks.
2. For each spatial task and each model, generate multiple scene descriptions using an LLM.
3. Post-process the model's output to remove internal commentary.
4. Log each generated prompt and metadata to both WandB and Neptune.ai.
5. Save each generated scene description and metadata to a JSONL file.
"""

import re
import os
import random
import json
import logging
import glob
from typing import List, Dict, Tuple, Any

import neptune.utils
import torch
import pandas as pd
from transformers import pipeline
from dotenv import load_dotenv
import wandb 
import neptune

# Load environment variables from the .env file.
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(filename="debug_prompts.log", encoding="utf-8", level=logging.DEBUG)

# -------------------------------
# Type Aliases
# -------------------------------
ObjectType = Dict[str, Any]
SpatialTaskType = Dict[str, str]
MessageType = Dict[str, str]

# -------------------------------
# Constants
# -------------------------------
SAVE_DIR = "output/prompts/"
TRIAL_NAME = "all-llm-prompts-trial_v2"
OUTPUT_FILENAME = (
    f"{SAVE_DIR}{TRIAL_NAME}.jsonl"  # All prompts will be appended here.
)

# -------------------------------
# 7. List of Models to Use
# -------------------------------
MODELS: List[str] = glob.glob("bin/models/llms/*")

# -------------------------------
# Initialize WandB
# -------------------------------
wandb.init(
    project="SpatialScenePrompts",
    name=TRIAL_NAME,
    config={
        "num_prompts_per_task": 10,
        "models": MODELS,
    },
)

# -------------------------------
# Initialize Neptune.ai Run
# -------------------------------
neptune_run = neptune.init_run(
    project="stogiannidis/create-prompts",  # Replace with your project name.
    api_token=os.getenv("NEPTUNE_API_TOKEN"),  # Replace with your Neptune API token.
    name=TRIAL_NAME,
)
neptune_run["config/num_prompts_per_task"] = 10
neptune_run["config/models"] = neptune.utils.stringify_unsupported(MODELS)

# -------------------------------
# 1. List of Simple Objects with Attributes
# -------------------------------
OBJECTS: List[str] = ["apples", "oranges", "bowling ball", "basket ball", "foot ball", "soccer ball", "tennis ball", "golf ball", "baseball"
    "bicycle", "motorcycle", "scooter", "skateboard", "car", "truck", "bus", "train", "airplane", "helicopter", "man", "woiman", "kid",
    "dog", "cat", "bird", "fish", "tree", "bush", "flower", "grass", "rock", "mountain", "hill", "valley", "river", "lake", "ocean", "sea",
]

# -------------------------------
# 2. List of Spatial Test Tasks
# -------------------------------
# Each task now uses a concise definition and an example.
SPATIAL_TASKS: List[SpatialTaskType] = [
    {
        "task_type": "mental rotation",
        "example": "A vintage red convertible facing right, parked on a sunlit cobblestone street. Background details include lush green trees and charming old buildings bathed in soft afternoon light.",
    },
    {
        "task_type": "physics and causality",
        "example": "Two vibrant red apples, one tumbling from a high branch, the other dropping from a low fence, set against a soft-focus green orchard, bathed in warm sunlight.",
    },
    # {
    #     "task_type": "compositionality",
    #     "example": "A tranquil park view, highlighting a weathered stone bench positioned near a towering pine tree, bathed in soft morning light, surrounded by lush, emerald grass and blooming wildflowers.",
    # },
    # {
    #     "task_type": "visualization",
    #     "example": "A piece of paper being crumpled into a ball, set against a stark white background, with soft shadows cast by a single light source.",
    # },
    {
        "task_type": "perspective_taking",
        "example": "A cozy cafe scene featuring two individuals seated at a wooden table looking at a cup of coffee",
    },
]

# -------------------------------
# 3. Enhanced System Prompt
# -------------------------------
# This prompt instructs the LLM to generate a test prompt for spatial reasoning.
SYSTEM_PROMPT: str = f"""
You are an advanced assistant tasked with writing image generation prompts. You have 20+ years of expertise in cognitive psychology.
Your task is to create a concise image description prompt that can be used to evaluate spatial reasoning abilities.

Here are some examples:
- Two identical apples are falling from different heights. Both apples are released at the same time.
- A vintage red convertible facing right, parked on a sunlit cobblestone street.
- Two men are sitting on a bench, one reading a book and the other is looking at distant mountains.
- A woman is walking a dog in a park with blooming flowers and lush green trees.

Give only the essential details needed to generate a detailed scene description.

Hint: Include specific spatial relationships, object attributes, and environmental details to guide the scene construction process.
Here are some optional objects you can use: {OBJECTS}. Feel free to add more objects as needed.

Provide ONLY a concise short prompt for the image generation task. Do not include anything else.
"""

# -------------------------------
# 4. Function to Construct a Prompt for a Given Task
# -------------------------------

def construct_prompt_for_task(task: SpatialTaskType) -> Tuple[str, str]:
    """
    Constructs a test prompt using the task's definition and example, and optionally
    includes one or two randomly selected simple objects with attributes.

    Args:
            task: A spatial test task containing "task_type", "definition", and "example".

    Returns:
            A tuple with the task type and the constructed user prompt.
    """
    task_type = task["task_type"]
    base_details = (
        f"Task Type: {task_type}\n"
        f"Example: {task['example']}\n"
    )
    user_prompt = (
        f"Generate a complete image generation prompt following on the example provided; however, be creative and come up with your own unique prompt.\n"
        f"{base_details}"
    )
    return task_type, user_prompt


# -------------------------------
# 5. Post-Processing Function
# -------------------------------
def post_process_output(output_text: str) -> str:
    """
    Removes any text between <think> and </think> tags and strips whitespace.

    Args:
            output_text: The raw output text from the model.

    Returns:
            The cleaned output text.
    """
    cleaned_text: str = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL)
    return cleaned_text.strip()


# -------------------------------
# 6. Helper Functions to Adapt to Different Pipeline Variants
# -------------------------------
def supports_system_prompt(model_name: str) -> bool:
    """
    Heuristic: if the model id contains "gemma" (case-insensitive) we assume it does not support chat-style prompts.
    """
    return "gemma" not in model_name.lower()


def prepare_pipeline_input(messages: List[MessageType], model_name: str) -> Any:
    """
    Prepares the input for the pipeline call based on whether the model supports system prompts.
    """
    if supports_system_prompt(model_name):
        return messages
    else:
        combined_prompt = "\n\n".join(msg["content"].strip() for msg in messages)
        return [{"role": "user", "content": combined_prompt.strip()}]


def extract_generated_text(outputs: Any) -> str:
    """
    Extracts the generated text from the output of the pipeline.

    Supports both nested (chat-style) and flat output.
    """
    if isinstance(outputs, list) and outputs:
        output_item = outputs[0]
        if "generated_text" in output_item:
            gen = output_item["generated_text"]
            if isinstance(gen, list):
                last = gen[-1]
                if isinstance(last, dict) and "content" in last:
                    return last["content"].strip()
                elif isinstance(last, str):
                    return last.strip()
            elif isinstance(gen, str):
                return gen.strip()
    return ""


# -------------------------------
# 8. Main Routine: Iterate Over All Model and Task Combinations
# -------------------------------
def main() -> None:
    logger.info("\n\nStarting main routine with WandB and Neptune.ai logging.")

    for model_name in MODELS:
        logger.info("Processing model: %s", model_name)
        # Get the original name of the model without the path.
        sanitized_model: str = model_name.split("/")[-1]
        logger.info("Sanitized model name: %s", sanitized_model)

        # Initialize the text-generation pipeline for the current model.
        llm_pipe: Any = pipeline(
            "text-generation",
            model=model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        logger.info("Pipeline initialized for %s", sanitized_model)

        for task in SPATIAL_TASKS:
            # Generate a number of prompts for each task.
            for i in range(2):  # Adjust number per task as needed.
                logger.info(
                    "Generating prompt %d for task: %s", i + 1, task["task_type"]
                )
                task_type, user_prompt = construct_prompt_for_task(task)
                logger.info("User prompt generated for task: %s", task_type)

                # Prepare the messages.
                messages: List[MessageType] = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
                prompt_input = prepare_pipeline_input(messages, model_name)

                # Generate the scene description using the LLM.
                outputs: Any = llm_pipe(prompt_input, max_new_tokens=2048)
                generated_text: str = extract_generated_text(outputs)
                if generated_text:
                    cleaned_text = post_process_output(generated_text)

                # Prepare output metadata.
                output_data: Dict[str, Any] = {
                    "model": sanitized_model,
                    "task_type": task_type,
                    "user_prompt": user_prompt,
                    "generated_scene_description": cleaned_text,
                    "iteration": i + 1,
                }

                # Log the metadata to WandB.
                wandb.log(output_data)

                # Log the metadata to Neptune.
                neptune_run[
                    f"prompt/{sanitized_model}/{task_type}/iteration_{i + 1}"
                ] = output_data

                # Save the generated description and metadata to a JSONL file.
                with open(OUTPUT_FILENAME, "a", encoding="utf-8") as f:
                    f.write(json.dumps(output_data, ensure_ascii=False) + "\n")

    # Finish logging for both platforms.
    logger.info("Main routine completed.")

    # Create a Table of Contents for the generated prompts.
    results_df: pd.DataFrame = pd.read_json(OUTPUT_FILENAME, lines=True)
    
    # Log the Table of Contents to both platforms.
    wandb.log({"results": wandb.Table(data=results_df)})
    neptune_run["output"].upload(neptune.types.File.as_html(results_df))

    # Finish the run for both platforms.
    wandb.finish()
    neptune_run.stop()


if __name__ == "__main__":
    main()
