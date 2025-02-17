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
load_dotenv(override=True)

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
TRIAL_NAME = "fiveshotv3-prompts-trial"
OUTPUT_FILENAME = f"{SAVE_DIR}{TRIAL_NAME}.jsonl"  # All prompts will be appended here.
PROMPTS_PER_TASK = 12
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
        "num_prompts_per_task": PROMPTS_PER_TASK,
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
neptune_run["config/num_prompts_per_task"] = PROMPTS_PER_TASK
neptune_run["config/models"] = neptune.utils.stringify_unsupported(MODELS)

# -------------------------------
# 1. List of Simple Objects with Attributes
# -------------------------------
OBJECTS: List[str] = [
    "apples",
    "oranges",
    "bowling ball",
    "basket ball",
    "foot ball",
    "soccer ball",
    "bicycle",
    "motorcycle",
    "scooter",
    "skateboard",
    "car",
    "truck",
    "bus",
    "train",
    "man",
    "woman",
    "kid",
    "dog",
    "cat",
    "bird",
    "fish",
    "tree",
    "bush",
    "flower",
    "grass",
    "rock",
    "mountain",
    "hill",
    "valley",
    "river",
    "lake",
    "ocean",
    "sea",
]

# -------------------------------
# 2. List of Spatial Test Tasks
# -------------------------------
# Each task now uses a concise definition and an example.
SPATIAL_TASKS: List[SpatialTaskType] = [
    # {
    #     "task_type": "physics and causality",
    #     "examples": [
    #         "Two red vibrant apples of the same size falling from different heights surrounded by a soft bokeh of autumn leaves, warm hues of orange and yellow, with the late afternoon sun casting long shadows.",
    #         "Photograph of a A laptop standing at the edge of a desk ready to fall",
    #         "Three mugs are stacked on top of each other leaning to the left. One more mug would cause the stack to fall",
    #         "Two playful dogs, a heartbeat before collision, in a photograph capturing the joyous energy of a candid moment; dynamic, high-resolution, impressionistic style reminiscent of Ed Ruschas street photography.",
    #         "A bowling ball and a basket ball rolling down a steep hill, bright blue natural light, 35mm photography, wide shot, highly detailed, 8K, art by artgerm and greg rutkowski and alphonse mucha",
    #     ],
    # },
    {
        "task_type": "perspective taking",
        "examples": [
            "A photograph of a family of four, two adults and two children, standing in a row, with the camera positioned at the height of the children, looking up at the adults, who are smiling down at them.",
            "Street artist, vividly painting a vibrant mural, surrounded by captivated pedestrians, in a stencil-like graffiti style, with a gritty urban setting, drenched in chiaroscuro lighting for a dramatic and lively atmosphere.",
            "Vibrant street vendors, laden with an array of ripe fruits, amidst the lively hustle of a farmers market - captured in the style of a vivid, Impressionist oil painting, with warm sunlight filtering through a cloud-speckled sky.",
            "Scuba diver capturing a vibrant, up-close moment with a majestic sea turtle among intricately detailed, luminous coral reef, in the style of a high-definition underwater photograph blending vivid hues and soft shadows, with a serene, lively atmosphere.",
            "Toddler and playful puppy in sunlit backyard, chasing iridescent bubbles in whimsical Impressionist style, vibrant colors, tender atmosphere, capturing the joy of childhood and canine companionship.",
        ],
    },
]

# -------------------------------
# 3. Enhanced System Prompt
# -------------------------------
# This prompt instructs the LLM to generate a test prompt for spatial reasoning.
SYSTEM_PROMPT: str = (
    "You are helpful AI assistant tasked to help people with their requests."
)

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
    examples = task["examples"]
    user_prompt = (
        f"Here are example prompts for a text-to-image generation model: {examples}.\n"
        f"Please Generate a prompt based on the examples above using the following objects: {OBJECTS}."
        "Answer in only the prompt."
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
            return_full_text=False,
            do_sample=True,
            top_k=10,
        )
        eos_token_id = llm_pipe.tokenizer.eos_token_id
        logger.info("Pipeline initialized for %s", sanitized_model)

        for task in SPATIAL_TASKS:
            # Generate a number of prompts for each task.
            for i in range(PROMPTS_PER_TASK):
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
                outputs: Any = llm_pipe(
                    prompt_input,
                    max_new_tokens=2048,
                    pad_token_id=eos_token_id,
                )
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
