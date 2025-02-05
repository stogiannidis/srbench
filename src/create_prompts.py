#!/usr/bin/env python
"""
This script demonstrates how to:
1. Define a list of simple objects with attributes and a set of spatial test tasks.
2. For each spatial task and each model, generate multiple scene descriptions using an LLM.
3. Post-process the model's output to remove internal commentary.
4. Log each generated prompt and metadata to both WandB and Neptune.ai.
5. Save each generated scene description and metadata to a JSONL file.

Additional command-line arguments allow overriding default values.
"""

import re
import os
import random
import json
import logging
import glob
import argparse
from typing import List, Dict, Tuple, Any

import torch
import pandas as pd
from transformers import pipeline
from dotenv import load_dotenv
import wandb
import neptune

# Load environment variables from the .env file.
load_dotenv()

# Configure logging.
logger = logging.getLogger(__name__)
logging.basicConfig(filename="debug.log", encoding="utf-8", level=logging.DEBUG)

# -------------------------------
# Type Aliases
# -------------------------------
ObjectType = Dict[str, Any]
SpatialTaskType = Dict[str, str]
MessageType = Dict[str, str]

# -------------------------------
# Default Constants
# -------------------------------
DEFAULT_SAVE_DIR = "output/prompts/"
DEFAULT_TRIAL_NAME = "small-llm-prompts-trial_v4"
DEFAULT_NUM_PROMPTS = 5
DEFAULT_MODELS = [
    "bin/models/llms/gemma-2-2b-it",
    "bin/models/llms/DeepSeek-R1-Distill-Qwen-7B",
    "bin/models/llms/DeepSeek-R1-Distill-Llama-8B",
    "bin/models/llms/Llama-3.2-1B-Instruct",
]


# The output filename is constructed from the save directory and trial name.
# (It will be overwritten if the command-line arguments override the defaults.)
# -------------------------------
# Argument Parser Function
# -------------------------------
def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments to override default settings.

    Returns:
        argparse.Namespace: Parsed command-line arguments containing:
            - save_dir: Directory to save the generated prompts.
            - trial_name: The name for this trial.
            - num_prompts: Number of prompts to generate per task.
            - models: Comma-separated list of model names/IDs.
    """
    parser = argparse.ArgumentParser(
        description="Generate spatial scene prompts using an LLM and log them to WandB and Neptune.ai."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=DEFAULT_SAVE_DIR,
        help="Directory to save generated prompts (default: %(default)s).",
    )
    parser.add_argument(
        "--trial_name",
        type=str,
        default=DEFAULT_TRIAL_NAME,
        help="Name for the trial (default: %(default)s).",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=DEFAULT_NUM_PROMPTS,
        help="Number of prompts to generate per task (default: %(default)s).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated list of model names/IDs (default: %(default)s).",
    )
    return parser.parse_args()


# -------------------------------
# 1. List of Simple Objects with Attributes
# -------------------------------
OBJECTS: List[ObjectType] = [
    {
        "name": "cube",
        "attributes": "red, blue, green, large, small, transparent, glossy",
    },
    {
        "name": "sphere",
        "attributes": "blue, metallic, glass, tiny, large, iridescent, matte",
    },
    {
        "name": "cylinder",
        "attributes": "red, tall, short, wooden, plastic, smooth, textured",
    },
    {
        "name": "cone",
        "attributes": "yellow, large, small, textured, shiny, matte, glossy",
    },
    {"name": "mug", "attributes": "ceramic, white, blue, vintage, modern, elegant"},
    {"name": "book", "attributes": "open, closed, thick, red, ancient, modern"},
    {
        "name": "chair",
        "attributes": "wooden, metal, upholstered, swivel, rocking, folding",
    },
    {"name": "lamp", "attributes": "desk, floor, table, modern, antique, bright"},
    {"name": "sofa", "attributes": "leather, fabric, sectional, small, large, comfy"},
    {"name": "pen", "attributes": "ballpoint, gel, fine, blue, red, disposable"},
    {"name": "bottle", "attributes": "glass, plastic, water, wine, large, small"},
    {"name": "laptop", "attributes": "black, silver, thin, heavy, new, used"},
    {
        "name": "person",
        "attributes": "adult, child, male, female, tall, short, friendly, quiet",
    },
    {
        "name": "shoe",
        "attributes": "sneaker, boot, sandal, leather, canvas, high-heeled",
    },
    {"name": "watch", "attributes": "digital, analog, wrist, pocket, sports, luxury"},
]

# -------------------------------
# 2. List of Spatial Test Tasks
# -------------------------------
SPATIAL_TASKS: List[SpatialTaskType] = [
    {
        "task_type": "mental_rotation",
        "definition": "Test the ability to mentally rotate an object.",
        "example": "Identify the correct rotated version of a cube.",
    },
    {
        "task_type": "physics_causality",
        "definition": "Test understanding of cause and effect in physical scenarios.",
        "example": "Predict where a rolling ball will land after colliding with an obstacle.",
    },
    {
        "task_type": "compositionality",
        "definition": "Test the ability to combine fragmented shapes into a cohesive whole.",
        "example": "Determine which pieces assemble into a complete mug.",
    },
    {
        "task_type": "visualization",
        "definition": "Test the ability to visualize transformations such as folding or unfolding.",
        "example": "Describe the final appearance of a folded paper with cutouts.",
    },
    {
        "task_type": "perspective_taking",
        "definition": "Test the ability to understand different viewpoints of a scene.",
        "example": "Explain what a room looks like from a top-down view versus a frontal view.",
    },
]

# -------------------------------
# 3. Enhanced System Prompt
# -------------------------------
SYSTEM_PROMPT: str = (
    "You are an advanced assistant tasked with generating test prompts for spatial reasoning.\n"
    "Using the provided task details (definition and example) and, optionally, a description of simple objects,\n"
    "generate a concise image description prompt that can be used to test spatial reasoning.\n"
    "Respond with ONLY the final prompt text."
)


# -------------------------------
# 4. Function to Construct a Prompt for a Given Task
# -------------------------------
def construct_prompt_for_task(task: SpatialTaskType) -> Tuple[str, str]:
    """
    Constructs a test prompt using the task's definition and example, and optionally
    includes one or two randomly selected simple objects with attributes.

    Args:
        task (SpatialTaskType): A spatial test task containing "task_type", "definition", and "example".

    Returns:
        Tuple[str, str]: A tuple with the task type and the constructed user prompt.
    """
    task_type = task["task_type"]
    base_details = (
        f"Task Type: {task_type}\n"
        f"Definition: {task['definition']}\n"
        f"Example: {task['example']}\n"
    )
    # Optionally include one or two simple objects.
    include_objects = random.choice([True, False])
    if include_objects:
        num_objs = random.choice([1, 2])
        selected_objs = random.sample(OBJECTS, num_objs)
        obj_descs = []
        for obj in selected_objs:
            # Randomly select one attribute from the comma-separated list.
            attributes = [attr.strip() for attr in obj["attributes"].split(",")]
            attr = random.choice(attributes)
            obj_descs.append(f"{attr} {obj['name']}")
        if num_objs == 2:
            relation = random.choice(
                ["next to", "above", "below", "to the left of", "to the right of"]
            )
            object_details = f"Objects: {obj_descs[0]} {relation} {obj_descs[1]}.\n"
        else:
            object_details = f"Object: {obj_descs[0]}.\n"
    else:
        object_details = ""
    user_prompt = f"{base_details}{object_details}\nGenerate a complete image description prompt based on the above."
    return task_type, user_prompt


# -------------------------------
# 5. Post-Processing Function
# -------------------------------
def post_process_output(output_text: str) -> str:
    """
    Removes any text between <think> and </think> tags and strips whitespace.

    Args:
        output_text (str): The raw output text from the model.

    Returns:
        str: The cleaned output text.
    """
    cleaned_text: str = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL)
    return cleaned_text.strip()


# -------------------------------
# 6. Helper Functions to Adapt to Different Pipeline Variants
# -------------------------------
def supports_system_prompt(model_name: str) -> bool:
    """
    Heuristic: if the model id contains "gemma" (case-insensitive) we assume it does not support chat-style prompts.

    Args:
        model_name (str): The name or identifier of the model.

    Returns:
        bool: True if the model supports system prompts, False otherwise.
    """
    return "gemma" not in model_name.lower()


def prepare_pipeline_input(messages: List[MessageType], model_name: str) -> Any:
    """
    Prepares the input for the pipeline call based on whether the model supports system prompts.

    Args:
        messages (List[MessageType]): A list of messages (each with a role and content).
        model_name (str): The name or identifier of the model.

    Returns:
        Any: The prepared input for the pipeline.
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

    Args:
        outputs (Any): The output from the text-generation pipeline.

    Returns:
        str: The extracted generated text.
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
    """
    Main routine that:
    1. Initializes WandB and Neptune.ai.
    2. Iterates over each model (as provided via command-line arguments).
    3. For each model, iterates over each spatial task and generates a number of prompts.
    4. Post-processes the generated prompt, logs the metadata, and appends it to a JSONL file.
    5. Finally, logs a table of all generated prompts to WandB and Neptune.ai.
    """
    args = parse_arguments()
    # Update save directory and trial name from arguments.
    save_dir = args.save_dir
    trial_name = args.trial_name
    num_prompts = args.num_prompts
    # Parse the models argument into a list.
    models_list = [m.strip() for m in args.models.split(",") if m.strip()]
    output_filename = f"{save_dir}{trial_name}.jsonl"

    logger.info("Starting main routine with WandB and Neptune.ai logging.")

    # Initialize WandB.
    wandb.init(
        project="SpatialScenePrompts",
        name=trial_name,
        config={
            "num_prompts_per_task": num_prompts,
            "models": models_list,
        },
    )

    # Initialize Neptune.ai Run.
    neptune_run = neptune.init_run(
        project="stogiannidis/create-prompts",  # Replace with your project name.
        api_token=os.getenv(
            "NEPTUNE_API_TOKEN"
        ),  # Replace with your Neptune API token.
        name=trial_name,
        tags=["llm", "spatial-reasoning"],
    )
    neptune_run["config/num_prompts_per_task"] = num_prompts
    neptune_run["config/models"] = models_list

    for model_name in models_list:
        logger.info("Processing model: %s", model_name)
        sanitized_model: str = model_name.replace("/", "_")
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
            for i in range(num_prompts):
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
                else:
                    cleaned_text = ""

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
                with open(output_filename, "a", encoding="utf-8") as f:
                    f.write(json.dumps(output_data, ensure_ascii=False) + "\n")

    logger.info("Main routine completed.")

    # Create a Table of Contents for the generated prompts.
    results_df: pd.DataFrame = pd.read_json(output_filename, lines=True)

    # Log the Table of Contents to both platforms.
    wandb.log({"results": wandb.Table(data=results_df)})
    neptune_run["results"].upload(neptune.types.File.as_html(results_df))

    # Finish the run for both platforms.
    wandb.finish()
    neptune_run.stop()


if __name__ == "__main__":
    main()
