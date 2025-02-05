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

import torch
import pandas as pd
from transformers import pipeline
from dotenv import load_dotenv
import wandb 
import neptune

# Load environment variables from the .env file.
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(filename="debug.log", encoding="utf-8", level=logging.DEBUG)

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
OUTPUT_FILENAME = (
    f"{SAVE_DIR}small_llm_prompts_v3.jsonl"  # All prompts will be appended here.
)
TRIAL_NAME = "small-llm-prompts-trial_v3"
# -------------------------------
# Initialize WandB
# -------------------------------
wandb.init(
    project="SpatialScenePrompts",
    name=TRIAL_NAME,
    config={
        "num_prompts_per_task": 10,
        "models": [
            "bin/models/llms/google_gemma-2-2b-it",
            "bin/models/llms/deepseek-ai_DeepSeek-R1-Distill-Qwen-7B",
            "bin/models/llms/deepseek-ai_DeepSeek-R1-Distill-Llama-8B",
            "bin/models/llms/meta-llama_Llama-3.2-1B-Instruct",
        ],
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
neptune_run["config/models"] = [
    "bin/models/llms/google_gemma-2-2b-it",
    "bin/models/llms/deepseek-ai_DeepSeek-R1-Distill-Qwen-7B",
    "bin/models/llms/deepseek-ai_DeepSeek-R1-Distill-Llama-8B",
    "bin/models/llms/meta-llama_Llama-3.2-1B-Instruct",
]

# -------------------------------
# 1. List of Simple Objects with Attributes
# -------------------------------
OBJECTS: List[ObjectType] = [
    {
        "name": "cube",
        "attributes": [
            "red",
            "blue",
            "green",
            "large",
            "small",
            "transparent",
            "glossy",
        ],
    },
    {
        "name": "sphere",
        "attributes": [
            "blue",
            "metallic",
            "glass",
            "tiny",
            "large",
            "iridescent",
            "matte",
        ],
    },
    {
        "name": "cylinder",
        "attributes": [
            "red",
            "tall",
            "short",
            "wooden",
            "plastic",
            "smooth",
            "textured",
        ],
    },
    {
        "name": "cone",
        "attributes": [
            "yellow",
            "large",
            "small",
            "textured",
            "shiny",
            "matte",
            "glossy",
        ],
    },
    {
        "name": "mug",
        "attributes": ["ceramic", "white", "blue", "vintage", "modern", "elegant"],
    },
    {
        "name": "book",
        "attributes": ["open", "closed", "thick", "red", "ancient", "modern"],
    },
]

# -------------------------------
# 2. List of Spatial Test Tasks
# -------------------------------
# Each task now uses a concise definition and an example.
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
# This prompt instructs the LLM to generate a test prompt for spatial reasoning.
SYSTEM_PROMPT: str = """\
You are an advanced assistant tasked with generating test prompts for spatial reasoning.
Using the provided task details (definition and example) and, optionally, a description of simple objects,
generate a concise image description prompt that can be used to test spatial reasoning.
Respond with ONLY the final prompt text.
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
            attr = random.choice(obj["attributes"])
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
# 7. List of Models to Use
# -------------------------------
MODELS: List[str] = [
    "bin/models/llms/google_gemma-2-2b-it",
    "bin/models/llms/deepseek-ai_DeepSeek-R1-Distill-Qwen-7B",
    "bin/models/llms/deepseek-ai_DeepSeek-R1-Distill-Llama-8B",
    "bin/models/llms/meta-llama_Llama-3.2-1B-Instruct",
]


# -------------------------------
# 8. Main Routine: Iterate Over All Model and Task Combinations
# -------------------------------
def main() -> None:
    logger.info("Starting main routine with WandB and Neptune.ai logging.")

    for model_name in MODELS:
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
            for i in range(5):  # Adjust number per task as needed.
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
    wandb.log({"toc": wandb.Table(data=results_df)})
    neptune_run["output"].upload(f"{SAVE_DIR}toc.csv")

    # Finish the run for both platforms.
    wandb.finish()
    neptune_run.stop()


if __name__ == "__main__":
    main()
