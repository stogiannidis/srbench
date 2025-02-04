#!/usr/bin/env python
"""
This script demonstrates how to:
1. Define a list of objects with attributes and a set of spatial tasks.
2. For each spatial task and each model, generate 200 scene descriptions using an LLM.
3. Post-process the model's output to remove internal commentary.
4. Log each generated prompt and metadata to both WandB and Neptune.ai.
5. Save each generated scene description and metadata to a JSON file.
"""

import re
import os
import random
import json
import logging
import glob
from typing import List, Dict, Tuple, Any

import torch
from transformers import pipeline
from dotenv import load_dotenv
import wandb                   # Import WandB
import neptune  # Import Neptune.ai

# Load environment variables from the .env file.
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(filename='debug.log', encoding='utf-8', level=logging.DEBUG)

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
OUTPUT_FILENAME = f"{SAVE_DIR}small_llm_prompts.json"  # All prompts will be appended here.

# -------------------------------
# Initialize WandB
# -------------------------------
wandb.init(
    project="SpatialScenePrompts",
    name="small-llm-prompts-trial",
    config={
        "num_prompts_per_task": 200,
        "models": [
            "bin/models/llms/google_gemma-2-2b-it",
            "bin/models/llms/deepseek-ai_DeepSeek-R1-Distill-Qwen-7B",
            "bin/models/llms/deepseek-ai_DeepSeek-R1-Distill-Llama-8B",
            "bin/models/llms/meta-llama_Llama-3.2-1B-Instruct"
            
        ]
    }
)

# -------------------------------
# Initialize Neptune.ai Run
# -------------------------------
neptune_run = neptune.init_run(
    project="stogiannidis/create-prompts",  # Replace with your project name.
    api_token=os.getenv("NEPTUNE_API_TOKEN")           # Replace with your Neptune API token.
)
neptune_run["config/num_prompts_per_task"] = 200
neptune_run["config/models"] = [
    "bin/models/llms/google_gemma-2-2b-it",
    "bin/models/llms/deepseek-ai_DeepSeek-R1-Distill-Qwen-7B",
    "bin/models/llms/deepseek-ai_DeepSeek-R1-Distill-Llama-8B",
    "bin/models/llms/meta-llama_Llama-3.2-1B-Instruct"
]

# -------------------------------
# 1. List of Objects with Attributes
# -------------------------------
OBJECTS: List[ObjectType] = [
    {"name": "cube", "attributes": ["red", "blue", "green", "large", "small", "transparent", "glossy"]},
    {"name": "sphere", "attributes": ["blue", "metallic", "glass", "tiny", "large", "iridescent", "matte"]},
    {"name": "cylinder", "attributes": ["red", "tall", "short", "wooden", "plastic", "smooth", "textured"]},
    {"name": "cone", "attributes": ["yellow", "large", "small", "textured", "shiny", "matte", "glossy"]},
    {"name": "pyramid", "attributes": ["white", "black", "metallic", "small", "large", "stone", "cracked"]},
    {"name": "chair", "attributes": ["wooden", "blue", "modern", "folding", "metallic", "comfortable", "vintage"]},
    {"name": "table", "attributes": ["round", "square", "wooden", "glass", "white", "polished", "rustic"]},
    {"name": "book", "attributes": ["open", "closed", "thick", "thin", "red", "ancient", "modern"]},
    {"name": "mug", "attributes": ["ceramic", "white", "blue", "large", "small", "vintage", "elegant"]},
    {"name": "phone", "attributes": ["black", "modern", "large-screen", "small", "silver", "sleek", "matte"]},
    {"name": "lamp", "attributes": ["bright", "dim", "vintage", "modern", "gold", "silver", "wooden"]},
    {"name": "bottle", "attributes": ["glass", "plastic", "colored", "transparent", "old", "modern"]},
    {"name": "vase", "attributes": ["ceramic", "ornate", "minimalist", "blue", "white", "red"]},
    {"name": "car", "attributes": ["red", "sleek", "vintage", "modern", "sporty", "luxurious"]},
    {"name": "tree", "attributes": ["tall", "short", "leafy", "bare", "lush", "ancient", "modern"]},
    {"name": "bicycle", "attributes": ["rustic", "modern", "red", "blue", "vintage", "sleek"]},
    {"name": "clock", "attributes": ["antique", "modern", "round", "square", "gold", "silver"]},
]

# -------------------------------
# 2. List of Spatial Tasks
# -------------------------------
SPATIAL_TASKS: List[SpatialTaskType] = [
    {
        "task_type": "mental_rotation",
        "instruction_template": (
            "Place a {object1} rotated by 45 degrees around its vertical axis, "
            "resting on top of a {object2}. Emphasize the rotated orientation clearly."
        ),
        "description": "Assessing mental rotation by a specific degree.",
    },
    {
        "task_type": "compositionality",
        "instruction_template": (
            "Merge a {object1} and a {object2} into a single composite object. "
            "Let the {object2} act as a base for the {object1}, forming a new unified entity."
        ),
        "description": "Evaluating compositional relationships between objects.",
    },
    {
        "task_type": "physics_causality",
        "instruction_template": (
            "Depict a scenario where a {object1} is precariously balanced on the edge of a {object2}, "
            "suggesting an imminent shift due to gravitational pull."
        ),
        "description": "Testing physical causality and tension between objects.",
    },
    {
        "task_type": "perspective_taking",
        "instruction_template": (
            "Illustrate the scene from an elevated viewpoint: a {object1} in the foreground, "
            "a {object2} directly behind it, and a {object3} off to the side, emphasizing depth and perspective."
        ),
        "description": "Evaluating perspective shifts and depth perception.",
    },
]

# -------------------------------
# 3. Enhanced System Prompt
# -------------------------------
SYSTEM_PROMPT: str = """\
You are an advanced assistant trained to generate detailed and vivid textual descriptions 
for spatial scenes that can later be rendered by image generation models.

Requirements:
- Explicitly state spatial relationships: use terms like "above", "below", "next to", 
  "in front of", "behind", "to the left of", "to the right of", and "centered on".
- Clearly describe any transformations (e.g., "rotated by 45 degrees", "scaled down").
- Incorporate details regarding lighting, perspective, and background if applicable.
- Keep the description concise yet rich in spatial details, ensuring clarity and unambiguity.
- Provide only the scene description in your response without extra commentary.

RESPOND ONLY WITH THE SCENE DESCRIPTION. 
DO NOT provide additional commentary or disclaimers.
"""

# -------------------------------
# 4. Function to Generate a Scene Instruction for a Given Task
# -------------------------------
def create_instruction_for_task(task: SpatialTaskType) -> Tuple[str, str]:
    """
    Fills in the placeholders of the given task's instruction template with randomly chosen object attributes.
    
    Args:
        task: A spatial task containing the instruction template.
    
    Returns:
        A tuple with the task type and the filled instruction.
    """
    placeholders: List[str] = ["{object1}", "{object2}", "{object3}"]
    needed_placeholders: List[str] = [ph for ph in placeholders if ph in task["instruction_template"]]
    selected_objects: List[ObjectType] = random.sample(OBJECTS, len(needed_placeholders))
    
    instruction: str = task["instruction_template"]
    for i, ph in enumerate(needed_placeholders, start=1):
        obj: ObjectType = selected_objects[i - 1]
        chosen_attribute: str = random.choice(obj["attributes"])
        placeholder_value: str = f"{chosen_attribute} {obj['name']}"
        instruction = instruction.replace(ph, placeholder_value)
    return task["task_type"], instruction

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
    A simple heuristic to decide if the model supports chat-style prompts with a system message.
    In this example, if the model id contains "gemma" (case-insensitive) we assume it does not.
    """
    return "gemma" not in model_name.lower()

def prepare_pipeline_input(messages: List[MessageType], model_name: str) -> Any:
    """
    Prepares the input for the pipeline call based on whether the model supports system prompts.
    
    If the model supports system prompts, we pass the messages list as-is.
    Otherwise, we combine all message contents (including system instructions) into a single prompt.
    """
    if supports_system_prompt(model_name):
        # Model supports chat-style messages.
        return messages
    else:
        # For models that do NOT support system prompts, merge all messages.
        combined_prompt = "\n\n".join(msg["content"].strip() for msg in messages)
        return [{"role": "user", "content": combined_prompt.strip()}]

def extract_generated_text(outputs: Any) -> str:
    """
    Extracts the generated text from the output of the pipeline.
    
    Supports both the structure from the gemma-like pipelines (nested "content")
    and instruct-based pipelines (flat string).
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
# MODELS: List[str] = glob.glob("bin/models/llms/*")
MODELS: List[str] = [
    "bin/models/llms/google_gemma-2-2b-it",
    "bin/models/llms/deepseek-ai_DeepSeek-R1-Distill-Qwen-7B",
    "bin/models/llms/deepseek-ai_DeepSeek-R1-Distill-Llama-8B",
    "bin/models/llms/meta-llama_Llama-3.2-1B-Instruct"
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
            # attn_implementation="flash_attention_2"
        )
        logger.info("Pipeline initialized for %s", sanitized_model)
        
        for task in SPATIAL_TASKS:
            # Generate 200 prompts for each task.
            for i in range(200):
                logger.info("Generating prompt %d for task: %s", i+1, task["task_type"])
                task_type, instruction = create_instruction_for_task(task)
                
                # Construct the user prompt.
                user_prompt: str = (
                    f"Task Type: {task_type}\n"
                    f"Instruction: {instruction}\n\n"
                    "Generate a spatial scene description as per the instruction above."
                )
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
                    "model": model_name,
                    "task_type": task_type,
                    "instruction": instruction,
                    "system_prompt": SYSTEM_PROMPT,
                    "user_prompt": user_prompt,
                    "generated_scene_description": cleaned_text,
                    "iteration": i + 1
                }
                
                # Log the metadata to WandB.
                wandb.log(output_data)
                
                # Log the metadata to Neptune.
                neptune_run[f"prompt/{sanitized_model}/{task_type}/iteration_{i+1}"] = output_data
                
                # Save the generated description and metadata to a JSON file.
                with open(OUTPUT_FILENAME, "a", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=4)
                    f.write("\n")
                
                print(f"Scene description and metadata saved to '{OUTPUT_FILENAME}'.\n")
    
    # Finish logging for both platforms.
    wandb.finish()
    neptune_run.stop()

if __name__ == "__main__":
    main()
