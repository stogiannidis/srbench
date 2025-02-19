import re
import pandas as pd
import logging
from tqdm import tqdm
from qwen_vl_utils import process_vision_info  # used for handling visual inputs
from utils import VLMWrapper as VLM
from datasets import load_dataset
import torch

# Configure logger
logging.basicConfig(filename="eval.log",level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Instantiate your vlm with the exact model_id you wish to test.
Qwen="Qwen/Qwen2.5-VL-3B-Instruct" #? Works  or "Qwen/Qwen2.5-VL-7B-Instruct"
Llava= "llava-hf/llava-1.5-7b-hf" #? Works
LlavaNext= "llava-hf/llava-v1.6-mistral-7b-hf" #? Works
InstructBlip= "Salesforce/instructblip-vicuna-7b" #? Works
Molmo= "allenai/Molmo-7B-D-0924" #! Does not work
InternVL= "OpenGVLab/InternVL2_5-1B-MPO" #? Works
Idefics= "HuggingFaceM4/Idefics3-8B-Llama3" #? Works
SmolVLM = "HuggingFaceTB/SmolVLM-Instruct" #? Works
Mllama = "meta-llama/Llama-3.2-11B-Vision-Instruct" #? Works    meta-llama/Llama-3.2-90B-Vision-Instruct"
Phi35= "microsoft/Phi-3.5-vision-instruct" #! Does not work
MiniCPM= "openbmb/MiniCPM-V-2_6" #! Does not work

model_id = InstructBlip

short_name = model_id.split("/")[-1]
logger.info(f"\n\nStarting evaluation script for model: {model_id}")

vlm = VLM(model_id, "auto")

data = load_dataset("stogian/mrt_fp", split="train")
vqa = load_dataset("MilaWang/SpatialEval", "vqa", split="test")

# sample 100 images of the mrt and pf splits
# data = data.filter(lambda x: x["scplit"] == "mrt" or x["split"] == "pf").select(range(64))

batch_size = 16  # Define your batch size

total = 0
correct = 0

results = []

# Loop through the dataset in batches.
for i in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
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
    
    # image_inputs, _ = process_vision_info(messages)
    
    inputs = vlm.preprocess(conversation=messages, image_input=images)
    logger.info("Generated output: %s", inputs)
    

    # # Run generation using the callable vlm interface.
    generated_ids = vlm(**inputs, max_new_tokens=128)
    logger.info("Generated output: %s", generated_ids)

    # Decode the generated ids    
    output_texts = vlm.decode(generated_ids)



    # Process the outputs
    for idx, raw_prediction in enumerate(output_texts):
        pred = raw_prediction.strip()
        ground_truth = ground_truths[idx]  # Get corresponding ground truth
        results.append(
            {
                "prediction": pred,
                "ground_truth": ground_truth,
            }
        )


# Save results to a CSV file.
results_df = pd.DataFrame(results)
results_df.to_csv(f"results_{short_name}_full.csv", index=False)
