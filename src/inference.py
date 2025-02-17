import re
import pandas as pd
from tqdm import tqdm
from qwen_vl_utils import process_vision_info  # used for handling visual inputs
from utils import VLMWrapper as VLM
from datasets import load_dataset
import torch

# Instantiate your vlm with the exact model_id you wish to test.
# (For instance, here we use Mllama.)
model_id = "HuggingFaceM4/Idefics3-8B-Llama3"
vlm = VLM(model_id, "auto")

data = load_dataset("stogian/srbench_v3", split="train")
#sample 100 images of the mrt and pf splits
data = data.filter(lambda x: x["split"] == "mrt" or x["split"] == "pf").select(range(100))

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
    
    # Prepare text and visual inputs using the vlm's processor.
    texts = [
        vlm.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        for message in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = vlm.processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(vlm.device)

    # Run generation using the callable vlm interface.
    generated_ids = vlm.model.generate(**inputs, max_new_tokens=128)

    # Trim the input tokens from the generated output.
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_texts = vlm.processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # Process the outputs
    for idx, raw_prediction in enumerate(output_texts):
        raw_prediction = raw_prediction.strip()
        ground_truth = ground_truths[idx]  # Get corresponding ground truth

    
        results.append(
            {
                "prediction": raw_prediction,
                "ground_truth": ground_truth,
            }
        )

# print("Detailed Results:")
# for idx, res in enumerate(results):
#     print(
#         f"Sample {idx + 1}: Predicted: {res['prediction']} | Ground Truth: {res['ground_truth']} | Correct: {res['correct']}"
#     )

# Save results to a CSV file.
results_df = pd.DataFrame(results)
results_df.to_csv("results_llama_full.csv", index=False)
