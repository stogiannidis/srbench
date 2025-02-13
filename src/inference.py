import re
import pandas as pd
from tqdm import tqdm
import torch
from qwen_vl_utils import process_vision_info  # used for handling visual inputs
from utils import VLMWrapper as VLM
from dataset inmport laod_dataset

# Instantiate your wrapper with the exact model_id you wish to test.
# (For instance, here we use Mllama.)
model_id = "meta-llama/Llama-3.2-90B-Vision-Instruct"
wrapper = VLM(model_id)

# Assume `data` is a dictionary with a "test" key containing samples.
# Each sample should have at least an "image" field and a "correct_candidate" field.
# For example, data["test"] might be a list of dictionaries.

total = 0
correct = 0
results = []

# Loop through each sample in the dataset.
for sample in tqdm(data["test"], desc="Processing samples"):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {
                    "type": "text",
                    "text": "Which of the candidates depicts the correct unfolded version of the paper after being punctured? Provide only the candidate and NOTHING else",
                },
            ],
        }
    ]

    # Prepare text and visual inputs using the wrapper's processor.
    text = wrapper.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = wrapper.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(wrapper.device)

    # Run generation using the callable wrapper interface.
    generated_ids = wrapper(**inputs, max_new_tokens=128)

    # Trim the input tokens from the generated output.
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = wrapper.processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    raw_prediction = output_text[0].strip()
    match = re.search(r"\d+", raw_prediction)
    if match:
        prediction = match.group()
    else:
        prediction = None

    ground_truth = sample["correct_candidate"]
    try:
        is_correct = (
            int(prediction) == ground_truth if prediction is not None else False
        )
    except Exception as e:
        is_correct = False
        print(
            f"Error comparing prediction '{prediction}' with ground truth '{ground_truth}': {e}"
        )

    results.append(
        {
            "prediction": prediction,
            "ground_truth": ground_truth,
            "correct": is_correct,
            "raw_prediction": raw_prediction,
        }
    )

    total += 1
    if is_correct:
        correct += 1

accuracy = 100 * correct / total if total > 0 else 0

print(f"Total Samples: {total}")
print(f"Correct Predictions: {correct}")
print(f"Accuracy: {accuracy:.2f}%\n")

print("Detailed Results:")
for idx, res in enumerate(results):
    print(
        f"Sample {idx + 1}: Predicted: {res['prediction']} | Ground Truth: {res['ground_truth']} | Correct: {res['correct']}"
    )

# Save results to a CSV file.
results_df = pd.DataFrame(results)
results_df["candidate_order"] = data["test"]["candidate_order"]
results_df.to_csv("results.csv", index=False)
