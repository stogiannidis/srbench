import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description="Evaluate MiniCPM model on a dataset")
parser.add_argument("-d", "--dataset", type=str, help="Name of the Hugging Face dataset")
args = parser.parse_args()

# --- 1. Load the Hugging Face dataset ---
print(f"Loading dataset: {args.dataset}")
hf_dataset = load_dataset(args.dataset, split="train")


class HFImageQADataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # If the "image" field is a file path, open it; otherwise assume it's a PIL image.
        if isinstance(item["image"], str):
            image = Image.open(item["image"]).convert("RGB")
        else:
            image = item["image"]
        question = item["question"]
        # Optionally include a ground truth answer if available.
        answer = item.get("answer", None)
        return {"id": idx, "image": image, "question": question, "answer": answer}


# Custom collate function that integrates the image into the message content.
def custom_collate_fn(batch):
    msgs_batch = []
    for sample in batch:
        # Convert the PIL image to a NumPy array (shape: H x W x C)
        np_img = np.array(sample["image"])
        # Transpose to channel-first format (C x H x W) so that the image processor can normalize it correctly.
        np_img = np_img.transpose(2, 0, 1)
        # Create a conversation history where the image and question are integrated in the same message.
        msgs = [{"role": "user", "content": [np_img, sample["question"]]}]
        msgs_batch.append(msgs)
    return msgs_batch, batch


dataset_obj = HFImageQADataset(hf_dataset)
data_loader = DataLoader(dataset_obj, batch_size=16, collate_fn=custom_collate_fn)

# --- 2. Load the MiniCPM model and tokenizer ---
model = AutoModel.from_pretrained(
    "openbmb/MiniCPM-V-2_6",
    trust_remote_code=True,
    attn_implementation="flash_attention_2",  # or flash_attention_2 if desired
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    "openbmb/MiniCPM-V-2_6", trust_remote_code=True, use_fast=True
)

# --- 3. Batched inference and saving results ---
results = []

for msgs_batch, original_batch in tqdm(data_loader, desc="Running inference", unit="batch", colour="blue"):
    try:
        # Note: When doing batch inference, the images must be integrated into msgs.
        responses = model.chat(image=None, msgs=msgs_batch, tokenizer=tokenizer, max_tokens=128, do_sample=False)
    except Exception as e:
        # Fallback: iterate over each sample in the batch.
        responses = []
        for msgs in msgs_batch:
            response = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)
            responses.append(response)

    # Merge each generated answer with its corresponding original sample.
    for sample, response in zip(original_batch, responses):
        result_entry = {
            "id": sample["id"],
            "question": sample["question"],
            "response": response,
        }
        if sample.get("answer") is not None:
            result_entry["answer"] = sample["answer"]
        results.append(result_entry)

# Save the combined results to a CSV file.
df = pd.DataFrame(results)
short_dataset_name = args.dataset.split("/")[-1]
df.to_csv(f"output/evaluations/{short_dataset_name}/MiniCPM-V-2_6.csv", index=False)
print(f"Saved results to output/evaluations/{short_dataset_name}/MiniCPM-V-2_6.csv")
# --- 4. Done ---

