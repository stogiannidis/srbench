import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import pandas as pd

# ---------------------------
# Preprocessing functions
# ---------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Define target aspect ratios based on tile counts
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Choose the closest matching aspect ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate target dimensions and number of blocks (tiles)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize and then split the image into tiles
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_input, input_size=448, max_num=12):
    """
    Modified load_image accepts:
    - a file path (str)
    - a dictionary (with "path") as returned by HF datasets Image feature
    - or a PIL Image directly
    """
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, dict) and "path" in image_input:
        image = Image.open(image_input["path"]).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise ValueError("Unsupported image input type")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# ---------------------------
# Model & Tokenizer Setup
# ---------------------------
path = "OpenGVLab/InternVL2_5-26B-MPO"
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
model = (
    AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map= "auto",
    ).eval()##.cuda()
)
generation_config = dict(max_new_tokens=1024, do_sample=False, pad_token_id=tokenizer.pad_token_id)

# ---------------------------
# Load Hugging Face Dataset
# ---------------------------
# Replace "your_dataset_name" with the actual dataset ID.
# It is assumed that each sample in the dataset has fields "question" and "image".
dataset = load_dataset("stogian/sr_test", split="train")


def collate_fn(batch):
    all_pixel_values = []
    questions = []
    num_patches_list = []
    for sample in batch:
        # Extract question and image from the dataset sample.
        question = sample["question"]
        image_input = sample["image"]
        pv = load_image(image_input, max_num=12)
        all_pixel_values.append(pv)
        num_patches_list.append(pv.size(0))
        questions.append(question)
    # Concatenate pixel values across samples.
    # Each pv has shape (num_patches, C, H, W); concatenation yields a tensor of shape (total_patches, C, H, W)
    all_pixel_values = torch.cat(all_pixel_values, dim=0)
    return {
        "pixel_values": all_pixel_values,
        "questions": questions,
        "num_patches_list": num_patches_list,
    }


# Create a DataLoader for batched inference.
dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)

# ---------------------------
# Batched Inference Loop
# ---------------------------
@torch.inference_mode()
def run_inference():
    results = []
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(torch.bfloat16).cuda()
        questions = batch["questions"]
        num_patches_list = batch["num_patches_list"]
        responses = model.batch_chat(
            tokenizer,
            pixel_values,
            num_patches_list=num_patches_list,
            questions=questions,
            generation_config=generation_config,
        )
        # Append the question and generated response for each sample in the batch.
        for question, response in zip(questions, responses):
            results.append({"question": question, "answer": response})
    # Convert results to a pandas DataFrame and print the table.
    results_df = pd.DataFrame(results)
    results_df["gt"] = dataset["answer"]
    results_df.to_csv("results_intern_8B.csv", index=False)


if __name__ == "__main__":
    run_inference()
