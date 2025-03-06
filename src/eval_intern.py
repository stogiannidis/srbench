import argparse
import os
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

# ---------------------------
# Preprocessing functions
# ---------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    """Creates the default transform for images."""
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Determines the best matching target aspect ratio."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Resizes the image, splits it into tiles, and optionally adds a thumbnail."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = {(i, j) for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1)
                     for j in range(1, n + 1)
                     if i * j <= max_num and i * j >= min_num}
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_input, input_size=448, max_num=12):
    """
    Loads an image from a file path, dictionary, or PIL Image and returns a tensor with pixel values.
    """
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, dict) and "path" in image_input:
        image = Image.open(image_input["path"]).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise ValueError("Unsupported image input type")

    transform = build_transform(input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


def collate_fn(batch):
    """Collates a batch of samples for batched inference."""
    all_pixel_values = []
    questions = []
    num_patches_list = []
    for sample in batch:
        question = sample["question"]
        image_input = sample["image"]
        pv = load_image(image_input, max_num=12)
        all_pixel_values.append(pv)
        num_patches_list.append(pv.size(0))
        questions.append(question)
    all_pixel_values = torch.cat(all_pixel_values, dim=0)
    return {
        "pixel_values": all_pixel_values,
        "questions": questions,
        "num_patches_list": num_patches_list,
    }


@torch.inference_mode()
def run_inference(dataloader, model, tokenizer, generation_config, dataset, output_dir, m_name):
    """Performs batched inference and writes results to CSV."""
    results = []
    for batch in tqdm(dataloader, desc="Running inference", unit="batch", colour="magenta"):
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
        for question, response in zip(questions, responses):
            results.append({"question": question, "response": response})
    results_df = pd.DataFrame(results)
    results_df["answer"] = dataset["answer"]
    results_df["split"] = dataset["split"]
    results_df.to_csv(os.path.join(output_dir, f"{m_name}.csv"), index=False)


def initialize_components(model_name, dataset_name, batch_size=32):
    """
    Initializes the model, tokenizer, generation config, dataset, and dataloader.
    Returns: model, tokenizer, generation_config, dataset, dataloader.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
        device_map="auto",
    ).eval()

    generation_config = dict(max_new_tokens=64, do_sample=False, pad_token_id=tokenizer.pad_token_id)

    dataset = load_dataset(dataset_name, split="train")
    
    d_name = dataset_name.split("/")[-1]
    m_name = model_name.split("/")[-1]
    
    output_dir = f"output/evaluations/{d_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    return model, tokenizer, generation_config, dataset, dataloader, output_dir, m_name


def main():
    parser = argparse.ArgumentParser(description="Run batched inference on a dataset with a given model.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name or path of the model (e.g., OpenGVLab/InternVL2_5-1B-MPO)")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name or id of the dataset (e.g., stogian/mrt_pf_mix)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    args = parser.parse_args()

    model, tokenizer, generation_config, dataset, dataloader, output_dir, m_name = initialize_components(
        args.model_name, args.dataset_name, args.batch_size
    )
    
    run_inference(dataloader, model, tokenizer, generation_config, dataset, output_dir, m_name)


if __name__ == "__main__":
    main()
