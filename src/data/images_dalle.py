"""
Generate images using DALL-E 3.

Note: DALL-E 3 requires version 1.0.0 of the openai-python library or later.
"""

import os
import argparse
import json
import requests
from tqdm import tqdm
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

OUTPUT_DIR = "output/images/dalle3"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_arguments():
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(
		description="Generate images using DALL-E 3."
	)
	parser.add_argument(
		"-f",
		"--metadata_file",
		type=str,
		required=True,
		help="Path to the JSON file containing the metadata for image generation.",
	)
	return parser.parse_args()


def load_metadata(metadata_file: str):
	"""Load metadata from a JSON Lines file."""
	with open(metadata_file, "r") as file:
		lines = file.readlines()
	return [json.loads(line.strip()) for line in lines if line.strip()]


def main():
    """Main function for generating images."""
    args = parse_arguments()
    metadata = load_metadata(args.metadata_file)

    client = AzureOpenAI(
        api_version="2024-02-01",
        azure_endpoint=os.environ["DALLE_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
    )

    for idx, item in enumerate(
        tqdm(metadata, desc="Generating images", unit="prompt", colour="#000080")
    ):
        prompt = item["generated_scene_description"]
        print(f"Generating image for prompt: {prompt}")
        output_filename = f"image_{idx}.png"
        result = client.images.generate(
            model="dall-e-3",  # the name of your DALL-E 3 deployment
            prompt="A photo-realistic image of " + prompt,
            n=1,
        )
        image_data = json.loads(result.model_dump_json())["data"][0]
        image_url = image_data["url"]

        # Placeholder for image download functionality.
        print(f"Generated image '{output_filename}' from URL: {image_url}")

        # Save the image to the output directory
        image_path = os.path.join(OUTPUT_DIR, output_filename)
        with open(image_path, "wb") as file:
            file.write(requests.get(image_url).content)
        print(f"Saved image to '{image_path}'")
		


if __name__ == "__main__":
	main()






