from datasets import Dataset, Image, Features, Value
import json
from PIL import Image as PILImage
import os
import glob


def generate_examples():
    for data in annotations:
        image_path = os.path.join("Spatial-MM/data/spatial_mm", data["image_name"])
        try:
            image = PILImage.open(image_path).convert("RGB")  # Load image as RGB
        except Exception as e:
            print("Error loading image: ", image_path)
            continue
        yield {
            "image": image,
            "question": data["question"],
            "answer": data["answer"],
        }

if __name__ == "__main__":
	# Load JSON annotations from multiple files
	annotations = []
	json_files = glob.glob("Spatial-MM/data/*.json")
	for json_file in json_files:
		with open(json_file, "r") as f:
			annotations.extend(json.load(f))

	# Define dataset features (adjust based on your JSON structure)
	features = Features(
		{
			"image": Image(),
			"question": Value("string"),
			"answer": Value("string"),
		}
	)


	# Create the dataset
	dataset = Dataset.from_generator(
		generate_examples,
		features=features,
	)

	dataset.push_to_hub("spatial_mm", private= True)  # Push to the Hub

