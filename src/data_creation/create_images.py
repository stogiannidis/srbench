import os
import json
import torch
import logging
from diffusers import FluxPipeline, StableDiffusion3Pipeline

# Configure logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE)],
)


class DiffusionPipelineWrapper:
    def __init__(self, model_id: str, steps: int = 40, scale: float = 4.5):
        self.model_id = model_id
        self.steps = steps
        self.scale = scale
        logging.info(f"Initializing wrapper for {model_id}")
        self.pipeline = self._create_pipeline(model_id)

    def _create_pipeline(self, model_id: str):
        if model_id == "stabilityai/stable-diffusion-3.5-large":
            pipe = StableDiffusion3Pipeline.from_pretrained(
                "bin/models/diffusers/stable-diffusion-3.5-large",
                device_map="balanced",
            )
        elif model_id == "black-forest-labs/FLUX.1-dev":
            pipe = FluxPipeline.from_pretrained(
                "bin/models/diffusers/FLUX.1-dev",
                torch_dtype=torch.bfloat16,
                device_map="balanced",
            )
        else:
            raise ValueError(
                "Unsupported model id. Use 'stabilityai/stable-diffusion-3.5-large' or 'black-forest-labs/FLUX.1-dev'."
            )
        return pipe

    def _call_pipeline(self, prompt: str):
        clip_prompt = prompt[:300]
        if isinstance(self.pipeline, StableDiffusion3Pipeline):
            return self.pipeline(
                prompt=clip_prompt,
                prompt_3=prompt,
                negative_prompt="bad anatomy, poorly drawn face, low resolution, blurry, artifacts, bad lighting, bad composition, cartoonish",
                num_inference_steps=self.steps,
                guidance_scale=self.scale,
            )
        elif isinstance(self.pipeline, FluxPipeline):
            logging.info("Using FLUX.1-dev: sending full prompt as 'prompt_2'.")
            return self.pipeline(
                prompt=clip_prompt,
                prompt_2=prompt,
                negative_prompt="bad anatomy, poorly drawn face, low resolution, blurry, artifacts, bad lighting, bad composition, cartoonish",
                num_inference_steps=self.steps,
                guidance_scale=self.scale,
            )
        else:
            raise ValueError("Unsupported pipeline type.")

    def generate_image(self, prompt: str, output_filename: str, output_dir: str) -> str:
        logging.info(
            f"Generating image with prompt (first 30 chars): '{prompt[:30]}...'"
        )
        with torch.no_grad():
            result = self._call_pipeline(prompt)
            image = result.images[0]
        os.makedirs(output_dir, exist_ok=True)
        image_path = os.path.join(output_dir, output_filename)
        image.save(image_path)
        logging.info(f"Image saved to {image_path}")
        return image_path

    def __call__(self, prompt: str, output_filename: str, output_dir: str) -> str:
        return self.generate_image(prompt, output_filename, output_dir)

    @staticmethod
    def load_metadata_from_json(json_path: str):
        logging.info(f"Loading metadata from {json_path}")
        with open(json_path, "r") as f:
            data = [json.loads(line.strip()) for line in f if line.strip()]
        logging.info(f"Loaded {len(data)} metadata items.")
        return data


def main():
    logging.info("Starting image generation process using DiffusionPipelineWrapper.")
    try:
        diffusion_models = [
            {"model_id": "black-forest-labs/FLUX.1-dev", "steps": 50, "scale": 7.5},
            {
                "model_id": "stabilityai/stable-diffusion-3.5-large",
                "steps": 50,
                "scale": 7.5,
            },
        ]
        json_file = "output/prompts/fiveshotv3-prompts-trial.jsonl"
        base_output_dir = "output/images" + json_file.split("/")[-1].split("-")[0]
        metadata_list = DiffusionPipelineWrapper.load_metadata_from_json(json_file)
        if not metadata_list:
            logging.error("No metadata found in the JSON file.")
            return

        output_metadata = []
        for diff_model in diffusion_models:
            model_id = diff_model["model_id"]
            steps = diff_model.get("steps", 40)
            scale = diff_model.get("scale", 4.5)
            safe_model_id = model_id.replace("/", "_")
            output_dir = os.path.join(base_output_dir, safe_model_id)
            logging.info(f"Initializing wrapper for model '{model_id}'")
            try:
                wrapper = DiffusionPipelineWrapper(model_id, steps, scale)
            except Exception as e:
                logging.error(f"Failed to initialize wrapper for model {model_id}: {e}")
                continue

            for idx, item in enumerate(metadata_list):
                prompt = item.get("generated_scene_description")
                if not prompt:
                    logging.warning(
                        f"Skipping prompt index {idx} due to missing description."
                    )
                    continue
                output_filename = f"image_{idx:03d}.png"
                try:
                    with torch.no_grad():
                        image_path = wrapper(prompt, output_filename, output_dir)
                except Exception as e:
                    logging.error(
                        f"Error generating image for prompt index {idx} with model {model_id}: {e}"
                    )
                    continue
                record = {
                    "model_id": model_id,
                    "prompt": prompt,
                    "image_path": image_path,
                    "steps": steps,
                    "scale": scale,
                }
                output_metadata.append(record)
                logging.info(
                    f"Generated image for prompt index {idx} using model '{model_id}'."
                )

        output_metadata_file = f"metadata_{json_file.split('/')[-1].split('-')[0]}.jsonl"
        with open(output_metadata_file, "a") as f:
            for record in output_metadata:
                f.write(json.dumps(record) + "\n")
        logging.info(f"Updated metadata saved to '{output_metadata_file}'.")

    except Exception as err:
        logging.exception(f"An unexpected error occurred: {err}")


if __name__ == "__main__":
    main()
