import torch
import requests
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
    AutoModelForCausalLM,
    GenerationConfig,
    AutoModelForVision2Seq,
    MllamaForConditionalGeneration
)
from qwen_vl_utils import process_vision_info


class VLMWrapper:
    def __init__(self, model_id: str):
        """
        model_id: exact model identifier (e.g.)
          - Qwen: "Qwen/Qwen2.5-VL-3B-Instruct" or "Qwen/Qwen2.5-VL-7B-Instruct"
          - Llava: "llava-hf/llava-1.5-7b-hf"
          - LlavaNext: "llava-hf/llava-v1.6-mistral-7b-hf"
          - PaliGemma: "google/paligemma2-3b-pt-896" or "google/paligemma2-10b-pt-896"
          - InstructBlip: "Salesforce/instructblip-vicuna-7b"
          - Molmo: "allenai/Molmo-7B-D-0924"
          - Idefics: "HuggingFaceM4/Idefics3-8B-Llama3"
          - Mllama: "meta-llama/Llama-3.2-90B-Vision-Instruct"
        """
        self.model_id = model_id
        self.dtype = torch.bfloat16

        if model_id.startswith("Qwen/"):
            self.model_type = "qwen"
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(model_id)
        elif "llava-hf/llava-1.5-7b-hf" in model_id:
            self.model_type = "llava"
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(model_id)
        elif "llava-hf/llava-v1.6-mistral-7b-hf" in model_id:
            self.model_type = "llava_next"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
        elif model_id.startswith("google/paligemma2"):
            self.model_type = "palgemma"
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                device_map="auto",
            ).eval()
            self.processor = PaliGemmaProcessor.from_pretrained(model_id)
        elif model_id.startswith("Salesforce/instructblip"):
            self.model_type = "instructblip"
            self.model = InstructBlipForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                device_map="auto",
            )
            self.processor = InstructBlipProcessor.from_pretrained(model_id)
        elif model_id.startswith("allenai/Molmo"):
            self.model_type = "molmo"
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=self.dtype,
                device_map="auto",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=self.dtype,
                device_map="auto",
            )
        elif model_id.startswith("HuggingFaceM4/Idefics"):
            self.model_type = "idefics"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                device_map="auto",
            )
        elif model_id.startswith("meta-llama"):
            self.model_type = "mllama"
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(model_id)
        else:
            raise ValueError(f"Unsupported model_id: {model_id}")

        self.device = self.model.device

    def load_image_from_url(self, image_url: str) -> Image.Image:
        response = requests.get(image_url, stream=True)
        return Image.open(response.raw)

    def generate_response(
        self, conversation: list, image_input=None, max_new_tokens=100
    ):
        """
        conversation: list of messages as dictionaries
        image_input: a single PIL Image or a list of PIL Images
        """
        if self.model_type == "qwen":
            prompt = self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(conversation)
            inputs = self.processor(
                text=[prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            return self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        elif self.model_type in ["mllama"]:
            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            inputs = self.processor(
                image_input,
                prompt,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(self.device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            return self.processor.decode(generated_ids[0], skip_special_tokens=True)
        elif self.model_type in ["llava", "llava_next"]:
            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            inputs = self.processor(
                images=image_input,
                text=prompt,
                return_tensors="pt",
            ).to(self.device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            # For some models (e.g. Llava) it might be necessary to trim tokens:
            if self.model_type == "llava":
                return self.processor.decode(
                    generated_ids[0][2:], skip_special_tokens=True
                )
            else:
                return self.processor.decode(generated_ids[0], skip_special_tokens=True)
        elif self.model_type == "palgemma":
            prompt = ""
            inputs = self.processor(
                text=prompt,
                images=image_input,
                return_tensors="pt",
            ).to(self.device)
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
                )
            generation = generation[0][input_len:]
            return self.processor.decode(generation, skip_special_tokens=True)
        elif self.model_type == "instructblip":
            prompt = conversation[0]["content"][0]["text"]
            inputs = self.processor(
                images=image_input,
                text=prompt,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_length=256,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
            )
            return self.processor.batch_decode(outputs, skip_special_tokens=True)[
                0
            ].strip()
        elif self.model_type == "molmo":
            prompt = conversation[0]["content"][0]["text"]
            imgs = image_input if isinstance(image_input, list) else [image_input]
            inputs = self.processor.process(
                images=imgs,
                text=prompt,
            )
            inputs = {k: v.to(self.device).unsqueeze(0) for k, v in inputs.items()}
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(
                    max_new_tokens=max_new_tokens, stop_strings="<|endoftext|>"
                ),
                tokenizer=self.processor.tokenizer,
            )
            input_len = inputs["input_ids"].shape[-1]
            generated_tokens = output[0, input_len:]
            return self.processor.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )
        elif self.model_type == "idefics":
            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            imgs = image_input if isinstance(image_input, list) else [image_input]
            inputs = self.processor(
                text=prompt,
                images=imgs,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            return self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def __call__(self, **kwargs):
        """
        Make the wrapper callable so that calling an instance directly delegates
        to the underlying model's .generate function.

        Example:
            output = wrapper(input_ids=..., attention_mask=..., max_new_tokens=30)
        """
        return self.model.generate(**kwargs)


# ---------------------------
# Example usage:
# ---------------------------
if __name__ == "__main__":
    # Example with Mllama
    model_id = "meta-llama/Llama-3.2-90B-Vision-Instruct"
    wrapper = VLMWrapper(model_id)

    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
    image = wrapper.load_image_from_url(url)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": "If I had to write a haiku for this one, it would be: ",
                },
            ],
        }
    ]
    input_text = wrapper.processor.apply_chat_template(
        messages, add_generation_prompt=True
    )
    inputs = wrapper.processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(wrapper.device)

    # Using the callable interface
    output = wrapper(**inputs, max_new_tokens=30)
    print(wrapper.processor.decode(output[0], skip_special_tokens=True))