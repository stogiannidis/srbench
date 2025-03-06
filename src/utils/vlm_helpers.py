import re
import torch
import requests
from PIL import Image
from transformers import (
	Qwen2_5_VLForConditionalGeneration,
	AutoProcessor,
	LlavaForConditionalGeneration,
	LlavaNextForConditionalGeneration,
	InstructBlipProcessor,
	InstructBlipForConditionalGeneration,
	AutoModelForCausalLM,
	GenerationConfig,
	AutoModelForVision2Seq,
	MllamaForConditionalGeneration,
	AutoModel,  # Added for MiniCPM
	AutoTokenizer,  # Added for MiniCPM
)
from qwen_vl_utils import process_vision_info


class VLMWrapper:
    def __init__(self, model_id: str, device_map="auto"):
        """
        model_id: exact model identifier (e.g.)
          - Qwen: "Qwen/Qwen2.5-VL-3B-Instruct" or "Qwen/Qwen2.5-VL-7B-Instruct"
         - Llava: "llava-hf/llava-1.5-7b-hf"
          - LlavaNext: "llava-hf/llava-v1.6-mistral-7b-hf"
          - InstructBlip: "Salesforce/instructblip-vicuna-7b"
          - Molmo: "allenai/Molmo-7B-D-0924"
          - Idefics: "HuggingFaceM4/Idefics3-8B-Llama3"
          - SmolVLM: "HuggingFaceTB/SmolVLM-Instruct"
          - Mllama: "meta-llama/Llama-3.2-11B-Vision-Instruct"
          - Phi3.5: "microsoft/Phi-3.5-vision-instruct"
          - MiniCPM: "openbmb/MiniCPM-V-2_6" (or similar)
        """
        self.model_id = model_id
        self.dtype = torch.bfloat16
        self.device_map = device_map

        if model_id.startswith("Qwen/"):
            self.model_type = "qwen"
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                attn_implementation="flash_attention_2",
                device_map=self.device_map,
            ).eval()
            self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
            # Ensure tokenizer uses left padding for batched generation with Flash Attention
            self.processor.tokenizer.padding_side = "left"
        elif "llava-hf/llava-1.5-7b-hf" in model_id:
            self.model_type = "llava"
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                device_map=self.device_map,
            ).eval()
            self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        elif "llava-hf/llava-v1.6-mistral-7b-hf" in model_id:
            self.model_type = "llava_next"
            self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                device_map=self.device_map,
            ).eval()
        # Removed Paligemma support
        elif model_id.startswith("Salesforce/instructblip"):
            self.model_type = "instructblip"
            self.model = InstructBlipForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                device_map=self.device_map,
            ).eval()
            self.processor = InstructBlipProcessor.from_pretrained(
                model_id, use_fast=True
            )
        elif model_id.startswith("allenai/Molmo"):
            self.model_type = "molmo"
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=self.dtype,
                device_map=self.device_map,
                use_fast=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=self.dtype,
                device_map=self.device_map,
            ).eval()
        elif model_id.startswith("HuggingFaceM4/Idefics"):
            self.model_type = "idefics"
            self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                device_map=self.device_map,
                attn_implementation="flash_attention_2",
            ).eval()
        elif model_id.startswith("HuggingFaceTB/SmolVLM"):
            self.model_type = "smolvlm"
            self.processor = AutoProcessor.from_pretrained(
                model_id, use_fast=True, padding_side="left"
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                device_map=self.device_map,
                attn_implementation="flash_attention_2",
            ).eval()
        elif model_id.startswith("meta-llama"):
            self.model_type = "mllama"
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                device_map=self.device_map,
            ).eval()
            self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True, padding_side="left")
        elif model_id.startswith("microsoft/Phi-3.5-vision-instruct"):
            self.model_type = "phi35"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=self.device_map,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                _attn_implementation="flash_attention_2",
            ).eval()
            self.processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True, num_crops=16
            )
        elif model_id.startswith("openbmb/MiniCPM"):
            self.model_type = "minicpm"
            self.model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map=self.device_map,
            ).eval()
            self.processor = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True
            )
        else:
            raise ValueError(f"Unsupported model_id: {model_id}")

        # Set pad token if available
        if hasattr(self.processor, "tokenizer") and hasattr(
            self.model, "generation_config"
        ):
            self.model.generation_config.pad_token_id = (
                self.processor.tokenizer.pad_token_id
            )

        self.device = self.model.device

    def load_image_from_url(self, image_url: str) -> Image.Image:
        response = requests.get(image_url, stream=True)
        return Image.open(response.raw)

    def preprocess(self, conversation: list, image_input=None):
        """
        Preprocess the conversation and image input.
        Returns the processed inputs that would normally be passed to the model.
        """
        # Normalize conversation and image_input into batches
        if conversation and isinstance(conversation[0], list):
            batch_conversations = conversation
        else:
            batch_conversations = [conversation]

        if image_input is not None:
            if isinstance(image_input, list) and hasattr(image_input[0], "format"):
                batch_images = image_input
            else:
                batch_images = [image_input] * len(batch_conversations)
        else:
            batch_images = None

        if self.model_type == "qwen":
            prompts = [
                self.processor.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=True
                )
                for conv in batch_conversations
            ]
            image_inputs, _ = process_vision_info(batch_conversations)
            inputs = self.processor(
                text=prompts,
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            return inputs

        elif self.model_type == "mllama":
            prompts = [
                self.processor.apply_chat_template(conv, add_generation_prompt=True)
                for conv in batch_conversations
            ]
            inputs = self.processor(
                batch_images,
                prompts,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            return inputs

        elif self.model_type in ["llava", "llava_next"]:
            prompts = [
                self.processor.apply_chat_template(conv, add_generation_prompt=True)
                for conv in batch_conversations
            ]
            inputs = self.processor(
                images=batch_images,
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            return inputs

        elif self.model_type == "phi35":
            processed_inputs_batch = []
            for conv, image in zip(
                batch_conversations, batch_images
            ):  # Iterate over batch
                placeholder = "<|image_1|>\n"
                prompt = placeholder + conv[0]["content"][1]["text"]
                messages = [{"role": "user", "content": prompt}]
                prompt_text = self.processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                processed_input = self.processor(
                    text=prompt_text, images=image, return_tensors="pt"
                ).to(self.device)  # Process single example
                processed_inputs_batch.append(processed_input)

            # Concatenate processed inputs
            batched_input_ids = torch.cat(
                [inputs["input_ids"] for inputs in processed_inputs_batch]
            )
            batched_attention_mask = torch.cat(
                [inputs["attention_mask"] for inputs in processed_inputs_batch]
            )
            batched_pixel_values = torch.cat(
                [inputs["pixel_values"] for inputs in processed_inputs_batch]
            )

            inputs = {
                "input_ids": batched_input_ids,
                "attention_mask": batched_attention_mask,
                "pixel_values": batched_pixel_values,
            }
            return inputs

        elif self.model_type == "instructblip":
            prompts = [conv[0]["content"][1]["text"] for conv in batch_conversations]
            inputs = self.processor(
                images=batch_images,
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            return inputs

        elif self.model_type == "molmo":
            # Build a list of prompt strings (one per conversation)
            prompts = [
                " ".join(conv[0]["content"][1]["text"])
                if isinstance(conv[0]["content"][1]["text"], list)
                else conv[0]["content"][1]["text"]
                for conv in batch_conversations
            ]
            imgs = batch_images if batch_images is not None else None
            inputs = self.processor.process(
                images=imgs,
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            return inputs

        elif self.model_type in ["idefics", "smolvlm"]:
            prompts = [
                self.processor.apply_chat_template(conv, add_generation_prompt=True)
                for conv in batch_conversations
            ]
            imgs = batch_images if batch_images is not None else None
            if imgs and not isinstance(imgs, list):
                imgs = [imgs] * len(batch_conversations)
            inputs = self.processor(
                text=prompts,
                images=imgs,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            return inputs

        elif self.model_type == "minicpm":
            msgs = batch_conversations[0]
            image_arg = batch_images[0] if batch_images else None
            return (msgs, image_arg)

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def decode(self, generated_ids, extra=None):
        """
        Decode the generated_ids based on the model type.

        Params:
                generated_ids: Output token ids from the model.
                extra: Additional parameter for decoding (e.g. input length for molmo).

        Returns:
                Decoded text as a list of strings.
        """
        if self.model_type == "qwen":
            return self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        elif self.model_type == "mllama":
            return [
                self.processor.decode(g, skip_special_tokens=True)
                for g in generated_ids
            ]
        elif self.model_type in ["llava", "llava_next"]:
            if self.model_type == "llava":
                return [
                    self.processor.decode(g[2:], skip_special_tokens=True)
                    for g in generated_ids
                ]
            else:
                return self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
        elif self.model_type == "phi35":
            return self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        elif self.model_type == "instructblip":
            decoded = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return [d.strip() for d in decoded]
        elif self.model_type == "molmo":
            input_len = extra if extra is not None else 0
            return [
                self.processor.tokenizer.decode(g[input_len:], skip_special_tokens=True)
                for g in generated_ids
            ]
        elif self.model_type in ["idefics", "smolvlm"]:
            return self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        elif self.model_type == "minicpm":
            text = ""
            for t in generated_ids:
                text += t
            return [text]
        else:
            return self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

    @torch.inference_mode()
    def __call__(self, *args, **kwargs):
        """
        Invoke the model's generate method with the provided keyword arguments.
        Parameters:
                **kwargs: Arbitrary keyword arguments that are passed directly to the model's generate method.
        Returns:
                The output produced by self.model.generate when called with the supplied arguments.
        """
        if self.model_type == "minicpm":
            return self.model.chat(*args, tokenizer=self.processor._tokenizer)
        return self.model.generate(**kwargs)
