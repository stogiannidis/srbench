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
    def __init__(self, model_id: str, device_map="auto"):
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
        self.device_map = device_map

        if model_id.startswith("Qwen/"):
            self.model_type = "qwen"
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                attn_implementation="flash_attention_2",
                device_map=self.device_map,
            ).eval()
            self.processor = AutoProcessor.from_pretrained(model_id)
        elif "llava-hf/llava-1.5-7b-hf" in model_id:
            self.model_type = "llava"
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                device_map=self.device_map,
            ).eval()
            self.processor = AutoProcessor.from_pretrained(model_id)
        elif "llava-hf/llava-v1.6-mistral-7b-hf" in model_id:
            self.model_type = "llava_next"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                device_map=self.device_map,
            ).eval()
        elif model_id.startswith("google/paligemma2"):
            self.model_type = "palgemma"
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                device_map=self.device_map,
            ).eval()
            self.processor = PaliGemmaProcessor.from_pretrained(model_id)
        elif model_id.startswith("Salesforce/instructblip"):
            self.model_type = "instructblip"
            self.model = InstructBlipForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                device_map=self.device_map,
            ).eval()
            self.processor = InstructBlipProcessor.from_pretrained(model_id)
        elif model_id.startswith("allenai/Molmo"):
            self.model_type = "molmo"
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=self.dtype,
                device_map=self.device_map,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=self.dtype,
                device_map=self.device_map,
            ).eval()
        elif model_id.startswith("HuggingFaceM4/Idefics"):
            self.model_type = "idefics"
            self.processor = AutoProcessor.from_pretrained(model_id)
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
            self.processor = AutoProcessor.from_pretrained(model_id)
        else:
            raise ValueError(f"Unsupported model_id: {model_id}")

        if hasattr(self.processor, "tokenizer") and hasattr(self.model, "generation_config"):
            self.model.generation_config.pad_token_id = self.processor.tokenizer.pad_token_id

        self.device = self.model.device

    def load_image_from_url(self, image_url: str) -> Image.Image:
        response = requests.get(image_url, stream=True)
        return Image.open(response.raw)

    def generate_response(
        self, conversation: list, image_input=None, max_new_tokens=100
    ):
        """
        conversation: can be a single conversation (list of message dicts) or a batch (list of conversations).
        image_input: a single PIL Image or a list of PIL Images.
        Returns a list of generated responses for each batched conversation.
        """
        # Normalize conversation and image_input into batches
        if conversation and isinstance(conversation[0], list):
            batch_conversations = conversation
        else:
            batch_conversations = [conversation]

        if image_input is not None:
            if isinstance(image_input, list) and not hasattr(image_input[0], "format"):
                # Already a batch of images
                batch_images = image_input
            else:
                batch_images = [image_input] * len(batch_conversations)
        else:
            batch_images = None

        if self.model_type == "qwen":
            prompts = [
                self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                for conv in batch_conversations
            ]
            image_inputs_list = []
            video_inputs_list = []
            for conv in batch_conversations:
                img, vid = process_vision_info(conv)
                image_inputs_list.append(img)
                video_inputs_list.append(vid)
            inputs = self.processor(
                text=prompts,
                images=image_inputs_list,
                videos=video_inputs_list,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            return self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        elif self.model_type in ["mllama"]:
            prompts = [
                self.processor.apply_chat_template(conv, add_generation_prompt=True)
                for conv in batch_conversations
            ]
            inputs = self.processor(
                batch_images,
                prompts,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(self.device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            return [self.processor.decode(gid, skip_special_tokens=True) for gid in generated_ids]

        elif self.model_type in ["llava", "llava_next"]:
            prompts = [
                self.processor.apply_chat_template(conv, add_generation_prompt=True)
                for conv in batch_conversations
            ]
            inputs = self.processor(
                images=batch_images,
                text=prompts,
                return_tensors="pt",
            ).to(self.device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            if self.model_type == "llava":
                return [self.processor.decode(ids[2:], skip_special_tokens=True) for ids in generated_ids]
            else:
                return [self.processor.decode(ids, skip_special_tokens=True) for ids in generated_ids]

        elif self.model_type == "palgemma":
            # Using an empty prompt for each conversation
            inputs = self.processor(
                text=[""] * len(batch_conversations),
                images=batch_images,
                return_tensors="pt",
            ).to(self.device)
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
                )
            results = []
            for gen in generation:
                gen_tokens = gen[input_len:]
                results.append(self.processor.decode(gen_tokens, skip_special_tokens=True))
            return results

        elif self.model_type == "instructblip":
            prompts = [conv[0]["content"][0]["text"] for conv in batch_conversations]
            inputs = self.processor(
                images=batch_images,
                text=prompts,
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
            decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)
            return [d.strip() for d in decoded]

        elif self.model_type == "molmo":
            prompts = [conv[0]["content"][0]["text"] for conv in batch_conversations]
            imgs = batch_images if batch_images is not None else None
            # If a single image was passed, replicate it for the batch
            if imgs and not isinstance(imgs, list):
                imgs = [imgs] * len(batch_conversations)
            inputs = self.processor.process(
                images=imgs,
                text=prompts,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=max_new_tokens, stop_strings="<|endoftext|>"),
                tokenizer=self.processor.tokenizer,
            )
            input_len = inputs["input_ids"].shape[-1]
            results = []
            for out in output:
                generated_tokens = out[input_len:]
                results.append(self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True))
            return results

        elif self.model_type == "idefics":
            prompts = [
                self.processor.apply_chat_template(conv, add_generation_prompt=True)
                for conv in batch_conversations
            ]
            imgs = batch_images if batch_images is not None else None
            # Replicate image if necessary:
            if imgs and not isinstance(imgs, list):
                imgs = [imgs] * len(batch_conversations)
            inputs = self.processor(
                text=prompts,
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
        to the underlying `generate_response` method.
        
        Usage:
        vlm = VLMWrapper("Qwen/Qwen2.5-VL-7B-Instruct")
        response = vlm(conversation=conversation, image_input=image_input)
        
        """
        return self.generate_response(**kwargs)