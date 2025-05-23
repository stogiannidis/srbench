import re
import torch
import requests
from typing import List, Dict, Any, Union, Optional, Tuple
from PIL import Image
from functools import lru_cache
import gc
from contextlib import contextmanager
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
    AutoModel,
    AutoTokenizer,
)
from qwen_vl_utils import process_vision_info
import random
import numpy as np
import os


def set_seed(seed: int = 42) -> None:
    """Set seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


@contextmanager
def torch_memory_manager():
    """Context manager for torch memory management."""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class VLMWrapper:
    _model_cache = {}  # Class-level cache for model reuse
    
    def __init__(self, model_id: str, device_map: str = "auto", 
                 torch_dtype: torch.dtype = torch.bfloat16,
                 low_cpu_mem_usage: bool = True,
                 trust_remote_code: bool = True):
        """
        Optimized VLM wrapper with caching and memory management.
        """
        self.model_id = model_id
        self.dtype = torch_dtype
        self.device_map = device_map
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.trust_remote_code = trust_remote_code
        
        # Check cache first
        cache_key = f"{model_id}_{device_map}_{torch_dtype}"
        if cache_key in self._model_cache:
            cached_model, cached_processor, cached_type = self._model_cache[cache_key]
            self.model = cached_model
            self.processor = cached_processor
            self.model_type = cached_type
        else:
            self._initialize_model()
            self._model_cache[cache_key] = (self.model, self.processor, self.model_type)
        
        self.device = next(self.model.parameters()).device
        self._setup_generation_config()

    def _initialize_model(self):
        """Initialize model and processor based on model_id."""
        with torch_memory_manager():
            if self.model_id.startswith("Qwen/"):
                self._init_qwen()
            elif "llava-hf/llava-1.5-7b-hf" in self.model_id:
                self._init_llava()
            elif "llava-hf/llava-v1.6-mistral-7b-hf" in self.model_id:
                self._init_llava_next()
            elif self.model_id.startswith("Salesforce/instructblip"):
                self._init_instructblip()
            elif self.model_id.startswith("allenai/Molmo"):
                self._init_molmo()
            elif self.model_id.startswith("HuggingFaceM4/Idefics"):
                self._init_idefics()
            elif self.model_id.startswith("HuggingFaceTB/SmolVLM"):
                self._init_smolvlm()
            elif self.model_id.startswith("meta-llama"):
                self._init_mllama()
            elif self.model_id.startswith("microsoft/Phi-3.5-vision-instruct"):
                self._init_phi35()
            elif self.model_id.startswith("openbmb/MiniCPM"):
                self._init_minicpm()
            else:
                raise ValueError(f"Unsupported model_id: {self.model_id}")

    def _init_qwen(self):
        """Initialize Qwen model with optimizations."""
        self.model_type = "qwen"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            attn_implementation="flash_attention_2",
            device_map=self.device_map,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
            trust_remote_code=self.trust_remote_code,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, use_fast=True, trust_remote_code=self.trust_remote_code
        )
        self.processor.tokenizer.padding_side = "left"

    def _init_llava(self):
        """Initialize LLaVA model with optimizations."""
        self.model_type = "llava"
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, use_fast=True, trust_remote_code=self.trust_remote_code
        )

    def _init_llava_next(self):
        """Initialize LLaVA-Next model with optimizations."""
        self.model_type = "llava_next"
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, use_fast=True, trust_remote_code=self.trust_remote_code
        )
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
        ).eval()

    def _init_instructblip(self):
        """Initialize InstructBLIP model with optimizations."""
        self.model_type = "instructblip"
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            device_map=self.device_map,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
            trust_remote_code=self.trust_remote_code,
        ).eval()
        self.processor = InstructBlipProcessor.from_pretrained(
            self.model_id, use_fast=True, trust_remote_code=self.trust_remote_code
        )

    def _init_molmo(self):
        """Initialize Molmo model with optimizations."""
        self.model_type = "molmo"
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=self.dtype,
            device_map=self.device_map,
            use_fast=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=self.dtype,
            device_map=self.device_map,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
        ).eval()

    def _init_idefics(self):
        """Initialize Idefics model with optimizations."""
        self.model_type = "idefics"
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, use_fast=True, trust_remote_code=self.trust_remote_code
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            device_map=self.device_map,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=self.low_cpu_mem_usage,
            trust_remote_code=self.trust_remote_code,
        ).eval()

    def _init_smolvlm(self):
        """Initialize SmolVLM model with optimizations."""
        self.model_type = "smolvlm"
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, use_fast=True, padding_side="left",
            trust_remote_code=self.trust_remote_code
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            device_map=self.device_map,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=self.low_cpu_mem_usage,
            trust_remote_code=self.trust_remote_code,
        ).eval()

    def _init_mllama(self):
        """Initialize Mllama model with optimizations."""
        self.model_type = "mllama"
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            device_map=self.device_map,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
            trust_remote_code=self.trust_remote_code,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, use_fast=True, padding_side="left",
            trust_remote_code=self.trust_remote_code
        )

    def _init_phi35(self):
        """Initialize Phi-3.5 model with optimizations."""
        self.model_type = "phi35"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=self.dtype,
            _attn_implementation="flash_attention_2",
            low_cpu_mem_usage=self.low_cpu_mem_usage,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code, num_crops=16
        )

    def _init_minicpm(self):
        """Initialize MiniCPM model with optimizations."""
        self.model_type = "minicpm"
        self.model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=self.dtype,
            attn_implementation="flash_attention_2",
            device_map=self.device_map,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
        ).eval()
        self.processor = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )

    def _setup_generation_config(self):
        """Setup generation configuration."""
        if hasattr(self.processor, "tokenizer") and hasattr(self.model, "generation_config"):
            if hasattr(self.processor.tokenizer, "pad_token_id"):
                self.model.generation_config.pad_token_id = self.processor.tokenizer.pad_token_id

    @lru_cache(maxsize=128)
    def load_image_from_url(self, image_url: str) -> Image.Image:
        """Cached image loading from URL."""
        response = requests.get(image_url, stream=True, timeout=30)
        response.raise_for_status()
        return Image.open(response.raw)

    def preprocess(self, conversation: List[List[Dict]], image_input: Optional[List] = None) -> Dict[str, torch.Tensor]:
        """
        Optimized preprocessing with memory management.
        """
        # Normalize inputs
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

        # Model-specific preprocessing with optimizations
        with torch_memory_manager():
            return self._preprocess_by_type(batch_conversations, batch_images)

    def _preprocess_by_type(self, batch_conversations: List, batch_images: Optional[List]) -> Dict[str, torch.Tensor]:
        """Model-specific preprocessing logic."""
        if self.model_type == "qwen":
            return self._preprocess_qwen(batch_conversations, batch_images)
        elif self.model_type == "mllama":
            return self._preprocess_mllama(batch_conversations, batch_images)
        elif self.model_type in ["llava", "llava_next"]:
            return self._preprocess_llava(batch_conversations, batch_images)
        elif self.model_type == "phi35":
            return self._preprocess_phi35(batch_conversations, batch_images)
        elif self.model_type == "instructblip":
            return self._preprocess_instructblip(batch_conversations, batch_images)
        elif self.model_type == "molmo":
            return self._preprocess_molmo(batch_conversations, batch_images)
        elif self.model_type in ["idefics", "smolvlm"]:
            return self._preprocess_idefics_smolvlm(batch_conversations, batch_images)
        elif self.model_type == "minicpm":
            return self._preprocess_minicpm(batch_conversations, batch_images)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _preprocess_qwen(self, batch_conversations: List, batch_images: Optional[List]) -> Dict[str, torch.Tensor]:
        """Optimized Qwen preprocessing."""
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

    def _preprocess_mllama(self, batch_conversations: List, batch_images: Optional[List]) -> Dict[str, torch.Tensor]:
        """Optimized Mllama preprocessing."""
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

    def _preprocess_llava(self, batch_conversations: List, batch_images: Optional[List]) -> Dict[str, torch.Tensor]:
        """Optimized LLaVA preprocessing."""
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

    def _preprocess_phi35(self, batch_conversations: List, batch_images: Optional[List]) -> Dict[str, torch.Tensor]:
        """Optimized Phi-3.5 preprocessing."""
        processed_inputs_batch = []
        for conv, image in zip(batch_conversations, batch_images):
            placeholder = "<|image_1|>\n"
            prompt = placeholder + conv[0]["content"][1]["text"]
            messages = [{"role": "user", "content": prompt}]
            prompt_text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            processed_input = self.processor(
                text=prompt_text, images=image, return_tensors="pt"
            ).to(self.device)
            processed_inputs_batch.append(processed_input)

        # Efficient tensor concatenation
        keys = processed_inputs_batch[0].keys()
        inputs = {
            key: torch.cat([inp[key] for inp in processed_inputs_batch], dim=0)
            for key in keys
        }
        return inputs

    def _preprocess_instructblip(self, batch_conversations: List, batch_images: Optional[List]) -> Dict[str, torch.Tensor]:
        """Optimized InstructBLIP preprocessing."""
        prompts = [conv[0]["content"][1]["text"] for conv in batch_conversations]
        inputs = self.processor(
            images=batch_images,
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        return inputs

    def _preprocess_molmo(self, batch_conversations: List, batch_images: Optional[List]) -> Dict[str, torch.Tensor]:
        """Optimized Molmo preprocessing."""
        prompts = [
            " ".join(conv[0]["content"][1]["text"])
            if isinstance(conv[0]["content"][1]["text"], list)
            else conv[0]["content"][1]["text"]
            for conv in batch_conversations
        ]
        inputs = self.processor.process(
            images=batch_images,
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def _preprocess_idefics_smolvlm(self, batch_conversations: List, batch_images: Optional[List]) -> Dict[str, torch.Tensor]:
        """Optimized Idefics/SmolVLM preprocessing."""
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

    def _preprocess_minicpm(self, batch_conversations: List, batch_images: Optional[List]) -> Tuple[List, Any]:
        """Optimized MiniCPM preprocessing."""
        msgs = batch_conversations[0]
        image_arg = batch_images[0] if batch_images else None
        return (msgs, image_arg)

    def decode(self, generated_ids: torch.Tensor, extra: Optional[int] = None) -> List[str]:
        """
        Optimized decoding with memory cleanup.
        """
        with torch_memory_manager():
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
                return ["".join(generated_ids)]
            else:
                return self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

    @torch.inference_mode()
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """
        Optimized model inference with memory management.
        """
        with torch_memory_manager():
            if self.model_type == "minicpm":
                return self.model.chat(*args, tokenizer=self.processor._tokenizer)
            return self.model.generate(**kwargs)

    def __del__(self):
        """Cleanup resources on deletion."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
