import re
import torch
import requests
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from functools import lru_cache
from contextlib import contextmanager
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
    AutoModel,
    AutoTokenizer,
)
from qwen_vl_utils import process_vision_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for different VLM models."""
    model_class: type
    processor_class: type
    requires_trust_remote_code: bool = False
    supports_flash_attention: bool = False
    padding_side: str = "left"
    special_args: Dict[str, Any] = None
    processor_args: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.special_args is None:
            self.special_args = {}
        if self.processor_args is None:
            self.processor_args = {}

# Model configurations registry
MODEL_CONFIGS = {
    "qwen": ModelConfig(
        model_class=Qwen2_5_VLForConditionalGeneration,
        processor_class=AutoProcessor,
        supports_flash_attention=True,
        processor_args={"use_fast": True}
    ),
    "llava": ModelConfig(
        model_class=LlavaForConditionalGeneration,
        processor_class=AutoProcessor,
        special_args={"low_cpu_mem_usage": True},
        processor_args={"use_fast": True}
    ),
    "llava_next": ModelConfig(
        model_class=LlavaNextForConditionalGeneration,
        processor_class=AutoProcessor,
        special_args={"low_cpu_mem_usage": True},
        processor_args={"use_fast": True}
    ),
    "instructblip": ModelConfig(
        model_class=InstructBlipForConditionalGeneration,
        processor_class=InstructBlipProcessor,
        processor_args={"use_fast": True}
    ),
    "molmo": ModelConfig(
        model_class=AutoModelForCausalLM,
        processor_class=AutoProcessor,
        requires_trust_remote_code=True,
        processor_args={"use_fast": True}
    ),
    "idefics": ModelConfig(
        model_class=AutoModelForVision2Seq,
        processor_class=AutoProcessor,
        supports_flash_attention=True,
        processor_args={"use_fast": True}
    ),
    "smolvlm": ModelConfig(
        model_class=AutoModelForVision2Seq,
        processor_class=AutoProcessor,
        supports_flash_attention=True,
        special_args={},
        processor_args={"use_fast": True, "padding_side": "left"}
    ),
    "mllama": ModelConfig(
        model_class=MllamaForConditionalGeneration,
        processor_class=AutoProcessor,
        processor_args={"use_fast": True, "padding_side": "left"}
    ),
    "phi35": ModelConfig(
        model_class=AutoModelForCausalLM,
        processor_class=AutoProcessor,
        requires_trust_remote_code=True,
        special_args={"num_crops": 16}
    ),
    "minicpm": ModelConfig(
        model_class=AutoModel,
        processor_class=AutoTokenizer,
        requires_trust_remote_code=True,
        supports_flash_attention=True
    ),
}

class VLMWrapper:
    """Optimized Vision-Language Model wrapper with improved performance and memory management."""
    
    def __init__(self, model_id: str, device_map: str = "auto", dtype: torch.dtype = torch.bfloat16):
        """
        Initialize VLM wrapper with automatic model type detection and optimized settings.
        
        Args:
            model_id: HuggingFace model identifier
            device_map: Device mapping strategy ("auto", "cpu", "cuda", etc.)
            dtype: Model precision (torch.bfloat16, torch.float16, etc.)
        """
        self.model_id = model_id
        self.model_type = self._detect_model_type(model_id)
        self.config = MODEL_CONFIGS[self.model_type]
        self.dtype = self._validate_dtype(dtype)
        self.device_map = self._optimize_device_map(device_map)
        
        # Lazy initialization
        self._model = None
        self._processor = None
        self._device = None
        
        logger.info(f"Initialized VLMWrapper for {self.model_type} model: {model_id}")

    @property
    def model(self):
        """Lazy loading of model."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    @property
    def processor(self):
        """Lazy loading of processor."""
        if self._processor is None:
            self._processor = self._load_processor()
        return self._processor

    @property
    def device(self):
        """Get model device."""
        if self._device is None:
            self._device = self.model.device
        return self._device

    def _validate_dtype(self, dtype: torch.dtype) -> torch.dtype:
        """Validate and optimize dtype based on hardware capabilities."""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using float32")
            return torch.float32
        
        # Force float16 for InstructBLIP to avoid dtype mismatch errors
        if self.model_type == "instructblip":
            logger.info("Using float16 for InstructBLIP to avoid dtype mismatch")
            return torch.float16
        
        if torch.cuda.get_device_capability()[0] < 8 and dtype == torch.bfloat16:
            logger.warning("GPU doesn't support bfloat16, falling back to float16")
            return torch.float16
            
        return dtype

    def _optimize_device_map(self, device_map: str) -> str:
        """Optimize device mapping based on available hardware."""
        if (device_map == "auto"):
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                if gpu_count > 1:
                    logger.info(f"Multiple GPUs detected ({gpu_count}), using auto device mapping")
                    return "auto"
                else:
                    return "cuda:0"
            else:
                logger.warning("CUDA not available, using CPU")
                return "cpu"
        return device_map

    def _detect_model_type(self, model_id: str) -> str:
        """Detect model type from model_id."""
        model_patterns = {
            "qwen": r"Qwen/",
            "llava": r"llava-hf/llava-1\.5",
            "llava_next": r"llava-hf/llava-v1\.6",
            "instructblip": r"Salesforce/instructblip",
            "molmo": r"allenai/Molmo",
            "idefics": r"HuggingFaceM4/Idefics",
            "smolvlm": r"HuggingFaceTB/SmolVLM",
            "mllama": r"meta-llama",
            "phi35": r"microsoft/Phi-3\.5-vision-instruct",
            "minicpm": r"openbmb/MiniCPM",
        }
        
        for model_type, pattern in model_patterns.items():
            if re.search(pattern, model_id):
                return model_type
                
        raise ValueError(f"Unsupported model_id: {model_id}")

    def _load_model(self):
        """Load model with optimized settings."""
        try:
            model_args = {
                "torch_dtype": self.dtype,
                "device_map": self.device_map,
            }
            
            if self.config.requires_trust_remote_code:
                model_args["trust_remote_code"] = True
                
            if self.config.supports_flash_attention and torch.cuda.is_available():
                model_args["attn_implementation"] = "flash_attention_2"
                
            # Add special arguments
            model_args.update(self.config.special_args)
            
            model = self.config.model_class.from_pretrained(self.model_id, **model_args)
            return model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_id}: {e}")
            raise

    def _load_processor(self):
        """Load processor with optimized settings."""
        try:
            processor_args = {}
            
            if self.config.requires_trust_remote_code:
                processor_args["trust_remote_code"] = True
                
            # Add special arguments
            processor_args.update(self.config.processor_args)
            
            processor = self.config.processor_class.from_pretrained(self.model_id, **processor_args)
            
            # Configure padding
            if hasattr(processor, "tokenizer"):
                processor.tokenizer.padding_side = self.config.padding_side
                
            # Set pad token if available
            if hasattr(processor, "tokenizer") and hasattr(self.model, "generation_config"):
                self.model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
                
            return processor
            
        except Exception as e:
            logger.error(f"Failed to load processor for {self.model_id}: {e}")
            raise

    @lru_cache(maxsize=128)
    def load_image_from_url(self, image_url: str) -> Image.Image:
        """Load image from URL with caching."""
        try:
            response = requests.get(image_url, stream=True, timeout=10)
            response.raise_for_status()
            return Image.open(response.raw)
        except Exception as e:
            logger.error(f"Failed to load image from {image_url}: {e}")
            raise

    def _normalize_inputs(self, conversation: List, image_input: Optional[Union[Image.Image, List[Image.Image]]]) -> Tuple[List[List], Optional[List[Image.Image]]]:
        """Normalize conversation and image inputs to batch format."""
        # Normalize conversation
        if conversation and isinstance(conversation[0], list):
            batch_conversations = conversation
        else:
            batch_conversations = [conversation]

        # Normalize images
        batch_images = None
        if image_input is not None:
            if isinstance(image_input, list) and hasattr(image_input[0], "format"):
                batch_images = image_input
            else:
                batch_images = [image_input] * len(batch_conversations)

        return batch_conversations, batch_images

    def preprocess(self, conversation: List, image_input: Optional[Union[Image.Image, List[Image.Image]]] = None) -> Dict[str, torch.Tensor]:
        """
        Preprocess conversation and images for model input.
        
        Args:
            conversation: List of conversation turns or batch of conversations
            image_input: Single image or list of images
            
        Returns:
            Preprocessed inputs ready for model
        """
        batch_conversations, batch_images = self._normalize_inputs(conversation, image_input)
        
        try:
            if self.model_type == "qwen":
                return self._preprocess_qwen(batch_conversations)
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
                
        except Exception as e:
            logger.error(f"Preprocessing failed for {self.model_type}: {e}")
            raise

    def _preprocess_qwen(self, batch_conversations: List[List]) -> Dict[str, torch.Tensor]:
        """Preprocess for Qwen models."""
        prompts = [
            self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            for conv in batch_conversations
        ]
        image_inputs, _ = process_vision_info(batch_conversations)
        inputs = self.processor(text=prompts, images=image_inputs, padding=True, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _preprocess_mllama(self, batch_conversations: List[List], batch_images: List[Image.Image]) -> Dict[str, torch.Tensor]:
        """Preprocess for Mllama models."""
        prompts = [
            self.processor.apply_chat_template(conv, add_generation_prompt=True)
            for conv in batch_conversations
        ]
        inputs = self.processor(
            batch_images, prompts, add_special_tokens=False,
            return_tensors="pt", padding=True, truncation=True
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _preprocess_llava(self, batch_conversations: List[List], batch_images: List[Image.Image]) -> Dict[str, torch.Tensor]:
        """Preprocess for Llava models."""
        prompts = [
            self.processor.apply_chat_template(conv, add_generation_prompt=True)
            for conv in batch_conversations
        ]
        inputs = self.processor(
            images=batch_images, text=prompts,
            return_tensors="pt", padding=True, truncation=True
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _preprocess_phi35(self, batch_conversations: List[List], batch_images: List[Image.Image]) -> Dict[str, torch.Tensor]:
        """Preprocess for Phi3.5 models."""
        processed_inputs_batch = []
        for conv, image in zip(batch_conversations, batch_images):
            placeholder = "<|image_1|>\n"
            prompt = placeholder + conv[0]["content"][1]["text"]
            messages = [{"role": "user", "content": prompt}]
            prompt_text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            processed_input = self.processor(text=prompt_text, images=image, return_tensors="pt")
            processed_inputs_batch.append({k: v.to(self.device) for k, v in processed_input.items()})

        # Efficient batching
        return {
            "input_ids": torch.cat([inp["input_ids"] for inp in processed_inputs_batch]),
            "attention_mask": torch.cat([inp["attention_mask"] for inp in processed_inputs_batch]),
            "pixel_values": torch.cat([inp["pixel_values"] for inp in processed_inputs_batch]),
        }

    def _preprocess_instructblip(self, batch_conversations: List[List], batch_images: List[Image.Image]) -> Dict[str, torch.Tensor]:
        """Preprocess for InstructBlip models."""
        prompts = [conv[0]["content"][1]["text"] for conv in batch_conversations]
        inputs = self.processor(
            images=batch_images, text=prompts,
            return_tensors="pt", padding=True, truncation=True
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _preprocess_molmo(self, batch_conversations: List[List], batch_images: List[Image.Image]) -> Dict[str, torch.Tensor]:
        """Preprocess for Molmo models."""
        prompts = [
            " ".join(conv[0]["content"][1]["text"]) if isinstance(conv[0]["content"][1]["text"], list)
            else conv[0]["content"][1]["text"] for conv in batch_conversations
        ]
        inputs = self.processor.process(
            images=batch_images, text=prompts,
            return_tensors="pt", padding=True, truncation=True
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _preprocess_idefics_smolvlm(self, batch_conversations: List[List], batch_images: List[Image.Image]) -> Dict[str, torch.Tensor]:
        """Preprocess for Idefics and SmolVLM models."""
        prompts = [
            self.processor.apply_chat_template(conv, add_generation_prompt=True)
            for conv in batch_conversations
        ]
        if batch_images and not isinstance(batch_images, list):
            batch_images = [batch_images] * len(batch_conversations)
        inputs = self.processor(
            text=prompts, images=batch_images,
            return_tensors="pt", padding=True, truncation=True
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _preprocess_minicpm(self, batch_conversations: List[List], batch_images: List[Image.Image]) -> Tuple:
        """Preprocess for MiniCPM models."""
        msgs = batch_conversations[0]
        image_arg = batch_images[0] if batch_images else None
        return (msgs, image_arg)

    def decode(self, generated_ids: torch.Tensor, extra: Optional[int] = None) -> List[str]:
        """
        Decode generated token IDs to text.
        
        Args:
            generated_ids: Generated token IDs from model
            extra: Additional parameter (e.g., input length for some models)
            
        Returns:
            List of decoded text strings
        """
        try:
            if self.model_type == "qwen":
                return self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            elif self.model_type == "mllama":
                return [self.processor.decode(g, skip_special_tokens=True) for g in generated_ids]
            elif self.model_type == "llava":
                return [self.processor.decode(g[2:], skip_special_tokens=True) for g in generated_ids]
            elif self.model_type == "llava_next":
                return self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            elif self.model_type == "phi35":
                return self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            elif self.model_type == "instructblip":
                decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                return [d.strip() for d in decoded]
            elif self.model_type == "molmo":
                input_len = extra if extra is not None else 0
                return [self.processor.tokenizer.decode(g[input_len:], skip_special_tokens=True) for g in generated_ids]
            elif self.model_type in ["idefics", "smolvlm"]:
                return self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            elif self.model_type == "minicpm":
                return ["".join(generated_ids)]
            else:
                return self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                
        except Exception as e:
            logger.error(f"Decoding failed for {self.model_type}: {e}")
            raise

    @torch.inference_mode()
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            *args: Positional arguments for generation
            **kwargs: Keyword arguments for generation
            
        Returns:
            Generated token IDs
        """
        try:
            if self.model_type == "minicpm":
                return self.model.chat(*args, tokenizer=self.processor._tokenizer)
            return self.model.generate(**kwargs)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    @contextmanager
    def memory_efficient_mode(self):
        """Context manager for memory-efficient inference."""
        original_grad_state = torch.is_grad_enabled()
        try:
            torch.set_grad_enabled(False)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            yield
        finally:
            torch.set_grad_enabled(original_grad_state)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __del__(self):
        """Cleanup resources."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
