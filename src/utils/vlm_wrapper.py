import re
import torch
import requests
import logging
import gc
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from functools import lru_cache
from contextlib import contextmanager
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    AutoModelForCausalLM,
    GenerationConfig,
    AutoModelForVision2Seq,
    MllamaForConditionalGeneration,
    AutoModel,
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Glm4vForConditionalGeneration
)
from qwen_vl_utils import process_vision_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for InternVL
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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
    inference_type: str = "standard"  # standard, internvl, minicpm
    
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
        processor_args={"use_fast": True, "padding_side": "left"},
    ),
    "kimi": ModelConfig(
        model_class=AutoModelForCausalLM,
        processor_class=AutoProcessor,
        requires_trust_remote_code=True,
        supports_flash_attention=True,
        processor_args={"use_fast": True, "padding_side": "left"},
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
    "minicpm": ModelConfig(
        model_class=AutoModel,
        processor_class=AutoTokenizer,
        requires_trust_remote_code=True,
        supports_flash_attention=True,
        inference_type="minicpm"
    ),
    "internvl": ModelConfig(
        model_class=AutoModel,
        processor_class=AutoTokenizer,
        requires_trust_remote_code=True,
        supports_flash_attention=True,
        inference_type="internvl"
    ),
    "gemma3": ModelConfig(
        model_class=Gemma3ForConditionalGeneration,
        processor_class=AutoProcessor,
        supports_flash_attention=True,
        processor_args={"use_fast": True, "padding_side": "left"},
    ),
    "glm4v": ModelConfig(
        model_class=Glm4vForConditionalGeneration,
        processor_class=AutoProcessor,
        requires_trust_remote_code=True,
        supports_flash_attention=True,
        padding_side="left",
        processor_args={"use_fast": True}
    ),
}

class VLMWrapper:
    """Unified Vision-Language Model wrapper supporting all model types."""
    
    def __init__(self, model_id: str, device_map: str = "auto", dtype: torch.dtype = torch.bfloat16):
        """
        Initialize unified VLM wrapper with automatic model type detection.
        
        Args:
            model_id: HuggingFace model identifier
            device_map: Device mapping strategy
            dtype: Model precision
        """
        self.model_id = model_id
        self.model_type = self._detect_model_type(model_id)
        self.config = MODEL_CONFIGS[self.model_type]
        self.dtype = dtype
        self.device_map = self._optimize_device_map(device_map)
        
        # Lazy initialization
        self._model = None
        self._processor = None
        self._device = None
        self._transform = None  # For InternVL
        
        logger.info(f"Initialized VLMWrapper: for {self.model_type} model: {model_id}")

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

    @property
    def transform(self):
        """Lazy load image transform for InternVL."""
        if self._transform is None and self.model_type == "internvl":
            self._transform = self._build_transform(input_size=448)
        return self._transform

    def _optimize_device_map(self, device_map: str) -> str:
        """Optimize device mapping based on available hardware."""
        if device_map == "auto":
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
            "idefics": r"HuggingFaceM4/Idefics",
            "smolvlm": r"HuggingFaceTB/SmolVLM",
            "mllama": r"meta-llama",
            "minicpm": r"openbmb/MiniCPM",
            "internvl": r"OpenGVLab/InternVL",
            "gemma3": r"google/gemma-3",
            "kimi": r"moonshotai/Kimi-VL",
            "glm4v": r"zai-org/GLM-4",
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
                if self.model_type == "internvl":
                    model_args["use_flash_attn"] = True
                else:
                    model_args["attn_implementation"] = "flash_attention_2"
                
            # Add special arguments
            model_args.update(self.config.special_args)
            
            model = self.config.model_class.from_pretrained(self.model_id, **model_args)
            # model = torch.compile(model, mode="default")  # Compile model for performance
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

    def _build_transform(self, input_size: int = 448) -> T.Compose:
        """Create optimized image transform pipeline for InternVL."""
        return T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

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

    def preprocess(self, conversation: List, image_input: Optional[Union[Image.Image, List[Image.Image]]] = None) -> Any:
        # Unified preprocessing for all inference types
        return self._preprocess_standard(conversation, image_input)

    def _preprocess_standard(self, conversation: List, image_input: Optional[Union[Image.Image, List[Image.Image]]] = None) -> Any:
        """Unified preprocessing for all VLM models, including internvl and minicpm."""
        # dispatch based on inference_type
        if self.config.inference_type == "internvl":
            return self._preprocess_internvl(conversation, image_input)
        if self.config.inference_type == "minicpm":
            return self._preprocess_minicpm(conversation, image_input)
        # default for standard models
        batch_conversations, batch_images = self._normalize_inputs(conversation, image_input)
        try:
            if self.model_type == "qwen":
                return self._preprocess_qwen(batch_conversations)
            elif self.model_type == "gemma3":
                return self._preprocess_gemma3(batch_conversations, batch_images)
            elif self.model_type == "kimi":
                return self._preprocess_kimi(batch_conversations, batch_images)
            elif self.model_type == "mllama":
                return self._preprocess_mllama(batch_conversations, batch_images)
            elif self.model_type in ["llava", "llava_next"]:
                return self._preprocess_llava(batch_conversations, batch_images)
            elif self.model_type == "glm4v":
                return self._preprocess_glm4v(batch_conversations, batch_images)
            elif self.model_type in ["idefics", "smolvlm"]:
                return self._preprocess_idefics_smolvlm(batch_conversations, batch_images)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        except Exception as e:
            logger.error(f"Preprocessing failed for {self.model_type}: {e}", exc_info=True)
            raise
        
        # Standard preprocessing methods (similar to original vlm_helpers.py)
    def _preprocess_qwen(self, batch_conversations: List[List]) -> Dict[str, torch.Tensor]:
        """Preprocess for Qwen models."""
        prompts = [
            self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            for conv in batch_conversations
        ]
        image_inputs, _ = process_vision_info(batch_conversations)
        inputs = self.processor(text=prompts, images=image_inputs, padding=True, return_tensors="pt").to(self.device, dtype=self.dtype)
        return inputs
    

    def _preprocess_mllama(self, batch_conversations: List[List], batch_images: List[Image.Image]) -> Dict[str, torch.Tensor]:
        """Preprocess for Mllama models."""
        prompts = []
        all_images = []
        
        for i, conv in enumerate(batch_conversations):
            prompt = self.processor.apply_chat_template(conv, add_generation_prompt=True)
            prompts.append(prompt)
            
            image_token_count = prompt.count('<|image|>')
            
            if batch_images and i < len(batch_images):
                current_image = batch_images[i]
                if image_token_count > 0:
                    all_images.extend([current_image] * image_token_count)
        
        if all_images:
            inputs = self.processor(
                all_images, prompts, add_special_tokens=False,
                return_tensors="pt", padding=True, truncation=True
            )
        else:
            inputs = self.processor(
                None, prompts, add_special_tokens=False,
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


    def _preprocess_idefics_smolvlm(self, batch_conversations: List[List], batch_images: List[Image.Image]) -> Dict[str, torch.Tensor]:
        """Preprocess for Idefics and SmolVLM models."""
        if batch_images and not isinstance(batch_images, list):
            batch_images = [batch_images] * len(batch_conversations)
        
        processed_inputs = []
        for i, conv in enumerate(batch_conversations):
            try:
                prompt = self.processor.apply_chat_template(conv, add_generation_prompt=True)
                image_token_count = prompt.count('<image>')
                
                if batch_images and i < len(batch_images):
                    current_image = batch_images[i]
                    if image_token_count > 1:
                        current_images = [current_image] * image_token_count
                    else:
                        current_images = [current_image] if image_token_count > 0 else None
                else:
                    current_images = None
                
                if current_images:
                    inputs = self.processor(
                        text=[prompt], 
                        images=current_images,
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True
                    )
                else:
                    inputs = self.processor(
                        text=[prompt],
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True
                    )
                
                processed_inputs.append(inputs)
                
            except Exception as e:
                logger.error(f"Error preprocessing conversation {i}: {e}")
                prompt = self.processor.apply_chat_template(conv, add_generation_prompt=True)
                inputs = self.processor(
                    text=[prompt],
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True
                )
                processed_inputs.append(inputs)
        
        if len(processed_inputs) == 1:
            return {k: v.to(self.device) for k, v in processed_inputs[0].items()}
        else:
            batched_inputs = {}
            for key in processed_inputs[0].keys():
                try:
                    batched_inputs[key] = torch.cat([inp[key] for inp in processed_inputs], dim=0)
                except Exception as e:
                    logger.warning(f"Could not batch {key}: {e}")
                    batched_inputs[key] = processed_inputs[0][key]
            
            return {k: v.to(self.device) for k, v in batched_inputs.items()}
        
    def _preprocess_glm4v(self, batch_conversations: List[List], batch_images: Optional[Union[Image.Image, List[Image.Image]]]) -> Dict[str, torch.Tensor]:
        """
        Preprocess for GLM-4V models using apply_chat_template.
        This method relies on a recent version of transformers that can handle
        image data (e.g., URLs) directly within the chat template.
        """
        prompts = [
            self.processor.apply_chat_template(
                conv,
                tokenize=True,
                add_generation_prompt=True
            )
            for conv in batch_conversations
        ]
        
        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device, dtype=self.dtype)
        
        return inputs
    
    def _preprocess_kimi(self, batch_conversations: List[List], batch_images: List[Image.Image]) -> Dict[str, torch.Tensor]:
        """Preprocess for Kimi models."""
        
        # Kimi's processor can handle multiple images per conversation
        # This implementation assumes that if multiple images are provided,
        # they all belong to the single conversation in the batch.
        if len(batch_conversations) > 1 and len(batch_images) > 1:
            logger.warning("Kimi wrapper handles multiple images for a single conversation, not for a batch of conversations. Processing one image per conversation.")
            images_to_process = [[img] for img in batch_images]
        else:
            images_to_process = [batch_images] * len(batch_conversations)

        processed_inputs_batch = []
        for i, conv in enumerate(batch_conversations):
            current_images = images_to_process[i]
            
            # 1. Construct the 'messages' list for the chat template
            messages = []
            for turn in conv:
                content_list = []
                # Kimi expects image content first
                if current_images:
                    content_list.extend([{"type": "image"}] * len(current_images))
                
                text_content = " ".join([item['text'] for item in turn['content'] if 'text' in item])
                content_list.append({"type": "text", "text": text_content})
                
                messages.append({"role": turn['role'], "content": content_list})

            # 2. First processing step: apply chat template
            # Note: The Kimi processor expects the tokenized output as input for the next step
            text_inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )

            # 3. Second processing step: combine images and tokenized text
            final_inputs = self.processor(
                images=current_images, text=text_inputs, return_tensors="pt", padding=True, truncation=True
            )
            processed_inputs_batch.append(final_inputs)

        # 4. Batch the results from each conversation
        if len(processed_inputs_batch) == 1:
            return {k: v.to(self.device) for k, v in processed_inputs_batch[0].items()}
        else:
            batched_inputs = {}
            for key in processed_inputs_batch[0].keys():
                batched_inputs[key] = torch.cat([inp[key] for inp in processed_inputs_batch], dim=0)
            return {k: v.to(self.device) for k, v in batched_inputs.items()}
    
    def _preprocess_gemma3(self, batch_conversations: List[List], batch_images: List[Image.Image]) -> Dict[str, torch.Tensor]:
        
        prompts = [
            self.processor.apply_chat_template(
                conv,
                add_generation_prompt=True,
            ) for conv in batch_conversations
        ]
        
        inputs = self.processor(
            images=batch_images,
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
            
        return inputs.to(self.device, dtype=self.dtype)

    def _preprocess_internvl(self, conversation: List, image_input: Optional[Union[Image.Image, List[Image.Image]]] = None) -> Dict[str, Any]:
        """
        A more robust preprocessing function for InternVL models.

        It handles single or batch conversations, safely parses content, and ensures
        correct dtype conversion based on the model's configuration.
        """
        # 1. Simplify batch handling: always work with a list of conversations
        conversation_list = conversation if isinstance(conversation[0], list) else [conversation]

        questions = []
        all_pixel_values = []
        num_patches_list = []

        # 2. Robustly parse each conversation
        for conv in conversation_list:
            # Safely find the image and text, ignoring their order
            image = None
            text_parts = []
            for item in conv[0]['content']:
                if 'image' in item:
                    image = item['image']
                elif 'text' in item:
                    text_parts.append(item['text'])
            
            if image is None:
                raise ValueError("No image found in conversation content.")

            question = " ".join(text_parts).strip()
            questions.append(question)
            
            # Process the image
            pixel_values = self._load_image_internvl(image, max_num=12)
            all_pixel_values.append(pixel_values)
            num_patches_list.append(pixel_values.size(0))

        # Concatenate all pixel values into a single tensor
        if all_pixel_values:
            pixel_values_tensor = torch.cat(all_pixel_values, dim=0)
        else:
            # Create an empty tensor with the correct number of dimensions
            pixel_values_tensor = torch.empty(0, 3, 448, 448) 

        # 3. Correctly move to device with the model's dtype
        # This prevents dtype mismatches regardless of the device (CPU, CUDA, MPS)
        model_dtype = self.model.dtype if hasattr(self, 'model') else torch.bfloat16
        pixel_values_tensor = pixel_values_tensor.to(device=self.device, dtype=model_dtype)

        return {
            "pixel_values": pixel_values_tensor,
            "questions": questions,
            "num_patches_list": num_patches_list,
        
        }

    def _preprocess_minicpm(self, conversation: List, image_input: Optional[Union[Image.Image, List[Image.Image]]] = None) -> Tuple:
        """Preprocessing for MiniCPM models."""
        if isinstance(conversation[0], list):
            # Batch format
            msgs_batch = []
            for conv in conversation:
                # Convert to MiniCPM format
                question = conv[0]["content"][1]["text"]
                image = conv[0]["content"][0]["image"]
                
                # Prepare image as numpy array
                np_img = self._prepare_image_minicpm(image)
                msgs = [{"role": "user", "content": [np_img, question]}]
                msgs_batch.append(msgs)
            return msgs_batch
        else:
            # Single conversation
            question = conversation[0]["content"][1]["text"]
            image = conversation[0]["content"][0]["image"]
            np_img = self._prepare_image_minicpm(image)
            msgs = [{"role": "user", "content": [np_img, question]}]
            return msgs

    def _load_image_internvl(self, image_input: Any, input_size: int = 448, max_num: int = 12) -> torch.Tensor:
        """Load and preprocess image for InternVL with dynamic preprocessing."""
        try:
            # Handle different image input types
            if isinstance(image_input, str):
                image = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, dict) and "path" in image_input:
                image = Image.open(image_input["path"]).convert("RGB")
            elif isinstance(image_input, Image.Image):
                image = image_input.convert("RGB")
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
            
            # Dynamic preprocessing
            images = self._dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
            
            # Apply transforms
            pixel_values = [self.transform(img) for img in images]
            return torch.stack(pixel_values)
            
        except Exception as e:
            logger.error(f"Failed to load image for InternVL: {e}")
            raise

    def _prepare_image_minicpm(self, image_input: Union[str, Image.Image, Dict]) -> np.ndarray:
        """Prepare image for MiniCPM processing."""
        try:
            # Handle different image input types
            if isinstance(image_input, str):
                image = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, dict) and "path" in image_input:
                image = Image.open(image_input["path"]).convert("RGB")
            elif isinstance(image_input, Image.Image):
                image = image_input.convert("RGB")
            else:
                image = image_input
            
            # Convert to numpy array with channel-first format (C x H x W)
            np_img = np.array(image)
            if len(np_img.shape) == 3:
                np_img = np_img.transpose(2, 0, 1)
            
            return np_img
            
        except Exception as e:
            logger.error(f"Failed to prepare image for MiniCPM: {e}")
            raise

    def _dynamic_preprocess(self, image: Image.Image, min_num: int = 1, max_num: int = 12, 
                           image_size: int = 448, use_thumbnail: bool = False) -> List[Image.Image]:
        """Dynamic preprocessing for InternVL."""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        
        # Generate target ratios
        target_ratios = [
            (i, j) for n in range(min_num, max_num + 1)
            for i in range(1, n + 1) for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        ]
        target_ratios.sort(key=lambda x: x[0] * x[1])
        
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )
        
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        
        # Resize image
        resized_img = image.resize((target_width, target_height))
        
        # Extract blocks
        processed_images = []
        cols = target_width // image_size
        
        for i in range(blocks):
            col = i % cols
            row = i // cols
            box = (
                col * image_size,
                row * image_size,
                (col + 1) * image_size,
                (row + 1) * image_size,
            )
            processed_images.append(resized_img.crop(box))
        
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
            
        return processed_images

    def _find_closest_aspect_ratio(self, aspect_ratio: float, target_ratios: List[Tuple[int, int]], 
                                  width: int, height: int, image_size: int) -> Tuple[int, int]:
        """Find the best matching target aspect ratio for InternVL."""
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif (ratio_diff == best_ratio_diff and 
                  area > 0.5 * image_size * image_size * ratio[0] * ratio[1]):
                best_ratio = ratio
                
        return best_ratio

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


    def decode(self, generated_ids: torch.Tensor) -> List[str]:
        """
        Decode generated token IDs to text.
        Assumes that any necessary slicing of the input tensor (e.g., removing input prompt tokens)
        has been performed before calling this function.
        
        Args:
            generated_ids: Generated token IDs from the model.
            
        Returns:
            List of decoded text strings.
        """
        try:
            # Handle models with truly unique decoding logic first
            if self.model_type == "minicpm":
                # This model type seems to expect a list of characters, not token IDs
                return ["".join(generated_ids)]
                
            if self.model_type == "internvl": 
                # Per the original comment, this is likely a placeholder
                logger.warning("decode() called for 'internvl', which might not be the intended path.")
                return [str(generated_ids)]
            
            if self.model_type == "gemma3":
                # Gemma 3 uses a different decoding method
                decoded_text = [
                    self.processor.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    for generated_id in generated_ids
                ]
                
                return decoded_text
            
            return self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
        except Exception as e:
            logger.error(f"Decoding failed for {self.model_type}: {e}")
            raise
    
    
    @torch.inference_mode()
    def generate(self, inputs: Any, **generation_kwargs) -> Any:
        """
        Generate text using the model with unified interface.
        
        Args:
            inputs: Preprocessed inputs
            **generation_kwargs: Generation parameters
            
        Returns:
            Generated outputs (format depends on model type)
        """
        try:
            if self.config.inference_type == "internvl":
                return self._generate_internvl(inputs, **generation_kwargs)
            elif self.config.inference_type == "minicpm":
                return self._generate_minicpm(inputs, **generation_kwargs)
            else:
                generated_ids = self.model.generate(**inputs, **generation_kwargs)
                
                logger.info(f"Inputs: {inputs}, type: {type(inputs)}, length: {len(inputs)}")
                logger.info(f"Generated IDs: {generated_ids}, type: {type(generated_ids)}, length: {len(generated_ids)}")
                
                return generated_ids
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
        
    def _generate_internvl(self, inputs: Dict[str, Any], **generation_kwargs) -> List[str]:
        """Generate using InternVL's batch_chat interface."""
        pixel_values = inputs["pixel_values"]
        questions = inputs["questions"]
        num_patches_list = inputs["num_patches_list"]
                
        # Default generation config
        generation_config = {
            "max_new_tokens": 128,
            "do_sample": False,
            "pad_token_id": self.processor.pad_token_id,
        }
        generation_config.update(generation_kwargs)
        
        responses = self.model.batch_chat(
            self.processor,
            pixel_values,
            num_patches_list=num_patches_list,
            questions=questions,
            generation_config=generation_config,
        )
    
        return responses

    def _generate_minicpm(self, msgs: Any, **generation_kwargs) -> List[str]:
        """Generate using MiniCPM's chat interface."""
        # Default generation config
        generation_config = {
            "max_tokens": 128,
            "do_sample": False,
        }
        generation_config.update(generation_kwargs)
        
        if isinstance(msgs, list) and isinstance(msgs[0], list):
            # Batch processing
            responses = []
            for msg in msgs:
                try:
                    response = self.model.chat(
                        image=None, 
                        msgs=msg, 
                        tokenizer=self.processor,
                        **generation_config
                    )
                    responses.append(response)
                except Exception as e:
                    logger.warning(f"Individual MiniCPM inference failed: {e}")
                    responses.append(f"ERROR: {str(e)}")
            return responses
        else:
            # Single processing
            return self.model.chat(
                image=None, 
                msgs=msgs, 
                tokenizer=self.processor,
                **generation_config
            )

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
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# Alias for backward compatibility
VLM = VLMWrapper