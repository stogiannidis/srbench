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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
        processor_args={"use_fast": True},
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
        processor_args={"use_fast": True},
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
                # Ensure padding_side
                processor.tokenizer.padding_side = self.config.padding_side
                # Ensure pad_token_id exists; fallback to eos if missing
                if getattr(processor.tokenizer, "pad_token_id", None) is None:
                    eos_id = getattr(processor.tokenizer, "eos_token_id", None)
                    if eos_id is not None:
                        processor.tokenizer.pad_token_id = eos_id
            
            # Set pad token on model generation config if available
            if hasattr(processor, "tokenizer") and hasattr(self.model, "generation_config"):
                if getattr(self.model.generation_config, "pad_token_id", None) is None:
                    self.model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
                else:
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
    

    def _preprocess_standard(self, batch_conversations: List, batch_images: Optional[Union[Image.Image, List[Image.Image]]] = None) -> Any:
        """Unified preprocessing for all VLM models, including internvl and minicpm."""
        if self.config.inference_type == "internvl":
            return self._preprocess_internvl(batch_conversations, batch_images)
        if self.config.inference_type == "minicpm":
            return self._preprocess_minicpm(batch_conversations, batch_images)
        # batch_conversations, batch_images = self._normalize_inputs(conversation, image_input)
        try:
            return self._preprocess(batch_conversations, batch_images)
    
        except Exception as e:
            logger.error(f"Preprocessing failed for {self.model_type}: {e}", exc_info=True)
            raise
        
    def _preprocess(self, batch_conversations: List[List], batch_images: Optional[Union[Image.Image, List[Image.Image]]]) -> Dict[str, torch.Tensor]:
        """Preprocess input data for the model.
        Args:
            batch_conversations: List of conversations, each a list of turns.
            batch_images: List of images corresponding to each conversation.
        Returns:
            Dictionary of preprocessed inputs ready for model inference.
        """

        # Apply chat template to each conversation
        prompts = [
            self.processor.apply_chat_template(
                conv, 
                add_generation_prompt=True,
                tokenize=False
            ) 
            for conv in batch_conversations
        ]
            
            
        if self.model_type in ["gemma3", "mllama"]:
            # For Gemma3, we need to ensure images are wrapped in a list
            images_to_process = [[img] for img in batch_images]
        else:
            images_to_process = batch_images
            
        assert len(prompts) == len(images_to_process), "Number of prompts must match number of image inputs"
        
        # Preprocess inputs
        inputs = self.processor(
            text=prompts,
            images=images_to_process,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        
    
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

    def _preprocess_internvl(self, conversation: List, image_input: Optional[Union[Image.Image, List[Image.Image]]] = None) -> Dict[str, Any]:
        """
        Robust preprocessing for InternVL models.
        - Supports batch or single conversations
        - Extracts the last user turn to avoid mixing one-shot examples
        - Builds a per-sample list of pixel tensors (later concatenated for batch_chat)
        - Ensures input dtype matches model parameters
        """
        # Normalize to list of conversations
        conversation_list = conversation if isinstance(conversation[0], list) else [conversation]
        
        questions: List[str] = []
        pixel_values_list: List[torch.Tensor] = []
        num_patches_list: List[int] = []
        
        # Determine model param dtype to avoid dtype mismatch (e.g., bf16 vs fp16)
        try:
            model_param_dtype = next(self.model.parameters()).dtype
        except Exception:
            model_param_dtype = self.dtype
        
        for conv in conversation_list:
            # Find last user turn
            user_turns = [turn for turn in conv if isinstance(turn, dict) and turn.get("role") == "user"]
            if not user_turns:
                raise ValueError("No user turn found in conversation for InternVL")
            last_user = user_turns[-1]
            
            # Extract image and text
            image = None
            text_parts: List[str] = []
            for item in last_user.get('content', []):
                if isinstance(item, dict) and 'image' in item and image is None:
                    image = item['image']
                elif isinstance(item, dict) and 'text' in item:
                    text_parts.append(item['text'])
            
            if image is None and image_input is not None:
                # Fallback if image not embedded
                image = image_input[0] if isinstance(image_input, list) and image_input else image_input
            if image is None:
                raise ValueError("No image found in conversation content for InternVL")
            
            question = " ".join(text_parts).strip()
            questions.append(question)
            
            # Load and preprocess image into patches (N, 3, 448, 448)
            patches = self._load_image_internvl(image, max_num=12)
            # Match dtype/device to model parameters to avoid conv2d dtype mismatch
            patches = patches.to(device=self.device, dtype=model_param_dtype)
            pixel_values_list.append(patches)
            num_patches_list.append(patches.size(0))
        
        return {
            "pixel_values": pixel_values_list,  # keep per-sample; we will concat at generation
            "questions": questions,
            "num_patches_list": num_patches_list,
        }

    def _preprocess_minicpm(self, conversation: List, image_input: Optional[Union[Image.Image, List[Image.Image]]] = None) -> Tuple:
        """Preprocessing for MiniCPM models. Extract the last user turn, then find image/text by keys."""
        # Normalize to list of conversations
        conv_list = conversation if isinstance(conversation[0], list) else [conversation]
        msgs_batch: List[List[Dict[str, Any]]] = []
        for conv in conv_list:
            # Find last user turn (to avoid one-shot example turns)
            user_turns = [turn for turn in conv if isinstance(turn, dict) and turn.get("role") == "user"]
            if not user_turns:
                raise ValueError("No user turn found in conversation")
            last_user = user_turns[-1]
            # Extract image and text from the content list, regardless of order
            img = None
            text = ""
            for item in last_user.get("content", []):
                if isinstance(item, dict) and "image" in item and img is None:
                    img = item["image"]
                elif isinstance(item, dict) and "text" in item and not text:
                    text = item["text"]
            # Fallback to image_input if not embedded in conversation
            if img is None and image_input is not None:
                if isinstance(image_input, list) and len(image_input) > 0:
                    img = image_input[0]
                else:
                    img = image_input
            if img is None:
                # Mirror the KeyError seen in logs for clarity
                raise KeyError("image")
            # Prepare image and build msgs
            np_img = self._prepare_image_minicpm(img)
            msgs = [{"role": "user", "content": [np_img, text]}]
            msgs_batch.append(msgs)
        # Return batch or single
        return msgs_batch if len(msgs_batch) > 1 else msgs_batch[0]

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



    def decode(self, generated_ids: Any) -> List[str]:
        """
        Decode generated outputs.
        - For standard models: decode token IDs to text (slice per-sample new tokens using prompt lengths).
        - For internvl/minicpm: pass through strings/lists.
        """
        try:
            # Pass-through for models that already return strings
            if self.config.inference_type in ("internvl", "minicpm"):
                if isinstance(generated_ids, list):
                    return generated_ids
                if isinstance(generated_ids, str):
                    return [generated_ids]
                return [str(generated_ids)]
            
            return self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
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
                sequences = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )
            
            # Slice prompt tokens from generated sequences
            prompt_length = inputs["input_ids"].shape[-1]
            if sequences.dim() == 1:
                generated_ids = sequences[prompt_length:]
            else:
                generated_ids = sequences[:, prompt_length:]
            
            return generated_ids
        
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
        
    def _generate_internvl(self, inputs: Dict[str, Any], **generation_kwargs) -> List[str]:
        """Generate using InternVL's chat API per-sample to avoid batch misalignment issues."""
        pixel_values_in = inputs["pixel_values"]
        questions = inputs["questions"]
        num_patches_list = inputs["num_patches_list"]
        
        # Build per-sample pixel tensors list
        if isinstance(pixel_values_in, list):
            pv_list = pixel_values_in
        else:
            # Split concatenated tensor by patch counts
            pv_list = list(torch.split(pixel_values_in, num_patches_list, dim=0))
        
        if len(pv_list) != len(questions):
            raise ValueError(
                f"InternVL mismatch: {len(pv_list)} pixel groups vs {len(questions)} questions"
            )
        
        # Default generation config
        generation_config = {
            "max_new_tokens": t,
            "do_sample": False,
            "pad_token_id": getattr(self.processor, 'pad_token_id', None),
            "eos_token_id": getattr(self.processor, 'eos_token_id', getattr(self.model.generation_config, 'eos_token_id', None)),
        }
        generation_config.update(generation_kwargs)
        
        responses: List[str] = []
        for pv, q, n in zip(pv_list, questions, num_patches_list):
            # Sanity checks
            if pv.dim() != 4 or pv.size(0) != int(n):
                raise ValueError(
                    f"InternVL per-sample mismatch: pixel batch={pv.size(0)} vs n={n}, shape={tuple(pv.shape)}"
                )
            out = self.model.batch_chat(
                self.processor,
                pv,
                num_patches_list=[int(n)],
                questions=[q],
                generation_config=generation_config,
            )
            # Normalize to string
            if isinstance(out, list):
                if len(out) == 1 and isinstance(out[0], dict) and 'response' in out[0]:
                    responses.append(out[0]['response'])
                elif len(out) == 1 and isinstance(out[0], str):
                    responses.append(out[0])
                else:
                    responses.append(str(out))
            else:
                responses.append(str(out))
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