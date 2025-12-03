"""
Simplified VLM Engine using AutoModelForImageTextToText.

This module provides a clean, simple interface for VLMs that use the
standard HuggingFace AutoModelForImageTextToText class. It's designed
for models that follow the standard HF vision-language pipeline.

Supported models include:
- Idefics (HuggingFaceM4/Idefics*)
- SmolVLM (HuggingFaceTB/SmolVLM*)
- InternVL HF-native (OpenGVLab/InternVL*-HF)
- LLaVA (llava-hf/*)
- And other AutoModelForImageTextToText compatible models
"""

import logging
import re
import torch
from typing import Any, Dict, List, Optional, Union

from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)

from .base import (
    BaseVLM,
    ModelConfig,
)

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Model Registry
# ============================================================================

# Models that are known to work with AutoModelForImageTextToText
SUPPORTED_MODEL_PATTERNS = {
    "idefics": r"HuggingFaceM4/Idefics",
    "smolvlm": r"HuggingFaceTB/SmolVLM",
    "internvl_hf": r"OpenGVLab/InternVL.*-HF",
    "llava": r"llava-hf/",
    "mllama": r"meta-llama/Llama.*Vision",
}

# Model-specific configurations
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "idefics": ModelConfig(
        model_class=AutoModelForImageTextToText,
        processor_class=AutoProcessor,
        supports_flash_attention=True,
        processor_args={"use_fast": True},
    ),
    "smolvlm": ModelConfig(
        model_class=AutoModelForImageTextToText,
        processor_class=AutoProcessor,
        supports_flash_attention=True,
        padding_side="left",
        processor_args={"use_fast": True, "padding_side": "left"},
    ),
    "internvl_hf": ModelConfig(
        model_class=AutoModelForImageTextToText,
        processor_class=AutoProcessor,
        requires_trust_remote_code=True,
        supports_flash_attention=True,
        processor_args={"use_fast": True},
    ),
    "llava": ModelConfig(
        model_class=AutoModelForImageTextToText,
        processor_class=AutoProcessor,
        special_args={"low_cpu_mem_usage": True},
        processor_args={"use_fast": True},
    ),
    "mllama": ModelConfig(
        model_class=AutoModelForImageTextToText,
        processor_class=AutoProcessor,
        padding_side="left",
        processor_args={"use_fast": True, "padding_side": "left"},
    ),
    # Default configuration for unknown models
    "default": ModelConfig(
        model_class=AutoModelForImageTextToText,
        processor_class=AutoProcessor,
        processor_args={"use_fast": True},
    ),
}


# ============================================================================
# VLM Engine
# ============================================================================

class VLMEngine(BaseVLM):
    """
    Simplified VLM engine using AutoModelForImageTextToText.
    
    This class provides a clean, unified interface for vision-language models
    that use the standard HuggingFace AutoModelForImageTextToText architecture.
    
    Key Features:
    - Automatic model type detection
    - Lazy model loading
    - Batch processing support
    - Memory-efficient inference mode
    - Simple preprocess -> generate -> decode pipeline
    
    Example:
        >>> engine = VLMEngine("HuggingFaceM4/Idefics3-8B-Llama3")
        >>> 
        >>> # Prepare conversation
        >>> conversation = [{
        ...     "role": "user",
        ...     "content": [
        ...         {"type": "image", "image": image},
        ...         {"type": "text", "text": "What is in this image?"}
        ...     ]
        ... }]
        >>> 
        >>> # Run inference
        >>> inputs = engine.preprocess([conversation], [image])
        >>> outputs = engine.generate(inputs, max_new_tokens=256)
        >>> responses = engine.decode(outputs)
    """
    
    def __init__(
        self, 
        model_id: str, 
        device_map: str = "auto", 
        dtype: torch.dtype = torch.bfloat16
    ):
        """
        Initialize the VLM Engine.
        
        Args:
            model_id: HuggingFace model identifier
            device_map: Device mapping strategy ("auto", "cuda:0", "cpu")
            dtype: Model precision (torch.bfloat16, torch.float16, etc.)
        """
        super().__init__(model_id, device_map, dtype)
        
        # Detect model type and get configuration
        self.model_type = self._detect_model_type(model_id)
        self.config = MODEL_CONFIGS.get(self.model_type, MODEL_CONFIGS["default"])
        
        logger.info(f"VLMEngine initialized for {self.model_type} model: {model_id}")
    
    def _detect_model_type(self, model_id: str) -> str:
        """
        Detect model type from model ID.
        
        Args:
            model_id: HuggingFace model identifier
            
        Returns:
            Model type string
        """
        for model_type, pattern in SUPPORTED_MODEL_PATTERNS.items():
            if re.search(pattern, model_id):
                return model_type
        
        logger.warning(
            f"Unknown model type for {model_id}, using default configuration. "
            f"Supported patterns: {list(SUPPORTED_MODEL_PATTERNS.keys())}"
        )
        return "default"
    
    def _load_model(self) -> Any:
        """
        Load the model with optimized settings.
        
        Returns:
            Loaded and configured model
        """
        try:
            model_args = {
                "torch_dtype": self.dtype,
                "device_map": self.device_map,
            }
            
            # Add trust_remote_code if required
            if self.config.requires_trust_remote_code:
                model_args["trust_remote_code"] = True
            
            # Enable Flash Attention 2 if supported and CUDA is available
            if self.config.supports_flash_attention and torch.cuda.is_available():
                model_args["attn_implementation"] = "flash_attention_2"
            
            # Add any model-specific arguments
            model_args.update(self.config.special_args)
            
            logger.info(f"Loading model {self.model_id} with args: {model_args}")
            
            model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, 
                **model_args
            )
            
            return model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_id}: {e}")
            raise
    
    def _load_processor(self) -> Any:
        """
        Load the processor with optimized settings.
        
        Returns:
            Loaded and configured processor
        """
        try:
            processor_args = {}
            
            if self.config.requires_trust_remote_code:
                processor_args["trust_remote_code"] = True
            
            processor_args.update(self.config.processor_args)
            
            logger.info(f"Loading processor for {self.model_id}")
            
            processor = AutoProcessor.from_pretrained(
                self.model_id, 
                **processor_args
            )
            
            # Configure padding
            self._configure_padding(processor)
            
            return processor
            
        except Exception as e:
            logger.error(f"Failed to load processor for {self.model_id}: {e}")
            raise
    
    def _configure_padding(self, processor: Any) -> None:
        """
        Configure padding settings for the processor.
        
        Args:
            processor: The loaded processor
        """
        # Set padding side
        if hasattr(processor, "tokenizer"):
            processor.tokenizer.padding_side = self.config.padding_side
            
            # Ensure pad_token_id exists
            if getattr(processor.tokenizer, "pad_token_id", None) is None:
                eos_id = getattr(processor.tokenizer, "eos_token_id", None)
                if eos_id is not None:
                    processor.tokenizer.pad_token_id = eos_id
        
        # Sync pad token with model generation config
        if hasattr(processor, "tokenizer") and hasattr(self.model, "generation_config"):
            pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
            if pad_token_id is not None:
                self.model.generation_config.pad_token_id = pad_token_id
    
    def preprocess(
        self, 
        conversation: List[List[Dict[str, Any]]], 
        image_input: Optional[Union[Image.Image, List[Image.Image]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess inputs for the model.
        
        Args:
            conversation: List of conversations, each being a list of turns.
                         Each turn is a dict with "role" and "content" keys.
            image_input: Optional list of images corresponding to conversations
            
        Returns:
            Dictionary of preprocessed tensors ready for model input
            
        Example:
            >>> conversation = [[{
            ...     "role": "user",
            ...     "content": [
            ...         {"type": "image", "image": img},
            ...         {"type": "text", "text": "Describe this image"}
            ...     ]
            ... }]]
            >>> inputs = engine.preprocess(conversation, [img])
        """
        try:
            # Apply chat template to each conversation
            prompts = [
                self.processor.apply_chat_template(
                    conv, 
                    add_generation_prompt=True,
                    tokenize=False
                )
                for conv in conversation
            ]
            
            # Prepare images - wrap each image in a list for processors that expect it
            if image_input is not None:
                images_to_process = [
                    [img] if not isinstance(img, list) else img 
                    for img in image_input
                ]
            else:
                images_to_process = None
            
            # Validate input lengths
            if images_to_process is not None:
                assert len(prompts) == len(images_to_process), \
                    f"Number of prompts ({len(prompts)}) must match images ({len(images_to_process)})"
            
            # Process inputs
            inputs = self.processor(
                text=prompts,
                images=images_to_process,
                return_tensors="pt",
                padding=True,
            ).to(self.device, dtype=self.dtype)
            
            return inputs
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}", exc_info=True)
            raise
    
    @torch.inference_mode()
    def generate(
        self, 
        inputs: Dict[str, torch.Tensor], 
        **generation_kwargs
    ) -> torch.Tensor:
        """
        Generate text from preprocessed inputs.
        
        Args:
            inputs: Preprocessed inputs from preprocess()
            **generation_kwargs: Additional generation parameters
                - max_new_tokens: Maximum tokens to generate (default: 1024)
                - do_sample: Whether to use sampling (default: False)
                - temperature: Sampling temperature
                - top_p: Nucleus sampling parameter
                - etc.
                
        Returns:
            Generated token IDs (prompt tokens removed)
        """
        try:
            # Set default generation parameters
            gen_params = {
                "max_new_tokens": 1024,
                "do_sample": False,
            }
            gen_params.update(generation_kwargs)
            
            # Generate sequences
            sequences = self.model.generate(
                **inputs,
                **gen_params
            )
            
            # Remove prompt tokens from generated sequences
            prompt_length = inputs["input_ids"].shape[-1]
            if sequences.dim() == 1:
                generated_ids = sequences[prompt_length:]
            else:
                generated_ids = sequences[:, prompt_length:]
            
            return generated_ids
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def decode(self, generated_ids: torch.Tensor) -> List[str]:
        """
        Decode generated token IDs to text.
        
        Args:
            generated_ids: Token IDs from generate()
            
        Returns:
            List of decoded text strings
        """
        try:
            return self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True 
            )
        except Exception as e:
            logger.error(f"Decoding failed: {e}")
            raise
    
    def __call__(
        self, 
        conversation: List[List[Dict[str, Any]]], 
        images: Optional[List[Image.Image]] = None,
        **generation_kwargs
    ) -> List[str]:
        """
        Convenience method for end-to-end inference.
        
        Args:
            conversation: List of conversations
            images: Optional list of images
            **generation_kwargs: Generation parameters
            
        Returns:
            List of generated responses
            
        Example:
            >>> responses = engine(conversations, images, max_new_tokens=512)
        """
        with self.memory_efficient_mode():
            inputs = self.preprocess(conversation, images)
            outputs = self.generate(inputs, **generation_kwargs)
            return self.decode(outputs)
    
    def __repr__(self) -> str:
        return (
            f"VLMEngine("
            f"model_id='{self.model_id}', "
            f"model_type='{self.model_type}', "
            f"dtype={self.dtype}, "
            f"device_map='{self.device_map}')"
        )


# ============================================================================
# Factory Function
# ============================================================================

def create_vlm_engine(
    model_id: str,
    device_map: str = "auto",
    dtype: torch.dtype = torch.bfloat16,
) -> VLMEngine:
    """
    Factory function to create a VLMEngine instance.
    
    Args:
        model_id: HuggingFace model identifier
        device_map: Device mapping strategy
        dtype: Model precision
        
    Returns:
        Configured VLMEngine instance
        
    Example:
        >>> engine = create_vlm_engine("HuggingFaceM4/Idefics3-8B-Llama3")
    """
    return VLMEngine(model_id, device_map, dtype)
