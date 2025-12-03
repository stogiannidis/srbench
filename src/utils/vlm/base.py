"""
Base abstractions for Vision-Language Models.

This module provides the foundational components for VLM implementations:
- Protocol definitions for type safety
- Configuration dataclasses
- Common utilities and constants
"""

import logging
import re
import torch
import requests
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import (
    Any, 
    Dict, 
    List, 
    Optional, 
    Protocol, 
    Tuple, 
    Type,
    Union,
    runtime_checkable,
)

from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

# ImageNet normalization constants (used by some models like InternVL)
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Default generation parameters
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_DO_SAMPLE = False


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """
    Configuration for VLM models.
    
    Attributes:
        model_class: The HuggingFace model class to use
        processor_class: The HuggingFace processor/tokenizer class
        requires_trust_remote_code: Whether to trust remote code
        supports_flash_attention: Whether the model supports Flash Attention 2
        padding_side: Token padding side ("left" or "right")
        special_args: Additional arguments for model loading
        processor_args: Additional arguments for processor loading
        inference_type: Type of inference ("standard", "internvl", "minicpm")
    """
    model_class: Type
    processor_class: Type
    requires_trust_remote_code: bool = False
    supports_flash_attention: bool = False
    padding_side: str = "left"
    special_args: Dict[str, Any] = field(default_factory=dict)
    processor_args: Dict[str, Any] = field(default_factory=dict)
    inference_type: str = "standard"


@dataclass
class GenerationConfig:
    """
    Configuration for text generation.
    
    Attributes:
        max_new_tokens: Maximum number of tokens to generate
        do_sample: Whether to use sampling
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        num_beams: Number of beams for beam search
        repetition_penalty: Repetition penalty factor
    """
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    do_sample: bool = DEFAULT_DO_SAMPLE
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    num_beams: int = 1
    repetition_penalty: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for model.generate() kwargs."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature if self.do_sample else None,
            "top_p": self.top_p if self.do_sample else None,
            "top_k": self.top_k if self.do_sample else None,
            "num_beams": self.num_beams,
            "repetition_penalty": self.repetition_penalty,
        }


# ============================================================================
# Protocols (Interface Definitions)
# ============================================================================

@runtime_checkable
class VLMProtocol(Protocol):
    """
    Protocol defining the interface for Vision-Language Models.
    
    Any VLM implementation should conform to this interface to ensure
    compatibility with the evaluation framework.
    """
    
    @property
    def model(self) -> Any:
        """Return the underlying model."""
        ...
    
    @property
    def processor(self) -> Any:
        """Return the processor/tokenizer."""
        ...
    
    @property
    def device(self) -> torch.device:
        """Return the device the model is on."""
        ...
    
    def preprocess(
        self, 
        conversation: List[Dict[str, Any]], 
        image_input: Optional[Union[Image.Image, List[Image.Image]]] = None
    ) -> Any:
        """Preprocess inputs for the model."""
        ...
    
    def generate(self, inputs: Any, **kwargs) -> Any:
        """Generate text from preprocessed inputs."""
        ...
    
    def decode(self, generated_ids: Any) -> List[str]:
        """Decode generated token IDs to text."""
        ...


# ============================================================================
# Base Classes
# ============================================================================

class BaseVLM(ABC):
    """
    Abstract base class for Vision-Language Models.
    
    Provides common functionality and enforces the VLM interface.
    Subclasses must implement the abstract methods.
    """
    
    def __init__(
        self, 
        model_id: str, 
        device_map: str = "auto", 
        dtype: torch.dtype = torch.bfloat16
    ):
        """
        Initialize the VLM.
        
        Args:
            model_id: HuggingFace model identifier
            device_map: Device mapping strategy ("auto", "cuda:0", "cpu", etc.)
            dtype: Model precision (torch.bfloat16, torch.float16, etc.)
        """
        self.model_id = model_id
        self.dtype = dtype
        self.device_map = self._optimize_device_map(device_map)
        
        # Lazy initialization
        self._model = None
        self._processor = None
        self._device = None
        
        logger.info(f"Initialized {self.__class__.__name__} for model: {model_id}")
    
    # =========================================================================
    # Properties (Lazy Loading)
    # =========================================================================
    
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
    def device(self) -> torch.device:
        """Get model device."""
        if self._device is None:
            self._device = self.model.device
        return self._device
    
    # =========================================================================
    # Abstract Methods (Must be implemented by subclasses)
    # =========================================================================
    
    @abstractmethod
    def _load_model(self) -> Any:
        """Load the model. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _load_processor(self) -> Any:
        """Load the processor. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def preprocess(
        self, 
        conversation: List[Dict[str, Any]], 
        image_input: Optional[Union[Image.Image, List[Image.Image]]] = None
    ) -> Any:
        """Preprocess inputs for the model."""
        pass
    
    @abstractmethod
    def generate(self, inputs: Any, **kwargs) -> Any:
        """Generate text from preprocessed inputs."""
        pass
    
    @abstractmethod
    def decode(self, generated_ids: Any) -> List[str]:
        """Decode generated token IDs to text."""
        pass
    
    # =========================================================================
    # Common Utilities
    # =========================================================================
    
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
    
    @staticmethod
    @lru_cache(maxsize=128)
    def load_image_from_url(image_url: str) -> Image.Image:
        """
        Load image from URL with caching.
        
        Args:
            image_url: URL of the image to load
            
        Returns:
            PIL Image object
        """
        try:
            response = requests.get(image_url, stream=True, timeout=10)
            response.raise_for_status()
            return Image.open(response.raw)
        except Exception as e:
            logger.error(f"Failed to load image from {image_url}: {e}")
            raise
    
    @staticmethod
    def ensure_rgb(image: Image.Image) -> Image.Image:
        """Ensure image is in RGB mode."""
        if image.mode != "RGB":
            return image.convert("RGB")
        return image
    
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


# ============================================================================
# Utility Functions
# ============================================================================

def detect_model_family(model_id: str) -> str:
    """
    Detect the model family from the model ID.
    
    Args:
        model_id: HuggingFace model identifier
        
    Returns:
        Model family string (e.g., "qwen", "llava", "idefics")
        
    Raises:
        ValueError: If model family cannot be determined
    """
    # Pattern matching for model families (ordered by specificity)
    model_patterns = {
        "internvl_hf": r"OpenGVLab/InternVL.*-HF",
        "qwen": r"Qwen/",
        "llava_1.5": r"llava-hf/llava-1\.5",
        "llava_next": r"llava-hf/llava-v1\.6",
        "idefics": r"HuggingFaceM4/Idefics",
        "smolvlm": r"HuggingFaceTB/SmolVLM",
        "mllama": r"meta-llama",
        "minicpm": r"openbmb/MiniCPM",
        "gemma3": r"google/gemma-3",
        "kimi": r"moonshotai/Kimi-VL",
        "glm4v": r"zai-org/GLM-4",
    }
    
    for family, pattern in model_patterns.items():
        if re.search(pattern, model_id):
            return family
    
    raise ValueError(f"Could not detect model family for: {model_id}")


def get_available_device() -> str:
    """Get the best available device for inference."""
    if torch.cuda.is_available():
        return "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
