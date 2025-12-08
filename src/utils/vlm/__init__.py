"""
Vision-Language Model (VLM) utilities package.

This package provides a simplified interface for working with VLMs:

VLMEngine - Simplified engine using only AutoModelForImageTextToText,
ideal for models with standard HuggingFace interfaces (Idefics, SmolVLM, etc.)

Quick Start:
    from utils.vlm import VLMEngine
    engine = VLMEngine("HuggingFaceM4/Idefics3-8B-Llama3")

Module Structure:
    - base.py: Base classes, protocols, and shared utilities
    - vlm_engine.py: Simplified AutoModelForImageTextToText-based engine
"""

# Base abstractions
from .base import (
    # Protocols and base classes
    BaseVLM,
    VLMProtocol,
    
    # Configuration
    ModelConfig,
    GenerationConfig,
    
    # Constants
    IMAGENET_MEAN,
    IMAGENET_STD,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_DO_SAMPLE,
    
    # Utilities
    detect_model_family,
    get_available_device,
)

# Simplified engine
from .vlm_engine import (
    VLMEngine,
    create_vlm_engine,
    SUPPORTED_MODEL_PATTERNS,
)

__all__ = [
    # Main class
    "VLMEngine",
    
    # Base abstractions
    "BaseVLM",
    "VLMProtocol",
    "ModelConfig",
    "GenerationConfig",
    
    # Constants
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_DO_SAMPLE",
    
    # Registry
    "SUPPORTED_MODEL_PATTERNS",
    
    # Factory functions
    "create_vlm_engine",
    
    # Utilities
    "detect_model_family",
    "get_available_device",
]

__version__ = "2.0.0"
