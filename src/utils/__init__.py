"""
Utilities package for SRBench.

This package provides utilities for Vision-Language Model evaluation:

- VLM subpackage: Model engines for inference
  - VLMEngine: Simplified engine for AutoModelForImageTextToText models
"""

# Expose the VLMEngine
from .vlm.vlm_engine import VLMEngine, create_vlm_engine

__all__ = [
    "VLMEngine",
    "create_vlm_engine",
]