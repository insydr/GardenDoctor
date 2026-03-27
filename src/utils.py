"""
Utility functions for Garden Doctor application.
"""

import os
import gc
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def get_device() -> str:
    """
    Get the best available device for inference.
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_dtype(device: str) -> torch.dtype:
    """
    Get the optimal dtype for the given device.
    
    Args:
        device: Device string
        
    Returns:
        Optimal torch dtype
    """
    if device == "cuda":
        return torch.float16
    elif device == "mps":
        return torch.float32  # MPS has limited float16 support
    else:
        return torch.float32


def clear_memory():
    """Clear GPU and system memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_memory_usage() -> dict:
    """
    Get current memory usage.
    
    Returns:
        Dictionary with memory statistics
    """
    result = {
        "system": None,
        "gpu": None
    }
    
    try:
        import psutil
        process = psutil.Process(os.getpid())
        result["system"] = {
            "rss_mb": process.memory_info().rss / 1024 / 1024,
            "vms_mb": process.memory_info().vms / 1024 / 1024,
        }
    except ImportError:
        pass
    
    if torch.cuda.is_available():
        result["gpu"] = {
            "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
            "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
        }
    
    return result


def estimate_model_memory(model_id: str) -> Optional[float]:
    """
    Estimate memory required to load a model.
    
    Args:
        model_id: Hugging Face model ID
        
    Returns:
        Estimated memory in GB, or None if unknown
    """
    # Common model sizes (approximate)
    size_estimates = {
        "7b": 14.0,  # 7B parameters, ~14GB in fp16
        "13b": 26.0,  # 13B parameters, ~26GB in fp16
        "70b": 140.0,  # 70B parameters, ~140GB in fp16
    }
    
    model_id_lower = model_id.lower()
    
    for key, size in size_estimates.items():
        if key in model_id_lower:
            return size
    
    return None


def check_memory_availability(required_gb: float) -> bool:
    """
    Check if sufficient memory is available.
    
    Args:
        required_gb: Required memory in GB
        
    Returns:
        True if sufficient memory available
    """
    try:
        import psutil
        available = psutil.virtual_memory().available / (1024 ** 3)
        return available >= required_gb * 1.2  # 20% buffer
    except ImportError:
        return True  # Assume OK if can't check


def optimize_for_inference(model) -> None:
    """
    Apply inference optimizations to a model.
    
    Args:
        model: PyTorch model to optimize
    """
    # Set to evaluation mode
    model.eval()
    
    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False
    
    # Apply torch.compile if available (PyTorch 2.0+)
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("Applied torch.compile optimization")
        except Exception as e:
            logger.warning(f"Could not apply torch.compile: {e}")


def get_recommended_settings(device: str) -> dict:
    """
    Get recommended inference settings for a device.
    
    Args:
        device: Device string
        
    Returns:
        Dictionary with recommended settings
    """
    if device == "cuda":
        return {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "use_flash_attention": True,
        }
    elif device == "mps":
        return {
            "torch_dtype": torch.float32,
            "device_map": None,
            "low_cpu_mem_usage": True,
            "use_flash_attention": False,
        }
    else:  # CPU
        return {
            "torch_dtype": torch.float32,
            "device_map": None,
            "low_cpu_mem_usage": True,
            "use_flash_attention": False,
        }
