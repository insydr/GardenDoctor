"""
Test utilities and helper functions for Garden Doctor tests.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import io

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_test_image(
    width: int = 336,
    height: int = 336,
    color: Tuple[int, int, int] = (34, 139, 34),
    mode: str = "RGB"
) -> Image.Image:
    """
    Create a test image with specified properties.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        color: RGB color tuple
        mode: PIL image mode (RGB, RGBA, L, etc.)
    
    Returns:
        PIL Image object
    """
    if mode == "RGB":
        return Image.new(mode, (width, height), color=color)
    elif mode == "RGBA":
        return Image.new(mode, (width, height), color=(*color, 255))
    elif mode == "L":
        gray = int(0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2])
        return Image.new(mode, (width, height), color=gray)
    else:
        return Image.new(mode, (width, height))


def create_diseased_leaf_image(
    disease_type: str = "blight",
    size: Tuple[int, int] = (336, 336)
) -> Image.Image:
    """
    Create a simulated diseased leaf image for testing.
    
    Args:
        disease_type: Type of disease to simulate
        size: Image dimensions
    
    Returns:
        PIL Image with simulated disease patterns
    """
    from PIL import ImageDraw
    
    # Base green leaf color
    base_color = (34, 139, 34)
    img = Image.new("RGB", size, base_color)
    draw = ImageDraw.Draw(img)
    
    if disease_type == "blight":
        # Add brown spots for blight
        import random
        random.seed(42)  # Reproducible
        for _ in range(20):
            x = random.randint(50, size[0] - 50)
            y = random.randint(50, size[1] - 50)
            r = random.randint(5, 15)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(139, 69, 19))
    
    elif disease_type == "rust":
        # Add orange spots for rust
        import random
        random.seed(43)
        for _ in range(30):
            x = random.randint(30, size[0] - 30)
            y = random.randint(30, size[1] - 30)
            r = random.randint(3, 8)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(210, 105, 30))
    
    elif disease_type == "mold":
        # Add white/gray patches for mold
        import random
        random.seed(44)
        for _ in range(15):
            x = random.randint(40, size[0] - 40)
            y = random.randint(40, size[1] - 40)
            r = random.randint(10, 25)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(200, 200, 180))
    
    return img


def image_to_bytes(img: Image.Image, format: str = "PNG") -> bytes:
    """
    Convert PIL Image to bytes.
    
    Args:
        img: PIL Image object
        format: Output format (PNG, JPEG, etc.)
    
    Returns:
        Image as bytes
    """
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    return buffer.getvalue()


def bytes_to_image(data: bytes) -> Image.Image:
    """
    Convert bytes to PIL Image.
    
    Args:
        data: Image data as bytes
    
    Returns:
        PIL Image object
    """
    buffer = io.BytesIO(data)
    return Image.open(buffer)


def create_temp_image_file(
    img: Optional[Image.Image] = None,
    suffix: str = ".jpg"
) -> str:
    """
    Create a temporary image file for testing file operations.
    
    Args:
        img: Optional PIL Image (creates default if None)
        suffix: File suffix (.jpg, .png, .webp)
    
    Returns:
        Path to temporary file (caller must delete)
    """
    if img is None:
        img = create_test_image()
    
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    
    # Handle format mapping
    format_map = {".jpg": "JPEG", ".png": "PNG", ".webp": "WEBP"}
    img.save(path, format=format_map.get(suffix.lower(), "PNG"))
    
    return path


def assert_valid_diagnosis_result(result) -> None:
    """
    Assert that a DiagnosisResult has valid structure.
    
    Args:
        result: DiagnosisResult to validate
    
    Raises:
        AssertionError if validation fails
    """
    from app import DiagnosisResult, ConfidenceLevel, ProcessingStatus
    
    assert isinstance(result, DiagnosisResult), "Result must be DiagnosisResult"
    assert isinstance(result.disease_name, str), "disease_name must be string"
    assert 0 <= result.confidence_score <= 1, "confidence_score must be 0-1"
    assert isinstance(result.confidence_level, ConfidenceLevel), "Invalid confidence_level"
    assert isinstance(result.status, ProcessingStatus), "Invalid status"
    assert isinstance(result.treatment_cultural, list), "treatment_cultural must be list"
    assert isinstance(result.treatment_organic, list), "treatment_organic must be list"
    assert isinstance(result.treatment_conventional, list), "treatment_conventional must be list"
    assert isinstance(result.prevention, list), "prevention must be list"


def assert_valid_confidence_dict(confidence: dict) -> None:
    """
    Assert that confidence dict has valid structure.
    
    Args:
        confidence: Confidence dictionary to validate
    
    Raises:
        AssertionError if validation fails
    """
    assert isinstance(confidence, dict), "Confidence must be a dict"
    
    for key, value in confidence.items():
        assert isinstance(key, str), f"Key {key} must be string"
        assert isinstance(value, (int, float)), f"Value {value} must be numeric"
        assert 0 <= value <= 1, f"Value {value} must be between 0 and 1"


def mock_model_response(
    disease: str = "Early Blight",
    confidence: str = "High",
    symptoms: str = "Test symptoms",
    cause: str = "Test cause"
) -> str:
    """
    Generate a mock model response for testing.
    
    Args:
        disease: Disease name
        confidence: Confidence level (High/Medium/Low)
        symptoms: Symptoms description
        cause: Cause description
    
    Returns:
        Formatted model response string
    """
    return f"""DISEASE: {disease}
CONFIDENCE: {confidence}
SYMPTOMS: {symptoms}
CAUSE: {cause}

CULTURAL_TREATMENTS:
1. Remove infected plant material
2. Improve air circulation
3. Adjust watering practices

ORGANIC_TREATMENTS:
1. Apply neem oil spray
2. Use copper-based fungicide

CONVENTIONAL_TREATMENTS:
1. Apply appropriate fungicide
2. Follow label instructions

PREVENTION:
• Practice crop rotation
• Remove plant debris
• Use resistant varieties"""


class MockProgress:
    """Mock progress callback for testing."""
    
    def __init__(self):
        self.calls = []
    
    def __call__(self, progress: float, desc: str = ""):
        """Record progress calls."""
        self.calls.append((progress, desc))
    
    def get_last_progress(self) -> float:
        """Get the last progress value."""
        return self.calls[-1][0] if self.calls else 0
    
    def get_last_description(self) -> str:
        """Get the last description."""
        return self.calls[-1][1] if self.calls else ""
    
    def assert_called(self, min_calls: int = 1):
        """Assert progress was called at least min_calls times."""
        assert len(self.calls) >= min_calls, f"Expected at least {min_calls} calls, got {len(self.calls)}"
