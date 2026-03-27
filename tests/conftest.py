"""
Pytest configuration and shared fixtures for Garden Doctor tests.
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image
import io

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set mock mode for testing
os.environ["GARDEN_DOCTOR_MOCK"] = "true"
os.environ["GARDEN_DOCTOR_TIMEOUT"] = "10"


# =============================================================================
# Image Fixtures
# =============================================================================

@pytest.fixture
def sample_image():
    """Create a simple sample image for testing."""
    img = Image.new('RGB', (336, 336), color=(34, 139, 34))  # Green image
    return img


@pytest.fixture
def sample_image_bytes():
    """Create sample image as bytes."""
    img = Image.new('RGB', (336, 336), color=(34, 139, 34))
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def small_image():
    """Create an image that's too small for valid diagnosis."""
    img = Image.new('RGB', (30, 30), color=(100, 100, 100))
    return img


@pytest.fixture
def large_image():
    """Create a large image for testing resize behavior."""
    img = Image.new('RGB', (1920, 1080), color=(139, 69, 19))  # Brown
    return img


@pytest.fixture
def non_rgb_image():
    """Create a non-RGB image (RGBA) for testing conversion."""
    img = Image.new('RGBA', (336, 336), color=(34, 139, 34, 200))
    return img


# =============================================================================
# Model Response Fixtures
# =============================================================================

@pytest.fixture
def mock_high_confidence_response():
    """Mock model response with high confidence."""
    return """DISEASE: Early Blight
CONFIDENCE: High
SYMPTOMS: Dark brown circular spots with concentric rings forming distinctive target-like patterns on the leaves. Yellowing of leaf tissue around the infected areas.
CAUSE: Early blight is caused by the fungus Alternaria solani. The pathogen survives in infected plant debris and soil for up to a year.

CULTURAL_TREATMENTS:
1. Remove and destroy all infected leaves immediately to prevent spore spread
2. Improve air circulation by proper plant spacing (24-36 inches apart)
3. Water at the base of plants early in the day to keep foliage dry

ORGANIC_TREATMENTS:
1. Apply copper-based fungicide every 7-10 days during active disease period
2. Use neem oil spray as a natural fungicide alternative

CONVENTIONAL_TREATMENTS:
1. Apply chlorothalonil or mancozeb fungicide according to label instructions
2. Use systemic fungicides containing azoxystrobin for severe infections

PREVENTION:
• Use disease-resistant varieties appropriate for Temperate climate
• Practice 3-4 year crop rotation avoiding solanaceous crops
• Remove and destroy all plant debris at end of growing season"""


@pytest.fixture
def mock_medium_confidence_response():
    """Mock model response with medium confidence."""
    return """DISEASE: Late Blight
CONFIDENCE: Medium
SYMPTOMS: Water-soaked lesions on leaves that turn brown and papery. White fungal growth may appear on leaf undersides in humid conditions.
CAUSE: Caused by the oomycete Phytophthora infestans, which thrives in cool, wet conditions.

CULTURAL_TREATMENTS:
1. Remove infected plant material immediately
2. Avoid overhead irrigation

ORGANIC_TREATMENTS:
1. Apply copper fungicide preventatively

CONVENTIONAL_TREATMENTS:
1. Use systemic fungicides as needed

PREVENTION:
• Ensure good drainage
• Space plants for air circulation"""


@pytest.fixture
def mock_low_confidence_response():
    """Mock model response with low confidence."""
    return """DISEASE: Unknown Leaf Condition
CONFIDENCE: Low
SYMPTOMS: Unclear symptoms visible in the image. The image quality or lighting may not be sufficient for accurate diagnosis.
CAUSE: Unable to determine the specific cause from the provided image.

CULTURAL_TREATMENTS:
1. Try uploading a clearer image with better lighting

ORGANIC_TREATMENTS:
1. Consult local agricultural extension

CONVENTIONAL_TREATMENTS:
1. Professional consultation recommended

PREVENTION:
• Regular plant monitoring"""


@pytest.fixture
def mock_healthy_response():
    """Mock model response for healthy plant."""
    return """DISEASE: Healthy Plant
CONFIDENCE: High
SYMPTOMS: No visible disease symptoms. The leaf appears healthy with uniform green coloration and no spots, lesions, or discoloration.
CAUSE: No disease present. The plant is in good health.

CULTURAL_TREATMENTS:
1. Continue regular watering schedule
2. Maintain good air circulation

ORGANIC_TREATMENTS:
1. No treatment needed

CONVENTIONAL_TREATMENTS:
1. No treatment needed

PREVENTION:
• Continue current care practices
• Monitor regularly for any changes"""


@pytest.fixture
def mock_unsupported_plant_response():
    """Mock model response for unsupported/unknown plant."""
    return """DISEASE: Unrecognized Plant Species
CONFIDENCE: Low
SYMPTOMS: The plant species in the image does not match our supported plant database.
CAUSE: This plant may not be in our training dataset.

CULTURAL_TREATMENTS:
1. Consult local gardening resources

ORGANIC_TREATMENTS:
1. Contact agricultural extension service

CONVENTIONAL_TREATMENTS:
1. Professional identification recommended

PREVENTION:
• Check our supported plant list"""


@pytest.fixture
def mock_malformed_response():
    """Mock malformed model response for error handling tests."""
    return "This is not a properly formatted response from the model."


# =============================================================================
# Climate Zone Fixtures
# =============================================================================

@pytest.fixture
def climate_zones():
    """Return all valid climate zones."""
    return ["Tropical", "Temperate", "Arid", "Cold"]


@pytest.fixture
def default_climate():
    """Return default climate zone."""
    return "Temperate"


# =============================================================================
# Model Manager Fixtures
# =============================================================================

@pytest.fixture
def mock_model_manager():
    """Create a mock model manager for testing."""
    from app import ModelManager
    
    manager = ModelManager(mock_mode=True)
    manager.is_loaded = True
    return manager


# =============================================================================
# App Client Fixture (for integration tests)
# =============================================================================

@pytest.fixture
def gradio_app():
    """Create a Gradio app instance for integration testing."""
    from app import ModelManager, create_interface
    
    model_manager = ModelManager(mock_mode=True)
    model_manager.is_loaded = True
    
    demo = create_interface(model_manager)
    return demo


@pytest.fixture
def gradio_client(gradio_app):
    """Create a Gradio test client."""
    from gradio.testing import TestClient
    
    client = TestClient(gradio_app)
    return client


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def expected_diseases():
    """List of expected disease names for validation."""
    return [
        "Early Blight",
        "Late Blight",
        "Apple Scab",
        "Common Rust",
        "Black Rot",
        "Healthy Plant",
        "Bacterial Spot",
        "Powdery Mildew",
        "Septoria Leaf Spot",
        "Target Spot",
    ]


@pytest.fixture
def supported_plants():
    """List of supported plant species."""
    return [
        "Tomato", "Potato", "Apple", "Grape", "Corn",
        "Pepper", "Strawberry", "Cherry", "Peach", "Orange"
    ]


# =============================================================================
# Cleanup Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_model_cache():
    """Clean up model cache after each test."""
    yield
    # Reset the model cache after each test
    try:
        from app import ModelCache
        ModelCache.reset()
    except Exception:
        pass
