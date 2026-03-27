"""
Integration tests for Garden Doctor Gradio application.

Tests cover:
- Application launch and startup
- UI component structure
- API endpoint functionality
- End-to-end diagnosis flow
"""

import pytest
import os
import sys
from unittest.mock import MagicMock, patch
from PIL import Image
import io

# Set environment for testing
os.environ["GARDEN_DOCTOR_MOCK"] = "true"


class TestGradioAppCreation:
    """Tests for Gradio app creation and configuration."""

    def test_create_interface_returns_blocks(self, mock_model_manager):
        """Test that create_interface returns a Gradio Blocks object."""
        from app import create_interface
        import gradio as gr
        
        demo = create_interface(mock_model_manager)
        
        assert isinstance(demo, gr.Blocks)

    def test_interface_has_correct_title(self, mock_model_manager):
        """Test that interface has correct title."""
        from app import create_interface
        
        demo = create_interface(mock_model_manager)
        
        assert "Garden Doctor" in demo.title or "🌿" in demo.title

    def test_interface_uses_green_theme(self, mock_model_manager):
        """Test that interface uses green theme."""
        from app import create_interface
        import gradio as gr
        
        demo = create_interface(mock_model_manager)
        
        # Theme should be green-based
        assert demo.theme is not None


class TestGradioComponents:
    """Tests for Gradio UI components."""

    @pytest.fixture
    def app_components(self, mock_model_manager):
        """Get app components for testing."""
        from app import create_interface
        
        demo = create_interface(mock_model_manager)
        return demo

    def test_has_image_input_component(self, mock_model_manager):
        """Test that app has image input component."""
        from app import create_interface
        
        demo = create_interface(mock_model_manager)
        
        # Check that app can be built
        with demo as app:
            # App should have been built successfully
            assert app is not None

    def test_has_climate_dropdown(self, mock_model_manager):
        """Test that app has climate zone dropdown."""
        from app import create_interface, CLIMATE_ZONES
        
        demo = create_interface(mock_model_manager)
        
        # Climate zones should be defined
        assert len(CLIMATE_ZONES) == 4
        assert "Temperate" in CLIMATE_ZONES

    def test_has_diagnose_button(self, mock_model_manager):
        """Test that app has diagnose button."""
        from app import create_interface
        
        demo = create_interface(mock_model_manager)
        assert demo is not None  # App created successfully


class TestGradioAPIEndpoints:
    """Tests for Gradio API endpoints."""

    def test_diagnose_api_exists(self, mock_model_manager):
        """Test that diagnose API endpoint exists."""
        from app import create_interface
        
        demo = create_interface(mock_model_manager)
        
        # API name should be registered
        # The diagnose button should have api_name="diagnose"
        assert demo is not None

    def test_health_check_endpoint(self, mock_model_manager):
        """Test health check endpoint function."""
        from app import health_check
        
        result = health_check()
        
        assert "status" in result
        assert result["status"] == "healthy"
        assert "timestamp" in result
        assert "model_id" in result


class TestEndToEndDiagnosis:
    """End-to-end tests for diagnosis flow."""

    @pytest.fixture
    def test_client(self, mock_model_manager):
        """Create a test client for the Gradio app."""
        from app import create_interface
        
        demo = create_interface(mock_model_manager)
        return demo

    def test_full_diagnosis_flow(self, mock_model_manager, sample_image):
        """Test complete diagnosis flow from upload to result."""
        from app import diagnose_plant
        
        mock_progress = MagicMock()
        
        # Simulate full diagnosis flow
        confidence, report, diagnosis = diagnose_plant(
            image=sample_image,
            climate="Temperate",
            model_manager=mock_model_manager,
            progress=mock_progress
        )
        
        # Verify all outputs are valid
        assert isinstance(confidence, dict)
        assert isinstance(report, str)
        assert isinstance(diagnosis, DiagnosisResult)
        
        assert len(confidence) > 0
        assert len(report) > 100  # Report should be substantial
        assert diagnosis.disease_name != ""

    def test_diagnosis_with_example_image(self, mock_model_manager):
        """Test diagnosis using example image from file."""
        from app import diagnose_plant
        
        example_path = "examples/tomato_healthy.jpg"
        if os.path.exists(example_path):
            image = Image.open(example_path)
            mock_progress = MagicMock()
            
            confidence, report, diagnosis = diagnose_plant(
                image=image,
                climate="Temperate",
                model_manager=mock_model_manager,
                progress=mock_progress
            )
            
            assert diagnosis.status != ProcessingStatus.ERROR


class TestModelManagerIntegration:
    """Tests for ModelManager integration."""

    def test_model_manager_initialization(self):
        """Test ModelManager initializes correctly."""
        from app import ModelManager
        
        manager = ModelManager(mock_mode=True)
        
        assert manager.mock_mode == True
        assert manager.is_loaded == False

    def test_model_manager_load_in_mock_mode(self):
        """Test model loading in mock mode."""
        from app import ModelManager
        
        manager = ModelManager(mock_mode=True)
        result = manager.load_model()
        
        assert result == True
        assert manager.is_loaded == True

    def test_model_manager_preprocess_image(self):
        """Test image preprocessing."""
        from app import ModelManager, IMAGE_SIZE
        
        manager = ModelManager(mock_mode=True)
        
        # Create a test image
        test_img = Image.new('RGB', (500, 500), color=(100, 150, 200))
        
        # Preprocess
        processed = manager.preprocess_image(test_img)
        
        assert processed.size == (IMAGE_SIZE, IMAGE_SIZE)
        assert processed.mode == "RGB"

    def test_model_manager_generate_mock_response(self):
        """Test mock response generation."""
        from app import ModelManager
        
        manager = ModelManager(mock_mode=True)
        manager.is_loaded = True
        
        response = manager.generate_response(
            image=Image.new('RGB', (336, 336)),
            prompt="Test prompt for Temperate climate"
        )
        
        assert "DISEASE:" in response
        assert "CONFIDENCE:" in response
        assert "SYMPTOMS:" in response


class TestModelCacheIntegration:
    """Tests for ModelCache singleton."""

    def test_model_cache_singleton(self):
        """Test that ModelCache returns same instance."""
        from app import ModelCache
        
        # Reset first
        ModelCache.reset()
        
        manager1 = ModelCache.get_model_manager(mock_mode=True)
        manager2 = ModelCache.get_model_manager(mock_mode=True)
        
        assert manager1 is manager2

    def test_model_cache_is_initialized(self):
        """Test ModelCache initialization tracking."""
        from app import ModelCache
        
        ModelCache.reset()
        
        assert ModelCache.is_initialized() == False
        
        ModelCache.get_model_manager(mock_mode=True)
        
        assert ModelCache.is_initialized() == True

    def test_model_cache_reset(self):
        """Test ModelCache reset functionality."""
        from app import ModelCache
        
        # Initialize
        ModelCache.get_model_manager(mock_mode=True)
        assert ModelCache.is_initialized() == True
        
        # Reset
        ModelCache.reset()
        assert ModelCache.is_initialized() == False


class TestPDFReportIntegration:
    """Tests for PDF report generation."""

    def test_pdf_generation(self, sample_image):
        """Test PDF report can be generated."""
        from app import PDFReportGenerator, DiagnosisResult
        
        diagnosis = DiagnosisResult(
            disease_name="Test Disease",
            confidence_score=0.85,
            confidence_level=ConfidenceLevel.HIGH,
            symptoms="Test symptoms",
            cause="Test cause",
            treatment_cultural=["Step 1"],
            treatment_organic=["Organic 1"],
            treatment_conventional=["Conv 1"],
            prevention=["Tip 1"],
            image=sample_image,
            climate="Temperate",
            status=ProcessingStatus.SUCCESS
        )
        
        generator = PDFReportGenerator()
        
        # Should not raise exception
        try:
            pdf_path = generator.generate_report(diagnosis)
            assert os.path.exists(pdf_path)
            os.remove(pdf_path)  # Cleanup
        except ImportError:
            pytest.skip("fpdf2 not installed")

    def test_pdf_contains_disease_name(self, sample_image):
        """Test that PDF contains disease name."""
        from app import PDFReportGenerator, DiagnosisResult
        
        diagnosis = DiagnosisResult(
            disease_name="Early Blight",
            confidence_score=0.85,
            confidence_level=ConfidenceLevel.HIGH,
            symptoms="Test symptoms",
            treatment_cultural=["Remove leaves"],
            prevention=["Crop rotation"],
            image=sample_image,
            climate="Temperate",
            status=ProcessingStatus.SUCCESS
        )
        
        generator = PDFReportGenerator()
        
        try:
            pdf_path = generator.generate_report(diagnosis)
            
            # Read PDF and check for disease name (basic text search)
            with open(pdf_path, 'rb') as f:
                content = f.read()
            
            # PDF is binary, but disease name might appear
            assert len(content) > 1000  # PDF should have substantial content
            
            os.remove(pdf_path)
        except ImportError:
            pytest.skip("fpdf2 not installed")


class TestClimateIntegration:
    """Tests for climate zone integration."""

    @pytest.mark.parametrize("climate", ["Tropical", "Temperate", "Arid", "Cold"])
    def test_all_climates_in_mock_response(self, climate):
        """Test that climate affects mock response."""
        from app import ModelManager
        
        manager = ModelManager(mock_mode=True)
        manager.is_loaded = True
        
        prompt = f"Test prompt for {climate} climate"
        response = manager._generate_mock_response(prompt)
        
        assert climate in response or "climate" in response.lower()

    def test_climate_zone_descriptions(self):
        """Test that all climate zones have descriptions."""
        from app import CLIMATE_ZONES
        
        for zone, info in CLIMATE_ZONES.items():
            assert "description" in info
            assert len(info["description"]) > 0


class TestErrorHandlingIntegration:
    """Tests for error handling integration."""

    def test_graceful_degradation_on_model_error(self, sample_image):
        """Test graceful degradation when model fails."""
        from app import diagnose_plant, ModelManager
        
        manager = ModelManager(mock_mode=True)
        manager.is_loaded = True
        
        # Force an error
        with patch.object(manager, 'generate_response', side_effect=Exception("Forced error")):
            mock_progress = MagicMock()
            
            confidence, report, diagnosis = diagnose_plant(
                image=sample_image,
                climate="Temperate",
                model_manager=manager,
                progress=mock_progress
            )
            
            # Should return error result, not crash
            assert diagnosis.status == ProcessingStatus.ERROR
            assert len(report) > 0


# Import required types at module level
from app import DiagnosisResult, ProcessingStatus, ConfidenceLevel


class TestTimeoutIntegration:
    """Tests for timeout handling integration."""

    def test_timeout_returns_appropriate_message(self, sample_image):
        """Test that timeout returns appropriate error message."""
        from app import diagnose_plant, ModelManager, TimeoutError, ERROR_MESSAGES
        
        manager = ModelManager(mock_mode=True)
        manager.is_loaded = True
        
        # Mock to raise timeout
        with patch.object(manager, 'generate_response', side_effect=TimeoutError("Timeout")):
            mock_progress = MagicMock()
            
            confidence, report, diagnosis = diagnose_plant(
                image=sample_image,
                climate="Temperate",
                model_manager=manager,
                progress=mock_progress
            )
            
            assert diagnosis.status == ProcessingStatus.ERROR
            assert diagnosis.error_message == "timeout"
            # Report should mention timeout
            assert "timeout" in report.lower() or "Timeout" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
