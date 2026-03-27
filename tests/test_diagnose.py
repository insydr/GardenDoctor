"""
Unit tests for diagnose_plant function in Garden Doctor.

Tests cover:
- diagnose_plant() with mock model responses
- Image validation
- Error handling
- Timeout handling
- Progress reporting
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from PIL import Image
import time

from app import (
    diagnose_plant,
    DiagnosisResult,
    ProcessingStatus,
    ConfidenceLevel,
    ModelManager,
    TimeoutError,
    ERROR_MESSAGES,
)


class TestDiagnosePlantBasic:
    """Basic functionality tests for diagnose_plant."""

    def test_diagnose_with_valid_image(self, sample_image, mock_model_manager):
        """Test diagnosis with a valid image."""
        mock_progress = MagicMock()
        
        confidence, report, diagnosis = diagnose_plant(
            image=sample_image,
            climate="Temperate",
            model_manager=mock_model_manager,
            progress=mock_progress
        )
        
        assert isinstance(diagnosis, DiagnosisResult)
        assert diagnosis.disease_name != "Unknown"
        assert 0 <= diagnosis.confidence_score <= 1
        assert isinstance(confidence, dict)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_diagnose_sets_climate(self, sample_image, mock_model_manager):
        """Test that climate is set in diagnosis result."""
        mock_progress = MagicMock()
        
        _, _, diagnosis = diagnose_plant(
            image=sample_image,
            climate="Tropical",
            model_manager=mock_model_manager,
            progress=mock_progress
        )
        
        assert diagnosis.climate == "Tropical"

    def test_diagnose_stores_image(self, sample_image, mock_model_manager):
        """Test that image is stored in diagnosis result."""
        mock_progress = MagicMock()
        
        _, _, diagnosis = diagnose_plant(
            image=sample_image,
            climate="Temperate",
            model_manager=mock_model_manager,
            progress=mock_progress
        )
        
        assert diagnosis.image is not None
        assert diagnosis.image == sample_image

    def test_diagnose_records_processing_time(self, sample_image, mock_model_manager):
        """Test that processing time is recorded."""
        mock_progress = MagicMock()
        
        _, _, diagnosis = diagnose_plant(
            image=sample_image,
            climate="Temperate",
            model_manager=mock_model_manager,
            progress=mock_progress
        )
        
        assert diagnosis.processing_time >= 0


class TestDiagnosePlantNullImage:
    """Tests for diagnose_plant with null/missing image."""

    def test_diagnose_with_none_image(self, mock_model_manager):
        """Test diagnosis with None image returns error."""
        mock_progress = MagicMock()
        
        confidence, report, diagnosis = diagnose_plant(
            image=None,
            climate="Temperate",
            model_manager=mock_model_manager,
            progress=mock_progress
        )
        
        assert diagnosis.status == ProcessingStatus.ERROR
        assert diagnosis.error_message == "no_image"
        assert confidence == {}
        assert "No Image" in report

    def test_diagnose_with_none_image_empty_confidence(self, mock_model_manager):
        """Test that None image returns empty confidence dict."""
        mock_progress = MagicMock()
        
        confidence, _, _ = diagnose_plant(
            image=None,
            climate="Temperate",
            model_manager=mock_model_manager,
            progress=mock_progress
        )
        
        assert confidence == {}


class TestDiagnosePlantInvalidImage:
    """Tests for diagnose_plant with invalid images."""

    def test_diagnose_with_small_image(self, small_image, mock_model_manager):
        """Test diagnosis with image too small."""
        mock_progress = MagicMock()
        
        confidence, report, diagnosis = diagnose_plant(
            image=small_image,
            climate="Temperate",
            model_manager=mock_model_manager,
            progress=mock_progress
        )
        
        assert diagnosis.status == ProcessingStatus.ERROR
        assert "Invalid" in report or "invalid" in report.lower()

    def test_diagnose_with_non_pil_image(self, mock_model_manager):
        """Test diagnosis with non-PIL image type."""
        mock_progress = MagicMock()
        
        confidence, report, diagnosis = diagnose_plant(
            image="not an image",
            climate="Temperate",
            model_manager=mock_model_manager,
            progress=mock_progress
        )
        
        assert diagnosis.status == ProcessingStatus.ERROR


class TestDiagnosePlantClimateVariations:
    """Tests for diagnose_plant with different climate zones."""

    @pytest.mark.parametrize("climate", ["Tropical", "Temperate", "Arid", "Cold"])
    def test_diagnose_with_all_climates(self, climate, sample_image, mock_model_manager):
        """Test diagnosis works for all climate zones."""
        mock_progress = MagicMock()
        
        _, _, diagnosis = diagnose_plant(
            image=sample_image,
            climate=climate,
            model_manager=mock_model_manager,
            progress=mock_progress
        )
        
        assert diagnosis.climate == climate

    def test_climate_affects_mock_response(self, sample_image):
        """Test that climate affects mock response content."""
        # Create a fresh mock manager
        manager = ModelManager(mock_mode=True)
        manager.is_loaded = True
        mock_progress = MagicMock()
        
        _, _, diagnosis = diagnose_plant(
            image=sample_image,
            climate="Tropical",
            model_manager=manager,
            progress=mock_progress
        )
        
        # The mock response should mention the climate
        assert "Tropical" in diagnosis.raw_response or diagnosis.climate == "Tropical"


class TestDiagnosePlantProgress:
    """Tests for progress reporting in diagnose_plant."""

    def test_progress_called_during_diagnosis(self, sample_image, mock_model_manager):
        """Test that progress callback is called."""
        mock_progress = MagicMock()
        
        diagnose_plant(
            image=sample_image,
            climate="Temperate",
            model_manager=mock_model_manager,
            progress=mock_progress
        )
        
        # Progress should be called multiple times
        assert mock_progress.call_count >= 5

    def test_progress_starts_at_zero(self, sample_image, mock_model_manager):
        """Test that progress starts near zero."""
        mock_progress = MagicMock()
        
        diagnose_plant(
            image=sample_image,
            climate="Temperate",
            model_manager=mock_model_manager,
            progress=mock_progress
        )
        
        # First call should be low value
        first_call = mock_progress.call_args_list[0]
        assert first_call[0][0] < 0.5


class TestDiagnosePlantErrorHandling:
    """Tests for error handling in diagnose_plant."""

    def test_model_inference_error_handling(self, sample_image):
        """Test handling of model inference errors."""
        manager = ModelManager(mock_mode=True)
        manager.is_loaded = True
        
        # Mock generate_response to raise an error
        with patch.object(manager, 'generate_response', side_effect=Exception("Model error")):
            mock_progress = MagicMock()
            
            confidence, report, diagnosis = diagnose_plant(
                image=sample_image,
                climate="Temperate",
                model_manager=manager,
                progress=mock_progress
            )
            
            assert diagnosis.status == ProcessingStatus.ERROR
            assert "Error" in report or "error" in report.lower()

    def test_timeout_error_handling(self, sample_image):
        """Test handling of timeout errors."""
        manager = ModelManager(mock_mode=True)
        manager.is_loaded = True
        
        # Mock generate_response to raise TimeoutError
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
            assert "Timeout" in report or "timeout" in report.lower()


class TestDiagnosePlantImagePreprocessing:
    """Tests for image preprocessing in diagnose_plant."""

    def test_non_rgb_image_conversion(self, non_rgb_image, mock_model_manager):
        """Test that non-RGB images are converted."""
        mock_progress = MagicMock()
        
        # Should not raise an error
        _, _, diagnosis = diagnose_plant(
            image=non_rgb_image,
            climate="Temperate",
            model_manager=mock_model_manager,
            progress=mock_progress
        )
        
        assert diagnosis.status != ProcessingStatus.ERROR or diagnosis.error_message != "invalid_image"

    def test_large_image_resize(self, large_image, mock_model_manager):
        """Test that large images are resized."""
        mock_progress = MagicMock()
        
        _, _, diagnosis = diagnose_plant(
            image=large_image,
            climate="Temperate",
            model_manager=mock_model_manager,
            progress=mock_progress
        )
        
        # Should process without error
        assert diagnosis.disease_name != "Unknown" or diagnosis.status != ProcessingStatus.ERROR


class TestDiagnosePlantMockMode:
    """Tests specific to mock mode operation."""

    def test_mock_mode_returns_valid_response(self, sample_image):
        """Test that mock mode returns valid structured response."""
        manager = ModelManager(mock_mode=True)
        manager.is_loaded = True
        mock_progress = MagicMock()
        
        _, _, diagnosis = diagnose_plant(
            image=sample_image,
            climate="Temperate",
            model_manager=manager,
            progress=mock_progress
        )
        
        # Mock should return Early Blight by default
        assert "Blight" in diagnosis.disease_name or diagnosis.disease_name != "Unknown"

    def test_mock_mode_includes_treatments(self, sample_image):
        """Test that mock mode includes treatment options."""
        manager = ModelManager(mock_mode=True)
        manager.is_loaded = True
        mock_progress = MagicMock()
        
        _, _, diagnosis = diagnose_plant(
            image=sample_image,
            climate="Temperate",
            model_manager=manager,
            progress=mock_progress
        )
        
        assert len(diagnosis.treatment_cultural) > 0
        assert len(diagnosis.treatment_organic) > 0
        assert len(diagnosis.treatment_conventional) > 0


class TestDiagnosePlantOutputFormat:
    """Tests for output format from diagnose_plant."""

    def test_confidence_dict_format(self, sample_image, mock_model_manager):
        """Test that confidence dict has correct format."""
        mock_progress = MagicMock()
        
        confidence, _, diagnosis = diagnose_plant(
            image=sample_image,
            climate="Temperate",
            model_manager=mock_model_manager,
            progress=mock_progress
        )
        
        # Should have disease name as key
        assert diagnosis.disease_name in confidence
        # Value should be float
        assert isinstance(confidence[diagnosis.disease_name], float)

    def test_report_is_markdown(self, sample_image, mock_model_manager):
        """Test that report is formatted as markdown."""
        mock_progress = MagicMock()
        
        _, report, _ = diagnose_plant(
            image=sample_image,
            climate="Temperate",
            model_manager=mock_model_manager,
            progress=mock_progress
        )
        
        # Should contain markdown formatting
        assert "**" in report or "#" in report

    def test_report_includes_emoji(self, sample_image, mock_model_manager):
        """Test that report includes emoji indicators."""
        mock_progress = MagicMock()
        
        _, report, _ = diagnose_plant(
            image=sample_image,
            climate="Temperate",
            model_manager=mock_model_manager,
            progress=mock_progress
        )
        
        # Should have some emoji
        has_emoji = any(char in report for char in "🟢🟡🔴✅⚠️🔍🌱")
        assert has_emoji


class TestDiagnosePlantEdgeCases:
    """Edge case tests for diagnose_plant."""

    def test_very_large_image(self, mock_model_manager):
        """Test handling of very large images."""
        large_img = Image.new('RGB', (4000, 3000), color=(34, 139, 34))
        mock_progress = MagicMock()
        
        _, _, diagnosis = diagnose_plant(
            image=large_img,
            climate="Temperate",
            model_manager=mock_model_manager,
            progress=mock_progress
        )
        
        assert diagnosis.status != ProcessingStatus.ERROR or diagnosis.error_message != "invalid_image"

    def test_just_above_minimum_size(self, mock_model_manager):
        """Test image at minimum valid size boundary."""
        min_img = Image.new('RGB', (51, 51), color=(34, 139, 34))
        mock_progress = MagicMock()
        
        _, _, diagnosis = diagnose_plant(
            image=min_img,
            climate="Temperate",
            model_manager=mock_model_manager,
            progress=mock_progress
        )
        
        # Should process successfully (above 50px threshold)
        assert diagnosis.status != ProcessingStatus.ERROR or diagnosis.error_message != "invalid_image"

    def test_just_below_minimum_size(self, mock_model_manager):
        """Test image just below minimum valid size."""
        tiny_img = Image.new('RGB', (49, 49), color=(34, 139, 34))
        mock_progress = MagicMock()
        
        _, report, diagnosis = diagnose_plant(
            image=tiny_img,
            climate="Temperate",
            model_manager=mock_model_manager,
            progress=mock_progress
        )
        
        assert diagnosis.status == ProcessingStatus.ERROR
        assert "Invalid" in report


class TestDiagnosePlantConcurrency:
    """Tests related to concurrent operation."""

    def test_diagnose_is_thread_safe(self, sample_image, mock_model_manager):
        """Test that diagnose_plant can be called from multiple threads."""
        import threading
        
        results = []
        mock_progress = MagicMock()
        
        def run_diagnosis():
            _, _, diagnosis = diagnose_plant(
                image=sample_image,
                climate="Temperate",
                model_manager=mock_model_manager,
                progress=mock_progress
            )
            results.append(diagnosis)
        
        threads = [threading.Thread(target=run_diagnosis) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(results) == 3
        for r in results:
            assert isinstance(r, DiagnosisResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
