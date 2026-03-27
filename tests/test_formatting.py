"""
Unit tests for output formatting functions in Garden Doctor.

Tests cover:
- parse_diagnosis_response() - parsing model output
- format_diagnosis_result() - formatting for display
- format_normal_result() - normal case formatting
- format_low_confidence_message() - low confidence formatting
- format_unsupported_plant_message() - unsupported plant formatting
"""

import pytest
from app import (
    parse_diagnosis_response,
    format_diagnosis_result,
    format_normal_result,
    format_low_confidence_message,
    format_unsupported_plant_message,
    DiagnosisResult,
    ConfidenceLevel,
    ProcessingStatus,
)


class TestParseDiagnosisResponse:
    """Tests for parse_diagnosis_response function."""

    def test_parse_high_confidence_response(self, mock_high_confidence_response):
        """Test parsing a well-formed high confidence response."""
        result = parse_diagnosis_response(mock_high_confidence_response, 2.5)
        
        assert result.disease_name == "Early Blight"
        assert result.confidence_score == 0.85
        assert result.confidence_level == ConfidenceLevel.HIGH
        assert result.status == ProcessingStatus.SUCCESS
        assert "circular spots" in result.symptoms
        assert "Alternaria solani" in result.cause
        assert len(result.treatment_cultural) >= 1
        assert len(result.treatment_organic) >= 1
        assert len(result.treatment_conventional) >= 1
        assert len(result.prevention) >= 1

    def test_parse_medium_confidence_response(self, mock_medium_confidence_response):
        """Test parsing a medium confidence response."""
        result = parse_diagnosis_response(mock_medium_confidence_response, 3.0)
        
        assert result.disease_name == "Late Blight"
        assert result.confidence_score == 0.65
        assert result.confidence_level == ConfidenceLevel.MEDIUM

    def test_parse_low_confidence_response(self, mock_low_confidence_response):
        """Test parsing a low confidence response."""
        result = parse_diagnosis_response(mock_low_confidence_response, 1.5)
        
        assert "Unknown" in result.disease_name
        assert result.confidence_score == 0.35
        assert result.confidence_level == ConfidenceLevel.LOW
        assert result.status == ProcessingStatus.LOW_CONFIDENCE

    def test_parse_healthy_response(self, mock_healthy_response):
        """Test parsing a healthy plant response."""
        result = parse_diagnosis_response(mock_healthy_response, 2.0)
        
        assert result.is_healthy()
        assert result.confidence_score == 0.85
        assert result.status == ProcessingStatus.SUCCESS

    def test_parse_unsupported_plant_response(self, mock_unsupported_plant_response):
        """Test parsing an unsupported plant response."""
        result = parse_diagnosis_response(mock_unsupported_plant_response, 1.0)
        
        assert result.is_unsupported_plant()
        assert result.status == ProcessingStatus.UNSUPPORTED

    def test_parse_malformed_response(self, mock_malformed_response):
        """Test parsing a malformed response."""
        result = parse_diagnosis_response(mock_malformed_response, 0.5)
        
        # Should return default values
        assert result.disease_name == "Unknown"
        assert result.confidence_score == 0.5  # Default

    def test_parse_preserves_processing_time(self, mock_high_confidence_response):
        """Test that processing time is preserved."""
        processing_time = 15.7
        result = parse_diagnosis_response(mock_high_confidence_response, processing_time)
        
        assert result.processing_time == processing_time

    def test_parse_stores_raw_response(self, mock_high_confidence_response):
        """Test that raw response is stored."""
        result = parse_diagnosis_response(mock_high_confidence_response, 2.0)
        
        assert result.raw_response == mock_high_confidence_response


class TestParseDiagnosisResponseEdgeCases:
    """Edge case tests for parse_diagnosis_response."""

    def test_empty_response(self):
        """Test parsing empty string."""
        result = parse_diagnosis_response("", 1.0)
        
        assert result.disease_name == "Unknown"
        assert result.confidence_score == 0.5

    def test_partial_response(self):
        """Test parsing partial response with only disease name."""
        partial = "DISEASE: Some Disease"
        result = parse_diagnosis_response(partial, 1.0)
        
        assert result.disease_name == "Some Disease"

    def test_case_insensitive_parsing(self):
        """Test that parsing is case-insensitive."""
        response = """disease: Tomato Blight
confidence: high
symptoms: Test symptoms"""
        result = parse_diagnosis_response(response, 1.0)
        
        assert result.disease_name == "Tomato Blight"
        assert result.confidence_score == 0.85  # High = 0.85

    def test_whitespace_handling(self):
        """Test handling of extra whitespace."""
        response = """
        DISEASE:   Potato Late Blight   
        
        CONFIDENCE:   Medium
        
        SYMPTOMS:   Water soaked lesions   
        """
        result = parse_diagnosis_response(response, 1.0)
        
        assert result.disease_name == "Potato Late Blight"


class TestFormatDiagnosisResult:
    """Tests for format_diagnosis_result function."""

    def test_format_null_result(self):
        """Test formatting null result."""
        output = format_diagnosis_result(None, "Temperate")
        
        assert "No Image Provided" in output or "no image" in output.lower()

    def test_format_error_result(self):
        """Test formatting error result."""
        result = DiagnosisResult(
            status=ProcessingStatus.ERROR,
            error_message="model_error"
        )
        output = format_diagnosis_result(result, "Temperate")
        
        assert "Error" in output or "error" in output.lower()

    def test_format_success_result(self):
        """Test formatting successful result."""
        result = DiagnosisResult(
            disease_name="Early Blight",
            confidence_score=0.85,
            confidence_level=ConfidenceLevel.HIGH,
            symptoms="Test symptoms",
            cause="Test cause",
            treatment_cultural=["Step 1", "Step 2"],
            treatment_organic=["Organic 1"],
            treatment_conventional=["Conv 1"],
            prevention=["Tip 1"],
            status=ProcessingStatus.SUCCESS
        )
        output = format_diagnosis_result(result, "Temperate")
        
        assert "Early Blight" in output
        assert "85%" in output or "High" in output
        assert "Temperate" in output

    def test_format_healthy_result(self):
        """Test formatting healthy plant result."""
        result = DiagnosisResult(
            disease_name="Healthy Plant",
            confidence_score=0.90,
            confidence_level=ConfidenceLevel.HIGH,
            status=ProcessingStatus.SUCCESS
        )
        output = format_diagnosis_result(result, "Tropical")
        
        assert "Healthy" in output
        assert "Great News" in output or "healthy" in output.lower()


class TestFormatNormalResult:
    """Tests for format_normal_result function."""

    def test_format_includes_disease_name(self):
        """Test that disease name is included."""
        result = DiagnosisResult(
            disease_name="Apple Scab",
            confidence_score=0.75,
            confidence_level=ConfidenceLevel.MEDIUM,
            status=ProcessingStatus.SUCCESS
        )
        output = format_normal_result(result, "Temperate")
        
        assert "Apple Scab" in output

    def test_format_includes_confidence(self):
        """Test that confidence score is included."""
        result = DiagnosisResult(
            disease_name="Test Disease",
            confidence_score=0.88,
            confidence_level=ConfidenceLevel.HIGH,
            status=ProcessingStatus.SUCCESS
        )
        output = format_normal_result(result, "Temperate")
        
        assert "88%" in output or "High" in output

    def test_format_includes_climate(self):
        """Test that climate zone is included."""
        result = DiagnosisResult(
            disease_name="Test Disease",
            confidence_score=0.80,
            confidence_level=ConfidenceLevel.HIGH,
            status=ProcessingStatus.SUCCESS
        )
        
        for climate in ["Tropical", "Temperate", "Arid", "Cold"]:
            output = format_normal_result(result, climate)
            assert climate in output

    def test_format_includes_symptoms(self):
        """Test that symptoms are included."""
        result = DiagnosisResult(
            disease_name="Test Disease",
            confidence_score=0.80,
            confidence_level=ConfidenceLevel.HIGH,
            symptoms="Brown spots on leaves with yellow halos",
            status=ProcessingStatus.SUCCESS
        )
        output = format_normal_result(result, "Temperate")
        
        assert "Brown spots" in output

    def test_format_includes_treatments(self):
        """Test that treatment options are included."""
        result = DiagnosisResult(
            disease_name="Test Disease",
            confidence_score=0.80,
            confidence_level=ConfidenceLevel.HIGH,
            treatment_cultural=["Remove infected leaves"],
            treatment_organic=["Apply neem oil"],
            treatment_conventional=["Use fungicide"],
            status=ProcessingStatus.SUCCESS
        )
        output = format_normal_result(result, "Temperate")
        
        assert "Remove infected leaves" in output
        assert "neem oil" in output
        assert "fungicide" in output

    def test_format_includes_prevention(self):
        """Test that prevention tips are included."""
        result = DiagnosisResult(
            disease_name="Test Disease",
            confidence_score=0.80,
            confidence_level=ConfidenceLevel.HIGH,
            prevention=["Practice crop rotation", "Remove plant debris"],
            status=ProcessingStatus.SUCCESS
        )
        output = format_normal_result(result, "Temperate")
        
        assert "crop rotation" in output

    def test_format_includes_processing_time(self):
        """Test that processing time is included."""
        result = DiagnosisResult(
            disease_name="Test Disease",
            confidence_score=0.80,
            confidence_level=ConfidenceLevel.HIGH,
            processing_time=12.5,
            status=ProcessingStatus.SUCCESS
        )
        output = format_normal_result(result, "Temperate")
        
        assert "12." in output  # Should show processing time

    def test_format_includes_disclaimer(self):
        """Test that disclaimer is included."""
        result = DiagnosisResult(
            disease_name="Test Disease",
            confidence_score=0.80,
            confidence_level=ConfidenceLevel.HIGH,
            status=ProcessingStatus.SUCCESS
        )
        output = format_normal_result(result, "Temperate")
        
        assert "informational" in output.lower()


class TestFormatLowConfidenceMessage:
    """Tests for format_low_confidence_message function."""

    def test_low_confidence_warning_present(self):
        """Test that low confidence warning is shown."""
        result = DiagnosisResult(
            disease_name="Uncertain Diagnosis",
            confidence_score=0.35,
            confidence_level=ConfidenceLevel.LOW,
            status=ProcessingStatus.LOW_CONFIDENCE
        )
        output = format_low_confidence_message(result, "Temperate")
        
        assert "Low Confidence" in output or "low confidence" in output.lower()

    def test_improvement_tips_present(self):
        """Test that image improvement tips are shown."""
        result = DiagnosisResult(
            disease_name="Uncertain Diagnosis",
            confidence_score=0.40,
            confidence_level=ConfidenceLevel.LOW,
            status=ProcessingStatus.LOW_CONFIDENCE
        )
        output = format_low_confidence_message(result, "Temperate")
        
        # Should include tips for improving image quality
        assert "lighting" in output.lower() or "image" in output.lower()

    def test_expert_help_suggested(self):
        """Test that expert help is suggested."""
        result = DiagnosisResult(
            disease_name="Uncertain Diagnosis",
            confidence_score=0.30,
            confidence_level=ConfidenceLevel.LOW,
            status=ProcessingStatus.LOW_CONFIDENCE
        )
        output = format_low_confidence_message(result, "Temperate")
        
        # Should suggest consulting experts
        assert "expert" in output.lower() or "extension" in output.lower()

    def test_possible_diagnosis_shown_if_available(self):
        """Test that possible diagnosis is shown even with low confidence."""
        result = DiagnosisResult(
            disease_name="Possible Early Blight",
            confidence_score=0.45,
            confidence_level=ConfidenceLevel.LOW,
            status=ProcessingStatus.LOW_CONFIDENCE
        )
        output = format_low_confidence_message(result, "Temperate")
        
        assert "Early Blight" in output


class TestFormatUnsupportedPlantMessage:
    """Tests for format_unsupported_plant_message function."""

    def test_unsupported_message_present(self):
        """Test that unsupported plant message is shown."""
        result = DiagnosisResult(
            disease_name="Unrecognized Plant",
            confidence_score=0.30,
            confidence_level=ConfidenceLevel.LOW,
            status=ProcessingStatus.UNSUPPORTED
        )
        output = format_unsupported_plant_message(result, "Temperate")
        
        assert "not recognized" in output.lower() or "unsupported" in output.lower()

    def test_supported_plants_listed(self):
        """Test that supported plants are listed."""
        result = DiagnosisResult(
            disease_name="Unknown Species",
            confidence_score=0.25,
            confidence_level=ConfidenceLevel.LOW,
            status=ProcessingStatus.UNSUPPORTED
        )
        output = format_unsupported_plant_message(result, "Temperate")
        
        # Should mention supported plants
        assert "Tomato" in output or "Potato" in output or "Apple" in output

    def test_tips_for_recognition(self):
        """Test that recognition tips are included."""
        result = DiagnosisResult(
            disease_name="Unknown Plant",
            confidence_score=0.30,
            confidence_level=ConfidenceLevel.LOW,
            status=ProcessingStatus.UNSUPPORTED
        )
        output = format_unsupported_plant_message(result, "Temperate")
        
        assert "leaf" in output.lower() or "image" in output.lower()


class TestDiagnosisResultDataClass:
    """Tests for DiagnosisResult dataclass methods."""

    def test_is_healthy_method(self):
        """Test is_healthy() method."""
        healthy = DiagnosisResult(disease_name="Healthy Plant")
        assert healthy.is_healthy()
        
        diseased = DiagnosisResult(disease_name="Early Blight")
        assert not diseased.is_healthy()

    def test_is_low_confidence_method(self):
        """Test is_low_confidence() method."""
        low = DiagnosisResult(confidence_score=0.35)
        assert low.is_low_confidence()
        
        high = DiagnosisResult(confidence_score=0.85)
        assert not high.is_low_confidence()

    def test_is_high_confidence_method(self):
        """Test is_high_confidence() method."""
        high = DiagnosisResult(confidence_score=0.90)
        assert high.is_high_confidence()
        
        low = DiagnosisResult(confidence_score=0.60)
        assert not low.is_high_confidence()

    def test_is_unsupported_plant_method(self):
        """Test is_unsupported_plant() method."""
        unsupported = DiagnosisResult(disease_name="Unknown Plant Species")
        assert unsupported.is_unsupported_plant()
        
        supported = DiagnosisResult(disease_name="Early Blight")
        assert not supported.is_unsupported_plant()

    def test_can_download_report(self):
        """Test can_download_report() method."""
        # Success case
        success = DiagnosisResult(
            disease_name="Early Blight",
            status=ProcessingStatus.SUCCESS
        )
        assert success.can_download_report()
        
        # Error case
        error = DiagnosisResult(
            disease_name="Unknown",
            status=ProcessingStatus.ERROR
        )
        assert not error.can_download_report()
        
        # Unknown disease case
        unknown = DiagnosisResult(
            disease_name="Unknown",
            status=ProcessingStatus.SUCCESS
        )
        assert not unknown.can_download_report()

    def test_default_timestamp_generated(self):
        """Test that timestamp is auto-generated."""
        result = DiagnosisResult()
        assert result.timestamp != ""
        assert len(result.timestamp) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
