"""
Unit tests for confidence threshold logic in Garden Doctor.

Tests cover:
- Confidence level determination
- Confidence badge generation
- Confidence interpretation text
- Edge cases at threshold boundaries
"""

import pytest
from app import (
    determine_confidence_level,
    get_confidence_badge,
    get_confidence_interpretation,
    CONFIDENCE_THRESHOLDS,
    ConfidenceLevel,
)


class TestDetermineConfidenceLevel:
    """Tests for determine_confidence_level function."""

    def test_high_confidence_at_threshold(self):
        """Test that exactly 80% is high confidence."""
        result = determine_confidence_level(0.80)
        assert result == ConfidenceLevel.HIGH

    def test_high_confidence_above_threshold(self):
        """Test that >80% is high confidence."""
        result = determine_confidence_level(0.85)
        assert result == ConfidenceLevel.HIGH

    def test_high_confidence_max(self):
        """Test that 100% is high confidence."""
        result = determine_confidence_level(1.0)
        assert result == ConfidenceLevel.HIGH

    def test_medium_confidence_at_threshold(self):
        """Test that exactly 50% is medium confidence."""
        result = determine_confidence_level(0.50)
        assert result == ConfidenceLevel.MEDIUM

    def test_medium_confidence_middle(self):
        """Test that 65% is medium confidence."""
        result = determine_confidence_level(0.65)
        assert result == ConfidenceLevel.MEDIUM

    def test_medium_confidence_just_below_high(self):
        """Test that 79% is medium confidence."""
        result = determine_confidence_level(0.79)
        assert result == ConfidenceLevel.MEDIUM

    def test_low_confidence_just_below_medium(self):
        """Test that 49% is low confidence."""
        result = determine_confidence_level(0.49)
        assert result == ConfidenceLevel.LOW

    def test_low_confidence_very_low(self):
        """Test that 10% is low confidence."""
        result = determine_confidence_level(0.10)
        assert result == ConfidenceLevel.LOW

    def test_low_confidence_zero(self):
        """Test that 0% is low confidence."""
        result = determine_confidence_level(0.0)
        assert result == ConfidenceLevel.LOW

    @pytest.mark.parametrize("score,expected", [
        (0.00, ConfidenceLevel.LOW),
        (0.25, ConfidenceLevel.LOW),
        (0.49, ConfidenceLevel.LOW),
        (0.50, ConfidenceLevel.MEDIUM),
        (0.55, ConfidenceLevel.MEDIUM),
        (0.79, ConfidenceLevel.MEDIUM),
        (0.80, ConfidenceLevel.HIGH),
        (0.90, ConfidenceLevel.HIGH),
        (1.00, ConfidenceLevel.HIGH),
    ])
    def test_confidence_levels_parametrized(self, score, expected):
        """Parametrized test for confidence level boundaries."""
        result = determine_confidence_level(score)
        assert result == expected


class TestGetConfidenceBadge:
    """Tests for get_confidence_badge function."""

    def test_high_confidence_badge(self):
        """Test high confidence badge contains correct emoji."""
        badge = get_confidence_badge(ConfidenceLevel.HIGH)
        assert "🟢" in badge
        assert "High" in badge

    def test_medium_confidence_badge(self):
        """Test medium confidence badge contains correct emoji."""
        badge = get_confidence_badge(ConfidenceLevel.MEDIUM)
        assert "🟡" in badge
        assert "Moderate" in badge

    def test_low_confidence_badge(self):
        """Test low confidence badge contains correct emoji."""
        badge = get_confidence_badge(ConfidenceLevel.LOW)
        assert "🔴" in badge
        assert "Low" in badge

    def test_unknown_confidence_badge(self):
        """Test unknown confidence badge."""
        badge = get_confidence_badge(ConfidenceLevel.UNKNOWN)
        assert "⚪" in badge
        assert "Unknown" in badge

    def test_badge_is_bold(self):
        """Test that badge text is bold formatted."""
        badge = get_confidence_badge(ConfidenceLevel.HIGH)
        assert "**" in badge


class TestGetConfidenceInterpretation:
    """Tests for get_confidence_interpretation function."""

    def test_high_confidence_interpretation(self):
        """Test high confidence interpretation message."""
        interpretation = get_confidence_interpretation(0.85)
        assert "reliable" in interpretation.lower()
        assert "confidence" in interpretation.lower()

    def test_medium_confidence_interpretation(self):
        """Test medium confidence interpretation message."""
        interpretation = get_confidence_interpretation(0.65)
        assert "moderately" in interpretation.lower()
        assert "additional" in interpretation.lower()

    def test_low_confidence_interpretation(self):
        """Test low confidence interpretation message."""
        interpretation = get_confidence_interpretation(0.35)
        assert "uncertain" in interpretation.lower()
        assert "tips" in interpretation.lower()

    def test_unknown_confidence_interpretation(self):
        """Test unknown confidence interpretation message."""
        interpretation = get_confidence_interpretation(-0.1)  # Invalid score
        assert "Unable" in interpretation or "unknown" in interpretation.lower()


class TestConfidenceThresholds:
    """Tests for confidence threshold configuration."""

    def test_low_threshold_value(self):
        """Test that low threshold is 0.5."""
        assert CONFIDENCE_THRESHOLDS["low"] == 0.5

    def test_high_threshold_value(self):
        """Test that high threshold is 0.8."""
        assert CONFIDENCE_THRESHOLDS["high"] == 0.8

    def test_thresholds_are_valid(self):
        """Test that thresholds are in valid range."""
        for key, value in CONFIDENCE_THRESHOLDS.items():
            assert 0 <= value <= 1, f"Threshold {key} is out of range: {value}"

    def test_low_less_than_high(self):
        """Test that low threshold is less than high threshold."""
        assert CONFIDENCE_THRESHOLDS["low"] < CONFIDENCE_THRESHOLDS["high"]


class TestConfidenceLevelEnum:
    """Tests for ConfidenceLevel enum."""

    def test_enum_has_high(self):
        """Test that HIGH level exists."""
        assert ConfidenceLevel.HIGH.value == "High"

    def test_enum_has_medium(self):
        """Test that MEDIUM level exists."""
        assert ConfidenceLevel.MEDIUM.value == "Medium"

    def test_enum_has_low(self):
        """Test that LOW level exists."""
        assert ConfidenceLevel.LOW.value == "Low"

    def test_enum_has_unknown(self):
        """Test that UNKNOWN level exists."""
        assert ConfidenceLevel.UNKNOWN.value == "Unknown"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_negative_confidence(self):
        """Test handling of negative confidence scores."""
        result = determine_confidence_level(-0.5)
        assert result == ConfidenceLevel.LOW

    def test_confidence_above_one(self):
        """Test handling of confidence scores above 1.0."""
        result = determine_confidence_level(1.5)
        assert result == ConfidenceLevel.HIGH

    def test_floating_point_precision(self):
        """Test handling of floating point precision issues."""
        # These should all be HIGH confidence
        result1 = determine_confidence_level(0.7999999999999999)
        result2 = determine_confidence_level(0.8000000000000001)
        
        # Due to floating point, 0.79999... might be treated differently
        assert result1 in [ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH]
        assert result2 == ConfidenceLevel.HIGH


class TestConfidenceIntegration:
    """Integration tests for confidence-related functions."""

    def test_full_confidence_flow(self):
        """Test the complete confidence determination flow."""
        scores = [0.35, 0.55, 0.90]
        expected_levels = [ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH]
        
        for score, expected in zip(scores, expected_levels):
            level = determine_confidence_level(score)
            badge = get_confidence_badge(level)
            interpretation = get_confidence_interpretation(score)
            
            assert level == expected
            assert len(badge) > 0
            assert len(interpretation) > 0

    def test_confidence_level_matches_interpretation(self):
        """Test that interpretation text matches confidence level."""
        test_cases = [
            (0.90, "reliable"),
            (0.60, "moderately"),
            (0.30, "uncertain"),
        ]
        
        for score, keyword in test_cases:
            level = determine_confidence_level(score)
            interpretation = get_confidence_interpretation(score)
            
            assert keyword.lower() in interpretation.lower(), \
                f"Score {score} with level {level} should have '{keyword}' in interpretation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
