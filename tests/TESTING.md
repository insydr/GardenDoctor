# Garden Doctor - Testing Guide

**Comprehensive Testing Documentation**

*Last Updated: March 2026*

---

## Overview

This document provides a complete guide to testing the Garden Doctor application, including:

- **Test Suite Structure**: How tests are organized
- **Running Tests**: Commands and options
- **Test Categories**: Unit, integration, and manual tests
- **CI/CD Integration**: Automated testing pipeline
- **Test Data**: Available fixtures and test images

---

## Quick Start

### Run All Tests

```bash
# Navigate to project directory
cd GardenDoctor

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=app --cov-report=html
```

### Run Specific Test Categories

```bash
# Unit tests only (fast)
pytest -m unit

# Integration tests only
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

### Run Specific Test Files

```bash
# Test confidence logic
pytest tests/test_confidence.py -v

# Test output formatting
pytest tests/test_formatting.py -v

# Test diagnosis function
pytest tests/test_diagnose.py -v

# Test integration
pytest tests/test_integration.py -v
```

---

## Test Suite Structure

```
tests/
├── __init__.py              # Package marker
├── conftest.py              # Shared fixtures and configuration
├── test_confidence.py       # Confidence threshold logic tests
├── test_formatting.py       # Output formatting function tests
├── test_diagnose.py         # diagnose_plant() function tests
├── test_integration.py      # Gradio app integration tests
└── MANUAL_TEST_SCENARIOS.md # Manual QA test scenarios
```

---

## Test Categories

### 1. Unit Tests (`@pytest.mark.unit`)

Fast, isolated tests for individual functions:

| File | What's Tested |
|------|---------------|
| `test_confidence.py` | `determine_confidence_level()`, `get_confidence_badge()`, `get_confidence_interpretation()` |
| `test_formatting.py` | `parse_diagnosis_response()`, `format_diagnosis_result()`, `format_normal_result()` |
| `test_diagnose.py` | `diagnose_plant()` with various inputs |

**Run unit tests:**
```bash
pytest -m unit -v
```

### 2. Integration Tests (`@pytest.mark.integration`)

Tests that verify components work together:

| File | What's Tested |
|------|---------------|
| `test_integration.py` | Gradio app creation, API endpoints, end-to-end flows |

**Run integration tests:**
```bash
pytest -m integration -v
```

### 3. Manual Tests

Human-executed test scenarios documented in:
- `tests/MANUAL_TEST_SCENARIOS.md`

---

## Test Fixtures

Shared fixtures are defined in `conftest.py`:

### Image Fixtures

| Fixture | Description |
|---------|-------------|
| `sample_image` | Valid 336x336 RGB image |
| `sample_image_bytes` | Image as bytes |
| `small_image` | Image below minimum size (30x30) |
| `large_image` | Large image for resize testing (1920x1080) |
| `non_rgb_image` | RGBA image for conversion testing |

### Response Fixtures

| Fixture | Description |
|---------|-------------|
| `mock_high_confidence_response` | Well-formed high confidence model response |
| `mock_medium_confidence_response` | Medium confidence model response |
| `mock_low_confidence_response` | Low confidence model response |
| `mock_healthy_response` | Healthy plant response |
| `mock_unsupported_plant_response` | Unrecognized plant response |
| `mock_malformed_response` | Invalid response for error testing |

### Model Fixtures

| Fixture | Description |
|---------|-------------|
| `mock_model_manager` | Mock ModelManager in mock mode |
| `gradio_app` | Gradio Blocks instance |
| `gradio_client` | Gradio test client |

---

## Test Coverage Goals

| Component | Target Coverage | Current |
|-----------|-----------------|---------|
| Confidence logic | 95% | - |
| Output formatting | 90% | - |
| Diagnosis function | 85% | - |
| Model manager | 80% | - |
| Overall | 85% | - |

### Generate Coverage Report

```bash
# Generate HTML coverage report
pytest --cov=app --cov-report=html

# View report
open htmlcov/index.html
```

---

## Writing New Tests

### Test File Naming

- Files: `test_*.py`
- Classes: `Test*`
- Functions: `test_*`

### Test Structure Template

```python
"""
Tests for [component name].
"""

import pytest
from app import function_to_test


class TestFunctionName:
    """Tests for function_name."""
    
    def test_normal_case(self):
        """Test normal operation."""
        result = function_to_test(valid_input)
        assert result == expected_output
    
    def test_edge_case(self):
        """Test edge case behavior."""
        result = function_to_test(edge_case_input)
        assert result == expected_edge_output
    
    def test_error_handling(self):
        """Test error handling."""
        with pytest.raises(ExpectedException):
            function_to_test(invalid_input)
```

### Using Fixtures

```python
class TestWithFixtures:
    """Tests using shared fixtures."""
    
    def test_with_sample_image(self, sample_image):
        """Test using sample image fixture."""
        # sample_image is automatically provided
        assert sample_image.size == (336, 336)
    
    def test_with_mock_manager(self, mock_model_manager):
        """Test using mock manager fixture."""
        # mock_model_manager is automatically provided
        assert mock_model_manager.mock_mode == True
```

### Parametrized Tests

```python
@pytest.mark.parametrize("score,expected", [
    (0.90, ConfidenceLevel.HIGH),
    (0.65, ConfidenceLevel.MEDIUM),
    (0.35, ConfidenceLevel.LOW),
])
def test_confidence_levels(score, expected):
    """Parametrized test for confidence levels."""
    result = determine_confidence_level(score)
    assert result == expected
```

---

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: pytest --cov=app --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Pre-commit Hook

```bash
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest -m unit
        language: system
        pass_filenames: false
        always_run: true
```

---

## Test Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GARDEN_DOCTOR_MOCK` | `true` | Enable mock mode for testing |
| `GARDEN_DOCTOR_TIMEOUT` | `10` | Reduced timeout for tests |
| `HF_HOME` | `/tmp/hf_cache` | Hugging Face cache directory |

---

## Debugging Failed Tests

### View Full Output

```bash
pytest -v --tb=long
```

### Enter Debugger on Failure

```bash
pytest --pdb
```

### Run Single Test with Output

```bash
pytest tests/test_confidence.py::TestDetermineConfidenceLevel::test_high_confidence_at_threshold -v -s
```

### Print Statements in Tests

```python
def test_with_debug(sample_image, mock_model_manager, capsys):
    """Test with captured output."""
    result = diagnose_plant(sample_image, "Temperate", mock_model_manager)
    
    # Print to stdout
    print(f"Result: {result}")
    
    # Capture and assert on output
    captured = capsys.readouterr()
    assert "Expected text" in captured.out
```

---

## Common Issues and Solutions

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'app'`

**Solution**: Ensure you're in the project root directory:
```bash
cd GardenDoctor
pytest
```

### Mock Mode Not Active

**Problem**: Tests try to load real model

**Solution**: Verify environment variable:
```bash
export GARDEN_DOCTOR_MOCK=true
pytest
```

### Fixture Not Found

**Problem**: `fixture 'xxx' not found`

**Solution**: Ensure fixture is defined in `conftest.py` or imported:
```python
# In conftest.py
@pytest.fixture
def my_fixture():
    return "value"
```

### Slow Tests

**Problem**: Tests take too long

**Solution**: Skip slow tests during development:
```bash
pytest -m "not slow"
```

---

## Test Maintenance

### Regular Tasks

| Task | Frequency |
|------|-----------|
| Update test images | As needed |
| Review coverage reports | Weekly |
| Update mock responses | When model changes |
| Review flaky tests | Weekly |
| Update manual test scenarios | Per release |

### Adding New Test Images

1. Add image to `tests/test_data/` or `examples/`
2. Create fixture in `conftest.py` if needed
3. Update relevant test files

---

## Test Reports

### HTML Report

```bash
pytest --html=report.html --self-contained-html
```

### JUnit XML (for CI)

```bash
pytest --junitxml=junit.xml
```

### JSON Report

```bash
pip install pytest-json-report
pytest --json-report --json-report-file=report.json
```

---

## Further Reading

- [pytest Documentation](https://docs.pytest.org/)
- [Gradio Testing Guide](https://www.gradio.app/guides/testing)
- [Python Testing Best Practices](https://realpython.com/python-testing/)

---

## Contact

For questions about testing:

- Create an issue on GitHub
- Contact the development team
- Check `MANUAL_TEST_SCENARIOS.md` for QA procedures
