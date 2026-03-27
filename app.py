"""
Garden Doctor: Plant Disease & Care Assistant
An AI-powered plant disease detection and care recommendation application.

This application uses a fine-tuned LLaVA vision-language model to identify
plant diseases from leaf images and provide treatment recommendations.

PRD Reference: Garden_Doctor_PRD.md Section 5.1 (Core Features)
Feature 4: Confidence Threshold Handling (PRD FR-4.1-4.4)
"""

import os
import re
import time
import logging
from typing import Tuple, Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

import gradio as gr
import torch
from PIL import Image

# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Model configuration
MODEL_ID = "YuchengShi/LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection"

# Device and dtype configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Mock mode for testing without loading the full model
MOCK_MODE = os.environ.get("GARDEN_DOCTOR_MOCK", "true").lower() == "true"

# Image preprocessing configuration
IMAGE_SIZE = 336

# Inference timeout (seconds)
INFERENCE_TIMEOUT = 60


# =============================================================================
# Confidence Thresholds (PRD FR-4.1-4.4)
# =============================================================================

CONFIDENCE_THRESHOLDS = {
    "low": 0.5,      # Below this: show warning + quality tips
    "high": 0.8      # Above this: show high confidence badge
}

# Supported plants (PRD Appendix 12.1)
SUPPORTED_PLANTS = [
    "Tomato", "Potato", "Apple", "Grape", "Corn",
    "Pepper", "Strawberry", "Cherry", "Peach", "Orange"
]


# =============================================================================
# Climate Zone Configuration (PRD FR-2.1-2.4)
# =============================================================================

CLIMATE_ZONES = {
    "Tropical": {
        "description": "Hot, humid climate with year-round growing season",
        "examples": "Southeast Asia, Central America, Caribbean",
        "characteristics": "High humidity, frequent rainfall, warm temperatures"
    },
    "Temperate": {
        "description": "Four distinct seasons with moderate humidity",
        "examples": "Eastern US, Europe, East Asia",
        "characteristics": "Seasonal variation, moderate rainfall, diverse crops"
    },
    "Arid": {
        "description": "Hot, dry climate with low rainfall",
        "examples": "Southwest US, Middle East, Australia",
        "characteristics": "Low humidity, infrequent rain, irrigation needed"
    },
    "Cold": {
        "description": "Short growing season with harsh winters",
        "examples": "Northern Canada, Scandinavia, Russia",
        "characteristics": "Cool summers, long winters, frost risk"
    }
}


# =============================================================================
# Error Messages (PRD FR-4.1-4.4)
# =============================================================================

ERROR_MESSAGES = {
    "no_image": """
### ⚠️ No Image Provided

Please upload a photo of a plant leaf to diagnose.

**Tips:**
- Take a clear, well-lit photo of the affected leaf
- Ensure the leaf fills most of the frame
- Avoid blurry or dark images
""",
    
    "model_error": """
### 🔧 Processing Error

An error occurred while analyzing your image.

**Please try:**
- Uploading a clearer, better-lit image
- Ensuring the image shows a plant leaf clearly
- Using a different image format (JPG, PNG)

If the problem persists, the AI model may be temporarily unavailable.
""",
    
    "timeout": """
### ⏱️ Analysis Timeout

The analysis took too long and was cancelled.

**Please try:**
- Using a smaller image file
- Uploading a simpler image with fewer details
- Trying again in a few moments
""",
    
    "low_confidence": """
### ⚠️ Low Confidence Result

The AI model is **not confident** in this diagnosis ({confidence:.1%}).

**This could mean:**
- The image quality is insufficient
- The plant species is not well-represented in our training data
- The disease symptoms are not clearly visible
- The image may not show a plant leaf

**Try these improvements:**
- ✅ Use better lighting (natural daylight is best)
- ✅ Focus on the most affected area of the leaf
- ✅ Ensure the leaf fills at least 70% of the frame
- ✅ Avoid shadows and reflections
- ✅ Take multiple photos from different angles
""",
    
    "unsupported_plant": """
### 🌿 Plant Not Recognized

This plant may not be in our disease database.

**We currently support these crops:**

| Fruits | Vegetables | Others |
|--------|-----------|--------|
| 🍎 Apple | 🍅 Tomato | 🌽 Corn |
| 🍇 Grape | 🥔 Potato | 🫘 Soybean |
| 🍊 Orange | 🌶️ Pepper | 🎃 Squash |
| 🍑 Peach | 🍓 Strawberry | |
| 🍒 Cherry | 🫐 Blueberry | |
| 🍇 Raspberry | | |

**Tips:**
- Try uploading an image of a supported crop
- Ensure the leaf is clearly visible
- Consider consulting a local agricultural extension service
""",
    
    "invalid_image": """
### 🚫 Invalid Image

The uploaded file could not be processed as an image.

**Please:**
- Upload a valid image file (JPG, PNG, WebP)
- Ensure the file is not corrupted
- Try a different image
"""
}


# =============================================================================
# Image Quality Tips (PRD FR-4.2)
# =============================================================================

IMAGE_QUALITY_TIPS = """
### 📷 Tips for Better Results

To improve diagnosis accuracy, please ensure your image meets these criteria:

| Factor | Good ✅ | Avoid ❌ |
|--------|---------|----------|
| **Lighting** | Natural daylight, even lighting | Dark rooms, harsh shadows |
| **Focus** | Sharp, clear details | Blurry, out-of-focus |
| **Framing** | Leaf fills 70%+ of frame | Leaf too small, too far |
| **Background** | Plain, neutral background | Cluttered, distracting |
| **Subject** | Single, clear leaf | Multiple overlapping leaves |

**Best Practices:**
1. Photograph in morning or late afternoon light
2. Hold camera steady or use a tripod
3. Include both healthy and affected areas if possible
4. Capture the most affected part of the leaf
5. Take photos from multiple angles if uncertain
"""


# =============================================================================
# Data Classes
# =============================================================================

class ConfidenceLevel(Enum):
    """Confidence level enumeration."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    UNKNOWN = "Unknown"


class ProcessingStatus(Enum):
    """Processing status for state tracking."""
    IDLE = "idle"
    PROCESSING = "processing"
    SUCCESS = "success"
    ERROR = "error"
    LOW_CONFIDENCE = "low_confidence"
    UNSUPPORTED = "unsupported"


@dataclass
class DiagnosisResult:
    """
    Structured diagnosis result.
    
    Attributes:
        disease_name: Identified disease or "Healthy"
        confidence_score: Numerical confidence (0.0-1.0)
        confidence_level: HIGH/MEDIUM/LOW classification
        symptoms: Description of visible symptoms
        cause: Explanation of disease cause
        treatment_cultural: Cultural practice treatments
        treatment_organic: Organic/natural treatments
        treatment_conventional: Chemical/conventional treatments
        prevention: Preventive measures list
        raw_response: Original model output
        processing_time: Time taken for inference
        status: Processing status
        error_message: Error message if any
    """
    disease_name: str = "Unknown"
    confidence_score: float = 0.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.UNKNOWN
    symptoms: str = ""
    cause: str = ""
    treatment_cultural: List[str] = field(default_factory=list)
    treatment_organic: List[str] = field(default_factory=list)
    treatment_conventional: List[str] = field(default_factory=list)
    prevention: List[str] = field(default_factory=list)
    raw_response: str = ""
    processing_time: float = 0.0
    status: ProcessingStatus = ProcessingStatus.SUCCESS
    error_message: str = ""
    
    def is_healthy(self) -> bool:
        """Check if the diagnosis indicates a healthy plant."""
        return "healthy" in self.disease_name.lower()
    
    def is_low_confidence(self) -> bool:
        """Check if confidence is below threshold."""
        return self.confidence_score < CONFIDENCE_THRESHOLDS["low"]
    
    def is_high_confidence(self) -> bool:
        """Check if confidence is above high threshold."""
        return self.confidence_score >= CONFIDENCE_THRESHOLDS["high"]
    
    def is_unsupported_plant(self) -> bool:
        """Check if the plant is likely unsupported."""
        unsupported_indicators = [
            "unknown", "unrecognized", "not identified", 
            "cannot determine", "unable to identify"
        ]
        return any(indicator in self.disease_name.lower() for indicator in unsupported_indicators)


# =============================================================================
# Prompt Template
# =============================================================================

DIAGNOSIS_PROMPT = """You are an expert plant pathologist. Analyze this plant leaf image for disease detection.

Please provide your analysis in this EXACT format:

DISEASE: [disease name or "Healthy Plant"]
CONFIDENCE: [High/Medium/Low]
SYMPTOMS: [detailed description of visible symptoms]
CAUSE: [explanation of what causes this condition]

CULTURAL_TREATMENTS:
1. [first cultural practice treatment]
2. [second cultural practice treatment]
3. [third cultural practice treatment if applicable]

ORGANIC_TREATMENTS:
1. [first organic treatment option]
2. [second organic treatment option]

CONVENTIONAL_TREATMENTS:
1. [first conventional/chemical treatment]
2. [second conventional/chemical treatment]

PREVENTION:
• [first prevention measure for {climate} climate]
• [second prevention measure]
• [third prevention measure]

Consider that this plant is growing in a {climate} climate zone. Provide practical, actionable advice appropriate for that climate's conditions."""


# =============================================================================
# Model Manager
# =============================================================================

class ModelManager:
    """Manages model loading and inference for plant disease detection."""
    
    def __init__(self, model_id: str = MODEL_ID, mock_mode: bool = MOCK_MODE):
        self.model_id = model_id
        self.mock_mode = mock_mode
        self.model = None
        self.processor = None
        self.is_loaded = False
        self.load_error: Optional[str] = None
        
        logger.info(f"ModelManager initialized (mock_mode={mock_mode}, device={DEVICE})")
    
    def load_model(self) -> bool:
        """Load the model and processor."""
        if self.mock_mode:
            logger.info("Mock mode enabled - skipping model loading")
            self.is_loaded = True
            return True
        
        logger.info(f"Loading model from {self.model_id}...")
        
        try:
            from transformers import LlavaForConditionalGeneration, AutoProcessor
            
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            
            load_kwargs = {
                "torch_dtype": DTYPE,
                "low_cpu_mem_usage": True,
            }
            
            if torch.cuda.is_available():
                load_kwargs["device_map"] = "auto"
                load_kwargs["torch_dtype"] = torch.float16
            
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_id, **load_kwargs
            )
            
            if not torch.cuda.is_available():
                self.model = self.model.to(DEVICE)
            
            self.model.eval()
            self.is_loaded = True
            logger.info("Model loaded successfully!")
            return True
            
        except Exception as e:
            self.load_error = f"Error loading model: {str(e)}"
            logger.error(self.load_error)
            return False
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for model input (336x336, RGB)."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        
        return image
    
    def generate_response(
        self, 
        image: Image.Image, 
        prompt: str, 
        max_new_tokens: int = 768,
        timeout: int = INFERENCE_TIMEOUT
    ) -> str:
        """Generate model response for an image and prompt."""
        if self.mock_mode:
            return self._generate_mock_response(prompt)
        
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            conversation = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
            ]
            text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(text=text_prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            generated_text = self.processor.decode(output[0], skip_special_tokens=True)
            response = generated_text.split("[/INST]")[-1].strip() if "[/INST]" in generated_text else generated_text.strip()
            return response
            
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError("GPU out of memory. Try using a smaller image or CPU mode.")
        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            raise
    
    def _generate_mock_response(self, prompt: str) -> str:
        """Generate mock response for testing."""
        climate = "Temperate"
        for zone in CLIMATE_ZONES.keys():
            if zone.lower() in prompt.lower():
                climate = zone
                break
        
        return f"""DISEASE: Early Blight
CONFIDENCE: High
SYMPTOMS: Dark brown circular spots with concentric rings forming distinctive target-like patterns on the leaves. Yellowing of leaf tissue around the infected areas, starting from the lower leaves. Lesions may merge together creating large dead patches. In severe cases, premature defoliation occurs.
CAUSE: Early blight is caused by the fungus Alternaria solani. The pathogen survives in infected plant debris and soil for up to a year. It thrives in warm, humid conditions with temperatures between 24-29°C (75-84°F) and high moisture levels. The disease spreads through wind-borne spores, water splash, and contaminated tools.

CULTURAL_TREATMENTS:
1. Remove and destroy all infected leaves immediately to prevent spore spread
2. Improve air circulation by proper plant spacing (24-36 inches apart) and staking plants
3. Water at the base of plants early in the day to keep foliage dry

ORGANIC_TREATMENTS:
1. Apply copper-based fungicide every 7-10 days during active disease period
2. Use neem oil spray as a natural fungicide alternative

CONVENTIONAL_TREATMENTS:
1. Apply chlorothalonil or mancozeb fungicide according to label instructions
2. Use systemic fungicides containing azoxystrobin for severe infections

PREVENTION:
• Use disease-resistant varieties appropriate for {climate} climate
• Practice 3-4 year crop rotation avoiding solanaceous crops
• Remove and destroy all plant debris at end of growing season
• Apply organic mulch to prevent soil splash onto lower leaves"""


# =============================================================================
# Result Formatting Functions (PRD Feature 4)
# =============================================================================

def determine_confidence_level(score: float) -> ConfidenceLevel:
    """
    Determine confidence level based on score.
    
    Args:
        score: Confidence score (0.0-1.0)
        
    Returns:
        ConfidenceLevel enum value
    """
    if score >= CONFIDENCE_THRESHOLDS["high"]:
        return ConfidenceLevel.HIGH
    elif score >= CONFIDENCE_THRESHOLDS["low"]:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW


def get_confidence_badge(level: ConfidenceLevel) -> str:
    """Get visual badge for confidence level."""
    badges = {
        ConfidenceLevel.HIGH: "🟢 **High Confidence**",
        ConfidenceLevel.MEDIUM: "🟡 **Moderate Confidence**",
        ConfidenceLevel.LOW: "🔴 **Low Confidence**",
        ConfidenceLevel.UNKNOWN: "⚪ **Unknown**"
    }
    return badges.get(level, "⚪ **Unknown**")


def get_confidence_interpretation(score: float) -> str:
    """Get human-readable interpretation of confidence score."""
    level = determine_confidence_level(score)
    
    interpretations = {
        ConfidenceLevel.HIGH: "Results are reliable. Follow treatment recommendations with confidence.",
        ConfidenceLevel.MEDIUM: "Results are moderately reliable. Consider consulting additional resources.",
        ConfidenceLevel.LOW: "Results are uncertain. Please review image quality tips below.",
        ConfidenceLevel.UNKNOWN: "Unable to determine confidence level."
    }
    
    return interpretations.get(level, "")


def format_diagnosis_result(
    result: Optional[DiagnosisResult],
    climate: str,
    confidence_thresholds: Dict[str, float] = CONFIDENCE_THRESHOLDS
) -> str:
    """
    Format diagnosis result with appropriate messages based on confidence.
    
    Implements PRD FR-4.1-4.4: Confidence Threshold Handling
    
    Args:
        result: DiagnosisResult object or None
        climate: Selected climate zone
        confidence_thresholds: Dictionary with 'low' and 'high' thresholds
        
    Returns:
        Formatted markdown string for display
    """
    # Handle None result (no diagnosis performed)
    if result is None:
        return ERROR_MESSAGES["no_image"]
    
    # Handle error status
    if result.status == ProcessingStatus.ERROR:
        return ERROR_MESSAGES.get(result.error_message, ERROR_MESSAGES["model_error"])
    
    # Handle timeout
    if result.status == ProcessingStatus.ERROR and "timeout" in result.error_message.lower():
        return ERROR_MESSAGES["timeout"]
    
    # Handle unsupported plant
    if result.is_unsupported_plant() or result.status == ProcessingStatus.UNSUPPORTED:
        return format_unsupported_plant_message(result, climate)
    
    # Handle low confidence (PRD FR-4.1)
    if result.confidence_score < confidence_thresholds["low"]:
        return format_low_confidence_message(result, climate)
    
    # Format normal result
    return format_normal_result(result, climate)


def format_low_confidence_message(result: DiagnosisResult, climate: str) -> str:
    """
    Format message for low confidence results.
    
    Implements PRD FR-4.1, FR-4.2
    """
    sections = []
    
    # Warning header
    sections.append(ERROR_MESSAGES["low_confidence"].format(
        confidence=result.confidence_score
    ))
    
    # Still show partial results if available
    if result.disease_name and result.disease_name != "Unknown":
        sections.append("\n---\n")
        sections.append("### 🤔 Possible Diagnosis")
        sections.append(f"**{result.disease_name}** (Confidence: {result.confidence_score:.1%})")
        sections.append("\n*Note: This diagnosis has low confidence. Please verify with additional sources.*\n")
    
    # Image quality tips
    sections.append("\n---\n")
    sections.append(IMAGE_QUALITY_TIPS)
    
    # Professional consultation suggestion
    sections.append("\n---\n")
    sections.append("### 🧑‍🌾 Need Expert Help?")
    sections.append("""
For uncertain diagnoses, consider:
- **Local Agricultural Extension Office** - Free expert advice for gardeners
- **Master Gardener Programs** - Community plant health experts
- **Professional Agronomist** - For commercial operations
- **Plant Disease Clinics** - University diagnostic services
""")
    
    return "\n".join(sections)


def format_unsupported_plant_message(result: DiagnosisResult, climate: str) -> str:
    """
    Format message for unsupported/unrecognized plants.
    
    Implements PRD FR-4.3
    """
    sections = []
    
    sections.append(ERROR_MESSAGES["unsupported_plant"])
    
    # Suggest alternatives
    sections.append("\n---\n")
    sections.append("### 📸 Tips for Better Recognition")
    sections.append("""
If you believe this is a supported crop:
- Ensure the image shows a **single leaf** clearly
- The leaf should be **well-lit and in focus**
- **Fill the frame** with the leaf (70%+ coverage)
- Avoid **shadows and reflections**
- Try a photo with **visible symptoms** if present
""")
    
    return "\n".join(sections)


def format_normal_result(result: DiagnosisResult, climate: str) -> str:
    """
    Format normal diagnosis result with full details.
    
    Implements PRD FR-3.1-3.6
    """
    sections = []
    
    # Header with confidence badge
    if result.is_healthy():
        sections.append("## ✅ Great News! Your Plant Appears Healthy\n")
        sections.append(f"**{result.disease_name}**")
    else:
        sections.append(f"## 🩺 Diagnosis: {result.disease_name}\n")
    
    # Confidence section with interpretation
    confidence_badge = get_confidence_badge(result.confidence_level)
    interpretation = get_confidence_interpretation(result.confidence_score)
    
    sections.append(f"**Confidence:** {result.confidence_score:.1%} — {confidence_badge}")
    sections.append(f"*{interpretation}*")
    sections.append(f"**Climate:** {climate}")
    sections.append(f"**Analysis Time:** {result.processing_time:.1f}s")
    sections.append("\n---\n")
    
    # What is this section
    if result.symptoms or result.cause:
        sections.append("### 🔍 What is this?\n")
        if result.symptoms:
            sections.append(f"**Symptoms Observed:**\n{result.symptoms}\n")
        if result.cause:
            sections.append(f"\n**Cause:**\n{result.cause}")
        sections.append("")
    
    # Treatment options
    has_treatments = (
        result.treatment_cultural or 
        result.treatment_organic or 
        result.treatment_conventional
    )
    
    if has_treatments:
        sections.append("### 🌱 Treatment Options\n")
        
        if result.treatment_cultural:
            sections.append("**Cultural Practices:**")
            for step in result.treatment_cultural:
                sections.append(f"- {step}")
            sections.append("")
        
        if result.treatment_organic:
            sections.append("**Organic Treatments:**")
            for step in result.treatment_organic:
                sections.append(f"- {step}")
            sections.append("")
        
        if result.treatment_conventional:
            sections.append("**Conventional Options:**")
            for step in result.treatment_conventional:
                sections.append(f"- {step}")
            sections.append("")
    
    # Prevention
    if result.prevention:
        sections.append(f"### 🛡️ Prevention for {climate} Climate\n")
        for tip in result.prevention:
            sections.append(f"- {tip}")
        sections.append("")
    
    # Disclaimer
    sections.append("---\n")
    sections.append("*⚠️ This diagnosis is for informational purposes only. For serious plant health issues, please consult a professional agronomist or your local agricultural extension service.*")
    
    return "\n".join(sections)


# =============================================================================
# Response Parsing
# =============================================================================

def parse_diagnosis_response(response: str, processing_time: float) -> DiagnosisResult:
    """Parse model response into structured diagnosis result."""
    disease_name = "Unknown"
    confidence_score = 0.5
    symptoms = ""
    cause = ""
    treatment_cultural = []
    treatment_organic = []
    treatment_conventional = []
    prevention = []
    
    # Extract disease name
    disease_match = re.search(r"DISEASE:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    if disease_match:
        disease_name = disease_match.group(1).strip()
    
    # Extract confidence
    confidence_match = re.search(r"CONFIDENCE:\s*(High|Medium|Low)", response, re.IGNORECASE)
    if confidence_match:
        conf_str = confidence_match.group(1).strip().capitalize()
        confidence_score = {"High": 0.85, "Medium": 0.65, "Low": 0.35}.get(conf_str, 0.5)
    
    # Determine confidence level
    confidence_level = determine_confidence_level(confidence_score)
    
    # Extract symptoms and cause
    symptoms_match = re.search(r"SYMPTOMS:\s*(.+?)(?=CAUSE:|$)", response, re.IGNORECASE | re.DOTALL)
    cause_match = re.search(r"CAUSE:\s*(.+?)(?=CULTURAL|TREATMENT|PREVENTION|$)", response, re.IGNORECASE | re.DOTALL)
    
    if symptoms_match:
        symptoms = symptoms_match.group(1).strip()
    if cause_match:
        cause = cause_match.group(1).strip()
    
    # Extract cultural treatments
    cultural_match = re.search(
        r"CULTURAL_TREATMENTS:\s*(.+?)(?=ORGANIC_|CONVENTIONAL_|PREVENTION:|$)", 
        response, re.IGNORECASE | re.DOTALL
    )
    if cultural_match:
        treatment_cultural = [
            step.strip() for step in 
            re.findall(r"\d+\.\s*(.+?)(?=\n\d+\.|$)", cultural_match.group(1), re.DOTALL)
        ]
        if not treatment_cultural:
            treatment_cultural = [
                line.strip().lstrip("0123456789. -")
                for line in cultural_match.group(1).split("\n")
                if line.strip() and not line.strip().startswith("[")
            ]
    
    # Extract organic treatments
    organic_match = re.search(
        r"ORGANIC_TREATMENTS:\s*(.+?)(?=CONVENTIONAL_|PREVENTION:|$)", 
        response, re.IGNORECASE | re.DOTALL
    )
    if organic_match:
        treatment_organic = [
            step.strip() for step in 
            re.findall(r"\d+\.\s*(.+?)(?=\n\d+\.|$)", organic_match.group(1), re.DOTALL)
        ]
    
    # Extract conventional treatments
    conventional_match = re.search(
        r"CONVENTIONAL_TREATMENTS:\s*(.+?)(?=PREVENTION:|$)", 
        response, re.IGNORECASE | re.DOTALL
    )
    if conventional_match:
        treatment_conventional = [
            step.strip() for step in 
            re.findall(r"\d+\.\s*(.+?)(?=\n\d+\.|$)", conventional_match.group(1), re.DOTALL)
        ]
    
    # Extract prevention tips
    prevention_match = re.search(r"PREVENTION:\s*(.+?)$", response, re.IGNORECASE | re.DOTALL)
    if prevention_match:
        prevention_text = prevention_match.group(1).strip()
        prevention = [
            tip.strip().lstrip("• -")
            for tip in prevention_text.split("\n")
            if tip.strip() and tip.strip() not in ["•", "-"]
        ]
    
    # Determine status
    status = ProcessingStatus.SUCCESS
    if confidence_score < CONFIDENCE_THRESHOLDS["low"]:
        status = ProcessingStatus.LOW_CONFIDENCE
    
    # Check for unsupported plant indicators
    unsupported_indicators = ["unknown", "unrecognized", "not identified", "cannot determine"]
    if any(indicator in disease_name.lower() for indicator in unsupported_indicators):
        status = ProcessingStatus.UNSUPPORTED
    
    return DiagnosisResult(
        disease_name=disease_name,
        confidence_score=confidence_score,
        confidence_level=confidence_level,
        symptoms=symptoms,
        cause=cause,
        treatment_cultural=treatment_cultural,
        treatment_organic=treatment_organic,
        treatment_conventional=treatment_conventional,
        prevention=prevention,
        raw_response=response,
        processing_time=processing_time,
        status=status
    )


# =============================================================================
# Main Diagnosis Function
# =============================================================================

def diagnose_plant(
    image: Optional[Image.Image],
    climate: str,
    model_manager: ModelManager,
    progress=gr.Progress()
) -> Tuple[Dict[str, float], str, ProcessingStatus]:
    """
    Process a plant image and generate diagnosis with comprehensive error handling.
    
    Implements PRD FR-1.1-1.6, FR-3.1-3.6, FR-4.1-4.4
    
    Args:
        image: PIL Image of the plant leaf
        climate: Selected climate zone
        model_manager: ModelManager instance
        progress: Gradio progress tracker
        
    Returns:
        Tuple of (confidence_dict, formatted_report, status)
    """
    # =====================================================================
    # Input Validation (PRD FR-4.1)
    # =====================================================================
    
    if image is None:
        logger.warning("No image provided")
        return {}, ERROR_MESSAGES["no_image"], ProcessingStatus.ERROR
    
    try:
        # Validate image
        if not isinstance(image, Image.Image):
            logger.error(f"Invalid image type: {type(image)}")
            return {}, ERROR_MESSAGES["invalid_image"], ProcessingStatus.ERROR
        
        # Check image dimensions
        if image.size[0] < 50 or image.size[1] < 50:
            logger.warning(f"Image too small: {image.size}")
            return {}, ERROR_MESSAGES["invalid_image"], ProcessingStatus.ERROR
        
        # =====================================================================
        # Progress Tracking
        # =====================================================================
        
        progress(0.1, desc="📷 Validating image...")
        time.sleep(0.3)  # Brief pause for UI update
        
        progress(0.15, desc="🔄 Preprocessing image...")
        
        # Preprocess image
        try:
            processed_image = model_manager.preprocess_image(image)
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            return {}, ERROR_MESSAGES["invalid_image"], ProcessingStatus.ERROR
        
        progress(0.2, desc="🤖 Loading AI model...")
        
        # Start timing
        start_time = time.time()
        
        # Construct prompt
        prompt = DIAGNOSIS_PROMPT.format(climate=climate)
        
        progress(0.3, desc="🔍 Analyzing leaf patterns...")
        
        # =====================================================================
        # Model Inference with Error Handling
        # =====================================================================
        
        try:
            response = model_manager.generate_response(processed_image, prompt)
        except RuntimeError as e:
            error_msg = str(e)
            logger.error(f"Model inference error: {error_msg}")
            
            if "out of memory" in error_msg.lower():
                return {}, ERROR_MESSAGES["timeout"], ProcessingStatus.ERROR
            else:
                return {}, ERROR_MESSAGES["model_error"], ProcessingStatus.ERROR
        
        except Exception as e:
            logger.error(f"Unexpected inference error: {str(e)}")
            return {}, ERROR_MESSAGES["model_error"], ProcessingStatus.ERROR
        
        progress(0.7, desc="📊 Processing diagnosis...")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Check for timeout
        if processing_time > INFERENCE_TIMEOUT:
            logger.warning(f"Inference timeout: {processing_time:.1f}s")
            return {}, ERROR_MESSAGES["timeout"], ProcessingStatus.ERROR
        
        # =====================================================================
        # Parse Response
        # =====================================================================
        
        progress(0.85, desc="📝 Formatting results...")
        
        diagnosis = parse_diagnosis_response(response, processing_time)
        
        logger.info(
            f"Diagnosis: {diagnosis.disease_name} "
            f"({diagnosis.confidence_level.value}, {diagnosis.confidence_score:.0%}) "
            f"in {processing_time:.2f}s"
        )
        
        progress(0.95, desc="✅ Finalizing...")
        
        # =====================================================================
        # Format Output (PRD FR-3.1-3.6)
        # =====================================================================
        
        # Confidence label
        confidence_dict = {diagnosis.disease_name: diagnosis.confidence_score}
        
        # Format report with threshold handling
        formatted_report = format_diagnosis_result(
            diagnosis, 
            climate, 
            CONFIDENCE_THRESHOLDS
        )
        
        progress(1.0, desc="Done!")
        
        return confidence_dict, formatted_report, diagnosis.status
        
    except Exception as e:
        logger.error(f"Unexpected error in diagnose_plant: {str(e)}", exc_info=True)
        return {}, ERROR_MESSAGES["model_error"], ProcessingStatus.ERROR


# =============================================================================
# Custom CSS
# =============================================================================

CUSTOM_CSS = """
/* Base styles */
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    background: linear-gradient(135deg, #F5F5DC 0%, #e8f5e9 100%) !important;
}

/* Header styling */
.app-header {
    text-align: center;
    padding: 25px;
    margin-bottom: 20px;
    background: linear-gradient(135deg, #228B22 0%, #2d5a27 100%);
    border-radius: 15px;
    color: white;
    box-shadow: 0 4px 15px rgba(34, 139, 34, 0.3);
}

.app-header h1 {
    margin: 0;
    font-size: 2.2em;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
}

.app-header p {
    margin: 10px 0 0 0;
    font-size: 1.1em;
    opacity: 0.95;
}

/* Card styling */
.input-card, .output-card {
    background: white;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    border: 2px solid #90EE90;
    transition: box-shadow 0.3s ease;
}

.input-card:hover, .output-card:hover {
    box-shadow: 0 4px 20px rgba(34, 139, 34, 0.2);
}

/* Button styling */
.primary-btn {
    background: linear-gradient(135deg, #228B22 0%, #2d5a27 100%) !important;
    border: none !important;
    font-size: 1.1em !important;
    padding: 12px 30px !important;
    border-radius: 10px !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}

.primary-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(34, 139, 34, 0.4) !important;
}

.secondary-btn {
    background: #f5f5f5 !important;
    border: 2px solid #228B22 !important;
    color: #228B22 !important;
    border-radius: 10px !important;
}

/* Image upload styling */
.image-upload {
    border: 3px dashed #90EE90 !important;
    border-radius: 15px !important;
    background: #fafffa !important;
    transition: border-color 0.3s !important;
}

.image-upload:hover {
    border-color: #228B22 !important;
}

/* Confidence badges */
.high-confidence { color: #228B22; }
.medium-confidence { color: #8B4513; }
.low-confidence { color: #DC143C; }

/* Status indicators */
.status-ready { background: #90EE90; color: #228B22; }
.status-processing { background: #FFBF00; color: #8B4513; }
.status-error { background: #FFB6C1; color: #DC143C; }

/* Examples gallery */
.examples-section {
    background: white;
    border-radius: 15px;
    padding: 20px;
    margin-top: 20px;
    border: 2px solid #90EE90;
}

/* Footer */
.app-footer {
    text-align: center;
    padding: 15px;
    margin-top: 20px;
    background: #f5f5dc;
    border-radius: 10px;
    font-size: 0.9em;
    color: #666;
}

/* Hide default footer */
footer { display: none !important; }

/* Responsive */
@media (max-width: 768px) {
    .app-header h1 { font-size: 1.8em; }
    .input-card, .output-card { padding: 15px; }
}

/* Warning box */
.warning-box {
    background: #FFF3CD;
    border-left: 4px solid #FFBF00;
    padding: 15px;
    margin: 10px 0;
    border-radius: 5px;
}

/* Error box */
.error-box {
    background: #FFE4E1;
    border-left: 4px solid #DC143C;
    padding: 15px;
    margin: 10px 0;
    border-radius: 5px;
}

/* Success box */
.success-box {
    background: #E8F5E9;
    border-left: 4px solid #228B22;
    padding: 15px;
    margin: 10px 0;
    border-radius: 5px;
}
"""


# =============================================================================
# Gradio Interface
# =============================================================================

def create_interface(model_manager: ModelManager) -> gr.Blocks:
    """Create the Gradio interface with comprehensive error handling."""
    
    with gr.Blocks(
        theme=gr.themes.Green(
            primary_hue="green",
            secondary_hue="emerald",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Segoe UI")
        ),
        css=CUSTOM_CSS,
        title="🌿 Garden Doctor AI",
    ) as demo:
        
        # =====================================================================
        # State for tracking processing status
        # =====================================================================
        
        processing_state = gr.State(value=ProcessingStatus.IDLE)
        
        # =====================================================================
        # Header Section
        # =====================================================================
        
        gr.HTML("""
        <div class="app-header">
            <h1>🌿 Garden Doctor AI</h1>
            <p>Your intelligent plant health assistant - Upload a leaf photo for instant disease diagnosis and care recommendations</p>
        </div>
        """)
        
        # Status indicator
        status_text = "🔧 Mock Mode (Testing)" if model_manager.mock_mode else f"✅ Ready ({DEVICE.upper()})"
        gr.HTML(f'<div style="text-align: center; margin-bottom: 15px;"><span class="status-indicator status-ready">{status_text}</span></div>')
        
        # =====================================================================
        # Main Content - Two Column Layout
        # =====================================================================
        
        with gr.Row(equal_height=True):
            
            # -----------------------------------------------------------------
            # LEFT COLUMN - Input Section
            # -----------------------------------------------------------------
            
            with gr.Column(scale=1, min_width=300):
                
                gr.HTML('<div class="input-card">')
                
                gr.Markdown("### 📷 Upload Plant Image")
                gr.Markdown("*Take a clear photo of the affected leaf for best results*")
                
                image_input = gr.Image(
                    type="pil",
                    label="Leaf Photo",
                    sources=["upload", "webcam"],
                    height=280,
                    show_label=True,
                    elem_classes=["image-upload"]
                )
                
                gr.Markdown("")
                
                gr.Markdown("### 🌍 Select Your Climate Zone")
                gr.Markdown("*Choose your growing region for tailored recommendations*")
                
                climate_input = gr.Dropdown(
                    choices=list(CLIMATE_ZONES.keys()),
                    value="Temperate",
                    label="Climate Zone",
                    info="Select your climate for region-specific advice",
                    elem_classes=["climate-dropdown"]
                )
                
                # Climate description display
                climate_info = gr.Markdown(
                    value=f"*{CLIMATE_ZONES['Temperate']['description']}*",
                    elem_classes=["climate-info"]
                )
                
                def update_climate_info(climate):
                    info = CLIMATE_ZONES.get(climate, {})
                    return f"*{info.get('description', '')}*\n\n*Examples: {info.get('examples', '')}*"
                
                climate_input.change(
                    fn=update_climate_info,
                    inputs=[climate_input],
                    outputs=[climate_info]
                )
                
                gr.Markdown("")
                
                # Action buttons
                with gr.Row():
                    diagnose_btn = gr.Button(
                        "🔬 Diagnose",
                        variant="primary",
                        size="lg",
                        elem_classes=["primary-btn"]
                    )
                    
                    clear_btn = gr.Button(
                        "🗑️ Clear",
                        variant="secondary",
                        size="lg",
                        elem_classes=["secondary-btn"]
                    )
                
                gr.HTML('</div>')
            
            # -----------------------------------------------------------------
            # RIGHT COLUMN - Output Section
            # -----------------------------------------------------------------
            
            with gr.Column(scale=1, min_width=300):
                
                gr.HTML('<div class="output-card">')
                
                gr.Markdown("### 📊 Diagnosis Results")
                
                confidence_output = gr.Label(
                    label="Disease Confidence",
                    num_top_classes=3,
                    show_label=True,
                    elem_classes=["confidence-label"]
                )
                
                gr.Markdown("")
                
                diagnosis_output = gr.Markdown(
                    value="*Upload an image and click **Diagnose** to see results...*",
                    label="Diagnosis Report",
                    elem_classes=["diagnosis-output"]
                )
                
                gr.HTML('</div>')
        
        # =====================================================================
        # Examples Gallery Section
        # =====================================================================
        
        gr.Markdown("")
        
        gr.HTML('<div class="examples-section">')
        
        gr.Markdown("### 🖼️ Example Images")
        gr.Markdown("*Click on an example to quickly test the diagnosis feature*")
        
        gr.Examples(
            examples=[
                ["examples/tomato_healthy.jpg", "Temperate"],
                ["examples/tomato_early_blight.jpg", "Temperate"],
                ["examples/potato_late_blight.jpg", "Temperate"],
                ["examples/apple_scab.jpg", "Temperate"],
                ["examples/corn_rust.jpg", "Tropical"],
                ["examples/grape_black_rot.jpg", "Tropical"],
            ],
            inputs=[image_input, climate_input],
            label="Example Plant Images",
            examples_per_page=6,
        )
        
        gr.HTML('</div>')
        
        # =====================================================================
        # Footer Section
        # =====================================================================
        
        gr.HTML(f"""
        <div class="app-footer">
            <p>
                <strong>Model:</strong> 
                <a href="https://huggingface.co/YuchengShi/LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection" target="_blank">
                    LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection
                </a>
                &nbsp;|&nbsp;
                <strong>Dataset:</strong> 
                <a href="https://www.plantvillage.psu.edu/" target="_blank">
                    PlantVillage
                </a>
            </p>
            <p>
                <strong>Supported Crops:</strong> {', '.join(SUPPORTED_PLANTS[:5])}, and more
            </p>
            <p>
                ⚠️ <em>This tool provides informational guidance only and does not replace professional agricultural consultation.</em>
            </p>
        </div>
        """)
        
        # =====================================================================
        # Event Handlers with Toast Notifications (PRD Feature 4)
        # =====================================================================
        
        def on_diagnose(
            image: Optional[Image.Image],
            climate: str,
            progress=gr.Progress()
        ) -> Tuple[Dict[str, float], str, ProcessingStatus]:
            """
            Handle diagnose button click with toast notifications.
            
            Implements PRD FR-1.1-1.6, FR-3.1-3.6, FR-4.1-4.4
            """
            # Check for empty input
            if image is None:
                gr.Warning("⚠️ Please upload an image first!")
                return {}, ERROR_MESSAGES["no_image"], ProcessingStatus.ERROR
            
            # Show processing toast
            gr.Info("🔍 Analyzing your plant image...")
            
            # Run diagnosis
            confidence, report, status = diagnose_plant(
                image, climate, model_manager, progress
            )
            
            # Show appropriate toast based on status
            if status == ProcessingStatus.SUCCESS:
                if confidence:
                    # Get the first (top) confidence score
                    top_confidence = list(confidence.values())[0] if confidence else 0
                    
                    if top_confidence >= CONFIDENCE_THRESHOLDS["high"]:
                        gr.Info("✅ Diagnosis complete! High confidence result.")
                    elif top_confidence >= CONFIDENCE_THRESHOLDS["low"]:
                        gr.Info("✅ Diagnosis complete! Moderate confidence result.")
                    else:
                        gr.Warning("⚠️ Low confidence result. Check the tips below.")
            
            elif status == ProcessingStatus.LOW_CONFIDENCE:
                gr.Warning("⚠️ Low confidence diagnosis. Please review the image quality tips.")
            
            elif status == ProcessingStatus.UNSUPPORTED:
                gr.Warning("🌿 This plant may not be in our database. Check supported crops.")
            
            elif status == ProcessingStatus.ERROR:
                gr.Warning("🔧 Processing error. Please try again with a different image.")
            
            return confidence, report, status
        
        def on_clear() -> Tuple[None, Dict[str, float], str, ProcessingStatus]:
            """Handle clear button click."""
            return None, {}, "*Upload an image and click **Diagnose** to see results...*", ProcessingStatus.IDLE
        
        # Wire Diagnose button
        diagnose_btn.click(
            fn=on_diagnose,
            inputs=[image_input, climate_input],
            outputs=[confidence_output, diagnosis_output, processing_state],
            api_name="diagnose"
        )
        
        # Wire Clear button
        clear_btn.click(
            fn=on_clear,
            outputs=[image_input, confidence_output, diagnosis_output, processing_state]
        )
        
        # Enable queue for progress tracking
        demo.queue(max_size=20)
    
    return demo


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for the application."""
    print("=" * 60)
    print("🌿 Garden Doctor AI: Plant Disease & Care Assistant")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    print(f"Device: {DEVICE}")
    print(f"Mock Mode: {MOCK_MODE}")
    print(f"Confidence Thresholds: Low < {CONFIDENCE_THRESHOLDS['low']}, High >= {CONFIDENCE_THRESHOLDS['high']}")
    print(f"Supported Plants: {', '.join(SUPPORTED_PLANTS[:5])}...")
    print("=" * 60)
    
    # Initialize model manager
    model_manager = ModelManager(
        model_id=MODEL_ID,
        mock_mode=MOCK_MODE
    )
    
    # Load model
    if not model_manager.load_model():
        print(f"❌ Failed to load model: {model_manager.load_error}")
        print("⚠️ Starting in mock mode for testing...")
        model_manager.mock_mode = True
        model_manager.is_loaded = True
    
    # Create and launch interface
    demo = create_interface(model_manager)
    
    # Launch configuration
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        show_api=True,
    )


if __name__ == "__main__":
    main()
