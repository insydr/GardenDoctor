"""
Garden Doctor: Plant Disease & Care Assistant
An AI-powered plant disease detection and care recommendation application.

This application uses a fine-tuned LLaVA vision-language model to identify
plant diseases from leaf images and provide treatment recommendations.

PRD Reference: 
- Section 5.1 Feature 4: Confidence Threshold Handling
- Section 5.2 Feature 5: Example Gallery
- Section 5.2 Feature 6: PDF Report Export
"""

import os
import re
import time
import logging
import tempfile
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO

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

# PDF Configuration
PDF_AUTHOR = "Garden Doctor AI"
PDF_CREATOR = "Garden Doctor - Plant Disease Detection System"
PDF_TITLE = "Plant Disease Diagnosis Report"


# =============================================================================
# Confidence Thresholds (PRD FR-4.1-4.4)
# =============================================================================

CONFIDENCE_THRESHOLDS = {
    "low": 0.5,
    "high": 0.8
}

# Supported plants (PRD Appendix 12.1)
SUPPORTED_PLANTS = [
    "Tomato", "Potato", "Apple", "Grape", "Corn",
    "Pepper", "Strawberry", "Cherry", "Peach", "Orange"
]


# =============================================================================
# Example Images Configuration (PRD Feature 5)
# =============================================================================

EXAMPLE_IMAGES = [
    {
        "path": "examples/tomato_healthy.jpg",
        "climate": "Temperate",
        "expected_disease": "Healthy",
        "description": "Healthy tomato leaf - no disease symptoms"
    },
    {
        "path": "examples/tomato_early_blight.jpg",
        "climate": "Temperate",
        "expected_disease": "Early Blight",
        "description": "Tomato leaf with early blight - target-like spots"
    },
    {
        "path": "examples/potato_late_blight.jpg",
        "climate": "Temperate",
        "expected_disease": "Late Blight",
        "description": "Potato leaf with late blight - water-soaked lesions"
    },
    {
        "path": "examples/apple_scab.jpg",
        "climate": "Temperate",
        "expected_disease": "Apple Scab",
        "description": "Apple leaf with scab - olive-brown velvety spots"
    },
    {
        "path": "examples/corn_rust.jpg",
        "climate": "Tropical",
        "expected_disease": "Common Rust",
        "description": "Corn leaf with rust - reddish-brown pustules"
    },
    {
        "path": "examples/grape_black_rot.jpg",
        "climate": "Tropical",
        "expected_disease": "Black Rot",
        "description": "Grape leaf with black rot - circular lesions"
    }
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
# Error Messages
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
    """Structured diagnosis result."""
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
    image: Optional[Image.Image] = None
    climate: str = "Temperate"
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def is_healthy(self) -> bool:
        return "healthy" in self.disease_name.lower()
    
    def is_low_confidence(self) -> bool:
        return self.confidence_score < CONFIDENCE_THRESHOLDS["low"]
    
    def is_high_confidence(self) -> bool:
        return self.confidence_score >= CONFIDENCE_THRESHOLDS["high"]
    
    def is_unsupported_plant(self) -> bool:
        unsupported_indicators = [
            "unknown", "unrecognized", "not identified", 
            "cannot determine", "unable to identify"
        ]
        return any(indicator in self.disease_name.lower() for indicator in unsupported_indicators)
    
    def can_download_report(self) -> bool:
        """Check if a downloadable report can be generated."""
        return (
            self.status in [ProcessingStatus.SUCCESS, ProcessingStatus.LOW_CONFIDENCE] 
            and self.disease_name != "Unknown"
        )


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
        if image.mode != "RGB":
            image = image.convert("RGB")
        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        return image
    
    def generate_response(self, image: Image.Image, prompt: str, max_new_tokens: int = 768) -> str:
        if self.mock_mode:
            return self._generate_mock_response(prompt)
        
        if not self.is_loaded:
            raise RuntimeError("Model not loaded.")
        
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
            
        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            raise
    
    def _generate_mock_response(self, prompt: str) -> str:
        climate = "Temperate"
        for zone in CLIMATE_ZONES.keys():
            if zone.lower() in prompt.lower():
                climate = zone
                break
        
        return f"""DISEASE: Early Blight
CONFIDENCE: High
SYMPTOMS: Dark brown circular spots with concentric rings forming distinctive target-like patterns on the leaves. Yellowing of leaf tissue around the infected areas, starting from the lower leaves. Lesions may merge together creating large dead patches. In severe cases, premature defoliation occurs.
CAUSE: Early blight is caused by the fungus Alternaria solani. The pathogen survives in infected plant debris and soil for up to a year. It thrives in warm, humid conditions with temperatures between 24-29°C (75-84°F) and high moisture levels.

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
• Use disease-resistant varieties appropriate for {climate} climate
• Practice 3-4 year crop rotation avoiding solanaceous crops
• Remove and destroy all plant debris at end of growing season"""


# =============================================================================
# PDF Report Generator (PRD Feature 6)
# =============================================================================

class PDFReportGenerator:
    """
    Generate PDF reports for plant disease diagnosis.
    
    Uses fpdf2 for PDF generation with support for:
    - Images and thumbnails
    - Multi-page reports
    - Custom formatting
    """
    
    def __init__(self):
        self.title = "Plant Disease Diagnosis Report"
        self.author = PDF_AUTHOR
    
    def generate_report(
        self,
        diagnosis: DiagnosisResult,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a PDF report for a diagnosis.
        
        Args:
            diagnosis: DiagnosisResult object
            output_path: Optional path to save PDF. If None, creates temp file.
            
        Returns:
            Path to the generated PDF file
        """
        try:
            from fpdf import FPDF
        except ImportError:
            logger.error("fpdf not installed. Install with: pip install fpdf2")
            raise ImportError("PDF generation requires fpdf2. Install with: pip install fpdf2")
        
        # Create PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # Set metadata
        pdf.set_title(self.title)
        pdf.set_author(self.author)
        pdf.set_creator(PDF_CREATOR)
        
        # Header with branding
        self._add_header(pdf)
        
        # Diagnosis summary section
        self._add_diagnosis_summary(pdf, diagnosis)
        
        # Image section (if available)
        if diagnosis.image:
            pdf.add_page()
            self._add_image_section(pdf, diagnosis.image)
        
        # Treatment recommendations
        pdf.add_page()
        self._add_treatment_section(pdf, diagnosis)
        
        # Prevention section
        self._add_prevention_section(pdf, diagnosis)
        
        # Footer disclaimer
        self._add_footer(pdf)
        
        # Save PDF
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                tempfile.gettempdir(),
                f"garden_doctor_report_{timestamp}.pdf"
            )
        
        pdf.output(output_path)
        logger.info(f"PDF report saved: {output_path}")
        
        return output_path
    
    def _add_header(self, pdf):
        """Add header with branding."""
        # Title
        pdf.set_font("Helvetica", "B", 24)
        pdf.set_text_color(34, 139, 34)  # Forest Green
        pdf.cell(0, 15, "Garden Doctor AI", ln=True, align="C")
        
        pdf.set_font("Helvetica", "", 14)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 8, "Plant Disease Diagnosis Report", ln=True, align="C")
        
        # Decorative line
        pdf.set_draw_color(144, 238, 144)  # Light Green
        pdf.set_line_width(1)
        pdf.line(20, pdf.get_y() + 5, 190, pdf.get_y() + 5)
        pdf.ln(15)
        
        # Timestamp
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", ln=True, align="R")
        pdf.ln(10)
    
    def _add_diagnosis_summary(self, pdf, diagnosis: DiagnosisResult):
        """Add diagnosis summary section."""
        # Section header
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(34, 139, 34)
        pdf.cell(0, 10, "Diagnosis Summary", ln=True)
        pdf.ln(5)
        
        # Disease name
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(0, 0, 0)
        
        if diagnosis.is_healthy():
            pdf.cell(0, 8, f"Result: {diagnosis.disease_name}", ln=True)
            pdf.set_font("Helvetica", "", 12)
            pdf.set_text_color(34, 139, 34)
            pdf.cell(0, 8, "Your plant appears to be healthy!", ln=True)
        else:
            pdf.cell(0, 8, f"Disease Identified: {diagnosis.disease_name}", ln=True)
        
        pdf.ln(5)
        
        # Confidence
        pdf.set_font("Helvetica", "", 12)
        pdf.set_text_color(0, 0, 0)
        
        confidence_text = f"Confidence Level: {diagnosis.confidence_level.value} ({diagnosis.confidence_score:.0%})"
        if diagnosis.is_high_confidence():
            pdf.set_text_color(34, 139, 34)
            confidence_text += " - High reliability"
        elif diagnosis.is_low_confidence():
            pdf.set_text_color(220, 20, 60)
            confidence_text += " - Results uncertain"
        else:
            pdf.set_text_color(184, 134, 11)
            confidence_text += " - Moderate reliability"
        
        pdf.cell(0, 8, confidence_text, ln=True)
        
        # Climate zone
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 8, f"Climate Zone: {diagnosis.climate}", ln=True)
        pdf.cell(0, 8, f"Analysis Time: {diagnosis.processing_time:.1f} seconds", ln=True)
        
        pdf.ln(10)
        
        # Symptoms
        if diagnosis.symptoms:
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(34, 139, 34)
            pdf.cell(0, 8, "Symptoms Observed:", ln=True)
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(60, 60, 60)
            pdf.multi_cell(0, 6, diagnosis.symptoms)
            pdf.ln(5)
        
        # Cause
        if diagnosis.cause:
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(34, 139, 34)
            pdf.cell(0, 8, "Cause:", ln=True)
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(60, 60, 60)
            pdf.multi_cell(0, 6, diagnosis.cause)
    
    def _add_image_section(self, pdf, image: Image.Image):
        """Add image section with thumbnail."""
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(34, 139, 34)
        pdf.cell(0, 10, "Analyzed Image", ln=True)
        pdf.ln(5)
        
        # Save image to bytes
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Add image as thumbnail (max 100mm width)
        pdf.image(img_byte_arr, x=30, w=150)
    
    def _add_treatment_section(self, pdf, diagnosis: DiagnosisResult):
        """Add treatment recommendations section."""
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(34, 139, 34)
        pdf.cell(0, 10, "Treatment Recommendations", ln=True)
        pdf.ln(5)
        
        # Cultural treatments
        if diagnosis.treatment_cultural:
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(0, 100, 0)
            pdf.cell(0, 8, "Cultural Practices:", ln=True)
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(60, 60, 60)
            for i, step in enumerate(diagnosis.treatment_cultural, 1):
                pdf.multi_cell(0, 6, f"  {i}. {step}")
            pdf.ln(5)
        
        # Organic treatments
        if diagnosis.treatment_organic:
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(0, 100, 0)
            pdf.cell(0, 8, "Organic Treatments:", ln=True)
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(60, 60, 60)
            for i, step in enumerate(diagnosis.treatment_organic, 1):
                pdf.multi_cell(0, 6, f"  {i}. {step}")
            pdf.ln(5)
        
        # Conventional treatments
        if diagnosis.treatment_conventional:
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(0, 100, 0)
            pdf.cell(0, 8, "Conventional Options:", ln=True)
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(60, 60, 60)
            for i, step in enumerate(diagnosis.treatment_conventional, 1):
                pdf.multi_cell(0, 6, f"  {i}. {step}")
            pdf.ln(5)
    
    def _add_prevention_section(self, pdf, diagnosis: DiagnosisResult):
        """Add prevention section."""
        if not diagnosis.prevention:
            return
        
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(34, 139, 34)
        pdf.cell(0, 10, f"Prevention for {diagnosis.climate} Climate", ln=True)
        pdf.ln(3)
        
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(60, 60, 60)
        for tip in diagnosis.prevention:
            pdf.multi_cell(0, 6, f"  * {tip}")
    
    def _add_footer(self, pdf):
        """Add footer with disclaimer."""
        pdf.ln(15)
        
        # Decorative line
        pdf.set_draw_color(144, 238, 144)
        pdf.line(20, pdf.get_y(), 190, pdf.get_y())
        pdf.ln(10)
        
        # Disclaimer
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(128, 128, 128)
        pdf.multi_cell(0, 5, 
            "DISCLAIMER: This diagnosis is for informational purposes only and does not replace "
            "professional agricultural consultation. For serious plant health issues, please consult "
            "a professional agronomist or your local agricultural extension service."
        )
        
        pdf.ln(10)
        
        # Branding
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(34, 139, 34)
        pdf.cell(0, 6, "Generated by Garden Doctor AI", ln=True, align="C")
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 5, "https://huggingface.co/spaces/garden-doctor", ln=True, align="C")
        pdf.cell(0, 5, f"Model: {MODEL_ID}", ln=True, align="C")


# =============================================================================
# Helper Functions
# =============================================================================

def determine_confidence_level(score: float) -> ConfidenceLevel:
    if score >= CONFIDENCE_THRESHOLDS["high"]:
        return ConfidenceLevel.HIGH
    elif score >= CONFIDENCE_THRESHOLDS["low"]:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW


def get_confidence_badge(level: ConfidenceLevel) -> str:
    badges = {
        ConfidenceLevel.HIGH: "🟢 **High Confidence**",
        ConfidenceLevel.MEDIUM: "🟡 **Moderate Confidence**",
        ConfidenceLevel.LOW: "🔴 **Low Confidence**",
        ConfidenceLevel.UNKNOWN: "⚪ **Unknown**"
    }
    return badges.get(level, "⚪ **Unknown**")


def get_confidence_interpretation(score: float) -> str:
    level = determine_confidence_level(score)
    interpretations = {
        ConfidenceLevel.HIGH: "Results are reliable. Follow treatment recommendations with confidence.",
        ConfidenceLevel.MEDIUM: "Results are moderately reliable. Consider consulting additional resources.",
        ConfidenceLevel.LOW: "Results are uncertain. Please review image quality tips below.",
        ConfidenceLevel.UNKNOWN: "Unable to determine confidence level."
    }
    return interpretations.get(level, "")


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
    
    disease_match = re.search(r"DISEASE:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    if disease_match:
        disease_name = disease_match.group(1).strip()
    
    confidence_match = re.search(r"CONFIDENCE:\s*(High|Medium|Low)", response, re.IGNORECASE)
    if confidence_match:
        conf_str = confidence_match.group(1).strip().capitalize()
        confidence_score = {"High": 0.85, "Medium": 0.65, "Low": 0.35}.get(conf_str, 0.5)
    
    confidence_level = determine_confidence_level(confidence_score)
    
    symptoms_match = re.search(r"SYMPTOMS:\s*(.+?)(?=CAUSE:|$)", response, re.IGNORECASE | re.DOTALL)
    cause_match = re.search(r"CAUSE:\s*(.+?)(?=CULTURAL|TREATMENT|PREVENTION|$)", response, re.IGNORECASE | re.DOTALL)
    
    if symptoms_match:
        symptoms = symptoms_match.group(1).strip()
    if cause_match:
        cause = cause_match.group(1).strip()
    
    cultural_match = re.search(r"CULTURAL_TREATMENTS:\s*(.+?)(?=ORGANIC_|CONVENTIONAL_|PREVENTION:|$)", response, re.IGNORECASE | re.DOTALL)
    if cultural_match:
        treatment_cultural = [step.strip() for step in re.findall(r"\d+\.\s*(.+?)(?=\n\d+\.|$)", cultural_match.group(1), re.DOTALL)]
        if not treatment_cultural:
            treatment_cultural = [line.strip().lstrip("0123456789. -") for line in cultural_match.group(1).split("\n") if line.strip()]
    
    organic_match = re.search(r"ORGANIC_TREATMENTS:\s*(.+?)(?=CONVENTIONAL_|PREVENTION:|$)", response, re.IGNORECASE | re.DOTALL)
    if organic_match:
        treatment_organic = [step.strip() for step in re.findall(r"\d+\.\s*(.+?)(?=\n\d+\.|$)", organic_match.group(1), re.DOTALL)]
    
    conventional_match = re.search(r"CONVENTIONAL_TREATMENTS:\s*(.+?)(?=PREVENTION:|$)", response, re.IGNORECASE | re.DOTALL)
    if conventional_match:
        treatment_conventional = [step.strip() for step in re.findall(r"\d+\.\s*(.+?)(?=\n\d+\.|$)", conventional_match.group(1), re.DOTALL)]
    
    prevention_match = re.search(r"PREVENTION:\s*(.+?)$", response, re.IGNORECASE | re.DOTALL)
    if prevention_match:
        prevention_text = prevention_match.group(1).strip()
        prevention = [tip.strip().lstrip("• -") for tip in prevention_text.split("\n") if tip.strip()]
    
    status = ProcessingStatus.SUCCESS
    if confidence_score < CONFIDENCE_THRESHOLDS["low"]:
        status = ProcessingStatus.LOW_CONFIDENCE
    
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


def format_diagnosis_result(result: Optional[DiagnosisResult], climate: str) -> str:
    """Format diagnosis result with appropriate messages."""
    if result is None:
        return ERROR_MESSAGES["no_image"]
    
    if result.status == ProcessingStatus.ERROR:
        return ERROR_MESSAGES.get(result.error_message, ERROR_MESSAGES["model_error"])
    
    if result.is_unsupported_plant():
        return format_unsupported_plant_message(result, climate)
    
    if result.confidence_score < CONFIDENCE_THRESHOLDS["low"]:
        return format_low_confidence_message(result, climate)
    
    return format_normal_result(result, climate)


def format_low_confidence_message(result: DiagnosisResult, climate: str) -> str:
    sections = []
    sections.append(ERROR_MESSAGES["low_confidence"].format(confidence=result.confidence_score))
    
    if result.disease_name and result.disease_name != "Unknown":
        sections.append("\n---\n")
        sections.append("### 🤔 Possible Diagnosis")
        sections.append(f"**{result.disease_name}** (Confidence: {result.confidence_score:.1%})")
        sections.append("\n*Note: This diagnosis has low confidence.*\n")
    
    sections.append("\n---\n")
    sections.append("### 🧑‍🌾 Need Expert Help?")
    sections.append("- **Local Agricultural Extension Office** - Free expert advice\n- **Master Gardener Programs** - Community experts\n- **Professional Agronomist** - For commercial operations")
    
    return "\n".join(sections)


def format_unsupported_plant_message(result: DiagnosisResult, climate: str) -> str:
    sections = []
    sections.append(ERROR_MESSAGES["unsupported_plant"])
    sections.append("\n---\n")
    sections.append("### 📸 Tips for Better Recognition")
    sections.append("- Ensure image shows a **single leaf** clearly\n- **Fill the frame** with the leaf (70%+ coverage)\n- Avoid **shadows and reflections**")
    return "\n".join(sections)


def format_normal_result(result: DiagnosisResult, climate: str) -> str:
    sections = []
    
    if result.is_healthy():
        sections.append("## ✅ Great News! Your Plant Appears Healthy\n")
        sections.append(f"**{result.disease_name}**")
    else:
        sections.append(f"## 🩺 Diagnosis: {result.disease_name}\n")
    
    confidence_badge = get_confidence_badge(result.confidence_level)
    interpretation = get_confidence_interpretation(result.confidence_score)
    
    sections.append(f"**Confidence:** {result.confidence_score:.1%} — {confidence_badge}")
    sections.append(f"*{interpretation}*")
    sections.append(f"**Climate:** {climate}")
    sections.append(f"**Analysis Time:** {result.processing_time:.1f}s")
    sections.append("\n---\n")
    
    if result.symptoms or result.cause:
        sections.append("### 🔍 What is this?\n")
        if result.symptoms:
            sections.append(f"**Symptoms:** {result.symptoms}\n")
        if result.cause:
            sections.append(f"\n**Cause:** {result.cause}")
        sections.append("")
    
    has_treatments = result.treatment_cultural or result.treatment_organic or result.treatment_conventional
    
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
    
    if result.prevention:
        sections.append(f"### 🛡️ Prevention for {climate} Climate\n")
        for tip in result.prevention:
            sections.append(f"- {tip}")
        sections.append("")
    
    sections.append("---\n")
    sections.append("*⚠️ This diagnosis is for informational purposes only.*")
    
    return "\n".join(sections)


# =============================================================================
# Main Diagnosis Function
# =============================================================================

def diagnose_plant(
    image: Optional[Image.Image],
    climate: str,
    model_manager: ModelManager,
    progress=gr.Progress()
) -> Tuple[Dict[str, float], str, DiagnosisResult]:
    """Process a plant image and generate diagnosis."""
    
    if image is None:
        logger.warning("No image provided")
        empty_result = DiagnosisResult(status=ProcessingStatus.ERROR, error_message="no_image")
        return {}, ERROR_MESSAGES["no_image"], empty_result
    
    try:
        if not isinstance(image, Image.Image):
            logger.error(f"Invalid image type: {type(image)}")
            error_result = DiagnosisResult(status=ProcessingStatus.ERROR, error_message="invalid_image")
            return {}, ERROR_MESSAGES["invalid_image"], error_result
        
        if image.size[0] < 50 or image.size[1] < 50:
            logger.warning(f"Image too small: {image.size}")
            error_result = DiagnosisResult(status=ProcessingStatus.ERROR, error_message="invalid_image")
            return {}, ERROR_MESSAGES["invalid_image"], error_result
        
        progress(0.1, desc="📷 Validating image...")
        progress(0.15, desc="🔄 Preprocessing...")
        
        processed_image = model_manager.preprocess_image(image)
        
        progress(0.2, desc="🤖 Loading AI model...")
        
        start_time = time.time()
        prompt = DIAGNOSIS_PROMPT.format(climate=climate)
        
        progress(0.3, desc="🔍 Analyzing leaf patterns...")
        
        try:
            response = model_manager.generate_response(processed_image, prompt)
        except Exception as e:
            logger.error(f"Model inference error: {str(e)}")
            error_result = DiagnosisResult(status=ProcessingStatus.ERROR, error_message="model_error")
            return {}, ERROR_MESSAGES["model_error"], error_result
        
        progress(0.7, desc="📊 Processing diagnosis...")
        
        processing_time = time.time() - start_time
        
        progress(0.85, desc="📝 Formatting results...")
        
        diagnosis = parse_diagnosis_response(response, processing_time)
        diagnosis.image = image
        diagnosis.climate = climate
        
        logger.info(f"Diagnosis: {diagnosis.disease_name} ({diagnosis.confidence_level.value}, {diagnosis.confidence_score:.0%})")
        
        progress(0.95, desc="✅ Finalizing...")
        
        confidence_dict = {diagnosis.disease_name: diagnosis.confidence_score}
        formatted_report = format_diagnosis_result(diagnosis, climate)
        
        progress(1.0, desc="Done!")
        
        return confidence_dict, formatted_report, diagnosis
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        error_result = DiagnosisResult(status=ProcessingStatus.ERROR, error_message="model_error")
        return {}, ERROR_MESSAGES["model_error"], error_result


# =============================================================================
# Custom CSS
# =============================================================================

CUSTOM_CSS = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    background: linear-gradient(135deg, #F5F5DC 0%, #e8f5e9 100%) !important;
}

.app-header {
    text-align: center;
    padding: 25px;
    margin-bottom: 20px;
    background: linear-gradient(135deg, #228B22 0%, #2d5a27 100%);
    border-radius: 15px;
    color: white;
    box-shadow: 0 4px 15px rgba(34, 139, 34, 0.3);
}

.app-header h1 { margin: 0; font-size: 2.2em; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); }
.app-header p { margin: 10px 0 0 0; font-size: 1.1em; opacity: 0.95; }

.input-card, .output-card {
    background: white;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    border: 2px solid #90EE90;
}

.primary-btn {
    background: linear-gradient(135deg, #228B22 0%, #2d5a27 100%) !important;
    border: none !important;
    font-size: 1.1em !important;
}

.download-btn {
    background: linear-gradient(135deg, #4169E1 0%, #2d4a9e 100%) !important;
    border: none !important;
    color: white !important;
}

.image-upload {
    border: 3px dashed #90EE90 !important;
    border-radius: 15px !important;
    background: #fafffa !important;
}

.examples-section {
    background: white;
    border-radius: 15px;
    padding: 20px;
    margin-top: 20px;
    border: 2px solid #90EE90;
}

.example-card {
    background: #f0fff0;
    border-radius: 10px;
    padding: 10px;
    margin: 5px;
    border: 1px solid #90EE90;
    transition: transform 0.2s;
}

.example-card:hover {
    transform: scale(1.02);
    box-shadow: 0 2px 10px rgba(34, 139, 34, 0.2);
}

.app-footer {
    text-align: center;
    padding: 15px;
    margin-top: 20px;
    background: #f5f5dc;
    border-radius: 10px;
    font-size: 0.9em;
    color: #666;
}

footer { display: none !important; }

@media (max-width: 768px) {
    .app-header h1 { font-size: 1.8em; }
}
"""


# =============================================================================
# Gradio Interface
# =============================================================================

def create_interface(model_manager: ModelManager) -> gr.Blocks:
    """Create the Gradio interface with examples and PDF export."""
    
    # Initialize PDF generator
    pdf_generator = PDFReportGenerator()
    
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
        
        # State for storing diagnosis result
        diagnosis_state = gr.State(value=None)
        
        # =====================================================================
        # Header
        # =====================================================================
        
        gr.HTML("""
        <div class="app-header">
            <h1>🌿 Garden Doctor AI</h1>
            <p>Your intelligent plant health assistant - Upload a leaf photo for instant diagnosis</p>
        </div>
        """)
        
        status_text = "🔧 Mock Mode" if model_manager.mock_mode else f"✅ Ready ({DEVICE.upper()})"
        gr.HTML(f'<div style="text-align: center; margin-bottom: 15px;"><span style="background: #90EE90; color: #228B22; padding: 5px 15px; border-radius: 20px; font-weight: bold;">{status_text}</span></div>')
        
        # =====================================================================
        # Main Content
        # =====================================================================
        
        with gr.Row(equal_height=True):
            
            # Left Column - Input
            with gr.Column(scale=1, min_width=300):
                gr.HTML('<div class="input-card">')
                
                gr.Markdown("### 📷 Upload Plant Image")
                
                image_input = gr.Image(
                    type="pil",
                    label="Leaf Photo",
                    sources=["upload", "webcam"],
                    height=280,
                    elem_classes=["image-upload"]
                )
                
                gr.Markdown("")
                gr.Markdown("### 🌍 Climate Zone")
                
                climate_input = gr.Dropdown(
                    choices=list(CLIMATE_ZONES.keys()),
                    value="Temperate",
                    label="Climate Zone",
                    info="Select for tailored recommendations"
                )
                
                climate_info = gr.Markdown(
                    value=f"*{CLIMATE_ZONES['Temperate']['description']}*"
                )
                
                climate_input.change(
                    fn=lambda c: f"*{CLIMATE_ZONES.get(c, {}).get('description', '')}*",
                    inputs=[climate_input],
                    outputs=[climate_info]
                )
                
                gr.Markdown("")
                
                with gr.Row():
                    diagnose_btn = gr.Button(
                        "🔬 Diagnose",
                        variant="primary",
                        size="lg",
                        elem_classes=["primary-btn"]
                    )
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="lg")
                
                gr.HTML('</div>')
            
            # Right Column - Output
            with gr.Column(scale=1, min_width=300):
                gr.HTML('<div class="output-card">')
                
                gr.Markdown("### 📊 Results")
                
                confidence_output = gr.Label(label="Confidence", num_top_classes=3)
                
                gr.Markdown("")
                
                diagnosis_output = gr.Markdown(
                    value="*Upload an image and click **Diagnose** to see results...*"
                )
                
                # PDF Download button (hidden initially)
                with gr.Row(visible=False) as download_row:
                    download_btn = gr.Button(
                        "📄 Download PDF Report",
                        variant="primary",
                        size="lg",
                        elem_classes=["download-btn"]
                    )
                
                # File output for PDF download
                pdf_output = gr.File(
                    label="Download Report",
                    visible=False,
                    file_count="single",
                    file_types=[".pdf"]
                )
                
                gr.HTML('</div>')
        
        # =====================================================================
        # Example Gallery (PRD Feature 5)
        # =====================================================================
        
        gr.Markdown("")
        gr.HTML('<div class="examples-section">')
        
        gr.Markdown("### 🖼️ Example Images")
        gr.Markdown("*Click an example to auto-fill and diagnose*")
        
        # Build examples list: [image_path, climate, disease_name]
        examples_list = [
            [ex["path"], ex["climate"], ex["expected_disease"]]
            for ex in EXAMPLE_IMAGES
        ]
        
        gr.Examples(
            examples=examples_list,
            inputs=[image_input, climate_input],
            outputs=[image_input, climate_input],
            fn=lambda img, climate, disease: (img, climate),
            cache_examples=False,
            label="Click to test diagnosis",
            examples_per_page=6,
            run_on_click=True
        )
        
        gr.HTML('</div>')
        
        # =====================================================================
        # Footer
        # =====================================================================
        
        gr.HTML(f"""
        <div class="app-footer">
            <p><strong>Model:</strong> LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection | <strong>Dataset:</strong> PlantVillage</p>
            <p><strong>Supported Crops:</strong> {', '.join(SUPPORTED_PLANTS[:5])}, and more</p>
            <p>⚠️ This tool provides informational guidance only.</p>
        </div>
        """)
        
        # =====================================================================
        # Event Handlers
        # =====================================================================
        
        def on_diagnose(image, climate, progress=gr.Progress()):
            """Handle diagnosis with toast notifications."""
            if image is None:
                gr.Warning("⚠️ Please upload an image first!")
                empty_result = DiagnosisResult(status=ProcessingStatus.ERROR)
                return {}, ERROR_MESSAGES["no_image"], empty_result, gr.Row(visible=False), gr.File(visible=False)
            
            gr.Info("🔍 Analyzing your plant image...")
            
            confidence, report, diagnosis = diagnose_plant(image, climate, model_manager, progress)
            
            # Show appropriate toast
            if diagnosis.status == ProcessingStatus.SUCCESS:
                if diagnosis.is_high_confidence():
                    gr.Info("✅ High confidence diagnosis complete!")
                elif diagnosis.is_low_confidence():
                    gr.Warning("⚠️ Low confidence result - check tips below.")
                else:
                    gr.Info("✅ Diagnosis complete!")
            elif diagnosis.status == ProcessingStatus.LOW_CONFIDENCE:
                gr.Warning("⚠️ Low confidence - review image quality tips.")
            elif diagnosis.status == ProcessingStatus.UNSUPPORTED:
                gr.Warning("🌿 Plant not in database - check supported crops.")
            else:
                gr.Warning("🔧 Processing error - try a different image.")
            
            # Show download button only for successful diagnoses
            show_download = diagnosis.can_download_report()
            
            return (
                confidence, 
                report, 
                diagnosis, 
                gr.Row(visible=show_download),
                gr.File(visible=False)  # Reset file output
            )
        
        def on_download(diagnosis: Optional[DiagnosisResult]):
            """Generate and return PDF report."""
            if diagnosis is None or not diagnosis.can_download_report():
                gr.Warning("⚠️ No diagnosis available to download.")
                return gr.File(visible=False)
            
            try:
                gr.Info("📄 Generating PDF report...")
                
                # Generate PDF
                pdf_path = pdf_generator.generate_report(diagnosis)
                
                gr.Info("✅ PDF report ready for download!")
                
                return gr.File(value=pdf_path, visible=True, label="Diagnosis Report")
                
            except Exception as e:
                logger.error(f"PDF generation error: {str(e)}")
                gr.Warning("🔧 Could not generate PDF. Please try again.")
                return gr.File(visible=False)
        
        def on_clear():
            """Clear all inputs and outputs."""
            return (
                None,  # image
                {},    # confidence
                "*Upload an image and click **Diagnose** to see results...*",  # report
                None,  # diagnosis state
                gr.Row(visible=False),  # hide download button
                gr.File(visible=False)  # hide file output
            )
        
        # Wire buttons
        diagnose_btn.click(
            fn=on_diagnose,
            inputs=[image_input, climate_input],
            outputs=[confidence_output, diagnosis_output, diagnosis_state, download_row, pdf_output],
            api_name="diagnose"
        )
        
        download_btn.click(
            fn=on_download,
            inputs=[diagnosis_state],
            outputs=[pdf_output]
        )
        
        clear_btn.click(
            fn=on_clear,
            outputs=[image_input, confidence_output, diagnosis_output, diagnosis_state, download_row, pdf_output]
        )
        
        demo.queue(max_size=20)
    
    return demo


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("🌿 Garden Doctor AI: Plant Disease & Care Assistant")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    print(f"Device: {DEVICE}")
    print(f"Mock Mode: {MOCK_MODE}")
    print(f"Examples: {len(EXAMPLE_IMAGES)} images")
    print("=" * 60)
    
    model_manager = ModelManager(model_id=MODEL_ID, mock_mode=MOCK_MODE)
    
    if not model_manager.load_model():
        print(f"❌ Failed to load model: {model_manager.load_error}")
        print("⚠️ Starting in mock mode...")
        model_manager.mock_mode = True
        model_manager.is_loaded = True
    
    demo = create_interface(model_manager)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        show_api=True,
    )


if __name__ == "__main__":
    main()
