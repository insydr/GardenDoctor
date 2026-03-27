"""
Garden Doctor: Plant Disease & Care Assistant
An AI-powered plant disease detection and care recommendation application.

This application uses a fine-tuned LLaVA vision-language model to identify
plant diseases from leaf images and provide treatment recommendations.

PRD Reference: Garden_Doctor_PRD.md Section 5.1 (Core Features)
"""

import os
import re
import time
import logging
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

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

# =============================================================================
# Confidence Thresholds (PRD FR-4.1-4.4)
# =============================================================================

CONFIDENCE_LOW = 0.5      # Below this: show warning + quality tips
CONFIDENCE_HIGH = 0.8     # Above this: show high confidence badge


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
# Color Scheme (PRD Section 8.2)
# =============================================================================

COLORS = {
    "primary": "#228B22",       # Forest Green
    "secondary": "#8B4513",     # Earth Brown
    "background": "#F5F5DC",    # Off-White
    "accent": "#90EE90",        # Leaf Green
    "warning": "#FFBF00",       # Amber
    "error": "#DC143C",         # Crimson
    "success": "#228B22",       # Green
}


# =============================================================================
# Data Classes
# =============================================================================

class ConfidenceLevel(Enum):
    """Confidence level enumeration."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


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
    """
    disease_name: str
    confidence_score: float
    confidence_level: ConfidenceLevel
    symptoms: str = ""
    cause: str = ""
    treatment_cultural: List[str] = field(default_factory=list)
    treatment_organic: List[str] = field(default_factory=list)
    treatment_conventional: List[str] = field(default_factory=list)
    prevention: List[str] = field(default_factory=list)
    raw_response: str = ""
    processing_time: float = 0.0
    
    def is_healthy(self) -> bool:
        """Check if the diagnosis indicates a healthy plant."""
        return "healthy" in self.disease_name.lower()
    
    def is_low_confidence(self) -> bool:
        """Check if confidence is below threshold."""
        return self.confidence_score < CONFIDENCE_LOW


# =============================================================================
# Image Quality Tips (PRD FR-4.2)
# =============================================================================

IMAGE_QUALITY_TIPS = """
### 📷 Tips for Better Results

To improve diagnosis accuracy, please ensure your image meets these criteria:

1. **Good Lighting** - Take photos in natural daylight or bright, even lighting
2. **Clear Focus** - Ensure the leaf is sharp and in focus
3. **Close-up View** - Fill the frame with the leaf, showing symptoms clearly
4. **Multiple Angles** - Try both sides of the leaf if symptoms are present
5. **Clean Background** - Use a plain background to reduce distraction
6. **Recent Photo** - Use fresh photos showing current symptoms

**Common Issues:**
- 🚫 Blurry or out-of-focus images
- 🚫 Poor lighting (too dark or harsh shadows)
- 🚫 Leaf too small in the frame
- 🚫 Multiple leaves overlapping
- 🚫 Non-plant objects in frame

**Best Practices:**
- ✅ Hold camera steady or use a tripod
- ✅ Photograph in morning or late afternoon light
- ✅ Include both healthy and affected areas if possible
- ✅ Capture the most affected part of the leaf
"""


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
    
    def generate_response(self, image: Image.Image, prompt: str, max_new_tokens: int = 768) -> str:
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
• Use disease-resistant varieties such as 'Mountain Magic' or 'Defiant' for {climate} conditions
• Practice 3-4 year crop rotation avoiding tomatoes, potatoes, and eggplant
• Remove all plant debris at end of season and destroy or hot compost
• Apply organic mulch to prevent soil splash onto lower leaves
• Monitor plants weekly during warm humid weather in {climate} climate"""


# =============================================================================
# Diagnosis Functions
# =============================================================================

def parse_confidence_level(confidence_str: str) -> Tuple[ConfidenceLevel, float]:
    """
    Parse confidence string to level and score.
    
    Args:
        confidence_str: Confidence string from model (High/Medium/Low)
        
    Returns:
        Tuple of (ConfidenceLevel, numerical_score)
    """
    confidence_map = {
        "high": (ConfidenceLevel.HIGH, 0.88),
        "medium": (ConfidenceLevel.MEDIUM, 0.65),
        "low": (ConfidenceLevel.LOW, 0.42),
        "moderate": (ConfidenceLevel.MEDIUM, 0.65),
    }
    return confidence_map.get(confidence_str.lower(), (ConfidenceLevel.MEDIUM, 0.60))


def parse_diagnosis_response(response: str, processing_time: float) -> DiagnosisResult:
    """
    Parse model response into structured diagnosis result.
    
    Args:
        response: Raw model response text
        processing_time: Time taken for inference
        
    Returns:
        Structured DiagnosisResult
    """
    # Default values
    disease_name = "Unknown"
    confidence_score = 0.5
    confidence_level = ConfidenceLevel.MEDIUM
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
    
    # Extract and parse confidence
    confidence_match = re.search(r"CONFIDENCE:\s*(High|Medium|Low|Moderate)", response, re.IGNORECASE)
    if confidence_match:
        conf_str = confidence_match.group(1).strip()
        confidence_level, confidence_score = parse_confidence_level(conf_str)
    
    # Extract symptoms
    symptoms_match = re.search(r"SYMPTOMS:\s*(.+?)(?=CAUSE:|CULTURAL|TREATMENT|$)", response, re.IGNORECASE | re.DOTALL)
    if symptoms_match:
        symptoms = symptoms_match.group(1).strip()
    
    # Extract cause
    cause_match = re.search(r"CAUSE:\s*(.+?)(?=CULTURAL|ORGANIC|CONVENTIONAL|TREATMENT|PREVENTION|$)", response, re.IGNORECASE | re.DOTALL)
    if cause_match:
        cause = cause_match.group(1).strip()
    
    # Extract cultural treatments
    cultural_match = re.search(r"CULTURAL_TREATMENTS?:\s*(.+?)(?=ORGANIC|CONVENTIONAL|PREVENTION|$)", response, re.IGNORECASE | re.DOTALL)
    if cultural_match:
        cultural_text = cultural_match.group(1).strip()
        treatment_cultural = [
            step.strip().lstrip("0123456789. ")
            for step in re.findall(r"\d+\.\s*(.+?)(?=\n\d+\.|$)", cultural_text, re.DOTALL)
        ]
        if not treatment_cultural:
            treatment_cultural = [line.strip().lstrip("0123456789.•- ") 
                                  for line in cultural_text.split("\n") if line.strip()]
    
    # Extract organic treatments
    organic_match = re.search(r"ORGANIC_TREATMENTS?:\s*(.+?)(?=CONVENTIONAL|PREVENTION|$)", response, re.IGNORECASE | re.DOTALL)
    if organic_match:
        organic_text = organic_match.group(1).strip()
        treatment_organic = [
            step.strip().lstrip("0123456789. ")
            for step in re.findall(r"\d+\.\s*(.+?)(?=\n\d+\.|$)", organic_text, re.DOTALL)
        ]
        if not treatment_organic:
            treatment_organic = [line.strip().lstrip("0123456789.•- ") 
                                 for line in organic_text.split("\n") if line.strip()]
    
    # Extract conventional treatments
    conventional_match = re.search(r"CONVENTIONAL_TREATMENTS?:\s*(.+?)(?=PREVENTION|$)", response, re.IGNORECASE | re.DOTALL)
    if conventional_match:
        conventional_text = conventional_match.group(1).strip()
        treatment_conventional = [
            step.strip().lstrip("0123456789. ")
            for step in re.findall(r"\d+\.\s*(.+?)(?=\n\d+\.|$)", conventional_text, re.DOTALL)
        ]
        if not treatment_conventional:
            treatment_conventional = [line.strip().lstrip("0123456789.•- ") 
                                      for line in conventional_text.split("\n") if line.strip()]
    
    # Fallback: Try to parse from general TREATMENT section if specific categories not found
    if not (treatment_cultural or treatment_organic or treatment_conventional):
        treatment_match = re.search(r"TREATMENT:\s*(.+?)(?=PREVENTION|$)", response, re.IGNORECASE | re.DOTALL)
        if treatment_match:
            treatment_text = treatment_match.group(1).strip()
            all_treatments = [
                step.strip().lstrip("0123456789. ")
                for step in re.findall(r"\d+\.\s*(.+?)(?=\n\d+\.|$)", treatment_text, re.DOTALL)
            ]
            # Distribute into categories
            treatment_cultural = all_treatments[:2] if len(all_treatments) > 0 else []
            treatment_organic = all_treatments[2:4] if len(all_treatments) > 2 else []
            treatment_conventional = all_treatments[4:] if len(all_treatments) > 4 else []
    
    # Extract prevention tips
    prevention_match = re.search(r"PREVENTION:\s*(.+?)$", response, re.IGNORECASE | re.DOTALL)
    if prevention_match:
        prevention_text = prevention_match.group(1).strip()
        prevention = [tip.strip().lstrip("• ") for tip in prevention_text.split("\n") if tip.strip()]
    
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
        processing_time=processing_time
    )


def get_confidence_interpretation(score: float) -> str:
    """
    Get confidence interpretation text.
    
    Args:
        score: Confidence score (0.0-1.0)
        
    Returns:
        Interpretation string
    """
    if score >= CONFIDENCE_HIGH:
        return "🟢 **High confidence** - Results are reliable"
    elif score >= CONFIDENCE_LOW:
        return "🟡 **Moderate confidence** - Consider consulting additional resources"
    else:
        return "🔴 **Low confidence** - Please try with a clearer image"


def format_diagnosis_report(diagnosis: DiagnosisResult, climate: str) -> str:
    """
    Format diagnosis result as markdown report per PRD Section 5.1.
    
    Args:
        diagnosis: Structured diagnosis result
        climate: Selected climate zone
        
    Returns:
        Formatted markdown string
    """
    sections = []
    
    # ==========================================================================
    # Header with Diagnosis (PRD FR-3.1, FR-3.2)
    # ==========================================================================
    
    if diagnosis.is_healthy():
        sections.append("### 🩺 Diagnosis: Healthy Plant")
        sections.append("")
        sections.append("Great news! Your plant appears to be **healthy** with no detectable disease symptoms.")
    else:
        sections.append(f"### 🩺 Diagnosis: {diagnosis.disease_name}")
    
    sections.append("")
    
    # ==========================================================================
    # Confidence with Interpretation (PRD FR-3.2, FR-4.1-4.4)
    # ==========================================================================
    
    confidence_interp = get_confidence_interpretation(diagnosis.confidence_score)
    sections.append(f"**Confidence:** {diagnosis.confidence_score:.1%} — {confidence_interp}")
    sections.append(f"**Climate:** {climate}")
    sections.append(f"**Analysis Time:** {diagnosis.processing_time:.1f}s")
    sections.append("")
    
    # ==========================================================================
    # Low Confidence Warning (PRD FR-4.1-4.4)
    # ==========================================================================
    
    if diagnosis.confidence_score < CONFIDENCE_LOW:
        sections.append("---")
        sections.append("### ⚠️ Low Confidence Warning")
        sections.append("")
        sections.append("The AI model is **not confident** in this diagnosis. This could mean:")
        sections.append("- The image quality is insufficient")
        sections.append("- The plant species is not in the training data")
        sections.append("- The disease symptoms are not clearly visible")
        sections.append("")
        sections.append(IMAGE_QUALITY_TIPS)
        sections.append("---")
        sections.append("")
    
    # ==========================================================================
    # Moderate Confidence Note (PRD FR-4.2)
    # ==========================================================================
    
    if CONFIDENCE_LOW <= diagnosis.confidence_score < CONFIDENCE_HIGH:
        sections.append("> 💡 **Note:** This diagnosis has moderate confidence. For critical decisions, consider consulting a professional agronomist or your local agricultural extension service.")
        sections.append("")
    
    # ==========================================================================
    # What is this? / Explanation (PRD FR-3.3)
    # ==========================================================================
    
    if diagnosis.symptoms or diagnosis.cause:
        sections.append("### 🔍 What is this?")
        sections.append("")
        
        if diagnosis.symptoms:
            sections.append("**Symptoms Observed:**")
            sections.append(diagnosis.symptoms)
            sections.append("")
        
        if diagnosis.cause:
            sections.append("**Cause:**")
            sections.append(diagnosis.cause)
            sections.append("")
    
    # ==========================================================================
    # Treatment Options (PRD FR-3.4)
    # ==========================================================================
    
    if not diagnosis.is_healthy():
        sections.append("### 🌱 Treatment Options")
        sections.append("")
        
        # Cultural Practices
        if diagnosis.treatment_cultural:
            sections.append("**Cultural Practices:**")
            for step in diagnosis.treatment_cultural:
                sections.append(f"- {step}")
            sections.append("")
        
        # Organic Treatments
        if diagnosis.treatment_organic:
            sections.append("**Organic Treatments:**")
            for step in diagnosis.treatment_organic:
                sections.append(f"- {step}")
            sections.append("")
        
        # Conventional Options
        if diagnosis.treatment_conventional:
            sections.append("**Conventional Options:**")
            for step in diagnosis.treatment_conventional:
                sections.append(f"- {step}")
            sections.append("")
    
    # ==========================================================================
    # Prevention (PRD FR-3.5)
    # ==========================================================================
    
    if diagnosis.prevention:
        sections.append(f"### 🛡️ Prevention for {climate} Climate")
        sections.append("")
        for tip in diagnosis.prevention:
            sections.append(f"- {tip}")
        sections.append("")
    
    # ==========================================================================
    # Footer Disclaimer (PRD FR-3.6)
    # ==========================================================================
    
    sections.append("---")
    sections.append("")
    sections.append("*⚠️ This diagnosis is for informational purposes only and does not replace professional agricultural consultation. For serious plant health issues, please consult a professional agronomist or your local agricultural extension service.*")
    
    return "\n".join(sections)


def diagnose_plant(
    image: Optional[Image.Image],
    climate: str,
    model_manager: ModelManager,
    progress=gr.Progress()
) -> Tuple[Dict[str, float], str]:
    """
    Process a plant image and generate diagnosis.
    
    This function implements the complete diagnosis pipeline:
    1. Validate input image (PRD FR-1.1-1.3)
    2. Preprocess image to 336x336 (PRD NFR-1.1)
    3. Generate diagnosis with progress tracking
    4. Parse and format results (PRD FR-3.1-3.6)
    5. Apply confidence threshold logic (PRD FR-4.1-4.4)
    
    Args:
        image: PIL Image of the plant leaf
        climate: Selected climate zone
        model_manager: ModelManager instance
        progress: Gradio progress tracker
        
    Returns:
        Tuple of (confidence_dict, formatted_report)
    """
    # ==========================================================================
    # Input Validation (PRD FR-1.1)
    # ==========================================================================
    
    if image is None:
        empty_label = {}
        empty_report = """### ⚠️ No Image Provided

Please upload a leaf image to diagnose.

**Supported formats:** JPEG, PNG, WebP
**Tips:** Use a clear, well-lit photo of the affected leaf for best results."""
        return empty_label, empty_report
    
    try:
        # ======================================================================
        # Image Preprocessing (PRD NFR-1.1)
        # ======================================================================
        
        progress(0.1, desc="📷 Processing image...")
        logger.info(f"Processing image: {image.size}, mode: {image.mode}")
        
        # Convert to RGB and resize
        processed_image = model_manager.preprocess_image(image)
        
        # ======================================================================
        # Model Inference
        # ======================================================================
        
        progress(0.2, desc="🤖 Loading AI model...")
        
        start_time = time.time()
        
        # Construct prompt with climate
        prompt = DIAGNOSIS_PROMPT.format(climate=climate)
        
        progress(0.3, desc="🔍 Analyzing leaf patterns...")
        
        # Generate response
        response = model_manager.generate_response(processed_image, prompt)
        
        progress(0.7, desc="📊 Processing diagnosis...")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # ======================================================================
        # Parse Response
        # ======================================================================
        
        diagnosis = parse_diagnosis_response(response, processing_time)
        
        logger.info(f"Diagnosis: {diagnosis.disease_name} ({diagnosis.confidence_level.value}, {diagnosis.confidence_score:.0%}) in {processing_time:.2f}s")
        
        progress(0.9, desc="📝 Formatting report...")
        
        # ======================================================================
        # Format Output (PRD FR-3.1-3.6)
        # ======================================================================
        
        # Confidence label (PRD FR-3.1)
        confidence_dict = {diagnosis.disease_name: diagnosis.confidence_score}
        
        # Format markdown report (PRD FR-3.2-3.6)
        formatted_report = format_diagnosis_report(diagnosis, climate)
        
        progress(1.0, desc="✅ Done!")
        
        return confidence_dict, formatted_report
        
    except Exception as e:
        error_msg = f"### ❌ Error Processing Image\n\nAn error occurred during diagnosis:\n\n```\n{str(e)}\n```\n\nPlease try again with a different image."
        logger.error(f"Diagnosis error: {str(e)}", exc_info=True)
        return {}, error_msg


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

/* Dropdown styling */
.climate-dropdown label {
    font-weight: 600;
}

/* Label output styling */
.confidence-label {
    border-radius: 15px;
    padding: 15px;
    background: linear-gradient(135deg, #f0fff0 0%, #e8f5e9 100%);
    border: 2px solid #90EE90;
}

/* Markdown output styling */
.diagnosis-output h3 {
    color: #228B22;
    border-bottom: 2px solid #90EE90;
    padding-bottom: 8px;
    margin-top: 15px;
}

.diagnosis-output h4 {
    color: #2d5a27;
}

/* Warning box styling */
.diagnosis-output h3:contains("Warning") {
    color: #FFBF00;
}

/* Examples gallery */
.examples-section {
    background: white;
    border-radius: 15px;
    padding: 20px;
    margin-top: 20px;
    border: 2px solid #90EE90;
}

/* Status indicator */
.status-indicator {
    display: inline-block;
    padding: 5px 15px;
    border-radius: 20px;
    font-size: 0.9em;
    font-weight: bold;
}

.status-ready {
    background: #90EE90;
    color: #228B22;
}

.status-mock {
    background: #FFBF00;
    color: #8B4513;
}

/* Confidence badges */
.high-confidence {
    background: #90EE90;
    color: #228B22;
    padding: 3px 10px;
    border-radius: 5px;
}

.medium-confidence {
    background: #FFF3CD;
    color: #856404;
    padding: 3px 10px;
    border-radius: 5px;
}

.low-confidence {
    background: #F8D7DA;
    color: #721C24;
    padding: 3px 10px;
    border-radius: 5px;
}

/* Footer styling */
.app-footer {
    text-align: center;
    padding: 15px;
    margin-top: 20px;
    background: #F5F5DC;
    border-radius: 10px;
    font-size: 0.9em;
    color: #666;
}

/* Responsive design */
@media (max-width: 768px) {
    .app-header h1 {
        font-size: 1.6em;
    }
    
    .input-card, .output-card {
        padding: 15px;
    }
}

/* Hide default footer */
footer {
    display: none !important;
}
"""


# =============================================================================
# Gradio Interface
# =============================================================================

def create_interface(model_manager: ModelManager) -> gr.Blocks:
    """Create the Gradio interface with PRD-specified design."""
    
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
        # Header Section
        # =====================================================================
        
        gr.HTML("""
        <div class="app-header">
            <h1>🌿 Garden Doctor AI</h1>
            <p>Your intelligent plant health assistant — Upload a leaf photo for instant disease diagnosis</p>
        </div>
        """)
        
        # Status indicator
        mode_class = "status-mock" if model_manager.mock_mode else "status-ready"
        mode_text = "🔧 Mock Mode (Testing)" if model_manager.mock_mode else f"✅ Ready ({DEVICE.upper()})"
        gr.HTML(f'<div style="text-align: center; margin-bottom: 15px;"><span class="status-indicator {mode_class}">{mode_text}</span></div>')
        
        # =====================================================================
        # Main Content - Two Column Layout
        # =====================================================================
        
        with gr.Row(equal_height=True):
            
            # -----------------------------------------------------------------
            # LEFT COLUMN - Input Section (PRD Section 8.3)
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
                
                climate_input = gr.Dropdown(
                    choices=list(CLIMATE_ZONES.keys()),
                    value="Temperate",
                    label="Climate Zone",
                    info="Select your climate for region-specific advice",
                    elem_classes=["climate-dropdown"]
                )
                
                # Climate description display (PRD FR-2.2)
                climate_info = gr.Markdown(
                    value=f"**{CLIMATE_ZONES['Temperate']['description']}**\n\n*Examples: {CLIMATE_ZONES['Temperate']['examples']}*",
                    elem_classes=["climate-info"]
                )
                
                def update_climate_info(climate):
                    info = CLIMATE_ZONES.get(climate, CLIMATE_ZONES["Temperate"])
                    return f"**{info['description']}**\n\n*Examples: {info['examples']}*"
                
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
            # RIGHT COLUMN - Output Section (PRD Section 8.3)
            # -----------------------------------------------------------------
            
            with gr.Column(scale=1, min_width=300):
                
                gr.HTML('<div class="output-card">')
                
                gr.Markdown("### 📊 Diagnosis Results")
                
                # Confidence label (PRD FR-3.1)
                confidence_output = gr.Label(
                    label="Disease Confidence",
                    num_top_classes=3,
                    show_label=True,
                    elem_classes=["confidence-label"]
                )
                
                gr.Markdown("")
                
                # Diagnosis report (PRD FR-3.2-3.6)
                diagnosis_output = gr.Markdown(
                    value="*Upload an image and click **Diagnose** to see results...*",
                    label="Diagnosis Report",
                    elem_classes=["diagnosis-output"]
                )
                
                gr.HTML('</div>')
        
        # =====================================================================
        # Examples Gallery (PRD Feature 5)
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
        
        gr.HTML("""
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
                ⚠️ <em>This tool provides informational guidance only and does not replace professional agricultural consultation.</em>
            </p>
        </div>
        """)
        
        # =====================================================================
        # Event Handlers (PRD Section 5.1)
        # =====================================================================
        
        def on_diagnose(image, climate, progress=gr.Progress()):
            """
            Handle diagnose button click.
            
            Implements PRD FR-1.1-1.6, FR-3.1-3.6, FR-4.1-4.4
            """
            return diagnose_plant(image, climate, model_manager, progress)
        
        def on_clear():
            """Handle clear button click."""
            return (
                None,  # Clear image
                {},    # Clear confidence label
                "*Upload an image and click **Diagnose** to see results...*"  # Reset markdown
            )
        
        # Wire Diagnose button to inference pipeline
        diagnose_btn.click(
            fn=on_diagnose,
            inputs=[image_input, climate_input],
            outputs=[confidence_output, diagnosis_output],
            api_name="diagnose"
        )
        
        # Wire Clear button
        clear_btn.click(
            fn=on_clear,
            outputs=[image_input, confidence_output, diagnosis_output]
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
    print(f"Confidence Thresholds: Low < {CONFIDENCE_LOW}, High >= {CONFIDENCE_HIGH}")
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
