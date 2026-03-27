"""
Garden Doctor: Plant Disease & Care Assistant
An AI-powered plant disease detection and care recommendation application.

This application uses a fine-tuned LLaVA vision-language model to identify
plant diseases from leaf images and provide treatment recommendations.
"""

import os
import re
import time
import logging
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
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

# Confidence thresholds
CONFIDENCE_THRESHOLD_LOW = 0.5
CONFIDENCE_THRESHOLD_HIGH = 0.8


# =============================================================================
# Climate Zone Configuration
# =============================================================================

CLIMATE_ZONES = {
    "Tropical": "Hot, humid climate with year-round growing season (e.g., Southeast Asia, Central America)",
    "Temperate": "Four distinct seasons with moderate humidity (e.g., Eastern US, Europe, East Asia)",
    "Arid": "Hot, dry climate with low rainfall (e.g., Southwest US, Middle East, Australia)",
    "Cold": "Short growing season with harsh winters (e.g., Northern Canada, Scandinavia, Russia)"
}


# =============================================================================
# Color Scheme (from PRD Section 8)
# =============================================================================

COLORS = {
    "primary": "#228B22",       # Forest Green
    "secondary": "#8B4513",     # Earth Brown
    "background": "#F5F5DC",    # Off-White
    "accent": "#90EE90",        # Leaf Green
    "warning": "#FFBF00",       # Amber
    "text_dark": "#1a1a2e",     # Dark text
    "text_light": "#ffffff",    # Light text
    "border": "#2d5a27",        # Dark green border
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
    """Structured diagnosis result."""
    disease_name: str
    confidence_score: float
    confidence_level: ConfidenceLevel
    explanation: str
    treatment_steps: List[str]
    prevention_tips: List[str]
    raw_response: str
    processing_time: float


# =============================================================================
# Prompt Template
# =============================================================================

DIAGNOSIS_PROMPT = """Analyze this plant leaf image for disease detection.

Instructions:
1. Identify the plant species if possible
2. Determine if the leaf is healthy or diseased
3. If diseased, identify the specific disease name
4. Provide a confidence level (High/Medium/Low)
5. Describe visible symptoms
6. Explain the cause of the condition
7. Provide treatment recommendations for {climate} climate
8. Suggest prevention measures

Please format your response as follows:
DISEASE: [disease name or "Healthy"]
CONFIDENCE: [High/Medium/Low]
SYMPTOMS: [description of visible symptoms]
CAUSE: [explanation of what causes this condition]
TREATMENT: [numbered list of treatment steps for {climate} climate]
PREVENTION: [bullet points for prevention tips]

Consider that this plant is growing in a {climate} climate zone when providing recommendations."""


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
        """Preprocess image for model input."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        
        return image
    
    def generate_response(self, image: Image.Image, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate model response for an image and prompt."""
        if self.mock_mode:
            return self._generate_mock_response(prompt)
        
        if not self.is_loaded:
            raise RuntimeError("Model not loaded.")
        
        conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(text=text_prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        
        generated_text = self.processor.decode(output[0], skip_special_tokens=True)
        response = generated_text.split("[/INST]")[-1].strip() if "[/INST]" in generated_text else generated_text.strip()
        return response
    
    def _generate_mock_response(self, prompt: str) -> str:
        """Generate mock response for testing."""
        climate = "Temperate"
        for zone in CLIMATE_ZONES.keys():
            if zone.lower() in prompt.lower():
                climate = zone
                break
        
        return f"""DISEASE: Early Blight
CONFIDENCE: High
SYMPTOMS: Dark brown circular spots with concentric rings forming target-like patterns on the leaves. Yellowing of leaf tissue around the infected areas. Lesions may merge causing large dead patches on affected leaves.
CAUSE: Early blight is caused by the fungus Alternaria solani. The pathogen survives in infected plant debris and soil. It thrives in warm, humid conditions with temperatures between 24-29°C (75-84°F). The disease spreads through wind-borne spores and water splash.
TREATMENT: 
1. Remove and destroy all infected leaves immediately to prevent further spread of the disease
2. Apply copper-based fungicide or chlorothalonil every 7-10 days during active growth
3. Improve air circulation around plants by proper spacing and removing lower leaves
4. Water at the base of plants early in the day to keep foliage dry
5. For {climate} climate, monitor plants closely during warm humid periods and apply preventive fungicides
PREVENTION: 
• Use disease-resistant tomato and potato varieties when available
• Practice 3-4 year crop rotation avoiding solanaceous crops
• Remove and destroy all plant debris at end of growing season
• Maintain adequate plant spacing (24-36 inches) for air circulation
• Apply organic mulch to prevent soil splash onto lower leaves
• Use drip irrigation or soaker hoses instead of overhead watering"""


# =============================================================================
# Diagnosis Functions
# =============================================================================

def parse_diagnosis_response(response: str, processing_time: float) -> DiagnosisResult:
    """Parse model response into structured diagnosis result."""
    disease_name = "Unknown"
    confidence_score = 0.5
    confidence_level = ConfidenceLevel.MEDIUM
    explanation = ""
    treatment_steps = []
    prevention_tips = []
    
    # Extract disease name
    disease_match = re.search(r"DISEASE:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    if disease_match:
        disease_name = disease_match.group(1).strip()
    
    # Extract confidence
    confidence_match = re.search(r"CONFIDENCE:\s*(High|Medium|Low)", response, re.IGNORECASE)
    if confidence_match:
        conf_str = confidence_match.group(1).strip().capitalize()
        confidence_level = ConfidenceLevel(conf_str)
        confidence_score = {"High": 0.85, "Medium": 0.65, "Low": 0.40}[conf_str]
    
    # Extract symptoms and cause
    symptoms_match = re.search(r"SYMPTOMS:\s*(.+?)(?=CAUSE:|$)", response, re.IGNORECASE | re.DOTALL)
    cause_match = re.search(r"CAUSE:\s*(.+?)(?=TREATMENT:|$)", response, re.IGNORECASE | re.DOTALL)
    
    explanation_parts = []
    if symptoms_match:
        explanation_parts.append(f"**Symptoms:** {symptoms_match.group(1).strip()}")
    if cause_match:
        explanation_parts.append(f"**Cause:** {cause_match.group(1).strip()}")
    explanation = "\n\n".join(explanation_parts)
    
    # Extract treatment steps
    treatment_match = re.search(r"TREATMENT:\s*(.+?)(?=PREVENTION:|$)", response, re.IGNORECASE | re.DOTALL)
    if treatment_match:
        treatment_text = treatment_match.group(1).strip()
        treatment_steps = [step.strip() for step in re.findall(r"\d+\.\s*(.+?)(?=\n\d+\.|$)", treatment_text, re.DOTALL)]
        if not treatment_steps:
            treatment_steps = [line.strip().lstrip("0123456789. ") for line in treatment_text.split("\n") if line.strip()]
    
    # Extract prevention tips
    prevention_match = re.search(r"PREVENTION:\s*(.+?)$", response, re.IGNORECASE | re.DOTALL)
    if prevention_match:
        prevention_text = prevention_match.group(1).strip()
        prevention_tips = [tip.strip().lstrip("• ") for tip in prevention_text.split("\n") if tip.strip()]
    
    return DiagnosisResult(
        disease_name=disease_name,
        confidence_score=confidence_score,
        confidence_level=confidence_level,
        explanation=explanation,
        treatment_steps=treatment_steps,
        prevention_tips=prevention_tips,
        raw_response=response,
        processing_time=processing_time
    )


def format_diagnosis_report(diagnosis: DiagnosisResult, climate: str) -> str:
    """Format diagnosis result as markdown report."""
    sections = []
    
    # Header with status icon
    if "healthy" in diagnosis.disease_name.lower():
        sections.append("## ✅ Great News! Your Plant Appears Healthy\n")
    else:
        sections.append("## 🔬 Disease Identified\n")
        sections.append(f"### **{diagnosis.disease_name}**\n")
    
    # Confidence indicator
    confidence_emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
    sections.append(f"**Confidence Level:** {confidence_emoji.get(diagnosis.confidence_level.value, '⚪')} {diagnosis.confidence_level.value} ({diagnosis.confidence_score:.0%})")
    sections.append(f"**Climate Zone:** {climate}")
    sections.append(f"**Analysis Time:** {diagnosis.processing_time:.1f}s")
    sections.append("\n---\n")
    
    # Explanation
    if diagnosis.explanation:
        sections.append("### 📋 Details\n")
        sections.append(diagnosis.explanation)
        sections.append("")
    
    # Treatment steps
    if diagnosis.treatment_steps:
        sections.append("### 💊 Treatment Recommendations\n")
        for i, step in enumerate(diagnosis.treatment_steps, 1):
            sections.append(f"**{i}.** {step}")
        sections.append("")
    
    # Prevention tips
    if diagnosis.prevention_tips:
        sections.append("### 🛡️ Prevention Tips\n")
        for tip in diagnosis.prevention_tips:
            sections.append(f"• {tip}")
        sections.append("")
    
    # Footer disclaimer
    sections.append("---\n")
    sections.append("*⚠️ This diagnosis is for informational purposes only. For serious plant health issues, please consult a professional agronomist or your local agricultural extension service.*")
    
    return "\n".join(sections)


def diagnose_plant(
    image: Optional[Image.Image],
    climate: str,
    model_manager: ModelManager,
    progress=gr.Progress()
) -> Tuple[Dict[str, float], str]:
    """
    Process a plant image and generate diagnosis with progress updates.
    
    Args:
        image: PIL Image of the plant leaf
        climate: Selected climate zone
        model_manager: ModelManager instance
        progress: Gradio progress tracker
        
    Returns:
        Tuple of (confidence_dict, formatted_report)
    """
    if image is None:
        return {}, "⚠️ **Please upload a leaf image to diagnose.**\n\nUpload a clear, well-lit photo of the affected leaf for best results."
    
    try:
        progress(0.1, desc="📷 Processing image...")
        
        # Preprocess image
        processed_image = model_manager.preprocess_image(image)
        
        progress(0.2, desc="🤖 Analyzing with AI model...")
        
        # Start timing
        start_time = time.time()
        
        # Construct prompt
        prompt = DIAGNOSIS_PROMPT.format(climate=climate)
        
        progress(0.4, desc="🔍 Detecting plant disease...")
        
        # Generate response
        response = model_manager.generate_response(processed_image, prompt)
        
        progress(0.7, desc="📊 Generating diagnosis report...")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Parse response
        diagnosis = parse_diagnosis_response(response, processing_time)
        
        progress(0.9, desc="✅ Finalizing results...")
        
        # Format output
        confidence_dict = {diagnosis.disease_name: diagnosis.confidence_score}
        formatted_report = format_diagnosis_report(diagnosis, climate)
        
        progress(1.0, desc="Done!")
        
        logger.info(f"Diagnosis completed in {processing_time:.2f}s: {diagnosis.disease_name}")
        
        return confidence_dict, formatted_report
        
    except Exception as e:
        error_msg = f"❌ **Error processing image:** {str(e)}\n\nPlease try again with a different image."
        logger.error(f"Diagnosis error: {str(e)}")
        return {}, error_msg


# =============================================================================
# Custom CSS
# =============================================================================

CUSTOM_CSS = """
/* Base styles */
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    background: linear-gradient(135deg, #f5f5dc 0%, #e8f5e9 100%) !important;
}

/* Header styling */
.app-header {
    text-align: center;
    padding: 20px;
    margin-bottom: 20px;
    background: linear-gradient(135deg, #228B22 0%, #2d5a27 100%);
    border-radius: 15px;
    color: white;
    box-shadow: 0 4px 15px rgba(34, 139, 34, 0.3);
}

.app-header h1 {
    margin: 0;
    font-size: 2.5em;
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
.climate-dropdown {
    border-radius: 10px !important;
}

/* Label output styling */
.confidence-label {
    border-radius: 15px;
    padding: 15px;
    background: linear-gradient(135deg, #f0fff0 0%, #e8f5e9 100%);
    border: 2px solid #90EE90;
}

/* Markdown output styling */
.diagnosis-output {
    background: white;
    border-radius: 15px;
    padding: 20px;
    border: 2px solid #e8f5e9;
}

/* Examples gallery */
.examples-section {
    background: white;
    border-radius: 15px;
    padding: 20px;
    margin-top: 20px;
    border: 2px solid #90EE90;
}

/* Footer styling */
.app-footer {
    text-align: center;
    padding: 15px;
    margin-top: 20px;
    background: #f5f5dc;
    border-radius: 10px;
    font-size: 0.9em;
    color: #666;
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

.status-loading {
    background: #FFBF00;
    color: #8B4513;
}

/* Responsive design */
@media (max-width: 768px) {
    .app-header h1 {
        font-size: 1.8em;
    }
    
    .input-card, .output-card {
        padding: 15px;
    }
}

/* Hide default footer */
footer {
    display: none !important;
}

/* Loading animation */
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

.loading {
    animation: pulse 1.5s ease-in-out infinite;
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
                    value=f"*{CLIMATE_ZONES['Temperate']}*",
                    elem_classes=["climate-info"]
                )
                
                def update_climate_info(climate):
                    return f"*{CLIMATE_ZONES.get(climate, '')}*"
                
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
                ["examples/tomato_healthy.jpg", "Temperate", "Healthy Tomato Leaf"],
                ["examples/tomato_early_blight.jpg", "Temperate", "Tomato Early Blight"],
                ["examples/potato_late_blight.jpg", "Temperate", "Potato Late Blight"],
                ["examples/apple_scab.jpg", "Temperate", "Apple Scab"],
                ["examples/corn_rust.jpg", "Tropical", "Corn Common Rust"],
                ["examples/grape_black_rot.jpg", "Tropical", "Grape Black Rot"],
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
        # Event Handlers
        # =====================================================================
        
        def on_diagnose(image, climate, progress=gr.Progress()):
            """Handle diagnose button click with progress."""
            return diagnose_plant(image, climate, model_manager, progress)
        
        def on_clear():
            """Handle clear button click."""
            return None, {}, "*Upload an image and click **Diagnose** to see results...*"
        
        # Connect event handlers
        diagnose_btn.click(
            fn=on_diagnose,
            inputs=[image_input, climate_input],
            outputs=[confidence_output, diagnosis_output],
            api_name="diagnose"
        )
        
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
