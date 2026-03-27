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

# Configure logging
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
HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE", "/tmp/huggingface_cache")

# Device and dtype configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Mock mode for testing without loading the full model
MOCK_MODE = os.environ.get("GARDEN_DOCTOR_MOCK", "false").lower() == "true"

# Image preprocessing configuration
IMAGE_SIZE = 336  # LLaVA uses 336x336 images

# Confidence thresholds
CONFIDENCE_THRESHOLD_LOW = 0.5
CONFIDENCE_THRESHOLD_HIGH = 0.8

# Climate zone options
CLIMATE_ZONES = [
    "Tropical",
    "Temperate", 
    "Arid",
    "Cold"
]


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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "disease_name": self.disease_name,
            "confidence_score": self.confidence_score,
            "confidence_level": self.confidence_level.value,
            "explanation": self.explanation,
            "treatment_steps": self.treatment_steps,
            "prevention_tips": self.prevention_tips,
            "raw_response": self.raw_response
        }


# =============================================================================
# Prompt Templates
# =============================================================================

DIAGNOSIS_PROMPT_TEMPLATE = """Analyze this plant leaf image for disease detection.

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
    """
    Manages model loading and inference for plant disease detection.
    
    Supports:
    - Lazy loading of model
    - CPU and GPU inference
    - Memory optimization for large models
    - Mock mode for testing
    """
    
    def __init__(self, model_id: str = MODEL_ID, mock_mode: bool = MOCK_MODE):
        """
        Initialize the model manager.
        
        Args:
            model_id: Hugging Face model ID
            mock_mode: If True, use mock responses instead of actual model
        """
        self.model_id = model_id
        self.mock_mode = mock_mode
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.is_loaded = False
        self.load_error: Optional[str] = None
        
        logger.info(f"ModelManager initialized (mock_mode={mock_mode}, device={DEVICE})")
    
    def load_model(self) -> bool:
        """
        Load the model and processor.
        
        Returns:
            True if loading successful, False otherwise
        """
        if self.mock_mode:
            logger.info("Mock mode enabled - skipping model loading")
            self.is_loaded = True
            return True
        
        logger.info(f"Loading model from {self.model_id}...")
        logger.info(f"Device: {DEVICE}, dtype: {DTYPE}")
        
        try:
            # Import transformers here to allow mock mode without dependencies
            from transformers import (
                LlavaForConditionalGeneration,
                AutoProcessor,
            )
            
            # Set cache directory
            os.environ["HF_HOME"] = HF_HUB_CACHE
            os.environ["TRANSFORMERS_CACHE"] = HF_HUB_CACHE
            
            # Load processor first (smaller, faster)
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            # Load model with memory optimization
            logger.info("Loading model (this may take a few minutes)...")
            
            # Memory optimization settings
            load_kwargs = {
                "torch_dtype": DTYPE,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
            }
            
            # GPU-specific optimizations
            if torch.cuda.is_available():
                load_kwargs["device_map"] = "auto"
                load_kwargs["torch_dtype"] = torch.float16
            else:
                # CPU optimizations
                load_kwargs["device_map"] = None
                
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_id,
                **load_kwargs
            )
            
            # Move to device if not using device_map
            if not torch.cuda.is_available():
                self.model = self.model.to(DEVICE)
            
            # Set to evaluation mode
            self.model.eval()
            
            self.is_loaded = True
            logger.info("Model loaded successfully!")
            return True
            
        except ImportError as e:
            self.load_error = f"Missing dependency: {str(e)}. Please install required packages."
            logger.error(self.load_error)
            return False
            
        except torch.OutOfMemoryError as e:
            self.load_error = f"Out of memory loading model. Try using mock mode or GPU. Error: {str(e)}"
            logger.error(self.load_error)
            return False
            
        except Exception as e:
            self.load_error = f"Error loading model: {str(e)}"
            logger.error(self.load_error)
            return False
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for model input.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Preprocessed PIL Image (336x336, RGB)
        """
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize to model input size (336x336 for LLaVA)
        # Use LANCZOS for high-quality downsampling
        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            image = image.resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                Image.Resampling.LANCZOS
            )
        
        # Note: Pixel normalization to [0,1] is handled by the processor
        # No need to do it manually here
        
        return image
    
    def generate_response(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1
    ) -> str:
        """
        Generate model response for an image and prompt.
        
        Args:
            image: Preprocessed PIL Image
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        if self.mock_mode:
            return self._generate_mock_response(prompt)
        
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Format conversation for LLaVA
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Apply chat template
            text_prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                text=text_prompt,
                images=image,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                # Use mixed precision for faster inference
                with torch.amp.autocast(device_type=DEVICE, dtype=DTYPE):
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=temperature > 0,
                        temperature=temperature if temperature > 0 else None,
                        top_p=0.9,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                    )
            
            # Decode output
            generated_text = self.processor.decode(
                output[0],
                skip_special_tokens=True
            )
            
            # Extract assistant's response
            if "[/INST]" in generated_text:
                response = generated_text.split("[/INST]")[-1].strip()
            elif "ASSISTANT:" in generated_text:
                response = generated_text.split("ASSISTANT:")[-1].strip()
            else:
                response = generated_text.strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            raise
    
    def _generate_mock_response(self, prompt: str) -> str:
        """Generate mock response for testing."""
        # Extract climate from prompt
        climate = "Temperate"
        for zone in CLIMATE_ZONES:
            if zone.lower() in prompt.lower():
                climate = zone
                break
        
        return f"""DISEASE: Early Blight
CONFIDENCE: High
SYMPTOMS: Dark brown circular spots with concentric rings (target-like pattern) on leaves. Yellowing of leaf tissue around the spots. Lesions may merge causing large dead areas.
CAUSE: Early blight is caused by the fungus Alternaria solani. It thrives in warm, humid conditions and spreads through infected plant debris, contaminated soil, and wind-borne spores.
TREATMENT:
1. Remove and destroy infected leaves immediately to prevent spread
2. Apply copper-based fungicide or chlorothalonil every 7-10 days
3. Improve air circulation around plants by proper spacing and pruning
4. Water at the base of plants to keep foliage dry
5. In {climate} climate, monitor plants closely during warm, humid periods
PREVENTION:
• Use disease-resistant tomato varieties when available
• Practice crop rotation (avoid planting tomatoes in same location for 2-3 years)
• Remove and destroy all plant debris at end of season
• Maintain adequate plant spacing for air circulation
• Use mulch to prevent soil splash onto leaves
• Apply preventive fungicide sprays during favorable disease conditions"""


# =============================================================================
# Diagnosis Functions
# =============================================================================

def parse_diagnosis_response(response: str) -> DiagnosisResult:
    """
    Parse model response into structured diagnosis result.
    
    Args:
        response: Raw model response text
        
    Returns:
        Structured DiagnosisResult
    """
    # Default values
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
        
        # Map to numerical score
        if confidence_level == ConfidenceLevel.HIGH:
            confidence_score = 0.85
        elif confidence_level == ConfidenceLevel.MEDIUM:
            confidence_score = 0.65
        else:
            confidence_score = 0.40
    
    # Extract symptoms/explanation
    symptoms_match = re.search(r"SYMPTOMS:\s*(.+?)(?=CAUSE:|$)", response, re.IGNORECASE | re.DOTALL)
    if symptoms_match:
        symptoms = symptoms_match.group(1).strip()
    
    cause_match = re.search(r"CAUSE:\s*(.+?)(?=TREATMENT:|$)", response, re.IGNORECASE | re.DOTALL)
    if cause_match:
        cause = cause_match.group(1).strip()
    
    # Combine into explanation
    explanation_parts = []
    if symptoms_match:
        explanation_parts.append(f"**Symptoms:** {symptoms}")
    if cause_match:
        explanation_parts.append(f"**Cause:** {cause}")
    explanation = "\n\n".join(explanation_parts)
    
    # Extract treatment steps
    treatment_match = re.search(r"TREATMENT:\s*(.+?)(?=PREVENTION:|$)", response, re.IGNORECASE | re.DOTALL)
    if treatment_match:
        treatment_text = treatment_match.group(1).strip()
        # Parse numbered list
        treatment_steps = [
            step.strip()
            for step in re.findall(r"\d+\.\s*(.+?)(?=\n\d+\.|$)", treatment_text, re.DOTALL)
        ]
        if not treatment_steps:
            # Fallback: split by newlines
            treatment_steps = [
                line.strip().lstrip("0123456789. ")
                for line in treatment_text.split("\n")
                if line.strip() and not line.strip().startswith("*")
            ]
    
    # Extract prevention tips
    prevention_match = re.search(r"PREVENTION:\s*(.+?)$", response, re.IGNORECASE | re.DOTALL)
    if prevention_match:
        prevention_text = prevention_match.group(1).strip()
        # Parse bullet points
        prevention_tips = [
            tip.strip().lstrip("• ")
            for tip in prevention_text.split("\n")
            if tip.strip()
        ]
    
    return DiagnosisResult(
        disease_name=disease_name,
        confidence_score=confidence_score,
        confidence_level=confidence_level,
        explanation=explanation,
        treatment_steps=treatment_steps,
        prevention_tips=prevention_tips,
        raw_response=response
    )


def diagnose_plant(
    image: Optional[Image.Image],
    climate: str,
    model_manager: ModelManager
) -> Tuple[Dict[str, float], str]:
    """
    Process a plant image and generate diagnosis.
    
    Args:
        image: PIL Image of the plant leaf
        climate: Selected climate zone
        model_manager: ModelManager instance
        
    Returns:
        Tuple of (confidence_dict, formatted_diagnosis_text)
    """
    # Validate input
    if image is None:
        return {}, "⚠️ Please upload an image to diagnose."
    
    try:
        # Start timing
        start_time = time.time()
        
        # Preprocess image
        processed_image = model_manager.preprocess_image(image)
        
        # Construct prompt
        prompt = DIAGNOSIS_PROMPT_TEMPLATE.format(climate=climate)
        
        # Generate response
        response = model_manager.generate_response(
            processed_image,
            prompt,
            max_new_tokens=512,
            temperature=0.1
        )
        
        # Parse response
        diagnosis = parse_diagnosis_response(response)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        logger.info(f"Diagnosis completed in {elapsed_time:.2f} seconds")
        
        # Create confidence label
        confidence_dict = {diagnosis.disease_name: diagnosis.confidence_score}
        
        # Format output
        formatted_output = format_diagnosis_output(diagnosis, climate, elapsed_time)
        
        return confidence_dict, formatted_output
        
    except Exception as e:
        error_msg = f"❌ Error processing image: {str(e)}"
        logger.error(error_msg)
        return {}, error_msg


def format_diagnosis_output(
    diagnosis: DiagnosisResult,
    climate: str,
    elapsed_time: float
) -> str:
    """
    Format diagnosis result for display.
    
    Args:
        diagnosis: Structured diagnosis result
        climate: Climate zone
        elapsed_time: Time taken for inference
        
    Returns:
        Formatted markdown text
    """
    sections = []
    
    # Header
    sections.append("## 🩺 Plant Diagnosis Report\n")
    
    # Metadata
    sections.append(f"**Climate Zone:** {climate}")
    sections.append(f"**Analysis Time:** {elapsed_time:.1f}s")
    sections.append(f"**Confidence:** {diagnosis.confidence_level.value} ({diagnosis.confidence_score:.0%})")
    sections.append("\n---\n")
    
    # Disease identification
    if "healthy" in diagnosis.disease_name.lower():
        sections.append("### ✅ Good News!")
        sections.append(f"Your plant appears to be **{diagnosis.disease_name}**")
    else:
        sections.append("### 🔬 Disease Identified")
        sections.append(f"**{diagnosis.disease_name}**")
    
    # Explanation
    if diagnosis.explanation:
        sections.append("\n### 📋 Details")
        sections.append(diagnosis.explanation)
    
    # Treatment steps
    if diagnosis.treatment_steps:
        sections.append("\n### 💊 Treatment Recommendations")
        for i, step in enumerate(diagnosis.treatment_steps, 1):
            sections.append(f"{i}. {step}")
    
    # Prevention tips
    if diagnosis.prevention_tips:
        sections.append("\n### 🛡️ Prevention Tips")
        for tip in diagnosis.prevention_tips:
            sections.append(f"• {tip}")
    
    # Footer
    sections.append("\n---")
    sections.append("*⚠️ This diagnosis is for informational purposes only. For serious plant health issues, consult a professional agronomist or local agricultural extension service.*")
    
    return "\n".join(sections)


# =============================================================================
# Gradio Interface
# =============================================================================

def create_interface(model_manager: ModelManager) -> gr.Blocks:
    """
    Create and configure the Gradio interface.
    
    Args:
        model_manager: ModelManager instance
        
    Returns:
        Configured Gradio Blocks interface
    """
    
    with gr.Blocks(
        theme=gr.themes.Green(),
        title="🌿 Garden Doctor",
        css="""
        .contain { display: flex; flex-direction: column; }
        .gradio-container { font-family: 'Segoe UI', sans-serif; }
        #component-0 { margin-bottom: 10px; }
        footer { display: none !important; }
        """
    ) as demo:
        
        # Header
        gr.Markdown(
            """
            # 🌿 Garden Doctor: Plant Disease & Care Assistant
            
            Upload a photo of a plant leaf to detect diseases and get personalized treatment recommendations.
            """
        )
        
        # Model status indicator
        if not model_manager.is_loaded:
            status_msg = "⚠️ Model not loaded. Click 'Load Model' first." if not model_manager.mock_mode else "🔧 Running in Mock Mode (for testing)"
            gr.Markdown(f"**Status:** {status_msg}")
        
        # Main Interface
        with gr.Row(equal_height=True):
            
            # Left Column - Inputs
            with gr.Column(scale=1):
                gr.Markdown("### 📷 Upload Plant Image")
                
                image_input = gr.Image(
                    type="pil",
                    label="Plant Leaf Photo",
                    sources=["upload", "webcam"],
                    height=300,
                )
                
                climate_input = gr.Dropdown(
                    choices=CLIMATE_ZONES,
                    value="Temperate",
                    label="🌍 Your Climate Zone",
                    info="Select your growing region for tailored recommendations"
                )
                
                with gr.Row():
                    diagnose_btn = gr.Button(
                        "🔬 Diagnose",
                        variant="primary",
                        size="lg"
                    )
                    
                    clear_btn = gr.Button(
                        "🗑️ Clear",
                        variant="secondary",
                        size="sm"
                    )
            
            # Right Column - Outputs
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Diagnosis Results")
                
                confidence_output = gr.Label(
                    label="Disease Confidence",
                    num_top_classes=3
                )
                
                diagnosis_output = gr.Markdown(
                    label="Care Instructions",
                    value="*Upload an image and click 'Diagnose' to see results...*"
                )
        
        # Examples Section
        gr.Markdown("---")
        gr.Markdown("### 🖼️ Example Images (Click to try)")
        
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
            label="Example Plant Images"
        )
        
        # Footer
        gr.Markdown(
            f"""
            ---
            **Model:** [LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection](https://huggingface.co/YuchengShi/LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection)
            | **Dataset:** [PlantVillage](https://www.plantvillage.psu.edu/)
            | **Mode:** {"Mock (Testing)" if model_manager.mock_mode else "Production"}
            
            ⚠️ *This tool provides informational guidance only and does not replace professional agricultural consultation.*
            """
        )
        
        # Event Handlers
        def on_diagnose(image, climate):
            return diagnose_plant(image, climate, model_manager)
        
        def on_clear():
            return None, {}, "*Upload an image and click 'Diagnose' to see results...*"
        
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
        
        # Enable queue for processing
        demo.queue(max_size=10)
    
    return demo


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for the application."""
    print("=" * 60)
    print("🌿 Garden Doctor: Plant Disease & Care Assistant")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    print(f"Device: {DEVICE}")
    print(f"Dtype: {DTYPE}")
    print(f"Mock Mode: {MOCK_MODE}")
    print(f"Python: {os.sys.version}")
    print("=" * 60)
    
    # Initialize model manager
    model_manager = ModelManager(
        model_id=MODEL_ID,
        mock_mode=MOCK_MODE
    )
    
    # Load model (or skip in mock mode)
    if not model_manager.load_model():
        print(f"❌ Failed to load model: {model_manager.load_error}")
        print("⚠️ Starting in mock mode for testing...")
        model_manager.mock_mode = True
        model_manager.is_loaded = True
    
    # Create and launch interface
    demo = create_interface(model_manager)
    
    # Launch configuration for Hugging Face Spaces
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        show_api=True,
    )


if __name__ == "__main__":
    main()
