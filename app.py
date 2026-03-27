"""
Garden Doctor: Plant Disease & Care Assistant
An AI-powered plant disease detection and care recommendation application.

This application uses a fine-tuned LLaVA vision-language model to identify
plant diseases from leaf images and provide treatment recommendations.
"""

import os
import time
from typing import Tuple, Optional, Dict, Any

import gradio as gr
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
)

# =============================================================================
# Configuration
# =============================================================================

MODEL_ID = "YuchengShi/LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Confidence threshold for uncertain predictions
CONFIDENCE_THRESHOLD_LOW = 0.5
CONFIDENCE_THRESHOLD_HIGH = 0.8

# Climate zone options
CLIMATE_ZONES = [
    "Tropical",
    "Temperate", 
    "Arid",
    "Cold"
]

# Disease detection prompt template
DIAGNOSIS_PROMPT = """You are a plant pathology expert. Analyze this plant leaf image and provide:
1. Disease Identification: Name the disease or condition (or state if healthy)
2. Confidence Level: Estimate your confidence (High/Medium/Low)
3. Symptoms Observed: Describe visible symptoms
4. Cause: Explain what causes this condition
5. Treatment Recommendations: Provide specific treatment steps
6. Prevention Tips: How to prevent this in the future

Consider that the plant is growing in a {climate} climate.

Provide a clear, practical response that a home gardener can understand and follow."""

# =============================================================================
# Model Loading
# =============================================================================

class ModelManager:
    """Manages model loading and inference for plant disease detection."""
    
    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.is_loaded = False
        
    def load_model(self) -> None:
        """Load the model and processor."""
        print(f"Loading model from {self.model_id}...")
        print(f"Using device: {DEVICE}, dtype: {DTYPE}")
        
        try:
            # Load processor for LLaVA model
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            
            # Load the LLaVA model
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=DTYPE,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
            )
            
            if not torch.cuda.is_available():
                self.model = self.model.to(DEVICE)
                
            self.model.eval()
            self.is_loaded = True
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def generate_diagnosis(
        self,
        image: Image.Image,
        climate: str,
        max_new_tokens: int = 512
    ) -> str:
        """Generate diagnosis for a plant image."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Format the prompt with climate info
        prompt = DIAGNOSIS_PROMPT.format(climate=climate)
        
        # Prepare conversation format for LLaVA
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process inputs
        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=text_prompt,
            images=image,
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.1,
            )
        
        # Decode output
        generated_text = self.processor.decode(
            output[0],
            skip_special_tokens=True
        )
        
        # Extract only the assistant's response
        if "[/INST]" in generated_text:
            response = generated_text.split("[/INST]")[-1].strip()
        else:
            response = generated_text.strip()
        
        return response


# Initialize model manager (lazy loading)
model_manager = ModelManager(MODEL_ID)


def load_model_on_startup():
    """Load model when the app starts."""
    try:
        model_manager.load_model()
    except Exception as e:
        print(f"Failed to load model: {e}")


# =============================================================================
# Processing Functions
# =============================================================================

def diagnose_plant(
    image: Optional[Image.Image],
    climate: str
) -> Tuple[Dict[str, float], str]:
    """
    Process a plant image and generate diagnosis.
    
    Args:
        image: PIL Image of the plant leaf
        climate: Selected climate zone
        
    Returns:
        Tuple of (confidence_dict, diagnosis_text)
    """
    # Validate input
    if image is None:
        return {}, "⚠️ Please upload an image to diagnose."
    
    try:
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize if too large (for performance)
        max_size = 512
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Generate diagnosis
        diagnosis = model_manager.generate_diagnosis(image, climate)
        
        # Parse confidence from response (placeholder logic)
        # In production, this would be extracted from model output
        confidence_score = 0.85  # Placeholder
        
        # Create confidence label
        if "healthy" in diagnosis.lower():
            label = "Healthy Plant"
            confidence_score = confidence_score
        else:
            # Extract disease name from response
            lines = diagnosis.split('\n')
            label = "Disease Detected"
            for line in lines:
                if "identification" in line.lower() or "disease" in line.lower():
                    # Try to extract disease name
                    if ':' in line:
                        label = line.split(':')[1].strip()[:50]
                    break
        
        confidence_dict = {label: confidence_score}
        
        # Format diagnosis output
        formatted_diagnosis = format_diagnosis_output(diagnosis, climate)
        
        return confidence_dict, formatted_diagnosis
        
    except Exception as e:
        return {}, f"❌ Error processing image: {str(e)}\n\nPlease try again with a different image."


def format_diagnosis_output(diagnosis: str, climate: str) -> str:
    """Format the diagnosis output for display."""
    
    output = f"""## 🩺 Plant Diagnosis Report

**Climate Zone:** {climate}

---

{diagnosis}

---

*⚠️ This diagnosis is for informational purposes only. For serious plant health issues, consult a professional agronomist or local agricultural extension service.*
"""
    return output


def create_example_gallery():
    """Create example images for the gallery."""
    # Placeholder - in production, these would be actual example images
    examples = [
        ["examples/tomato_healthy.jpg", "Temperate"],
        ["examples/potato_late_blight.jpg", "Temperate"],
        ["examples/apple_scab.jpg", "Temperate"],
        ["examples/grape_black_rot.jpg", "Tropical"],
    ]
    return examples


# =============================================================================
# Gradio Interface
# =============================================================================

def create_interface() -> gr.Blocks:
    """Create and configure the Gradio interface."""
    
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
            """
            ---
            **Model:** [LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection](https://huggingface.co/YuchengShi/LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection)
            | **Dataset:** [PlantVillage](https://www.plantvillage.psu.edu/)
            
            ⚠️ *This tool provides informational guidance only and does not replace professional agricultural consultation.*
            """
        )
        
        # Event Handlers
        def clear_outputs():
            return None, {}, "*Upload an image and click 'Diagnose' to see results...*"
        
        diagnose_btn.click(
            fn=diagnose_plant,
            inputs=[image_input, climate_input],
            outputs=[confidence_output, diagnosis_output],
            api_name="diagnose"
        )
        
        clear_btn.click(
            fn=clear_outputs,
            outputs=[image_input, confidence_output, diagnosis_output]
        )
        
        # Enable queue for processing
        demo.queue(max_size=10)
    
    return demo


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🌿 Garden Doctor: Plant Disease & Care Assistant")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    print(f"Device: {DEVICE}")
    print(f"Python: {os.sys.version}")
    print("=" * 60)
    
    # Load model before starting interface
    load_model_on_startup()
    
    # Create and launch interface
    demo = create_interface()
    
    # Launch configuration for Hugging Face Spaces
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        show_api=True,
    )
