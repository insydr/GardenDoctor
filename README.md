---
title: 🌿 Garden Doctor: Plant Disease & Care Assistant
emoji: 🌿
colorFrom: green
colorTo: emerald
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
models:
  - YuchengShi/LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection
tags:
  - plant-disease-detection
  - computer-vision
  - llava
  - multimodal
  - agriculture
short_description: AI-powered plant disease detection and care recommendations
---

# 🌿 Garden Doctor: Plant Disease & Care Assistant

<div align="center">

![Garden Doctor Banner](https://img.shields.io/badge/Plant%20Disease%20Detection-AI%20Powered-brightgreen?style=for-the-badge)
![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge)
![Gradio](https://img.shields.io/badge/Gradio-4.x-orange?style=for-the-badge)

**AI-powered plant disease detection and care recommendation system**

[Demo](#-demo) • [Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Model](#-ai-model)

</div>

---

## 📋 Description

Garden Doctor is an AI-powered application that helps gardeners, farmers, and agricultural enthusiasts identify plant diseases from leaf images and receive actionable treatment recommendations. Leveraging a fine-tuned multimodal vision-language model, the application provides:

- **Accurate Disease Detection** for 38 plant diseases across 14 crop species
- **Detailed Explanations** of identified conditions and their causes
- **Climate-Aware Recommendations** tailored to your growing environment
- **Treatment Protocols** including organic and conventional options

### How It Works

1. **Upload** a photo of an affected plant leaf
2. **Select** your climate zone for tailored recommendations
3. **Receive** instant diagnosis with treatment guidance
4. **Follow** step-by-step care instructions

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📷 **Image Upload** | Upload or capture photos of affected plant leaves |
| 🌡️ **Climate Selection** | Get recommendations tailored to your climate zone |
| 🔬 **Disease Detection** | Identify plant diseases with confidence scores |
| 📝 **Care Instructions** | Receive detailed treatment and prevention advice |
| 📊 **Example Gallery** | Test with pre-loaded example images |
| 📄 **PDF Export** | Download diagnosis reports for record-keeping |

---

## 🤖 AI Model

This application uses **[LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection](https://huggingface.co/YuchengShi/LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection)** - a fine-tuned multimodal vision-language model optimized for plant disease detection.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GRADIO INTERFACE                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Image Input │  │ Climate     │  │ Diagnosis Output    │ │
│  │ (Upload/Cam)│  │ Dropdown    │  │ (Label + Markdown)  │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              LLaVA-v1.5-7B-Plant-Leaf-Diseases              │
│                   (Vision-Language Model)                   │
│  ┌─────────────────┐          ┌─────────────────┐          │
│  │  Vision Encoder │  ──────▶ │ Language Model  │          │
│  │   (CLIP-ViT)    │          │    (Vicuna)     │          │
│  └─────────────────┘          └─────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### Supported Plants & Diseases

<details>
<summary>View full list (14 species, 38 conditions)</summary>

| Plant | Diseases |
|-------|----------|
| 🍎 Apple | Apple scab, Black rot, Cedar apple rust |
| 🫐 Blueberry | Healthy |
| 🍒 Cherry | Powdery mildew |
| 🌽 Corn | Cercospora leaf spot, Common rust, Northern Leaf Blight |
| 🍇 Grape | Black rot, Esca, Leaf blight |
| 🍊 Orange | Huanglongbing (Citrus greening) |
| 🍑 Peach | Bacterial spot |
| 🌶️ Pepper | Bacterial spot |
| 🥔 Potato | Early blight, Late blight |
| 🍇 Raspberry | Healthy |
| 🫘 Soybean | Healthy |
| 🎃 Squash | Powdery mildew |
| 🍓 Strawberry | Leaf scorch |
| 🍅 Tomato | Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, Spider mites, Target spot, Yellow leaf curl virus, Mosaic virus |

</details>

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for faster inference

### Setup

```bash
# Clone the repository
git clone https://github.com/insydr/GardenDoctor.git
cd GardenDoctor

# Create virtual environment
python -m venv vv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### Dependencies

```
gradio==4.44.0
torch==2.4.0
transformers==4.44.2
Pillow==10.4.0
accelerate==0.33.0
```

---

## 📖 Usage

### Running Locally

```bash
python app.py
```

The application will be available at `http://localhost:7860`

### Using the Interface

1. **Upload Image** - Drag & drop or click to upload a plant leaf photo
2. **Select Climate** - Choose your growing region (Tropical, Temperate, Arid, Cold)
3. **Click Diagnose** - Wait for AI analysis (5-30 seconds depending on hardware)
4. **View Results** - See disease identification and treatment recommendations

### API Usage

```python
import requests

# Upload image for diagnosis
with open("plant_leaf.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:7860/api/diagnose",
        files={"image": f},
        data={"climate": "Temperate"}
    )

result = response.json()
print(f"Disease: {result['disease']}")
print(f"Confidence: {result['confidence']}")
print(f"Treatment: {result['treatment']}")
```

---

## 🌐 Deployment

### Hugging Face Spaces

This application is designed for deployment on Hugging Face Spaces:

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space)
2. Select "Gradio" as the SDK
3. Upload all files to the Space:
   - `app.py` - Main application
   - `requirements.txt` - Dependencies
   - `README.md` - This file (with YAML header)
4. The Space will automatically build and deploy

### Hardware Requirements

| Tier | RAM | Inference Time |
|------|-----|----------------|
| CPU (Free) | 16GB+ | 10-30 seconds |
| GPU (T4) | 8GB VRAM | 2-5 seconds |

### Environment Variables (Optional)

For advanced configuration, set these environment variables:

```bash
export HF_HOME=/path/to/cache  # Hugging Face cache directory
export TRANSFORMERS_CACHE=/path/to/cache  # Transformers cache
```

---

## 📁 Project Structure

```
GardenDoctor/
├── app.py                 # Main Gradio application
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore patterns
├── docs/
│   └── Garden_Doctor_PRD.md   # Product Requirements Document
└── examples/              # Example images for testing
    ├── tomato_healthy.jpg
    ├── potato_blight.jpg
    └── ...
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Model Accuracy (PlantVillage) | ~92% |
| Supported Diseases | 38 |
| Supported Plants | 14 |
| Avg. Inference (CPU) | 15 seconds |
| Avg. Inference (GPU) | 3 seconds |

---

## ⚠️ Disclaimer

This application provides **informational guidance only** and does not replace professional agricultural consultation. The AI model has known limitations:

- Best results with clear, well-lit photographs
- May not detect early-stage or rare diseases
- Limited to conditions in the training dataset

**Always consult a professional agronomist for critical crop decisions.**

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Model:** [YuchengShi/LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection](https://huggingface.co/YuchengShi/LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection)
- **Dataset:** [PlantVillage Dataset](https://www.plantvillage.psu.edu/)
- **Framework:** [Gradio](https://www.gradio.app/)
- **Base Model:** [LLaVA](https://llava-vl.github.io/)

---

<div align="center">

**Made with 🌱 for gardeners everywhere**

[⬆ Back to Top](#-garden-doctor-plant-disease--care-assistant)

</div>
