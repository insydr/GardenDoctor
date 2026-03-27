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
  - vision-language-model
short_description: AI-powered plant disease detection and care recommendations
---

# 🌿 Garden Doctor: Plant Disease & Care Assistant

<div align="center">

[![Gradio](https://img.shields.io/badge/Gradio-4.44.0-FF6B35?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app/)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Spaces-CPU%20Tier-FFD21E?style=for-the-badge)](https://huggingface.co/spaces)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-000000?style=for-the-badge)](LICENSE)

**Your intelligent plant health companion — Upload a leaf photo, get instant diagnosis and treatment guidance powered by AI.**

[🎯 Live Demo](#-demo) • [✨ Features](#-features) • [📖 How to Use](#-how-to-use) • [🔧 Technical Details](#-technical-details)

</div>

---

## 📸 Screenshot

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│   ┌─────────────────────────────┐    ┌─────────────────────────────┐    │
│   │                             │    │                             │    │
│   │    📷 IMAGE UPLOAD          │    │    📊 RESULTS               │    │
│   │    ┌───────────────────┐    │    │                             │    │
│   │    │                   │    │    │    🩺 Diagnosis: Early      │    │
│   │    │   🍃 Leaf Photo   │    │    │       Blight               │    │
│   │    │                   │    │    │                             │    │
│   │    └───────────────────┘    │    │    🟢 Confidence: 85%       │    │
│   │                             │    │                             │    │
│   │    🌍 Climate: Temperate    │    │    🌱 Treatment Options     │    │
│   │                             │    │       • Cultural practices  │    │
│   │    [🔬 Diagnose] [🗑️ Clear] │    │       • Organic options     │    │
│   │                             │    │       • Conventional care   │    │
│   └─────────────────────────────┘    │                             │    │
│                                       │    📄 Download PDF Report   │    │
│   ┌─────────────────────────────────┐│                             │    │
│   │ 🖼️ Example Images (click to test)│└─────────────────────────────┘    │
│   │ [🍅] [🥔] [🍎] [🌽] [🍇] ...    │                                   │
│   └─────────────────────────────────┘                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

*Screenshot placeholder - Actual UI features a green-themed Gradio interface with responsive layout*

</div>

---

## 🎯 Project Overview

### The Problem

Plant diseases cause significant crop losses for home gardeners and small-scale farmers. Identifying these diseases early is crucial for effective treatment, but most people lack access to agricultural experts. Delayed or incorrect diagnosis often leads to:

- **Preventable crop losses** from treatable conditions
- **Misuse of pesticides** from incorrect self-diagnosis  
- **Wasted time and money** on ineffective treatments
- **Spread of preventable diseases** to healthy plants

### Our Solution

**Garden Doctor** democratizes plant disease diagnosis using state-of-the-art AI. Simply upload a photo of an affected leaf, and receive:

- **Instant disease identification** with confidence scoring
- **Detailed symptom explanations** to help you understand the condition
- **Climate-tailored treatment recommendations** including organic and conventional options
- **Prevention strategies** specific to your growing environment
- **Downloadable PDF reports** for record-keeping or consultation

### Value Proposition

| User Need | Garden Doctor Solution |
|-----------|------------------------|
| Quick diagnosis | ⚡ Results in 5-30 seconds |
| Trusted guidance | 🎯 ~92% accuracy on known diseases |
| Actionable advice | 🌱 Step-by-step treatment protocols |
| Accessibility | 📱 Works on any device with a browser |
| Cost-effective | 💰 Free to use on Hugging Face Spaces |

---

## ✨ Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| 📷 **Multi-Source Image Input** | Upload files, drag & drop, or use your device camera |
| 🌡️ **Climate-Aware Analysis** | Recommendations tailored to Tropical, Temperate, Arid, or Cold zones |
| 🔬 **Confidence Scoring** | Visual indicators (High/Medium/Low) help assess result reliability |
| 📝 **Comprehensive Reports** | Symptoms, causes, and treatment options in clear language |
| 🖼️ **Example Gallery** | Pre-loaded samples to test the system instantly |
| 📄 **PDF Export** | Download detailed reports for offline reference |

### Smart Confidence System

The application provides transparent confidence indicators:

| Confidence Level | Score | Interpretation |
|------------------|-------|----------------|
| 🟢 High | ≥80% | Reliable diagnosis - follow recommendations with confidence |
| 🟡 Moderate | 50-79% | Reasonable detection - consider additional verification |
| 🔴 Low | <50% | Uncertain result - review image quality tips provided |

---

## 📖 How to Use

### Step-by-Step Guide

1. **📷 Upload Your Image**
   - Click the upload area or drag and drop a photo
   - Supported formats: JPEG, PNG, WebP
   - For best results: clear, well-lit photos showing the affected leaf clearly

2. **🌍 Select Your Climate Zone**
   - **Tropical**: Hot, humid (Southeast Asia, Caribbean)
   - **Temperate**: Four seasons (Eastern US, Europe, East Asia)
   - **Arid**: Hot, dry (Southwest US, Middle East)
   - **Cold**: Short growing season (Northern regions)

3. **🔬 Click "Diagnose"**
   - Wait 5-30 seconds while AI analyzes your image
   - Progress indicators show analysis stages

4. **📊 Review Your Results**
   - Disease identification with confidence score
   - Detailed symptoms and cause explanation
   - Treatment options: Cultural, Organic, Conventional
   - Prevention tips for your climate

5. **📄 Download Report (Optional)**
   - Click "Download PDF Report" for a formatted document
   - Includes all diagnosis details, timestamp, and disclaimer

### Tips for Best Results

<details>
<summary>📸 Photography Guidelines</summary>

- ✅ **Lighting**: Use natural daylight when possible
- ✅ **Focus**: Ensure the affected area is sharp and clear
- ✅ **Coverage**: Leaf should fill at least 70% of the frame
- ✅ **Background**: Use a plain, contrasting background
- ✅ **Multiple angles**: Take 2-3 photos of different affected leaves
- ❌ **Avoid**: Shadows, reflections, blurry images, dark photos

</details>

---

## 🔧 Technical Details

### AI Model

| Aspect | Specification |
|--------|---------------|
| **Model** | [LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection](https://huggingface.co/YuchengShi/LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection) |
| **Architecture** | Vision-Language Model (VLM) |
| **Vision Encoder** | CLIP ViT-L/14 (336×336 resolution) |
| **Language Model** | Vicuna-7B (fine-tuned for plant pathology) |
| **Training Data** | PlantVillage Dataset (54,000+ images) |
| **Base Accuracy** | ~92% on PlantVillage test set |

### Framework Stack

| Component | Technology | Version |
|-----------|------------|---------|
| UI Framework | Gradio | 4.44.0 |
| ML Backend | PyTorch | 2.4.0 |
| Model Hub | Hugging Face Transformers | 4.44.2 |
| Image Processing | Pillow | 10.4.0 |
| PDF Generation | fpdf2 | 2.7.9 |

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           GRADIO INTERFACE                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────────┐  │
│  │ Image Input │  │ Climate     │  │ Diagnosis Output                │  │
│  │ (Upload/    │  │ Dropdown    │  │ • Label (confidence score)      │  │
│  │  Webcam)    │  │             │  │ • Markdown (detailed report)    │  │
│  └─────────────┘  └─────────────┘  │ • PDF Download                  │  │
│                                     └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    MODEL MANAGER (Singleton Cache)                       │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Quantization: 4-bit (GPU) / float16 (GPU) / float32 (CPU)       │    │
│  │ Timeout: 45 seconds with graceful cancellation                   │    │
│  │ Caching: Global singleton prevents reload per request            │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection                │
│                         (Vision-Language Model)                          │
│  ┌───────────────────────┐          ┌───────────────────────┐           │
│  │   Vision Encoder      │          │    Language Model     │           │
│  │   (CLIP ViT-L/14)     │  ──────▶ │    (Vicuna-7B)        │           │
│  │   336×336 input       │          │    Plant-tuned        │           │
│  └───────────────────────┘          └───────────────────────┘           │
└─────────────────────────────────────────────────────────────────────────┘
```

### Supported Plants & Diseases

<details>
<summary>📋 View full list (14 species, 38 conditions)</summary>

| Plant | Diseases Detected |
|-------|-------------------|
| 🍎 **Apple** | Apple scab, Black rot, Cedar apple rust |
| 🫐 **Blueberry** | Healthy detection |
| 🍒 **Cherry** | Powdery mildew |
| 🌽 **Corn** | Cercospora leaf spot, Common rust, Northern Leaf Blight |
| 🍇 **Grape** | Black rot, Esca, Leaf blight |
| 🍊 **Orange** | Huanglongbing (Citrus greening) |
| 🍑 **Peach** | Bacterial spot |
| 🌶️ **Pepper** | Bacterial spot |
| 🥔 **Potato** | Early blight, Late blight |
| 🍇 **Raspberry** | Healthy detection |
| 🫘 **Soybean** | Healthy detection |
| 🎃 **Squash** | Powdery mildew |
| 🍓 **Strawberry** | Leaf scorch |
| 🍅 **Tomato** | Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, Spider mites, Target spot, Yellow leaf curl virus, Mosaic virus, Healthy |

</details>

---

## 🚀 Deployment

### Hugging Face Spaces (Recommended)

This application is optimized for Hugging Face Spaces:

1. **Fork or clone** this Space
2. The Space automatically builds using the YAML header in this README
3. Model weights download automatically on first run (~13GB)
4. Initial startup may take 5-10 minutes; subsequent runs are faster

### Hardware Tiers

| Tier | RAM | Startup Time | Inference Time | Monthly Cost |
|------|-----|--------------|----------------|--------------|
| **CPU Basic** (Free) | 16GB | 3-5 min | 15-30s | Free |
| **CPU Upgrade** | 32GB | 2-3 min | 10-20s | $5-10 |
| **GPU T4** | 16GB VRAM | 1-2 min | 2-5s | $20-30 |
| **GPU A10G** | 24GB VRAM | <1 min | 1-3s | $50-70 |

### Environment Variables

Configure behavior via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GARDEN_DOCTOR_MOCK` | `true` | Set to `false` for production |
| `GARDEN_DOCTOR_TIMEOUT` | `45` | Inference timeout in seconds |
| `GARDEN_DOCTOR_4BIT` | `auto` | Quantization: `auto`, `true`, `false` |
| `HEALTH_CHECK_ENABLED` | `true` | Enable `/health` endpoint |

### GitHub Actions CI/CD

This repository includes automated workflows for continuous integration and deployment:

#### Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `sync-to-hf.yml` | Push to `main` | Auto-sync to Hugging Face Spaces |
| `run-tests.yml` | Push/PR to `main` | Run pytest test suite |

#### Setup Required

To enable automatic syncing to Hugging Face Spaces:

1. **Create HF Token**: Go to [Hugging Face Settings](https://huggingface.co/settings/tokens) → Create token with **Write** permissions
2. **Add GitHub Secret**: Repository → Settings → Secrets → Actions → New secret:
   - **Name**: `HF_TOKEN`
   - **Value**: Your Hugging Face write token
3. **Verify**: Push to main branch and check Actions tab

<details>
<summary>🔧 Manual Deployment Trigger</summary>

You can also manually trigger deployment:

1. Go to **Actions** tab in GitHub
2. Select **"Sync to Hugging Face Spaces"**
3. Click **"Run workflow"**
4. Enable **"Force full sync"** if needed
5. Click **"Run workflow"**

</details>

See [`.github/GITHUB_ACTIONS_SETUP.md`](.github/GITHUB_ACTIONS_SETUP.md) for detailed setup instructions.

---

## 📊 Performance Metrics

| Metric | CPU Tier | GPU Tier |
|--------|----------|----------|
| Cold Start | 3-5 min | 1-2 min |
| Warm Start | <10s | <5s |
| Avg Inference | 15-25s | 2-5s |
| Memory Usage | ~8GB | ~6GB VRAM |
| Concurrent Users | 3-5 | 10-15 |

---

## ⚠️ Disclaimer

<div align="center">

**📋 IMPORTANT: This application provides informational guidance only.**

</div>

**Garden Doctor is NOT a substitute for professional agricultural consultation.**

### Limitations

- Best results require clear, well-lit photographs of visible symptoms
- The model may not detect early-stage or asymptomatic diseases
- Limited to conditions represented in the PlantVillage training dataset
- Unusual or rare plant diseases may not be correctly identified
- Results with low confidence (<50%) should be treated as uncertain

### Recommendations

- **For critical crop decisions**: Consult a certified agronomist or agricultural extension service
- **For commercial operations**: Verify diagnoses with laboratory testing
- **For home gardening**: Use as a first-check tool before seeking expert advice

**The developers assume no liability for actions taken based on this application's output.**

---

## 📁 Project Structure

```
GardenDoctor/
├── app.py                      # Main Gradio application (optimized for HF Spaces)
├── requirements.txt            # Python dependencies
├── README.md                   # This file (HF Spaces metadata header)
├── space_config.yaml           # Detailed HF Spaces configuration
├── .gitignore                  # Git ignore patterns
├── docs/
│   ├── Garden_Doctor_PRD.md    # Product Requirements Document
│   ├── TESTING_CHECKLIST.md    # Pre-launch testing checklist
│   └── MONITORING_PLAN.md      # Post-launch monitoring plan
├── examples/                   # Pre-loaded example images
│   ├── tomato_healthy.jpg
│   ├── tomato_early_blight.jpg
│   ├── potato_late_blight.jpg
│   ├── apple_scab.jpg
│   ├── corn_rust.jpg
│   └── grape_black_rot.jpg
└── src/                        # Utility modules
    ├── __init__.py
    ├── prompts.py              # Prompt templates
    └── utils.py                # Helper functions
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'feat: add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

| Resource | Attribution |
|----------|-------------|
| **AI Model** | [YuchengShi/LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection](https://huggingface.co/YuchengShi/LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection) |
| **Training Dataset** | [PlantVillage Dataset](https://www.plantvillage.psu.edu/) by Penn State University |
| **UI Framework** | [Gradio](https://www.gradio.app/) |
| **Base Model** | [LLaVA](https://llava-vl.github.io/) - Large Language and Vision Assistant |

---

<div align="center">

**Made with 🌱 for gardeners, farmers, and plant enthusiasts everywhere**

*If this tool helped you, please consider giving it a ⭐!*

[⬆ Back to Top](#-garden-doctor-plant-disease--care-assistant)

</div>
