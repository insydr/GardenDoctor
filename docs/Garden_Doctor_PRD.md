# Product Requirements Document (PRD)

## Garden Doctor: Plant Disease & Care Assistant

**Document Version:** 1.0  
**Date:** March 27, 2026  
**Author:** Product Development Team  
**Status:** Draft for Review

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Product Vision & Goals](#3-product-vision--goals)
4. [Target Users & Personas](#4-target-users--personas)
5. [Product Features & Requirements](#5-product-features--requirements)
6. [Technical Architecture](#6-technical-architecture)
7. [AI Model Specification](#7-ai-model-specification)
8. [User Interface Design](#8-user-interface-design)
9. [Implementation Phases](#9-implementation-phases)
10. [Success Metrics & KPIs](#10-success-metrics--kpis)
11. [Risks & Mitigation](#11-risks--mitigation)
12. [Appendix](#12-appendix)

---

## 1. Executive Summary

### 1.1 Product Overview

Garden Doctor is an AI-powered plant disease detection and care recommendation application designed to help gardeners, farmers, and agricultural enthusiasts identify plant health issues and receive actionable treatment advice. The application leverages advanced multimodal AI technology to analyze plant leaf images, detect diseases, and generate comprehensive care instructions tailored to the user's regional climate conditions.

### 1.2 Value Proposition

The application addresses a critical gap in accessible agricultural expertise by providing instant, accurate plant disease diagnosis without requiring specialized botanical knowledge. By combining state-of-the-art vision-language models with practical horticultural guidance, Garden Doctor democratizes plant healthcare for users ranging from home gardeners to small-scale farmers and agricultural students.

### 1.3 Key Differentiators

Garden Doctor distinguishes itself from existing solutions through its multi-modal AI pipeline that combines image understanding with natural language generation. Unlike simple image classification tools that only return disease labels, our application provides detailed explanations of the identified condition, its causes, progression patterns, and comprehensive treatment protocols. The integration of climate-specific recommendations ensures that care advice is practical and applicable to the user's specific growing environment.

---

## 2. Problem Statement

### 2.1 Current Challenges

Plant diseases represent a significant threat to agricultural productivity and home gardening success. Research indicates that crop diseases account for up to 40% of global crop yield losses annually, with many of these losses being preventable through early detection and appropriate intervention. However, several barriers prevent effective plant disease management in practice.

**Accessibility Gap:** Professional plant pathologists and agricultural extension services are not readily available to most home gardeners and small-scale farmers. The expertise required to identify plant diseases accurately typically requires years of specialized training, leaving many plant health issues either unrecognized or misdiagnosed.

**Time Sensitivity:** Plant diseases can progress rapidly, with early intervention being crucial for successful treatment. The delay between symptom observation and professional diagnosis often results in disease progression beyond treatable stages. Home gardeners may wait days or weeks for expert consultation appointments, during which time their plants may succumb to entirely preventable conditions.

**Information Overload:** While internet resources provide abundant plant disease information, the quality and accuracy vary dramatically. Users without botanical training struggle to distinguish between similar-looking conditions, leading to inappropriate treatments that may worsen the problem or waste resources on ineffective remedies.

### 2.2 Target Problem

Garden Doctor addresses the fundamental challenge of bridging the gap between plant disease symptoms and actionable treatment guidance. The application enables users to upload a simple photograph of an affected plant leaf and receive immediate, expert-level diagnosis accompanied by specific treatment recommendations tailored to their growing conditions.

---

## 3. Product Vision & Goals

### 3.1 Vision Statement

To empower every gardener and farmer with instant access to plant disease expertise, enabling healthier plants, reduced crop losses, and more sustainable gardening practices through the power of artificial intelligence.

### 3.2 Primary Goals

**Goal 1: Accurate Disease Detection**  
Develop and deploy a disease detection system capable of identifying at least 38 distinct plant disease conditions with a minimum accuracy threshold of 85% on the PlantVillage dataset benchmark. The system must effectively distinguish between healthy plants and those affected by various fungal, bacterial, and viral pathogens.

**Goal 2: Actionable Care Guidance**  
Generate comprehensive, easy-to-follow treatment recommendations that users can immediately implement. Each recommendation must include identification confirmation, disease explanation, treatment options (organic and conventional), preventive measures, and climate-specific considerations.

**Goal 3: Accessible User Experience**  
Create an intuitive interface that requires no technical expertise to operate. Users should be able to complete a diagnosis session—from image upload to receiving recommendations—in under 60 seconds, with clear visual feedback throughout the process.

**Goal 4: Deployment on Hugging Face Spaces**  
Deploy the application on Hugging Face Spaces infrastructure to ensure broad accessibility, reliable performance, and integration with the open-source AI community. The deployment must function effectively on free-tier CPU resources while maintaining acceptable response times.

### 3.3 Success Criteria

The product will be considered successful when it achieves the following criteria within the first three months of deployment: processing at least 1,000 successful diagnosis requests, maintaining a user satisfaction rating above 4.0 out of 5.0, demonstrating disease detection accuracy above 80% on user-submitted images with verified ground truth, and receiving positive feedback from the Hugging Face community with active engagement metrics.

---

## 4. Target Users & Personas

### 4.1 Primary User Segments

**Home Gardeners:** Individuals who maintain personal gardens, vegetable patches, or ornamental plant collections. These users typically have basic horticultural knowledge but lack specialized plant pathology expertise. They value quick, reliable advice and prefer organic or natural treatment options when available.

**Small-Scale Farmers:** Agricultural operators managing limited acreage who cannot afford professional crop consulting services. These users need practical, cost-effective treatment recommendations and may have access to commercial agricultural products but require guidance on proper application.

**Agricultural Students & Educators:** Academic users seeking to learn about plant pathology through hands-on interaction with AI diagnostic tools. These users value detailed explanations of disease mechanisms and treatment rationale for educational purposes.

**Community Garden Managers:** Individuals responsible for maintaining shared garden spaces who must address plant health issues across diverse crop types. These users need scalable solutions for identifying and treating diseases across multiple plant varieties.

### 4.2 Primary Persona: "Home Gardener Helen"

**Demographics:** Helen is a 45-year-old working professional who maintains a 200-square-foot vegetable garden in a temperate climate zone. She has five years of gardening experience and grows tomatoes, peppers, lettuce, and various herbs.

**Goals & Motivations:** Helen wants to grow healthy, organic vegetables for her family. She takes pride in her garden and feels frustrated when plants struggle despite her best efforts. She seeks to understand what went wrong and how to prevent future occurrences.

**Pain Points:** Helen often notices unusual spots, discoloration, or wilting in her plants but cannot determine the cause. Online searches return overwhelming and often contradictory information. Local nurseries provide inconsistent advice, and professional consultation services are expensive and inconvenient.

**Usage Scenario:** Helen notices brown spots appearing on her tomato plant leaves. She opens Garden Doctor on her smartphone, takes a photo of the affected leaf, selects her temperate climate zone, and receives an instant diagnosis of Early Blight with specific treatment steps including proper pruning techniques, recommended fungicide options (both organic and conventional), and preventive measures for future seasons.

### 4.3 Secondary Persona: "Student Sam"

**Demographics:** Sam is a 22-year-old agricultural science student completing a plant pathology course. He has theoretical knowledge but limited practical field experience in disease identification.

**Goals & Motivations:** Sam wants to develop practical diagnostic skills and verify his assessments against expert analysis. He uses Garden Doctor as a learning tool to understand disease presentations and treatment protocols.

**Pain Points:** Textbook images often show idealized disease presentations that differ from real-world observations. Sam needs exposure to varied disease presentations to build diagnostic competency.

---

## 5. Product Features & Requirements

### 5.1 Core Features

#### Feature 1: Image-Based Disease Detection (Priority: P0 - Critical)

**Description:** Users can upload or capture images of plant leaves for automated disease analysis. The system processes the image through a fine-tuned vision-language model to identify potential diseases and assess confidence levels.

**Functional Requirements:**
- FR-1.1: Accept image uploads in JPEG, PNG, and WebP formats
- FR-1.2: Support images captured via device camera (webcam on desktop, camera app on mobile)
- FR-1.3: Process images with minimum resolution of 224x224 pixels
- FR-1.4: Display processing status with visual progress indicator
- FR-1.5: Return disease classification results within 30 seconds on standard hardware
- FR-1.6: Support batch upload for analyzing multiple leaves (future enhancement)

**Non-Functional Requirements:**
- NFR-1.1: Maintain model inference accuracy above 85% on PlantVillage test set
- NFR-1.2: Support concurrent users without degradation in response time
- NFR-1.3: Handle gracefully when image quality is insufficient for diagnosis

#### Feature 2: Climate-Aware Recommendations (Priority: P0 - Critical)

**Description:** The application tailors treatment recommendations based on the user's selected climate zone, recognizing that environmental conditions significantly influence disease progression and treatment efficacy.

**Functional Requirements:**
- FR-2.1: Provide climate zone dropdown with four primary options: Tropical, Temperate, Arid, Cold
- FR-2.2: Include climate zone descriptions to assist users in correct selection
- FR-2.3: Adjust treatment recommendations based on selected climate (e.g., humidity considerations, seasonal timing)
- FR-2.4: Provide region-specific preventive measures for upcoming seasons

#### Feature 3: Comprehensive Diagnosis Report (Priority: P0 - Critical)

**Description:** Following image analysis, users receive a detailed diagnosis report containing disease identification, confidence levels, disease explanation, and step-by-step treatment protocols.

**Functional Requirements:**
- FR-3.1: Display disease name and confidence score as primary output
- FR-3.2: Include confidence interpretation guide (e.g., "High confidence: above 80%")
- FR-3.3: Provide plain-language explanation of the identified disease
- FR-3.4: List treatment options categorized by approach (cultural practices, organic treatments, conventional treatments)
- FR-3.5: Include preventive measures to avoid recurrence
- FR-3.6: Format output using markdown for enhanced readability

#### Feature 4: Confidence Threshold Handling (Priority: P1 - High)

**Description:** When confidence scores fall below acceptable thresholds, the system provides appropriate feedback rather than potentially misleading diagnoses.

**Functional Requirements:**
- FR-4.1: Display warning when confidence score is below 50%
- FR-4.2: Suggest image quality improvements (lighting, focus, angle)
- FR-4.3: Recommend consulting professional resources for uncertain cases
- FR-4.4: Provide option to submit image for expert review (future enhancement)

### 5.2 Enhanced Features

#### Feature 5: Example Gallery (Priority: P1 - High)

**Description:** Pre-loaded example images demonstrating common plant diseases help users understand what to look for and provide quick testing functionality.

**Functional Requirements:**
- FR-5.1: Display gallery of 4-6 example images showing healthy and diseased leaf conditions
- FR-5.2: Include images representing different plant types (tomato, potato, apple, grape)
- FR-5.3: Enable one-click analysis of example images
- FR-5.4: Show expected diagnosis alongside example for educational purposes

#### Feature 6: PDF Report Export (Priority: P2 - Medium)

**Description:** Users can download a formatted PDF report of their diagnosis for record-keeping or sharing with gardening advisors.

**Functional Requirements:**
- FR-6.1: Generate PDF report containing uploaded image, diagnosis, and recommendations
- FR-6.2: Include timestamp and application branding
- FR-6.3: Support download functionality through browser interface

#### Feature 7: Community Data Sharing (Priority: P3 - Low - Future Enhancement)

**Description:** Optional feature allowing users to contribute their diagnosis data to a community dataset, supporting model improvement and agricultural research.

**Functional Requirements:**
- FR-7.1: Present clear opt-in consent for data sharing
- FR-7.2: Anonymize all uploaded images and metadata
- FR-7.3: Store contributions in Hugging Face Dataset
- FR-7.4: Display contribution count for community engagement

---

## 6. Technical Architecture

### 6.1 System Overview

The Garden Doctor application follows a modular architecture designed for deployment on Hugging Face Spaces. The system comprises three primary layers: the user interface layer built with Gradio, the inference layer powered by the LLaVA vision-language model, and the output processing layer that formats and presents diagnostic results.

### 6.2 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  Image      │  │  Climate    │  │  Diagnosis Results      │ │
│  │  Upload     │  │  Selector   │  │  Display                │ │
│  │  Component  │  │  Dropdown   │  │  (Label + Markdown)     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│                         │                                       │
│                         ▼                                       │
│                    GRADIO INTERFACE                             │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE LAYER                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           LLaVA-v1.5-7B-Plant-Leaf-Diseases             │   │
│  │                    (Vision-Language Model)               │   │
│  │  ┌─────────────┐     ┌─────────────┐                    │   │
│  │  │   Vision    │────▶│   Language  │                    │   │
│  │  │   Encoder   │     │   Decoder   │                    │   │
│  │  │  (CLIP-ViT) │     │   (Vicuna)  │                    │   │
│  │  └─────────────┘     └─────────────┘                    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                 OUTPUT PROCESSING LAYER                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  Confidence │  │  Disease    │  │  Treatment Plan         │ │
│  │  Calculator │  │  Classifier │  │  Generator              │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 Technology Stack

**Frontend Framework:** Gradio 4.x - Selected for its native Hugging Face Spaces integration, rapid prototyping capabilities, and built-in support for image upload and markdown rendering components.

**Core AI Model:** LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection (YuchengShi) - A fine-tuned multimodal vision-language model specifically optimized for plant disease detection and explanation.

**Model Serving:** Transformers library with PyTorch backend for model loading and inference execution.

**Programming Language:** Python 3.10+ - Standard for ML/AI applications with extensive library support.

**Deployment Platform:** Hugging Face Spaces with Gradio SDK - Provides free hosting, GPU upgrade options, and community visibility.

### 6.4 Data Flow

1. User uploads image through Gradio interface
2. Image is preprocessed to model-compatible format (resize, normalize)
3. Preprocessed image is passed to LLaVA vision encoder
4. Vision features are combined with text prompt for disease analysis
5. Language model generates structured diagnostic output
6. Output is parsed and formatted for Gradio display components
7. Results rendered to user with confidence scores and recommendations

---

## 7. AI Model Specification

### 7.1 Model Selection Rationale

**Selected Model:** `YuchengShi/LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection`

The chosen model offers several critical advantages for this application:

**Multimodal Capability:** Unlike pure image classification models, LLaVA is a vision-language model that can understand images AND generate natural language explanations. This enables not just disease identification but also detailed description of symptoms, causes, and treatment recommendations in a single inference pass.

**Domain-Specific Fine-Tuning:** The model has been specifically fine-tuned on the PlantVillage dataset, containing over 50,000 images of plant leaves across 14 crop species and 38 disease categories. This specialized training provides superior accuracy for agricultural applications compared to general-purpose vision models.

**Explanatory Output:** The model generates human-readable explanations alongside classifications, eliminating the need for a separate text generation component. Users receive context-aware responses that explain what they're seeing and why specific treatments are recommended.

**Open Source Availability:** Hosted on Hugging Face with permissive licensing, the model is freely accessible for both development and deployment, aligning with the open-source philosophy of the target platform.

### 7.2 Model Architecture

**Base Architecture:** LLaVA-1.5-7B (Large Language-and-Vision Assistant)

**Vision Encoder:** CLIP ViT-L/14 - A vision transformer pre-trained on 400 million image-text pairs, providing robust visual feature extraction capabilities.

**Language Model:** Vicuna-7B v1.5 - A fine-tuned variant of LLaMA optimized for conversational interactions and instruction following.

**Parameter Count:** Approximately 7 billion parameters, balancing capability with inference efficiency.

**Input Specification:**
- Image: 336×336 pixels, RGB format, normalized to [0,1] range
- Text: Natural language prompts for conditioning model output

**Output Specification:**
- Structured text containing disease classification, confidence indicators, and explanatory content

### 7.3 Model Performance Characteristics

**Accuracy Metrics:**
- PlantVillage test set accuracy: ~92% (reported by model card)
- Cross-validation performance: Consistent across major crop types
- Known limitations: Reduced accuracy on early-stage disease symptoms and images with poor lighting/focus

**Inference Requirements:**
- Minimum RAM: 16GB for CPU inference
- Recommended RAM: 32GB for optimal performance
- GPU Memory: 8GB VRAM enables faster inference (optional)
- Inference Time: 5-15 seconds on CPU (depending on hardware)

**Known Limitations:**
- Trained primarily on single-leaf images; reduced accuracy on whole-plant or field-scale images
- Best performance with clear, well-lit photographs showing disease symptoms prominently
- Limited to diseases represented in PlantVillage dataset (38 conditions across 14 species)

### 7.4 Alternative Models Considered

**Option A: Standard Image Classification (e.g., ResNet-50, MobileNetV2)**
- Pros: Faster inference, lower resource requirements
- Cons: Only outputs class labels, requires separate text generation for recommendations, less nuanced understanding

**Option B: Specialized Plant Disease Classification Models**
- Pros: Potentially higher classification accuracy, smaller model size
- Cons: Lack explanatory capabilities, would require multi-model pipeline

**Option C: GPT-4 Vision API**
- Pros: Superior general reasoning, no hosting requirements
- Cons: API costs, dependency on external service, privacy concerns

**Selected Approach Justification:** LLaVA-Plant-Disease provides the optimal balance of accuracy, explanatory capability, and deployment flexibility for the Hugging Face Spaces platform.

---

## 8. User Interface Design

### 8.1 Interface Layout

The application utilizes a two-column layout optimized for desktop and mobile responsiveness:

**Left Column (Input Area):**
- Application header with title and brief instructions
- Image upload component with drag-and-drop support
- Camera capture option for mobile users
- Climate zone dropdown selector
- Primary action button ("Diagnose")

**Right Column (Output Area):**
- Disease confidence label display (visual confidence bar)
- Diagnosis summary in markdown format
- Treatment recommendations with bullet points
- Preventive care section

**Bottom Section:**
- Example image gallery for quick testing
- Application information and credits

### 8.2 Visual Design Specifications

**Color Scheme:**
- Primary: Forest Green (#228B22) - Represents growth and plant health
- Secondary: Earth Brown (#8B4513) - Natural, grounded aesthetic
- Background: Off-White (#F5F5DC) - Clean, paper-like appearance
- Accent: Leaf Green (#90EE90) - Highlighting important elements
- Warning: Amber (#FFBF00) - Low confidence alerts

**Typography:**
- Headers: Sans-serif, bold, 18-24pt
- Body: Sans-serif, regular, 14-16pt
- Labels: Sans-serif, medium, 12-14pt

**Component Styling:**
- Gradio theme: `gr.themes.Green()` for gardening aesthetic
- Rounded corners on input/output containers
- Subtle shadows for depth
- Progress indicators during processing

### 8.3 Interaction Flow

```
Start → Upload/Capture Image → Select Climate → Click Diagnose → 
Processing (with progress indicator) → View Results → 
[Optional] Download Report → End
```

### 8.4 Responsive Design Considerations

- Stacked layout on mobile devices (below 768px width)
- Touch-optimized button sizes (minimum 44×44 pixels)
- Camera integration for smartphone users
- Compressed image upload for bandwidth efficiency

---

## 9. Implementation Phases

### Phase 1: Core Functionality (Week 1-2)

**Objective:** Deploy minimal viable product with core disease detection capability.

**Deliverables:**
- Basic Gradio interface with image upload
- Model loading and inference pipeline
- Simple confidence label output
- Climate selector dropdown
- Basic error handling

**Acceptance Criteria:**
- Successfully process uploaded images and return disease predictions
- Application accessible and functional on Hugging Face Spaces
- Inference time under 30 seconds on CPU

### Phase 2: Enhanced Output (Week 3)

**Objective:** Improve output quality and user experience.

**Deliverables:**
- Formatted markdown output with treatment recommendations
- Confidence threshold warnings
- Example image gallery
- Improved UI styling with Gradio theme
- Loading indicators during processing

**Acceptance Criteria:**
- Output includes structured treatment recommendations
- Confidence warnings display appropriately
- Example images function correctly

### Phase 3: Polish & Optimization (Week 4)

**Objective:** Optimize performance and add finishing touches.

**Deliverables:**
- PDF report generation feature
- Performance optimization for faster inference
- Comprehensive error handling
- Documentation and README
- Community feedback integration

**Acceptance Criteria:**
- PDF download functionality working
- Average inference time reduced by 20%
- Documentation complete and accurate

---

## 10. Success Metrics & KPIs

### 10.1 Technical Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Model Accuracy | ≥85% | Validation on PlantVillage test set |
| Inference Latency | ≤30 seconds | Application logging |
| Uptime | ≥99% | Hugging Face Spaces monitoring |
| Error Rate | ≤5% | Application error logging |

### 10.2 User Engagement Metrics

| Metric | Target (3-month) | Measurement Method |
|--------|------------------|-------------------|
| Total Diagnoses | 1,000+ | Analytics tracking |
| Unique Users | 500+ | Anonymous usage tracking |
| Return Users | 20% | Session analysis |
| Session Duration | ≥2 minutes | Time tracking |

### 10.3 Quality Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| User Satisfaction | ≥4.0/5.0 | Feedback collection |
| Diagnosis Helpfulness | ≥80% positive | Post-diagnosis survey |
| Recommendation Clarity | ≥85% positive | User feedback |

---

## 11. Risks & Mitigation

### 11.1 Technical Risks

**Risk: Model Inference Too Slow on Free CPU Tier**
- Probability: Medium
- Impact: High (poor user experience)
- Mitigation: Implement model quantization, optimize inference pipeline, consider upgrade to GPU tier if necessary

**Risk: Model Accuracy Insufficient for Real-World Images**
- Probability: Medium
- Impact: High (user trust erosion)
- Mitigation: Implement confidence thresholds, provide clear uncertainty warnings, collect user feedback for improvement

**Risk: Hugging Face Spaces Resource Limits**
- Probability: Low
- Impact: Medium (availability issues)
- Mitigation: Optimize memory usage, implement request queuing if needed

### 11.2 Product Risks

**Risk: Users Misinterpret AI Diagnosis as Professional Advice**
- Probability: High
- Impact: High (liability concerns)
- Mitigation: Clear disclaimers, encourage professional consultation for serious issues

**Risk: Limited Disease Coverage Frustrates Users**
- Probability: Medium
- Impact: Medium (user dissatisfaction)
- Mitigation: Communicate supported crops clearly, provide helpful resources for unsupported cases

### 11.3 Legal & Ethical Considerations

**Disclaimer Requirements:** All outputs must include clear statements that the application provides informational guidance only and does not replace professional agricultural consultation.

**Data Privacy:** Uploaded images must not be stored permanently without explicit user consent. Privacy policy must be clearly accessible.

**Model Limitations:** Users must be informed that the AI model has known limitations and may not detect all plant health issues, particularly those outside its training distribution.

---

## 12. Appendix

### 12.1 Supported Plant Diseases (PlantVillage Dataset)

The model supports detection across 14 plant species:

1. **Apple:** Apple scab, Black rot, Cedar apple rust, Healthy
2. **Blueberry:** Healthy
3. **Cherry:** Powdery mildew, Healthy
4. **Corn:** Cercospora leaf spot, Common rust, Northern Leaf Blight, Healthy
5. **Grape:** Black rot, Esca, Leaf blight, Healthy
6. **Orange:** Huanglongbing (Citrus greening), Healthy
7. **Peach:** Bacterial spot, Healthy
8. **Pepper:** Bacterial spot, Healthy
9. **Potato:** Early blight, Late blight, Healthy
10. **Raspberry:** Healthy
11. **Soybean:** Healthy
12. **Squash:** Powdery mildew
13. **Strawberry:** Leaf scorch, Healthy
14. **Tomato:** Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, Spider mites, Target spot, Yellow leaf curl virus, Mosaic virus, Healthy

### 12.2 Climate Zone Definitions

| Zone | Characteristics | Examples |
|------|----------------|----------|
| Tropical | Hot, humid, year-round growing | Southeast Asia, Central America |
| Temperate | Four seasons, moderate humidity | Eastern US, Europe, East Asia |
| Arid | Hot, dry, low rainfall | Southwest US, Middle East, Australia |
| Cold | Short growing season, harsh winters | Northern Canada, Scandinavia, Russia |

### 12.3 References

- PlantVillage Dataset: https://www.plantvillage.psu.edu/
- LLaVA Model Paper: "Visual Instruction Tuning" (Liu et al., 2023)
- Hugging Face Spaces Documentation: https://huggingface.co/docs/hub/spaces
- Gradio Documentation: https://www.gradio.app/docs/

---

**Document Approval:**

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Product Manager | _______________ | __________ | __________ |
| Technical Lead | _______________ | __________ | __________ |
| Designer | _______________ | __________ | __________ |

---

*This PRD is a living document and will be updated as the project progresses through development phases.*
