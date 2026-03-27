# Garden Doctor - Testing Checklist

**Pre-Launch Testing Guide for Hugging Face Spaces Deployment**

*Last Updated: March 2026*

---

## 📋 Pre-Deployment Checklist

Before deploying to Hugging Face Spaces, verify all items below.

### ✅ Core Functionality

#### Image Upload

| Test | Expected Behavior | Status |
|------|------------------|--------|
| JPEG upload | Image loads and displays correctly | ☐ |
| PNG upload | Image loads and displays correctly | ☐ |
| WebP upload | Image loads and displays correctly | ☐ |
| Large image (>5MB) | Image is accepted and processed | ☐ |
| Small image (<100KB) | Image is accepted and processed | ☐ |
| Drag and drop | Image uploads via drag & drop | ☐ |
| File picker | File picker dialog opens correctly | ☐ |
| Webcam capture | Camera captures and uploads image | ☐ |
| No image + Diagnose | Warning toast: "Please upload an image first!" | ☐ |
| Invalid file type | Error message displayed appropriately | ☐ |

#### Climate Dropdown

| Test | Expected Behavior | Status |
|------|------------------|--------|
| Default selection | "Temperate" is pre-selected | ☐ |
| Dropdown opens | All 4 climate zones are listed | ☐ |
| Select Tropical | Climate info updates with tropical description | ☐ |
| Select Temperate | Climate info updates with temperate description | ☐ |
| Select Arid | Climate info updates with arid description | ☐ |
| Select Cold | Climate info updates with cold description | ☐ |
| Climate affects output | Treatment recommendations mention climate | ☐ |

#### Diagnosis Flow

| Test | Expected Behavior | Status |
|------|------------------|--------|
| Progress indicator | Shows during diagnosis (0%, 25%, 50%, 75%, 100%) | ☐ |
| Progress messages | Shows descriptive status messages | ☐ |
| Diagnose button | Disabled during processing | ☐ |
| Successful diagnosis | Results display with confidence label | ☐ |
| Error handling | Graceful error messages on failure | ☐ |

---

### ✅ Confidence Threshold Logic

#### High Confidence (≥80%)

| Test | Expected Behavior | Status |
|------|------------------|--------|
| Clear disease image | Shows 🟢 High Confidence badge | ☐ |
| Confidence percentage | Displays correctly (e.g., "85%") | ☐ |
| Interpretation text | "Results are reliable. Follow treatment recommendations" | ☐ |
| Toast notification | "✅ High confidence diagnosis complete!" | ☐ |
| Download button | Visible and clickable | ☐ |

#### Medium Confidence (50-79%)

| Test | Expected Behavior | Status |
|------|------------------|--------|
| Moderate disease image | Shows 🟡 Moderate Confidence badge | ☐ |
| Interpretation text | "Results are moderately reliable..." | ☐ |
| Toast notification | "✅ Diagnosis complete!" | ☐ |
| Download button | Visible and clickable | ☐ |

#### Low Confidence (<50%)

| Test | Expected Behavior | Status |
|------|------------------|--------|
| Unclear image | Shows 🔴 Low Confidence badge | ☐ |
| Warning displayed | Full low-confidence warning message appears | ☐ |
| Improvement tips | Tips for better image quality shown | ☐ |
| Expert help section | Links to agricultural extension services | ☐ |
| Toast notification | "⚠️ Low confidence result - check tips below." | ☐ |
| Download button | May or may not appear depending on result | ☐ |

---

### ✅ Performance Testing

#### Inference Latency

| Test | Target | Actual | Status |
|------|--------|--------|--------|
| Cold start (first request) | <300s | ____s | ☐ |
| Warm inference (CPU) | <30s | ____s | ☐ |
| Warm inference (GPU) | <5s | ____s | ☐ |
| Timeout at 45s | Request cancelled | | ☐ |
| Timeout message | "Analysis Timeout" error displayed | | ☐ |

#### Memory Usage

| Test | Target | Actual | Status |
|------|--------|--------|--------|
| Idle memory | <4GB | ____GB | ☐ |
| Model loaded | <12GB | ____GB | ☐ |
| Peak during inference | <14GB | ____GB | ☐ |
| Memory after inference | Returns to baseline | | ☐ |

#### Concurrent Users

| Test | Expected Behavior | Status |
|------|------------------|--------|
| Single user | Works normally | ☐ |
| 2 concurrent users | Queue processes requests | ☐ |
| 5 concurrent users | Queue handles requests (may wait) | ☐ |
| Queue full (20+) | "Queue full" message or graceful handling | ☐ |

---

### ✅ Mobile Responsiveness

#### Layout

| Test | Expected Behavior | Status |
|------|------------------|--------|
| Desktop (>1200px) | Two-column layout displays correctly | ☐ |
| Tablet (768-1200px) | Layout adjusts appropriately | ☐ |
| Mobile (<768px) | Single column, stacked layout | ☐ |
| Header scales | Title and description readable | ☐ |
| Buttons scale | Touch targets adequate (≥44px) | ☐ |

#### Mobile Functionality

| Test | Expected Behavior | Status |
|------|------------------|--------|
| Camera capture | Device camera opens on mobile | ☐ |
| File upload | Mobile file picker works | ☐ |
| Dropdown interaction | Touch-friendly dropdown | ☐ |
| Scrolling | No horizontal scroll on narrow screens | ☐ |
| PDF download | PDF opens/downloads on mobile | ☐ |

---

### ✅ Error States

| Test | Expected Behavior | Status |
|------|------------------|--------|
| No image provided | Warning toast + error message in results | ☐ |
| Invalid image format | "Invalid Image" error message | ☐ |
| Image too small (<50px) | "Invalid Image" error message | ☐ |
| Model not loaded | Graceful fallback to mock mode or error | ☐ |
| Network error | Appropriate error handling | ☐ |
| Timeout (>45s) | "Analysis Timeout" error with suggestions | ☐ |
| PDF generation failure | Toast warning + retry option | ☐ |

---

### ✅ Example Gallery

| Test | Expected Behavior | Status |
|------|------------------|--------|
| Examples display | 6 example images visible | ☐ |
| Click example | Image and climate auto-populate | ☐ |
| Auto-diagnose on click | Diagnosis triggers automatically | ☐ |
| Example thumbnails | Clear, identifiable plant images | ☐ |
| Gallery layout | Responsive grid arrangement | ☐ |

---

### ✅ PDF Export

| Test | Expected Behavior | Status |
|------|------------------|--------|
| Download button appears | After successful diagnosis | ☐ |
| Click download | PDF generates and downloads | ☐ |
| PDF filename | Contains timestamp (garden_doctor_report_YYYYMMDD_HHMMSS.pdf) | ☐ |
| PDF opens | Valid PDF that opens in readers | ☐ |
| PDF contains header | "Garden Doctor AI" branding | ☐ |
| PDF contains diagnosis | Disease name and confidence | ☐ |
| PDF contains treatments | All treatment options listed | ☐ |
| PDF contains image | Analyzed leaf image included | ☐ |
| PDF contains disclaimer | Disclaimer text at bottom | ☐ |
| Toast on generate | "PDF report ready for download!" | ☐ |

---

### ✅ Health Check Endpoint

| Test | Expected Behavior | Status |
|------|------------------|--------|
| GET /health | Returns 200 OK | ☐ |
| Response format | Valid JSON object | ☐ |
| Status field | "healthy" | ☐ |
| Model loaded field | true after model loads | ☐ |
| Timestamp field | Valid ISO datetime | ☐ |
| Memory info | Memory usage in MB (if psutil available) | ☐ |

---

### ✅ Browser Compatibility

| Browser | Version | Status |
|---------|---------|--------|
| Chrome | Latest | ☐ |
| Firefox | Latest | ☐ |
| Safari | Latest | ☐ |
| Edge | Latest | ☐ |
| Mobile Chrome | Latest | ☐ |
| Mobile Safari | Latest | ☐ |

---

## 🧪 Test Images

Use these test cases for comprehensive testing:

### High Confidence Expected

| Image Type | Expected Disease | Notes |
|------------|------------------|-------|
| Tomato early blight | Early Blight | Clear concentric rings |
| Potato late blight | Late Blight | Water-soaked lesions |
| Apple scab | Apple Scab | Olive-brown spots |
| Healthy tomato leaf | Healthy | No symptoms |

### Low Confidence Expected

| Image Type | Expected Behavior | Notes |
|------------|-------------------|-------|
| Blurry image | Low confidence warning | Out of focus |
| Dark image | Low confidence warning | Poor lighting |
| Non-plant image | Unsupported plant or low confidence | Random object |
| Multiple leaves | May show low confidence | Too complex |
| Flower (not leaf) | May show unsupported plant | Not a leaf |

---

## 📊 Test Results Log

| Date | Tester | Environment | Result | Notes |
|------|--------|-------------|--------|-------|
| ____ | ______ | ___________ | ☐ Pass ☐ Fail | _________________ |
| ____ | ______ | ___________ | ☐ Pass ☐ Fail | _________________ |
| ____ | ______ | ___________ | ☐ Pass ☐ Fail | _________________ |

---

## ✅ Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Developer | ________ | ______ | ________ |
| QA Tester | ________ | ______ | ________ |
| Product Owner | ________ | ______ | ________ |

---

*Testing checklist complete? Proceed to deployment!*
