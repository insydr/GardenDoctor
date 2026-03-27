# Manual Test Scenarios for Garden Doctor

**QA Testing Guide with Step-by-Step Instructions**

*Last Updated: March 2026*

---

## Overview

This document provides detailed manual test scenarios for quality assurance testing of Garden Doctor. Each scenario includes:

- **Objective**: What we're testing
- **Prerequisites**: Setup requirements
- **Steps**: Detailed instructions
- **Expected Results**: What should happen
- **Pass Criteria**: How to determine success

---

## Test Environment Setup

### Prerequisites

1. **Application Access**: Garden Doctor deployed on Hugging Face Spaces or running locally
2. **Test Images**: Sample plant leaf images (provided in `examples/` folder)
3. **Browsers**: Chrome, Firefox, Safari, Edge (latest versions)
4. **Devices**: Desktop, tablet, and mobile for responsive testing

### Test Data

| Image File | Description | Expected Disease |
|------------|-------------|------------------|
| `tomato_healthy.jpg` | Healthy tomato leaf | Healthy / No disease |
| `tomato_early_blight.jpg` | Tomato with early blight | Early Blight |
| `potato_late_blight.jpg` | Potato with late blight | Late Blight |
| `apple_scab.jpg` | Apple leaf with scab | Apple Scab |
| `corn_rust.jpg` | Corn leaf with rust | Common Rust |
| `grape_black_rot.jpg` | Grape leaf with black rot | Black Rot |

---

## Scenario 1: Happy Path - Clear Tomato Blight Diagnosis

### Objective
Verify that a clear, well-lit image of diseased tomato leaf produces an accurate high-confidence diagnosis.

### Prerequisites
- Application is running and accessible
- Test image: `tomato_early_blight.jpg` or clear photo of tomato early blight

### Steps

1. Open Garden Doctor application in browser
2. Locate the image upload area (left panel)
3. Click upload area or drag and drop `tomato_early_blight.jpg`
4. Verify image appears in the upload preview
5. Confirm "Temperate" is selected in Climate Zone dropdown
6. Click the "🔬 Diagnose" button
7. Wait for analysis to complete (watch progress indicators)

### Expected Results

| Checkpoint | Expected Behavior |
|------------|-------------------|
| Image upload | Image loads and displays within 2 seconds |
| Progress indicator | Shows stages: "Validating", "Preprocessing", "Analyzing", "Processing", "Finalizing" |
| Completion time | 5-30 seconds on CPU, 2-5 seconds on GPU |
| Diagnosis name | Should show "Early Blight" or similar disease name |
| Confidence badge | Should show 🟢 High Confidence (≥80%) |
| Treatment section | Should display Cultural, Organic, and Conventional options |
| Prevention tips | Should include climate-specific recommendations |
| Toast notification | Should show "✅ High confidence diagnosis complete!" |
| Download button | Should appear below results |

### Pass Criteria
- [ ] Diagnosis identifies Early Blight (or correct disease)
- [ ] Confidence is High (≥80%)
- [ ] All treatment categories are populated
- [ ] PDF download works
- [ ] No error messages displayed

### Screenshot Location
`test_results/scenario1_happy_path.png`

---

## Scenario 2: Edge Case - Blurry Image Low Confidence

### Objective
Verify that low-quality images are handled gracefully with appropriate warnings.

### Prerequisites
- Intentionally blurry or dark image of plant leaf
- Or use an out-of-focus photo

### Steps

1. Open Garden Doctor application
2. Upload a blurry/out-of-focus plant leaf image
3. Select any climate zone (e.g., "Temperate")
4. Click "🔬 Diagnose"
5. Wait for analysis to complete

### Expected Results

| Checkpoint | Expected Behavior |
|------------|-------------------|
| Diagnosis returned | May show tentative diagnosis or "Unknown" |
| Confidence level | Should show 🔴 Low Confidence (<50%) |
| Warning message | Full low-confidence warning should appear |
| Improvement tips | Tips for better image quality should be listed |
| Toast notification | Should show "⚠️ Low confidence result" |
| Expert help | Section suggesting professional consultation should appear |

### Pass Criteria
- [ ] Low confidence warning is displayed
- [ ] Image quality tips are provided
- [ ] User is directed to expert resources
- [ ] No application crash or unhandled errors

### Screenshot Location
`test_results/scenario2_low_confidence.png`

---

## Scenario 3: Edge Case - Non-Plant Image

### Objective
Verify that non-plant images are rejected with appropriate messaging.

### Prerequisites
- Non-plant image (e.g., car, person, landscape, food)
- Or random object photo

### Steps

1. Open Garden Doctor application
2. Upload a non-plant image (e.g., photo of a cat, car, or building)
3. Select any climate zone
4. Click "🔬 Diagnose"
5. Wait for analysis to complete

### Expected Results

| Checkpoint | Expected Behavior |
|------------|-------------------|
| Diagnosis | Should indicate "Unrecognized" or similar |
| Confidence | Should be Low |
| Error message | Should indicate plant not supported or not recognized |
| Supported plants | Should list supported crop types |
| Tips section | Should provide tips for better recognition |
| Toast notification | Should show "🌿 Plant not in database" |

### Pass Criteria
- [ ] Clear message that plant is not recognized
- [ ] List of supported plants is shown
- [ ] No crash or confusing error messages

### Screenshot Location
`test_results/scenario3_non_plant.png`

---

## Scenario 4: Climate Variation - Same Disease, Different Recommendations

### Objective
Verify that climate zone selection affects treatment recommendations.

### Prerequisites
- Same test image (e.g., `tomato_early_blight.jpg`)
- Run test twice with different climate zones

### Steps

#### Test Run 1 - Tropical Climate

1. Open Garden Doctor application
2. Upload `tomato_early_blight.jpg`
3. Select "Tropical" from Climate Zone dropdown
4. Click "🔬 Diagnose"
5. Note the prevention recommendations
6. Copy or screenshot the prevention section

#### Test Run 2 - Cold Climate

1. Clear previous results (click "🗑️ Clear")
2. Upload same image (`tomato_early_blight.jpg`)
3. Select "Cold" from Climate Zone dropdown
4. Click "🔬 Diagnose"
5. Note the prevention recommendations
6. Compare with Test Run 1

### Expected Results

| Checkpoint | Tropical | Cold |
|------------|----------|------|
| Climate-specific tip | Should mention humid/tropical considerations | Should mention frost/cold considerations |
| Prevention tips | Different emphasis on drainage, humidity | Different emphasis on season length, protection |
| Treatment options | May vary based on availability in region | May vary based on growing season |

### Pass Criteria
- [ ] Prevention tips differ between climates
- [ ] At least one tip mentions climate context
- [ ] Both results show same disease (consistency)
- [ ] Climate zone is displayed in output

### Screenshot Locations
- `test_results/scenario4_tropical.png`
- `test_results/scenario4_cold.png`

---

## Scenario 5: Image Format Support

### Objective
Verify that all supported image formats work correctly.

### Prerequisites
- Test images in JPEG, PNG, and WebP formats
- Can convert example images using image editor

### Steps

#### JPEG Format

1. Upload `tomato_early_blight.jpg` (or any .jpg image)
2. Click "🔬 Diagnose"
3. Verify successful diagnosis
4. Clear results

#### PNG Format

1. Upload a PNG version of same image
2. Click "🔬 Diagnose"
3. Verify successful diagnosis
4. Clear results

#### WebP Format

1. Upload a WebP version of same image
2. Click "🔬 Diagnose"
3. Verify successful diagnosis

### Expected Results

| Format | Upload | Diagnosis |
|--------|--------|-----------|
| JPEG | ✅ Works | ✅ Success |
| PNG | ✅ Works | ✅ Success |
| WebP | ✅ Works | ✅ Success |

### Pass Criteria
- [ ] All three formats upload successfully
- [ ] All three produce valid diagnoses
- [ ] No format-specific errors

---

## Scenario 6: PDF Report Generation

### Objective
Verify PDF report download functionality works correctly.

### Prerequisites
- Successful diagnosis completed (use Scenario 1)
- PDF viewer installed on test machine

### Steps

1. Complete a diagnosis (follow Scenario 1)
2. Locate "📄 Download PDF Report" button
3. Click the download button
4. Wait for PDF generation (should be instant)
5. Open downloaded PDF file
6. Review PDF contents

### Expected Results

| PDF Section | Content |
|-------------|---------|
| Header | "Garden Doctor AI" branding |
| Timestamp | Generation date and time |
| Disease name | Matches diagnosis result |
| Confidence | Shows level and percentage |
| Climate zone | Selected climate displayed |
| Symptoms | Detailed description |
| Treatments | All three categories listed |
| Prevention | Climate-specific tips |
| Disclaimer | "Informational purposes only" |
| Image | Uploaded leaf image included |

### Pass Criteria
- [ ] PDF downloads without error
- [ ] File opens correctly
- [ ] All sections present and readable
- [ ] Image is included
- [ ] Disclaimer is visible

---

## Scenario 7: Example Gallery Functionality

### Objective
Verify that example images work correctly for quick testing.

### Prerequisites
- Application running with example images available

### Steps

1. Scroll down to "🖼️ Example Images" section
2. Click on `tomato_healthy.jpg` example
3. Verify image auto-fills in upload area
4. Verify climate auto-selects
5. Verify diagnosis runs automatically
6. Clear results
7. Click on `corn_rust.jpg` example
8. Verify different result

### Expected Results

| Checkpoint | Expected Behavior |
|------------|-------------------|
| Click example | Image uploads automatically |
| Climate selection | Pre-selects appropriate climate |
| Auto-diagnosis | Runs automatically after selection |
| Different examples | Different diseases identified |
| Gallery layout | All 6 examples visible |

### Pass Criteria
- [ ] All example images are clickable
- [ ] Each produces a valid diagnosis
- [ ] Results clear between different examples

---

## Scenario 8: Mobile Responsive Layout

### Objective
Verify application works on mobile devices.

### Prerequisites
- Mobile device or browser mobile emulation
- Test on portrait and landscape orientations

### Steps

1. Open Garden Doctor on mobile device (or use Chrome DevTools mobile emulation)
2. Set viewport to mobile size (<768px width)
3. Test image upload
4. Test climate dropdown
5. Test diagnosis button
6. Review results layout
7. Test PDF download
8. Test example gallery scrolling

### Expected Results

| Element | Mobile Behavior |
|---------|-----------------|
| Layout | Single column (stacked) |
| Upload area | Touch-friendly, full width |
| Buttons | Large touch targets (≥44px) |
| Results | Readable without horizontal scroll |
| Example gallery | Horizontal scroll or stacked grid |
| Dropdown | Native mobile selector |

### Pass Criteria
- [ ] All functionality works on mobile
- [ ] No horizontal scrolling on main content
- [ ] Touch targets are appropriately sized
- [ ] Text is readable at mobile scale

---

## Scenario 9: Error Recovery

### Objective
Verify graceful error handling and recovery.

### Prerequisites
- Application running

### Steps

#### Test 9a: No Image Error

1. Open application
2. Do NOT upload any image
3. Click "🔬 Diagnose"
4. Observe error handling
5. Upload an image
6. Click "🔬 Diagnose" again
7. Verify recovery

#### Test 9b: Multiple Rapid Requests

1. Upload an image
2. Click "🔬 Diagnose"
3. Immediately click "🔬 Diagnose" again while processing
4. Observe queue behavior

### Expected Results

| Test | Expected Behavior |
|------|-------------------|
| No image | Warning toast + clear error message |
| Recovery | Works correctly after uploading image |
| Rapid requests | Queue handles gracefully or button disabled |

### Pass Criteria
- [ ] Clear error messages shown
- [ ] User can recover and continue
- [ ] No application crash
- [ ] No stuck loading states

---

## Scenario 10: Browser Compatibility

### Objective
Verify cross-browser compatibility.

### Prerequisites
- Access to Chrome, Firefox, Safari, Edge browsers

### Steps

For each browser:

1. Open Garden Doctor
2. Upload test image
3. Run diagnosis
4. Download PDF
5. Test example gallery

### Expected Results

| Browser | Upload | Diagnosis | PDF |
|---------|--------|-----------|-----|
| Chrome (latest) | ✅ | ✅ | ✅ |
| Firefox (latest) | ✅ | ✅ | ✅ |
| Safari (latest) | ✅ | ✅ | ✅ |
| Edge (latest) | ✅ | ✅ | ✅ |

### Pass Criteria
- [ ] All browsers complete full workflow
- [ ] No browser-specific visual bugs
- [ ] PDF downloads in all browsers

---

## Test Results Summary

| Scenario | Description | Status | Notes |
|----------|-------------|--------|-------|
| 1 | Happy Path - Clear Diagnosis | ☐ Pass ☐ Fail | |
| 2 | Edge Case - Blurry Image | ☐ Pass ☐ Fail | |
| 3 | Edge Case - Non-Plant Image | ☐ Pass ☐ Fail | |
| 4 | Climate Variation | ☐ Pass ☐ Fail | |
| 5 | Image Format Support | ☐ Pass ☐ Fail | |
| 6 | PDF Report Generation | ☐ Pass ☐ Fail | |
| 7 | Example Gallery | ☐ Pass ☐ Fail | |
| 8 | Mobile Responsive | ☐ Pass ☐ Fail | |
| 9 | Error Recovery | ☐ Pass ☐ Fail | |
| 10 | Browser Compatibility | ☐ Pass ☐ Fail | |

---

## Tester Sign-Off

| Field | Value |
|-------|-------|
| Tester Name | |
| Test Date | |
| Environment | ☐ Local ☐ HF Spaces ☐ Other |
| Browser(s) Used | |
| Overall Result | ☐ All Pass ☐ Some Fail ☐ Major Issues |
| Notes | |

---

## Issue Reporting Template

If a test fails, document using this template:

```markdown
### Issue Report

**Scenario**: [Scenario number and name]
**Severity**: ☐ Critical ☐ High ☐ Medium ☐ Low
**Description**: [What happened vs what was expected]

**Steps to Reproduce**:
1. 
2. 
3. 

**Actual Result**: 
**Expected Result**: 

**Screenshots**: [Attach screenshots]
**Environment**: [Browser, OS, Device]
**Date/Time**: 
```
