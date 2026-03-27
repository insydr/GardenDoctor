# Garden Doctor - Post-Launch Monitoring Plan

**Operational Monitoring and Continuous Improvement Framework**

*Last Updated: March 2026*

---

## 📊 Monitoring Overview

This document outlines the monitoring strategy for Garden Doctor after deployment to Hugging Face Spaces. Effective monitoring ensures service reliability, identifies improvement opportunities, and tracks user satisfaction.

---

## 1. Key Performance Indicators (KPIs)

### 1.1 System Health Metrics

| Metric | Target | Alert Threshold | Collection Method |
|--------|--------|-----------------|-------------------|
| **Availability** | ≥99% | <95% triggers alert | HF Spaces status + `/health` endpoint |
| **Cold Start Time** | <5 min | >7 min | HF Spaces logs |
| **Inference Latency (CPU)** | <30s | >45s | Application logs |
| **Inference Latency (GPU)** | <5s | >10s | Application logs |
| **Memory Usage** | <12GB | >14GB | `/health` endpoint |
| **Error Rate** | <5% | >10% | Application logs |
| **Queue Wait Time** | <60s | >120s | Gradio metrics |

### 1.2 User Engagement Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Daily Active Users** | Unique users per day | 50+ after launch |
| **Diagnoses per Session** | Average diagnoses per visit | 1.5-2.0 |
| **Session Duration** | Time spent on app | 3-5 minutes |
| **PDF Downloads** | Reports generated per day | 10% of diagnoses |
| **Example Gallery Usage** | Clicks on examples | 20% of sessions |

### 1.3 Quality Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **High Confidence Rate** | Diagnoses ≥80% confidence | ≥60% |
| **Low Confidence Rate** | Diagnoses <50% confidence | <15% |
| **Unsupported Plant Rate** | Unrecognized plant species | <10% |
| **Timeout Rate** | Requests exceeding 45s | <5% |

---

## 2. Monitoring Infrastructure

### 2.1 Health Check Endpoint

The `/health` endpoint provides real-time system status:

```json
{
  "status": "healthy",
  "timestamp": "2026-03-27T10:30:00.000Z",
  "service": "Garden Doctor AI",
  "version": "1.0.0",
  "model_id": "YuchengShi/LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection",
  "device": "cpu",
  "mock_mode": false,
  "timeout": 45,
  "model_loaded": true,
  "quantization_mode": "float32_cpu",
  "memory_mb": 8542.3,
  "cpu_percent": 45.2
}
```

**Monitoring Setup:**

```bash
# Example: Uptime monitoring with curl
curl -s https://your-space.hf.space/health | jq '.status'

# Expected: "healthy"
```

### 2.2 Logging Strategy

#### Application Logs

```python
# Log levels in use:
# INFO  - Normal operations, diagnoses completed
# WARNING - Low confidence results, timeouts
# ERROR - Failures, exceptions
```

#### Key Log Events

| Event | Log Level | Example Message |
|-------|-----------|-----------------|
| Model loaded | INFO | "Model loaded successfully! Quantization mode: float32_cpu" |
| Diagnosis start | INFO | "Starting inference with 45s timeout..." |
| Diagnosis complete | INFO | "Diagnosis: Early Blight (High, 85%)" |
| Low confidence | WARNING | "Low confidence diagnosis: 35%" |
| Timeout | WARNING | "Inference timeout after 45 seconds" |
| Error | ERROR | "Model inference error: CUDA out of memory" |

### 2.3 Hugging Face Spaces Metrics

Monitor via HF Spaces dashboard:

| Dashboard Metric | What It Shows |
|------------------|---------------|
| **CPU/Memory Graph** | Resource utilization over time |
| **Request Count** | API calls and UI interactions |
| **Build Logs** | Deployment and startup status |
| **Error Logs** | Application errors and exceptions |

---

## 3. Data Collection

### 3.1 Automatic Metrics Collection

Metrics are collected automatically through:

1. **Health Check Polling** - External monitoring service pings `/health` every 30s
2. **Application Logs** - Structured logs capture all operations
3. **HF Spaces Analytics** - Built-in usage statistics

### 3.2 Anonymized Image Collection (Opt-In)

For model improvement, consider collecting low-confidence images:

#### Implementation Approach

```python
# Pseudocode for opt-in image collection
if confidence < 0.5 and user_consent == True:
    # Anonymize image
    anonymized_image = remove_metadata(image)
    # Store for review
    store_for_improvement(anonymized_image, diagnosis_result)
```

#### Privacy Considerations

| Consideration | Mitigation |
|---------------|------------|
| **Image Metadata** | Strip all EXIF data before storage |
| **Personal Information** | Images contain no personal data by design |
| **Storage Duration** | Auto-delete after 90 days or after review |
| **User Consent** | Explicit opt-in checkbox required |
| **Transparency** | Clear explanation in privacy policy |

#### Data Collection Form (Future Enhancement)

```markdown
### Help Us Improve

We'd like to use anonymous leaf images to improve our AI. 
No personal information is collected.

[ ] I consent to sharing my leaf image for research purposes

[Submit] [Decline]
```

### 3.3 User Feedback Collection

#### In-App Feedback

```markdown
### Was this diagnosis helpful?

👍 Yes, this was helpful
👎 No, needs improvement

[Optional: Tell us more...]
```

#### Feedback Categories

| Category | Questions |
|----------|-----------|
| **Accuracy** | "Was the diagnosis correct?" |
| **Helpfulness** | "Were the treatment recommendations useful?" |
| **Ease of Use** | "Was the app easy to use?" |
| **Speed** | "Was the response time acceptable?" |

---

## 4. Alerting Strategy

### 4.1 Alert Rules

| Alert | Condition | Severity | Response |
|-------|-----------|----------|----------|
| **Service Down** | Health check fails 3x consecutive | Critical | Immediate investigation |
| **High Error Rate** | Error rate >10% for 5 min | High | Check logs, investigate |
| **Memory Critical** | Memory >14GB for 5 min | High | Consider restart or upgrade |
| **Slow Response** | Avg latency >30s for 10 min | Medium | Check load, consider scaling |
| **Low Confidence Spike** | Low confidence >30% for 1 hour | Low | Review recent inputs |

### 4.2 Notification Channels

| Channel | Use Case |
|---------|----------|
| **Email** | Daily summary, critical alerts |
| **Slack/Discord** | Real-time alerts |
| **HF Spaces Dashboard** | Build failures, resource warnings |

### 4.3 Alert Template

```markdown
## 🚨 Garden Doctor Alert

**Alert Type:** [Service Down / High Error Rate / etc.]
**Severity:** [Critical / High / Medium / Low]
**Timestamp:** [ISO datetime]
**Space:** [your-space-name]

**Details:**
- Error rate: [X]%
- Avg latency: [X]s
- Memory: [X]GB

**Recent Errors:**
```
[paste relevant log lines]
```

**Action Required:** [Specific steps]
```

---

## 5. Continuous Improvement

### 5.1 Regular Reviews

| Review Type | Frequency | Participants |
|-------------|-----------|--------------|
| **Daily Health Check** | Daily | Automated |
| **Weekly Metrics Review** | Weekly | Dev team |
| **Monthly Performance Report** | Monthly | Dev + Product |
| **Quarterly Model Review** | Quarterly | Dev + ML team |

### 5.2 Improvement Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    IMPROVEMENT PIPELINE                          │
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐     │
│  │  COLLECT │ → │ ANALYZE  │ → │ PRIORITIZE│ → │ IMPLEMENT│     │
│  │  Data    │   │ Patterns │   │ Backlog   │   │ Changes  │     │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘     │
│       ↑                                              │          │
│       └──────────────────────────────────────────────┘          │
│                        FEEDBACK LOOP                             │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Improvement Categories

#### Short-Term (1-2 weeks)

- Fix bugs and edge cases
- Optimize inference speed
- Improve error messages
- UI/UX tweaks based on feedback

#### Medium-Term (1-2 months)

- Add support for new plant species
- Improve low-confidence handling
- Implement offline model caching
- Add multi-language support

#### Long-Term (3-6 months)

- Retrain model with collected data
- Implement disease progression tracking
- Add community features (shared diagnoses)
- Mobile app development

---

## 6. Incident Response

### 6.1 Incident Severity Levels

| Level | Description | Response Time | Examples |
|-------|-------------|---------------|----------|
| **P1 - Critical** | Service completely down | 15 minutes | Model won't load, all requests fail |
| **P2 - High** | Major functionality broken | 1 hour | Inference failing, memory issues |
| **P3 - Medium** | Degraded performance | 4 hours | Slow responses, increased errors |
| **P4 - Low** | Minor issues | 24 hours | UI glitches, non-critical bugs |

### 6.2 Incident Response Playbook

#### Step 1: Detection

- Automated alert triggers
- User reports
- Routine monitoring

#### Step 2: Assessment

1. Check `/health` endpoint
2. Review HF Spaces logs
3. Check error rates and latency
4. Determine severity level

#### Step 3: Mitigation

| Issue | Immediate Action |
|-------|------------------|
| Memory issues | Restart Space |
| Model won't load | Fall back to mock mode |
| High latency | Reduce concurrent users |
| Complete outage | Check HF status, contact support |

#### Step 4: Resolution

1. Identify root cause
2. Implement fix
3. Deploy fix
4. Verify resolution

#### Step 5: Post-Incident Review

- Document incident in runbook
- Identify preventive measures
- Update monitoring if needed

---

## 7. Reporting

### 7.1 Daily Report

```markdown
# Garden Doctor Daily Report - [Date]

## System Health
- Uptime: [X]%
- Avg Latency: [X]s
- Error Rate: [X]%

## Usage
- Total Diagnoses: [X]
- Unique Users: [X]
- PDF Downloads: [X]

## Notable Events
- [Event 1]
- [Event 2]
```

### 7.2 Weekly Metrics Dashboard

| Metric | This Week | Last Week | Change |
|--------|-----------|-----------|--------|
| Total Diagnoses | ___ | ___ | ___% |
| Avg Latency | ___s | ___s | ___% |
| Error Rate | ___% | ___% | ___% |
| High Confidence Rate | ___% | ___% | ___% |
| User Satisfaction | ___% | ___% | ___% |

### 7.3 Monthly Summary

- Executive summary of performance
- Key achievements and challenges
- User feedback highlights
- Planned improvements
- Resource utilization trends

---

## 8. Tools and Resources

### 8.1 Monitoring Tools

| Tool | Purpose | Cost |
|------|---------|------|
| HF Spaces Dashboard | Built-in metrics | Free |
| UptimeRobot | External uptime monitoring | Free tier available |
| Grafana | Custom dashboards | Self-hosted |
| Sentry | Error tracking | Free tier available |

### 8.2 Useful Commands

```bash
# Check Space health
curl -s https://your-space.hf.space/health | jq

# View recent logs (if SSH access)
tail -f /var/log/gradio/app.log

# Test API endpoint
curl -X POST https://your-space.hf.space/api/diagnose \
  -F "image=@test_leaf.jpg" \
  -F "climate=Temperate"
```

---

## 9. Contact and Escalation

| Role | Responsibility | Contact |
|------|----------------|---------|
| **On-Call Engineer** | First response to alerts | [Contact info] |
| **Lead Developer** | Technical decisions | [Contact info] |
| **Product Owner** | Business decisions | [Contact info] |
| **HF Support** | Platform issues | support@huggingface.co |

---

## Appendix A: Sample Monitoring Dashboard

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    GARDEN DOCTOR MONITORING DASHBOARD                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │   🟢 STATUS     │  │   ⏱️ LATENCY    │  │   💾 MEMORY     │          │
│  │   HEALTHY       │  │   12.3s avg     │  │   8.2 GB        │          │
│  │   99.8% uptime  │  │   ↓ 5%          │  │   ────░░░░      │          │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘          │
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │   📊 TODAY      │  │   🎯 CONFIDENCE │  │   ⚠️ ERRORS     │          │
│  │   247 diagnoses │  │   68% high      │  │   2.3% rate     │          │
│  │   ↑ 12%         │  │   22% medium    │  │   ↓ from 3.1%   │          │
│  └─────────────────┘  │   10% low       │  └─────────────────┘          │
│                       └─────────────────┘                               │
│                                                                          │
│  Recent Diagnoses (Last Hour)                                            │
│  ─────────────────────────────────────────────────────                   │
│  10:45 🍅 Tomato Early Blight (High)                                     │
│  10:42 🥔 Potato Late Blight (High)                                      │
│  10:38 🍎 Apple Scab (Medium)                                            │
│  10:35 🌽 Corn Rust (High)                                               │
│  10:30 🍇 Grape Black Rot (Low) ⚠️                                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

*Document maintained by: Garden Doctor Development Team*
*Next review date: Quarterly*
