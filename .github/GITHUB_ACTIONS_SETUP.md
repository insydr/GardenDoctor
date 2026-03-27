# GitHub Actions Setup Guide

## Overview

This repository uses GitHub Actions for:
1. **Sync to Hugging Face Spaces** - Automatic deployment on push to main
2. **Run Tests** - Automated testing on push and pull requests

---

## Required Secrets

### HF_TOKEN (Hugging Face API Token)

The `HF_TOKEN` secret is required for syncing to Hugging Face Spaces.

#### Step 1: Create a Hugging Face Token

1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Click **"Create new token"**
3. Configure the token:
   - **Name**: `github-actions-sync` (or similar)
   - **Type**: "Write" (required for pushing to Spaces)
   - **Permissions**: 
     - ✅ Write access to contents
     - ✅ Manage Spaces
4. Click **"Generate token"**
5. **Copy the token** (you won't see it again!)

#### Step 2: Add Secret to GitHub Repository

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **"New repository secret"**
4. Configure:
   - **Name**: `HF_TOKEN`
   - **Secret**: Paste your Hugging Face token
5. Click **"Add secret"**

---

## Workflows

### 1. Sync to Hugging Face Spaces

**File**: `.github/workflows/sync-to-hf.yml`

**Triggers**:
- Push to `main` branch
- Manual workflow dispatch

**What it does**:
1. Checks for relevant code changes (excludes docs/tests)
2. Uploads files to Hugging Face Space
3. Verifies deployment status

**Files excluded from sync**:
- `.git/`, `.github/`
- `tests/`, `docs/`
- `*.md`, `.gitkeep`
- `pytest.ini`, `spaces_app_config.json`

### 2. Run Tests

**File**: `.github/workflows/run-tests.yml`

**Triggers**:
- Push to `main` or `develop`
- Pull requests to `main`

**What it does**:
1. Sets up Python environment
2. Installs dependencies
3. Runs pytest with coverage
4. Uploads coverage report

---

## Manual Deployment

You can manually trigger a sync to Hugging Face Spaces:

1. Go to **Actions** tab in GitHub
2. Select **"Sync to Hugging Face Spaces"**
3. Click **"Run workflow"**
4. Optionally enable **"Force full sync"**
5. Click **"Run workflow"**

---

## Troubleshooting

### "HF_TOKEN not found in environment"

**Solution**: Ensure you've added the `HF_TOKEN` secret to your repository settings.

### "Repository not found" or "403 Forbidden"

**Solution**: 
1. Verify your HF_TOKEN has **write** permissions
2. Ensure the token is not expired
3. Check if the Space name matches your configuration

### Space not updating after push

**Solutions**:
1. Check the Actions tab for workflow status
2. Verify the files you changed aren't in the exclude list
3. Try manual workflow dispatch with "Force full sync"

### Tests failing in CI but passing locally

**Solutions**:
1. Ensure `GARDEN_DOCTOR_MOCK=true` environment variable is set
2. Check for missing dependencies in the workflow
3. Verify Python version matches

---

## Configuration

### Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `HF_USERNAME` | `insydr` | Hugging Face username |
| `HF_SPACE_NAME` | `garden-doctor` | Space name |
| `HF_SPACE_REPO` | `insydr/garden-doctor` | Full repository ID |
| `GARDEN_DOCTOR_MOCK` | `true` | Mock mode for tests |
| `GARDEN_DOCTOR_TIMEOUT` | `10` | Reduced timeout for CI |

### Permissions Required

The workflow requires these permissions:
```yaml
permissions:
  contents: read  # Read repository contents
```

---

## Monitoring

### Check Workflow Status

1. Go to **Actions** tab in GitHub
2. Click on the workflow run
3. View logs and step summary

### Space Status

After successful sync, check:
- **Space URL**: https://huggingface.co/spaces/insydr/garden-doctor
- **Build logs**: Available in Hugging Face Space settings

---

## Security Best Practices

1. ✅ Use repository secrets (not environment variables in workflow)
2. ✅ Use minimal required permissions
3. ✅ Use write tokens only when needed
4. ✅ Rotate tokens periodically
5. ✅ Never commit tokens to the repository

---

## Need Help?

- **GitHub Actions Documentation**: https://docs.github.com/en/actions
- **Hugging Face Hub Documentation**: https://huggingface.co/docs/hub
- **Create an Issue**: Report problems in the repository Issues tab
