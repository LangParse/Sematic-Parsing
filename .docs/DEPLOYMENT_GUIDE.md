# 🚀 Deployment Guide - Vietnamese AMR Semantic Parsing

Complete guide for deploying your Vietnamese AMR model to Hugging Face and creating web interfaces.

## 📋 Prerequisites

1. **Trained Model**: You need a trained AMR model (e.g., `models/minimal_model/`)
2. **Hugging Face Account**: Create account at [huggingface.co](https://huggingface.co)
3. **HF Token**: Get your token from [Settings > Access Tokens](https://huggingface.co/settings/tokens)

## 🔧 Installation

Install deployment dependencies:

```bash
# Activate environment
source .venv/bin/activate

# Install deployment packages
pip install gradio huggingface_hub

# Or use the deployment script
python deploy.py install
```

## 🤗 Push Model to Hugging Face

### Method 1: Using CLI

```bash
# Set your HF token (optional - can use --token instead)
export HF_TOKEN="your_huggingface_token_here"

# Push model to HF Hub
python main.py push-model \
  --model-path models/minimal_model \
  --repo-name "your-username/vietnamese-amr-model" \
  --private  # Optional: make repository private
```

### Method 2: Using Deploy Script

```bash
# Push model with deploy script
python deploy.py push \
  --model-path models/minimal_model \
  --repo-name "your-username/vietnamese-amr-model" \
  --token "your_token_here" \
  --private
```

### What Gets Uploaded:

- ✅ **Model files** (pytorch_model.bin, config.json)
- ✅ **Tokenizer files** (tokenizer.json, special_tokens_map.json)
- ✅ **Model card** (README.md with usage instructions)
- ✅ **Metadata** (model tags, language info)

## 🌐 Gradio Web Interface

### Local Gradio App

#### Option 1: Mock Model (for testing)
```bash
# Run with mock model (no real model needed)
python gradio_app.py --port 7860

# Or via CLI
python main.py gradio --port 7860
```

#### Option 2: Local Model
```bash
# Run with your trained local model
python gradio_app.py --model-path models/minimal_model --port 7860

# Or via CLI
python main.py gradio --model-path models/minimal_model --port 7860
```

#### Option 3: Hugging Face Model
```bash
# Run with model from HF Hub
python gradio_app.py --hf-model "your-username/vietnamese-amr-model" --port 7860

# Or via CLI
python main.py gradio --hf-model "your-username/vietnamese-amr-model" --port 7860
```

#### Create Public Link
```bash
# Create public shareable link
python gradio_app.py --share --port 7860
```

### Gradio Interface Features

- 🇻🇳 **Vietnamese Interface**: Optimized for Vietnamese text
- 📝 **Text Input**: Large text area for Vietnamese sentences
- 🎯 **AMR Output**: Formatted AMR representation
- 🔄 **Clear/Submit**: Easy-to-use buttons
- 📚 **Examples**: Pre-loaded Vietnamese examples
- 🎨 **Custom Styling**: Clean, professional design

## 🏗️ Hugging Face Spaces Deployment

### Step 1: Create Space Files

```bash
# Generate HF Space files
python deploy.py space --repo-name "your-username/vietnamese-amr-model"
```

This creates:
- `app.py` - Main Gradio app for HF Spaces
- `requirements_space.txt` - Dependencies for Spaces
- `README_space.md` - Space description

### Step 2: Create Hugging Face Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose **Gradio** as SDK
3. Name your space (e.g., `vietnamese-amr-parser`)
4. Upload the generated files:
   - `app.py` → root directory
   - `requirements_space.txt` → rename to `requirements.txt`
   - `README_space.md` → rename to `README.md`
   - `gradio_app.py` → upload as is

### Step 3: Configure Space

Your Space will be available at:
`https://huggingface.co/spaces/your-username/your-space-name`

## 🚀 Full Deployment Workflow

### Complete deployment in one command:

```bash
python deploy.py full \
  --model-path models/minimal_model \
  --repo-name "your-username/vietnamese-amr-model" \
  --token "your_token_here" \
  --port 7860 \
  --share
```

This will:
1. ✅ Install requirements
2. ✅ Push model to HF Hub
3. ✅ Create Space files
4. ✅ Launch local Gradio app

## 📊 Usage Examples

### 1. Test with Vietnamese Sentences

```
Input: "Tôi yêu Việt Nam"
Output: (y / yêu 
   :ARG0 (t / tôi) 
   :ARG1 (v / Việt_Nam))

Input: "Cô ấy đang học tiếng Anh"
Output: (h / học 
   :ARG0 (c / cô_ấy) 
   :ARG1 (t / tiếng_Anh) 
   :aspect (p / progressive))
```

### 2. API Usage (after HF deployment)

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load your deployed model
tokenizer = AutoTokenizer.from_pretrained("your-username/vietnamese-amr-model")
model = AutoModelForSeq2SeqLM.from_pretrained("your-username/vietnamese-amr-model")

# Predict AMR
text = "Tôi yêu Việt Nam"
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(**inputs, max_length=512, num_beams=4)
amr = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(amr)
```

## 🔧 Configuration Options

### Gradio App Options

```bash
python gradio_app.py \
  --model-path models/your_model \     # Local model path
  --hf-model "username/model-name" \   # HF model name
  --port 7860 \                        # Port number
  --host 0.0.0.0 \                     # Host (for server deployment)
  --share \                            # Create public link
  --debug                              # Enable debug mode
```

### Push Model Options

```bash
python main.py push-model \
  --model-path models/your_model \     # Local model path
  --repo-name "username/model-name" \  # HF repository name
  --token "your_token" \               # HF token
  --private \                          # Make repository private
  --commit-message "Upload AMR model"  # Custom commit message
```

## 🎯 Production Deployment

### For Production Servers:

```bash
# Run on all interfaces
python gradio_app.py --host 0.0.0.0 --port 8080

# With environment variables
export HF_MODEL="your-username/vietnamese-amr-model"
export PORT=8080
python gradio_app.py --hf-model $HF_MODEL --port $PORT --host 0.0.0.0
```

### Docker Deployment:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt
EXPOSE 7860

CMD ["python", "gradio_app.py", "--hf-model", "your-username/vietnamese-amr-model", "--host", "0.0.0.0", "--port", "7860"]
```

## 🔍 Troubleshooting

### Common Issues:

1. **Model not found**: Ensure model path exists or HF model name is correct
2. **Token issues**: Check HF token permissions and validity
3. **Memory issues**: Use smaller batch sizes or reduce model size
4. **Port conflicts**: Change port number if 7860 is occupied

### Debug Mode:

```bash
# Enable debug logging
python gradio_app.py --debug

# Check model loading
python -c "
from gradio_app import AMRGradioApp
app = AMRGradioApp(hf_model_name='your-username/vietnamese-amr-model')
print('Model loaded successfully!')
"
```

## 🎉 Success!

After deployment, you'll have:

- ✅ **HF Model Repository**: `https://huggingface.co/your-username/vietnamese-amr-model`
- ✅ **Local Gradio App**: `http://localhost:7860`
- ✅ **Public Gradio Link**: `https://xxxxx.gradio.live` (if using --share)
- ✅ **HF Space**: `https://huggingface.co/spaces/your-username/your-space-name`

Your Vietnamese AMR model is now accessible worldwide! 🌍
