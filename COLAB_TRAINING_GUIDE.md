# 🚀 Google Colab Training Guide - Vietnamese AMR

Complete guide for training Vietnamese AMR models efficiently on Google Colab.

## 🎯 Quick Setup (5 minutes)

### Step 1: Get Better GPU Runtime

```python
# Check current GPU
!nvidia-smi

# If you see Tesla T4 (15GB) or V100 (16GB) - GOOD!
# If you see K80 (12GB) - try to get better runtime:
# Runtime > Change runtime type > Hardware accelerator > GPU > High-RAM
```

### Step 2: Clone and Setup

```python
# Clone repository
!git clone https://github.com/your-username/nlp-semantic-parsing.git
%cd nlp-semantic-parsing

# Install dependencies (optimized for Colab)
!pip install -q torch transformers datasets tokenizers
!pip install -q pandas numpy scikit-learn PyYAML tqdm
!pip install -q nltk matplotlib seaborn jsonlines
!pip install -q gradio huggingface_hub

# Download NLTK data
import nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
```

### Step 3: Upload Your Data (if needed)

```python
# Option 1: Upload files directly
from google.colab import files
uploaded = files.upload()

# Option 2: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy data from Drive
!cp -r "/content/drive/MyDrive/amr_data/*" data/train/
```

## ⚡ Fast Training Configuration

### Create Colab-Optimized Config

```python
# Create fast training config
colab_config = """
# Colab Optimized Configuration
model:
  name: "VietAI/vit5-base"
  max_input_length: 256      # Reduced from 512
  max_output_length: 256     # Reduced from 512

training:
  batch_size: 8              # Optimal for T4 GPU
  gradient_accumulation_steps: 4  # Effective batch size = 32
  learning_rate: 3e-4        # Higher LR for faster convergence
  num_epochs: 3              # Fewer epochs
  warmup_steps: 100          # Quick warmup
  save_steps: 500
  eval_steps: 500
  logging_steps: 50
  
  # Memory optimization
  fp16: true                 # Mixed precision training
  dataloader_num_workers: 2  # Parallel data loading
  remove_unused_columns: true
  
  # Early stopping
  early_stopping_patience: 2
  early_stopping_threshold: 0.01

data:
  train_ratio: 0.85          # More training data
  val_ratio: 0.15
  test_ratio: 0.0            # Skip test split for faster training
  max_samples: 1000          # Limit samples for quick training
  shuffle_data: true
  random_seed: 42
  max_input_length: 256
  max_output_length: 256
  min_input_length: 3
  min_output_length: 3

paths:
  model_save_dir: "/content/models"
  log_dir: "/content/logs"
  data_dir: "/content/nlp-semantic-parsing/data"
"""

# Save config
with open('config/colab_fast_config.yaml', 'w') as f:
    f.write(colab_config)

print("✅ Colab config created!")
```

## 🚀 Training Commands

### Quick Training (15-20 minutes)

```python
# Process data first
!python main.py process-data --input-dir data/train --output-dir data/processed --split-data

# Train with fast config
!python main.py train --config config/colab_fast_config.yaml
```

### Monitor Training Progress

```python
# Real-time monitoring
import time
import os
from IPython.display import clear_output

def monitor_training():
    log_file = "/content/logs/colab_fast/amr_project_latest.log"
    
    while True:
        clear_output(wait=True)
        
        # Show GPU usage
        !nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
        
        # Show latest logs
        if os.path.exists(log_file):
            !tail -20 {log_file}
        
        time.sleep(10)

# Run in background
# monitor_training()  # Uncomment to use
```

## 💾 Memory Optimization Tips

### 1. Reduce Model Size

```python
# Use smaller model variant
config_small = """
model:
  name: "VietAI/vit5-small"  # Smaller than vit5-base
  max_input_length: 128      # Even smaller sequences
  max_output_length: 128
"""
```

### 2. Gradient Checkpointing

```python
# Add to training config
training_args = """
training:
  gradient_checkpointing: true  # Trade compute for memory
  fp16: true                    # Mixed precision
  batch_size: 4                 # Smaller batch if needed
  gradient_accumulation_steps: 8 # Maintain effective batch size
"""
```

### 3. Clear Memory Between Runs

```python
# Clear GPU memory
import torch
import gc

def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()
    print("🧹 Memory cleared!")

clear_memory()
```

## 📊 Training Strategies

### Strategy 1: Quick Prototype (10 minutes)

```yaml
# Ultra-fast config for testing
data:
  max_samples: 100           # Very small dataset
training:
  num_epochs: 1
  batch_size: 16
  learning_rate: 5e-4
```

### Strategy 2: Balanced Training (30 minutes)

```yaml
# Good balance of speed and quality
data:
  max_samples: 500
training:
  num_epochs: 3
  batch_size: 8
  learning_rate: 3e-4
  fp16: true
```

### Strategy 3: Full Training (1-2 hours)

```yaml
# Full dataset training
data:
  max_samples: null          # Use all data
training:
  num_epochs: 5
  batch_size: 4              # Smaller batch for full data
  gradient_accumulation_steps: 8
  fp16: true
```

## 🔧 Troubleshooting

### Out of Memory Errors

```python
# Reduce batch size
training:
  batch_size: 2              # Minimum viable batch size
  gradient_accumulation_steps: 16  # Maintain effective batch size

# Or reduce sequence length
model:
  max_input_length: 128
  max_output_length: 128
```

### Slow Training

```python
# Check GPU utilization
!nvidia-smi

# Increase batch size if GPU not fully utilized
training:
  batch_size: 16             # If you have headroom
  dataloader_num_workers: 4  # More parallel loading
```

### Connection Timeout

```python
# Keep Colab alive
import time
from IPython.display import Javascript

def keep_alive():
    display(Javascript('''
    function ClickConnect(){
        console.log("Working");
        document.querySelector("colab-toolbar-button#connect").click()
    }
    setInterval(ClickConnect,60000)
    '''))

keep_alive()
```

## 📈 Performance Monitoring

### Training Metrics Dashboard

```python
import matplotlib.pyplot as plt
import json

def plot_training_metrics():
    # Load training metrics
    with open('/content/models/training_metrics.json', 'r') as f:
        metrics = json.load(f)
    
    # Plot loss curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['eval_loss'], label='Validation Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics['learning_rate'])
    plt.title('Learning Rate')
    
    plt.tight_layout()
    plt.show()

# Call after training
# plot_training_metrics()
```

## 💾 Save and Download Model

### Save to Google Drive

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy trained model
!cp -r /content/models "/content/drive/MyDrive/amr_models/"
print("✅ Model saved to Google Drive!")
```

### Download Directly

```python
# Zip and download
!zip -r trained_model.zip /content/models/
from google.colab import files
files.download('trained_model.zip')
```

## 🚀 Quick Start Script

```python
# Complete training script for Colab
def quick_train():
    print("🚀 Starting Quick AMR Training on Colab")
    
    # 1. Setup
    !pip install -q transformers datasets torch tqdm PyYAML jsonlines
    
    # 2. Process data
    !python main.py process-data --input-dir data/train --output-dir data/processed --split-data
    
    # 3. Train
    !python main.py train --config config/colab_fast_config.yaml
    
    # 4. Test prediction
    !python main.py predict --model-path /content/models --text "Tôi yêu Việt Nam"
    
    print("✅ Training completed!")

# Run everything
quick_train()
```

## 🎯 Expected Performance

| Configuration | Time | GPU Memory | Quality |
|---------------|------|------------|---------|
| Quick Prototype | 10 min | 4GB | Basic |
| Balanced | 30 min | 8GB | Good |
| Full Training | 1-2 hours | 12GB | Best |

## 💡 Pro Tips

1. **Use T4 or V100**: Much faster than K80
2. **Enable Mixed Precision**: `fp16: true` for 2x speedup
3. **Batch Size**: Start with 8, adjust based on memory
4. **Gradient Accumulation**: Maintain effective batch size of 32
5. **Early Stopping**: Prevent overfitting and save time
6. **Monitor GPU**: Keep utilization >80%
7. **Save Frequently**: Colab can disconnect anytime

## 🔗 Quick Links

- **Colab Notebook**: [Open in Colab](https://colab.research.google.com/github/your-repo/AMR_Universal_Notebook.ipynb)
- **Model Hub**: Upload to [Hugging Face](https://huggingface.co)
- **Gradio Demo**: Test your model instantly

---

**Happy Training! 🎉**
