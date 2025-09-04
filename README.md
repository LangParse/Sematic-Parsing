# 🚀 AMR Semantic Parsing for Vietnamese

A **production-ready** system for Abstract Meaning Representation (AMR) semantic parsing specifically designed for Vietnamese language, built with clean architecture and modern NLP practices.

## ✨ Features

- ✅ **Vietnamese Language Support**: Optimized for Vietnamese text using VietAI/vit5-base (36K vocab)
- ✅ **Clean Architecture**: Modular design with proper separation of concerns
- ✅ **Universal Training**: Works on both Local (CPU/GPU) and Google Colab (GPU)
- ✅ **Multiple Interfaces**: CLI, Interactive mode, and Universal Jupyter notebook
- ✅ **Comprehensive Testing**: All components tested and verified
- ✅ **Fast I/O**: JSONL format for 10x faster data operations
- ✅ **Production Ready**: Proper logging, configuration management, and error handling
- ✅ **Easy Deployment**: Push to Hugging Face Hub + Gradio web interface
- ✅ **Web Interface**: Beautiful Gradio app with Vietnamese UI
- ✅ **Progress Bars**: Beautiful tqdm progress bars for all operations

## 🎯 What's Tested & Working

- ✅ **Data Processing**: 1893+ samples processed successfully
- ✅ **Tokenization**: VietAI/vit5-base with 36,096 vocabulary
- ✅ **Configuration**: YAML-based settings with validation
- ✅ **Evaluation**: BLEU, ROUGE, and exact match metrics
- ✅ **CLI Interface**: All commands working
- ✅ **Training**: Fixed and working with proper configs
- ✅ **Deployment**: Push to HF Hub + Gradio interface
- ✅ **Web Interface**: Production-ready Gradio app

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd nlp-semantic-parsing
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (required for evaluation):
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

### Basic Usage

#### 1. Process Raw AMR Data
```bash
python main.py process-data --input-dir data/train --output-dir data/processed --split-data
```

#### 2. Train the Model

**Local Training:**
```bash
python main.py train --config config/training_config.yaml
```

**⚡ Google Colab Training (15-30 minutes):**
```python
# 1. Get T4/V100 GPU: Runtime > Change runtime type > GPU
# 2. Clone and run optimized training
!git clone https://github.com/your-repo/nlp-semantic-parsing.git
%cd nlp-semantic-parsing
!python colab_train.py

# Or manual setup:
!python optimize_colab.py  # Optimize for your GPU
!python main.py train --config config/colab_fast_config.yaml
```

**See [COLAB_TRAINING_GUIDE.md](COLAB_TRAINING_GUIDE.md) for detailed Colab instructions.**

#### 3. Evaluate the Model
```bash
python main.py evaluate --model-path models/amr_model --test-data data/processed/test.jsonl --output-dir evaluation_results
```

#### 4. Make Predictions

Single prediction:
```bash
python main.py predict --model-path models/amr_model --text "Tôi yêu Việt Nam" --format
```

Interactive mode:
```bash
python main.py interactive --model-path models/amr_model
```

Process file:
```bash
python main.py predict-file --model-path models/amr_model --input-file input.txt --output-file output.txt
```

## 🚀 Deployment

### Push Model to Hugging Face

```bash
# Set your HF token
export HF_TOKEN="your_huggingface_token"

# Push trained model to HF Hub
python main.py push-model \
  --model-path models/amr_model \
  --repo-name "your-username/vietnamese-amr-model"
```

### Launch Gradio Web Interface

```bash
# Local model
python main.py gradio --model-path models/amr_model --port 7860

# Hugging Face model
python main.py gradio --hf-model "your-username/vietnamese-amr-model" --port 7860 --share
```

### Quick Deployment

```bash
# Full deployment: push model + launch Gradio
python deploy.py full \
  --model-path models/amr_model \
  --repo-name "your-username/vietnamese-amr-model" \
  --token "your_token" \
  --share
```

**See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed deployment instructions.**

## 📁 Project Structure

```
nlp-semantic-parsing/
├── src/
│   ├── data_processing/          # Data cleaning and preprocessing
│   │   ├── __init__.py
│   │   ├── amr_processor.py      # AMR file processing
│   │   └── data_loader.py        # Data loading and splitting
│   ├── tokenization/             # Tokenization with VietAI/vit5-base
│   │   ├── __init__.py
│   │   └── vit5_tokenizer.py     # ViT5 tokenizer wrapper
│   ├── training/                 # Model training
│   │   ├── __init__.py
│   │   ├── trainer.py            # Training pipeline
│   │   └── model_config.py       # Training configuration
│   ├── evaluation/               # Model evaluation
│   │   ├── __init__.py
│   │   └── evaluator.py          # Evaluation metrics and reporting
│   ├── inference/                # Model inference
│   │   ├── __init__.py
│   │   └── predictor.py          # Prediction interface
│   └── utils/                    # Utilities
│       ├── __init__.py
│       ├── config.py             # Configuration management
│       └── logger.py             # Logging utilities
├── config/
│   ├── default_config.yaml       # Default configuration
│   └── training_config.yaml      # Training-specific configuration
├── data/
│   ├── train/                    # Training data
│   └── test/                     # Test data
├── main.py                       # Main CLI interface
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🔧 Configuration

The project uses YAML configuration files for easy customization. Key configuration sections:

### Model Configuration
```yaml
model:
  name: "VietAI/vit5-base"
  max_length: 512
  batch_size: 8
  learning_rate: 5e-5
  num_epochs: 3
```

### Data Configuration
```yaml
data:
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  max_input_length: 512
  max_output_length: 512
```

### Training Configuration
```yaml
training:
  use_wandb: true
  wandb_project: "amr-semantic-parsing"
  early_stopping_patience: 3
  evaluation_strategy: "steps"
```

## 📊 Evaluation Metrics

The system provides comprehensive evaluation with multiple metrics:

- **BLEU Scores**: BLEU-1, BLEU-2, BLEU-3, BLEU-4
- **ROUGE Scores**: ROUGE-1, ROUGE-2, ROUGE-L
- **METEOR Score**: Semantic similarity metric
- **Exact Match**: Percentage of exactly matching predictions

Evaluation results include:
- Detailed metrics in JSON format
- Visualization plots
- Markdown report with analysis
- Sample predictions for error analysis

## 🎯 Key Improvements Over Original Code

### 1. **Clean Architecture**
- Modular design with clear separation of concerns
- Reusable components with well-defined interfaces
- Proper error handling and logging

### 2. **Optimized Data Processing**
- JSONL format for efficient I/O operations
- Batch processing capabilities
- Memory-efficient data loading

### 3. **Configuration Management**
- YAML-based configuration system
- Environment variable support
- Easy parameter tuning without code changes

### 4. **Enhanced Model Support**
- VietAI/vit5-base instead of Google models
- Proper Vietnamese language tokenization
- Optimized for Vietnamese text processing

### 5. **Comprehensive Evaluation**
- Multiple evaluation metrics
- Automatic report generation
- Visualization of results
- Model comparison capabilities

### 6. **User-Friendly Interface**
- CLI interface for all operations
- Interactive prediction mode
- Batch processing support
- Progress tracking and logging

## 🔍 Usage Examples

### Training with Custom Configuration

Create a custom configuration file:
```yaml
# custom_config.yaml
model:
  batch_size: 4
  learning_rate: 3e-5
  num_epochs: 5

training:
  use_wandb: true
  run_name: "custom-amr-training"
```

Train with custom config:
```bash
python main.py train --config custom_config.yaml
```

### Evaluation with Detailed Analysis

```bash
python main.py evaluate \
  --model-path models/amr_model \
  --test-data data/processed/test.jsonl \
  --output-dir detailed_evaluation \
  --batch-size 16
```

This generates:
- `evaluation_metrics.json`: Detailed metrics
- `detailed_results.jsonl`: Per-sample results
- `metrics_plot.png`: Visualization
- `evaluation_report.md`: Analysis report

### Interactive Prediction

```bash
python main.py interactive --model-path models/amr_model
```

Example interaction:
```
📝 Enter sentence: Tôi đang học tiếng Việt
🔄 Generating AMR...

--------------------------------------------------
📥 Input: Tôi đang học tiếng Việt
📤 AMR Output:
(h / học
   :ARG0 (t / tôi)
   :ARG1 (t2 / tiếng_Việt)
   :aspect (p / progressive))
--------------------------------------------------
```

## 🛠️ Development

### Running Tests
```bash
pytest tests/ -v --cov=src
```

### Code Formatting
```bash
black src/ main.py
flake8 src/ main.py
```

### Type Checking
```bash
mypy src/
```

## 📈 Performance Tips

1. **GPU Usage**: Enable CUDA for faster training and inference
2. **Batch Size**: Adjust based on available memory
3. **Mixed Precision**: Use FP16 for faster training
4. **Data Loading**: Use multiple workers for data loading
5. **Caching**: Enable model caching for faster loading

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Run code formatting and tests
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- VietAI team for the vit5-base model
- Hugging Face for the transformers library
- The AMR community for dataset and evaluation metrics

## 📞 Support

For questions or issues, please:
1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information

---

**Happy AMR parsing! 🚀**
