#!/usr/bin/env python3
"""
Colab Training Script
====================

Optimized training script for Google Colab with progress monitoring.
"""

import os
import sys
import time
import subprocess
import json
from pathlib import Path

def check_gpu():
    """Check GPU availability and type."""
    print("🔍 Checking GPU availability...")
    
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            print(f"✅ GPU detected: {gpu_info}")
            
            # Check if it's a good GPU
            if any(gpu in gpu_info.lower() for gpu in ['t4', 'v100', 'a100']):
                print("🚀 Great GPU for training!")
            elif 'k80' in gpu_info.lower():
                print("⚠️  K80 detected - training will be slower. Consider getting T4/V100.")
            
            return True
        else:
            print("❌ No GPU detected!")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found - no GPU available")
        return False

def install_dependencies():
    """Install required packages for Colab."""
    print("📦 Installing dependencies...")
    
    packages = [
        "torch",
        "transformers", 
        "datasets",
        "tokenizers",
        "pandas",
        "numpy", 
        "scikit-learn",
        "PyYAML",
        "tqdm",
        "nltk",
        "matplotlib",
        "seaborn",
        "jsonlines",
        "gradio",
        "huggingface_hub"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)
    
    # Download NLTK data
    print("📚 Downloading NLTK data...")
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    
    print("✅ All dependencies installed!")

def setup_colab_environment():
    """Setup Colab environment."""
    print("🔧 Setting up Colab environment...")
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Set environment variables for better performance
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid warnings
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"        # Async CUDA
    
    print("✅ Environment setup completed!")

def monitor_training_progress(log_file, max_wait_minutes=60):
    """Monitor training progress with real-time updates."""
    print(f"📊 Monitoring training progress (max {max_wait_minutes} minutes)...")
    
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    
    while True:
        elapsed = time.time() - start_time
        
        if elapsed > max_wait_seconds:
            print(f"⏰ Training timeout after {max_wait_minutes} minutes")
            break
        
        # Check if log file exists and show progress
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    
                # Look for training completion
                for line in reversed(lines[-20:]):  # Check last 20 lines
                    if "Training completed successfully" in line:
                        print("🎉 Training completed successfully!")
                        return True
                    elif "Error" in line or "Failed" in line:
                        print(f"❌ Training error detected: {line.strip()}")
                        return False
                
                # Show latest progress
                progress_lines = [line.strip() for line in lines[-5:] if line.strip()]
                if progress_lines:
                    latest = progress_lines[-1]
                    if "epoch" in latest.lower() or "step" in latest.lower():
                        print(f"📈 Progress: {latest}")
                        
            except Exception as e:
                print(f"⚠️  Error reading log: {e}")
        
        time.sleep(10)  # Check every 10 seconds
    
    return False

def run_fast_training():
    """Run fast training optimized for Colab."""
    print("🚀 Starting fast AMR training...")
    
    # Step 1: Process data
    print("\n📊 Step 1: Processing data...")
    cmd = [sys.executable, "main.py", "process-data", 
           "--input-dir", "data/train", 
           "--output-dir", "data/processed", 
           "--split-data"]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Data processing failed: {result.stderr}")
        return False
    
    print("✅ Data processing completed!")
    
    # Step 2: Train model
    print("\n🎯 Step 2: Training model...")
    cmd = [sys.executable, "main.py", "train", 
           "--config", "config/colab_fast_config.yaml"]
    
    # Start training in background
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Monitor progress
    log_file = "logs/colab_fast/amr_project_latest.log"
    
    # Wait for log file to be created
    for _ in range(30):  # Wait up to 30 seconds
        if os.path.exists(log_file):
            break
        time.sleep(1)
    
    # Monitor training
    success = monitor_training_progress(log_file, max_wait_minutes=45)
    
    # Wait for process to complete
    process.wait()
    
    if success and process.returncode == 0:
        print("✅ Training completed successfully!")
        return True
    else:
        print(f"❌ Training failed with return code: {process.returncode}")
        return False

def test_trained_model():
    """Test the trained model."""
    print("\n🧪 Testing trained model...")
    
    test_sentences = [
        "Tôi yêu Việt Nam",
        "Cô ấy đang học tiếng Anh", 
        "Hôm nay trời đẹp"
    ]
    
    model_path = "models/colab_fast_model"
    
    for sentence in test_sentences:
        print(f"\n📝 Input: {sentence}")
        
        cmd = [sys.executable, "main.py", "predict", 
               "--model-path", model_path, 
               "--text", sentence]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"🔮 AMR: {result.stdout.strip()}")
        else:
            print(f"❌ Prediction failed: {result.stderr}")

def show_training_summary():
    """Show training summary and next steps."""
    print("\n" + "="*60)
    print("🎊 TRAINING COMPLETED!")
    print("="*60)
    
    # Check if model exists
    model_path = Path("models/colab_fast_model")
    if model_path.exists():
        print(f"✅ Model saved at: {model_path}")
        
        # Show model files
        model_files = list(model_path.glob("*"))
        print(f"📁 Model files ({len(model_files)}):")
        for file in model_files[:5]:  # Show first 5 files
            print(f"   - {file.name}")
        if len(model_files) > 5:
            print(f"   ... and {len(model_files) - 5} more files")
    
    # Show logs
    log_dir = Path("logs/colab_fast")
    if log_dir.exists():
        log_files = list(log_dir.glob("*.log"))
        if log_files:
            print(f"📋 Training logs: {log_files[0]}")
    
    print("\n🚀 Next Steps:")
    print("1. Test predictions: python main.py predict --model-path models/colab_fast_model --text 'Your text'")
    print("2. Launch Gradio: python main.py gradio --model-path models/colab_fast_model")
    print("3. Push to HF Hub: python main.py push-model --model-path models/colab_fast_model --repo-name your-username/model-name")
    
    print("\n💾 Save your model:")
    print("- Download: Zip the models/ folder")
    print("- Google Drive: Copy to /content/drive/MyDrive/")
    print("- Hugging Face: Use push-model command")

def main():
    """Main training function for Colab."""
    print("🚀 Vietnamese AMR Training - Colab Optimized")
    print("=" * 60)
    
    try:
        # Step 1: Check GPU
        if not check_gpu():
            print("⚠️  No GPU detected. Training will be very slow!")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                return
        
        # Step 2: Install dependencies
        install_dependencies()
        
        # Step 3: Setup environment
        setup_colab_environment()
        
        # Step 4: Run training
        success = run_fast_training()
        
        if success:
            # Step 5: Test model
            test_trained_model()
            
            # Step 6: Show summary
            show_training_summary()
        else:
            print("❌ Training failed. Check logs for details.")
            
    except KeyboardInterrupt:
        print("\n👋 Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
