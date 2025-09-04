#!/usr/bin/env python3
"""
Colab Optimization Script
========================

Optimize Google Colab environment for faster AMR training.
"""

import os
import sys
import subprocess
import psutil
import json
from pathlib import Path

def check_system_resources():
    """Check available system resources."""
    print("🔍 Checking system resources...")
    
    # CPU info
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    print(f"💻 CPU: {cpu_count} cores @ {cpu_freq.current:.1f}MHz")
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"🧠 RAM: {memory.total / (1024**3):.1f}GB total, {memory.available / (1024**3):.1f}GB available")
    
    # Disk info
    disk = psutil.disk_usage('/')
    print(f"💾 Disk: {disk.total / (1024**3):.1f}GB total, {disk.free / (1024**3):.1f}GB free")
    
    # GPU info
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            print(f"🎮 GPU: {gpu_info}")
        else:
            print("❌ No GPU detected")
    except FileNotFoundError:
        print("❌ nvidia-smi not found")

def optimize_pytorch():
    """Optimize PyTorch settings for Colab."""
    print("⚡ Optimizing PyTorch settings...")
    
    # Set environment variables for better performance
    optimizations = {
        "CUDA_LAUNCH_BLOCKING": "0",           # Async CUDA operations
        "TOKENIZERS_PARALLELISM": "false",    # Avoid tokenizer warnings
        "OMP_NUM_THREADS": "2",               # Limit OpenMP threads
        "MKL_NUM_THREADS": "2",               # Limit MKL threads
        "NUMEXPR_NUM_THREADS": "2",           # Limit NumExpr threads
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"  # Better memory management
    }
    
    for key, value in optimizations.items():
        os.environ[key] = value
        print(f"  {key} = {value}")
    
    print("✅ PyTorch optimizations applied!")

def create_optimized_configs():
    """Create optimized training configurations."""
    print("⚙️  Creating optimized configurations...")
    
    # Check available GPU memory
    gpu_memory_gb = 15  # Default T4 memory
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_memory_mb = int(result.stdout.strip())
            gpu_memory_gb = gpu_memory_mb / 1024
    except:
        pass
    
    print(f"🎮 GPU Memory: {gpu_memory_gb:.1f}GB")
    
    # Determine optimal batch size based on GPU memory
    if gpu_memory_gb >= 24:      # A100/V100 32GB
        batch_size = 16
        max_length = 512
    elif gpu_memory_gb >= 15:    # T4/V100 16GB
        batch_size = 8
        max_length = 256
    elif gpu_memory_gb >= 11:    # RTX 2080 Ti
        batch_size = 4
        max_length = 256
    else:                        # K80 or smaller
        batch_size = 2
        max_length = 128
    
    # Create memory-optimized config
    config_content = f"""# Memory-Optimized Colab Configuration
# Auto-generated based on {gpu_memory_gb:.1f}GB GPU memory

model:
  name: "VietAI/vit5-base"
  max_input_length: {max_length}
  max_output_length: {max_length}
  early_stopping_patience: 2
  early_stopping_threshold: 0.01

training:
  # Optimized batch configuration
  batch_size: {batch_size}
  gradient_accumulation_steps: {32 // batch_size}  # Effective batch size = 32
  eval_batch_size: {batch_size * 2}
  
  # Learning configuration
  learning_rate: 3e-4
  num_epochs: 3
  warmup_steps: 100
  weight_decay: 0.01
  
  # Memory optimization
  fp16: true
  gradient_checkpointing: {"true" if gpu_memory_gb < 12 else "false"}
  dataloader_num_workers: 2
  remove_unused_columns: true
  
  # Logging and saving
  save_steps: 200
  eval_steps: 200
  logging_steps: 25
  save_total_limit: 2
  
  # Optimization
  adam_epsilon: 1e-8
  max_grad_norm: 1.0
  
  # Early stopping
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"
  greater_is_better: false

data:
  train_ratio: 0.85
  val_ratio: 0.15
  test_ratio: 0.0
  max_samples: {800 if gpu_memory_gb >= 15 else 400}
  shuffle_data: true
  random_seed: 42
  max_input_length: {max_length}
  max_output_length: {max_length}
  min_input_length: 3
  min_output_length: 3
  train_data_path: "data/train"
  test_data_path: "data/test"
  processed_data_dir: "data/processed"

paths:
  model_save_dir: "models/optimized_model"
  log_dir: "logs/optimized"
  data_dir: "data"
  
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file_prefix: "optimized_training"
"""
    
    # Save optimized config
    config_path = Path("config/colab_optimized_config.yaml")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Optimized config saved: {config_path}")
    print(f"   Batch size: {batch_size}")
    print(f"   Max length: {max_length}")
    print(f"   Gradient checkpointing: {'ON' if gpu_memory_gb < 12 else 'OFF'}")

def clear_cache():
    """Clear various caches to free memory."""
    print("🧹 Clearing caches...")
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("  ✅ CUDA cache cleared")
    except ImportError:
        pass
    
    try:
        import gc
        gc.collect()
        print("  ✅ Python garbage collected")
    except:
        pass
    
    # Clear pip cache
    try:
        subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], 
                      capture_output=True, check=False)
        print("  ✅ Pip cache cleared")
    except:
        pass

def setup_monitoring():
    """Setup resource monitoring."""
    print("📊 Setting up resource monitoring...")
    
    monitoring_script = '''
import psutil
import time
import subprocess
from datetime import datetime

def monitor_resources():
    """Monitor system resources during training."""
    print("📊 Resource Monitor Started")
    print("Time\\t\\tCPU%\\tRAM%\\tGPU%\\tGPU_MEM")
    print("-" * 50)
    
    while True:
        try:
            # CPU and RAM
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            ram_percent = memory.percent
            
            # GPU info
            gpu_percent = "N/A"
            gpu_memory = "N/A"
            
            try:
                result = subprocess.run([
                    'nvidia-smi', 
                    '--query-gpu=utilization.gpu,memory.used,memory.total', 
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    gpu_info = result.stdout.strip().split(',')
                    gpu_percent = f"{gpu_info[0]}%"
                    gpu_memory = f"{int(gpu_info[1])}/{int(gpu_info[2])}MB"
            except:
                pass
            
            # Print status
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"{timestamp}\\t{cpu_percent:.1f}%\\t{ram_percent:.1f}%\\t{gpu_percent}\\t{gpu_memory}")
            
            time.sleep(30)  # Update every 30 seconds
            
        except KeyboardInterrupt:
            print("\\n📊 Monitoring stopped")
            break
        except Exception as e:
            print(f"\\n❌ Monitoring error: {e}")
            break

if __name__ == "__main__":
    monitor_resources()
'''
    
    with open("monitor_resources.py", "w") as f:
        f.write(monitoring_script)
    
    print("✅ Resource monitor created: monitor_resources.py")
    print("   Run: python monitor_resources.py (in separate terminal)")

def show_optimization_summary():
    """Show optimization summary."""
    print("\n" + "="*60)
    print("⚡ COLAB OPTIMIZATION COMPLETED!")
    print("="*60)
    
    print("\n🎯 Optimizations Applied:")
    print("✅ PyTorch environment variables optimized")
    print("✅ Memory-optimized training config created")
    print("✅ Caches cleared")
    print("✅ Resource monitoring setup")
    
    print("\n🚀 Next Steps:")
    print("1. Train with optimized config:")
    print("   python main.py train --config config/colab_optimized_config.yaml")
    print("\n2. Monitor resources (optional):")
    print("   python monitor_resources.py")
    print("\n3. If you get OOM errors:")
    print("   - Reduce batch_size in config")
    print("   - Enable gradient_checkpointing")
    print("   - Reduce max_input_length/max_output_length")

def main():
    """Main optimization function."""
    print("⚡ Colab Optimization Tool")
    print("=" * 40)
    
    try:
        # Check system resources
        check_system_resources()
        print()
        
        # Apply optimizations
        optimize_pytorch()
        print()
        
        # Create optimized configs
        create_optimized_configs()
        print()
        
        # Clear caches
        clear_cache()
        print()
        
        # Setup monitoring
        setup_monitoring()
        
        # Show summary
        show_optimization_summary()
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
