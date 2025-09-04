#!/usr/bin/env python3
"""
Simple Test Script
==================

Test basic functionality without complex training.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def test_imports():
    """Test if all modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from src.utils.config import Config
        print("✅ Config")
        
        from src.data_processing import AMRProcessor, DataLoader
        print("✅ Data processing")
        
        from src.tokenization import ViT5Tokenizer
        print("✅ Tokenization")
        
        from src.training.model_config import ModelConfig
        print("✅ Model config")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config():
    """Test configuration loading."""
    print("\n🔧 Testing configuration...")
    
    try:
        from src.utils.config import Config
        config = Config(config_file="config/local_test_config.yaml")
        
        print(f"Model: {config.get('model', 'name', 'VietAI/vit5-base')}")
        print(f"Batch size: {config.get('model', 'batch_size', 2)}")
        print(f"Max samples: {config.get('data', 'max_samples', 100)}")
        
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_data_processing():
    """Test data processing."""
    print("\n📊 Testing data processing...")
    
    try:
        from src.data_processing import DataLoader
        
        # Check if processed data exists
        if Path("data/processed_local/train.jsonl").exists():
            loader = DataLoader()
            data = loader.load_jsonl("data/processed_local/train.jsonl")
            stats = loader.get_data_statistics(data)
            
            print(f"Loaded {len(data)} samples")
            print(f"Avg input length: {stats['avg_input_length']:.1f}")
            print(f"Avg output length: {stats['avg_output_length']:.1f}")
            
            return True
        else:
            print("⚠️  No processed data found")
            return False
            
    except Exception as e:
        print(f"❌ Data processing test failed: {e}")
        return False

def test_tokenizer():
    """Test tokenizer."""
    print("\n🔤 Testing tokenizer...")
    
    try:
        from src.tokenization import ViT5Tokenizer
        
        tokenizer = ViT5Tokenizer()
        
        # Test tokenization
        sample_text = "Tôi yêu Việt Nam"
        sample_amr = "(y / yêu :ARG0 (t / tôi) :ARG1 (v / Việt_Nam))"
        
        result = tokenizer.tokenize_sample(sample_text, sample_amr)
        
        print(f"Input: {sample_text}")
        print(f"Input tokens: {len(result.input_ids)}")
        print(f"Label tokens: {len(result.labels)}")
        print(f"Vocab size: {tokenizer.get_vocab_size()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tokenizer test failed: {e}")
        return False

def main():
    """Run simple tests."""
    print("🚀 Simple AMR Test Suite")
    print("=" * 40)
    
    tests = [
        ("Import test", test_imports),
        ("Configuration test", test_config),
        ("Data processing test", test_data_processing),
        ("Tokenizer test", test_tokenizer)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"🔍 {test_name}")
        print("=" * 40)
        
        if test_func():
            print(f"✅ {test_name} PASSED")
            passed += 1
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n{'='*40}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    print("=" * 40)
    
    if passed == total:
        print("🎉 All tests passed! System is ready for use.")
        print("\n📋 Next steps:")
        print("1. For local training: python main.py train --config config/local_test_config.yaml")
        print("2. For Colab training: Use AMR_Training_Colab.ipynb")
        print("3. For prediction: python main.py predict --model-path <model> --text 'Your text'")
    else:
        print(f"⚠️  {total - passed} tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
