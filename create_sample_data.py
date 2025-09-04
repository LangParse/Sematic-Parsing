#!/usr/bin/env python3
"""
Create Sample Data for Local Testing
====================================

This script creates a small sample dataset for local testing.
"""

import os
import json
from pathlib import Path

def create_sample_amr_data():
    """Create sample AMR data for testing."""
    
    # Sample Vietnamese sentences with their AMR representations
    sample_data = [
        {
            "sentence": "Tôi yêu Việt Nam",
            "amr": "(y / yêu\n    :ARG0 (t / tôi)\n    :ARG1 (v / Việt_Nam))"
        },
        {
            "sentence": "Cô ấy đang học tiếng Anh",
            "amr": "(h / học\n    :ARG0 (c / cô_ấy)\n    :ARG1 (t / tiếng_Anh)\n    :aspect (p / progressive))"
        },
        {
            "sentence": "Hôm nay trời đẹp",
            "amr": "(đ / đẹp\n    :ARG1 (t / trời)\n    :time (h / hôm_nay))"
        },
        {
            "sentence": "Anh ấy đi làm bằng xe buýt",
            "amr": "(đ / đi\n    :ARG0 (a / anh_ấy)\n    :ARG4 (l / làm)\n    :instrument (x / xe_buýt))"
        },
        {
            "sentence": "Chúng tôi ăn cơm tối ở nhà hàng",
            "amr": "(ă / ăn\n    :ARG0 (c / chúng_tôi)\n    :ARG1 (c2 / cơm_tối)\n    :location (n / nhà_hàng))"
        },
        {
            "sentence": "Em bé đang ngủ trong phòng",
            "amr": "(n / ngủ\n    :ARG0 (e / em_bé)\n    :location (p / phòng)\n    :aspect (p2 / progressive))"
        },
        {
            "sentence": "Bố mẹ tôi sống ở Hà Nội",
            "amr": "(s / sống\n    :ARG0 (b / bố_mẹ\n        :poss (t / tôi))\n    :location (h / Hà_Nội))"
        },
        {
            "sentence": "Sinh viên đọc sách trong thư viện",
            "amr": "(đ / đọc\n    :ARG0 (s / sinh_viên)\n    :ARG1 (s2 / sách)\n    :location (t / thư_viện))"
        },
        {
            "sentence": "Cậu bé chơi bóng đá với bạn",
            "amr": "(c / chơi\n    :ARG0 (c2 / cậu_bé)\n    :ARG1 (b / bóng_đá)\n    :accompanier (b2 / bạn))"
        },
        {
            "sentence": "Cô giáo dạy toán cho học sinh",
            "amr": "(d / dạy\n    :ARG0 (c / cô_giáo)\n    :ARG1 (t / toán)\n    :ARG2 (h / học_sinh))"
        },
        {
            "sentence": "Ông ấy mua xe mới hôm qua",
            "amr": "(m / mua\n    :ARG0 (ô / ông_ấy)\n    :ARG1 (x / xe\n        :mod (m2 / mới))\n    :time (h / hôm_qua))"
        },
        {
            "sentence": "Chúng ta cần bảo vệ môi trường",
            "amr": "(c / cần\n    :ARG0 (c2 / chúng_ta)\n    :ARG1 (b / bảo_vệ\n        :ARG1 (m / môi_trường)))"
        },
        {
            "sentence": "Cô ấy nấu ăn rất ngon",
            "amr": "(n / nấu_ăn\n    :ARG0 (c / cô_ấy)\n    :manner (n2 / ngon\n        :degree (r / rất)))"
        },
        {
            "sentence": "Anh trai tôi làm bác sĩ",
            "amr": "(l / làm\n    :ARG0 (a / anh_trai\n        :poss (t / tôi))\n    :ARG1 (b / bác_sĩ))"
        },
        {
            "sentence": "Trẻ em thích xem phim hoạt hình",
            "amr": "(t / thích\n    :ARG0 (t2 / trẻ_em)\n    :ARG1 (x / xem\n        :ARG1 (p / phim_hoạt_hình)))"
        },
        {
            "sentence": "Bà ngoại kể chuyện cho cháu",
            "amr": "(k / kể\n    :ARG0 (b / bà_ngoại)\n    :ARG1 (c / chuyện)\n    :ARG2 (c2 / cháu))"
        },
        {
            "sentence": "Chúng tôi đi du lịch mùa hè",
            "amr": "(đ / đi_du_lịch\n    :ARG0 (c / chúng_tôi)\n    :time (m / mùa_hè))"
        },
        {
            "sentence": "Cậu ấy chạy rất nhanh",
            "amr": "(c / chạy\n    :ARG0 (c2 / cậu_ấy)\n    :manner (n / nhanh\n        :degree (r / rất)))"
        },
        {
            "sentence": "Mẹ tôi trồng hoa trong vườn",
            "amr": "(t / trồng\n    :ARG0 (m / mẹ\n        :poss (t2 / tôi))\n    :ARG1 (h / hoa)\n    :location (v / vườn))"
        },
        {
            "sentence": "Học sinh làm bài tập về nhà",
            "amr": "(l / làm\n    :ARG0 (h / học_sinh)\n    :ARG1 (b / bài_tập_về_nhà))"
        }
    ]
    
    return sample_data

def create_amr_file(data, output_file):
    """Create AMR file in the expected format."""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, item in enumerate(data):
            # Write sentence
            f.write(f"#::snt {item['sentence']}\n")
            # Write AMR
            f.write(f"{item['amr']}\n")
            
            # Add separator except for last item
            if i < len(data) - 1:
                f.write("\n")

def create_jsonl_file(data, output_file):
    """Create JSONL file for direct training."""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            json_item = {
                "input": item['sentence'],
                "output": item['amr']
            }
            json.dump(json_item, f, ensure_ascii=False)
            f.write('\n')

def main():
    """Create sample data files."""
    
    print("🔄 Creating sample data for local testing...")
    
    # Create directories
    data_dir = Path("data")
    train_dir = data_dir / "train"
    processed_dir = data_dir / "processed_local"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Get sample data
    sample_data = create_sample_amr_data()
    
    # Split data
    total_samples = len(sample_data)
    train_size = int(total_samples * 0.7)  # 70% for training
    val_size = int(total_samples * 0.2)    # 20% for validation
    
    train_data = sample_data[:train_size]
    val_data = sample_data[train_size:train_size + val_size]
    test_data = sample_data[train_size + val_size:]
    
    print(f"📊 Data split:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Validation: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    
    # Create AMR file for processing demo
    amr_file = train_dir / "sample_amr.txt"
    create_amr_file(sample_data, amr_file)
    print(f"✅ Created AMR file: {amr_file}")
    
    # Create JSONL files for direct training
    train_file = processed_dir / "train.jsonl"
    val_file = processed_dir / "val.jsonl"
    test_file = processed_dir / "test.jsonl"
    
    create_jsonl_file(train_data, train_file)
    create_jsonl_file(val_data, val_file)
    create_jsonl_file(test_data, test_file)
    
    print(f"✅ Created training files:")
    print(f"  {train_file}")
    print(f"  {val_file}")
    print(f"  {test_file}")
    
    # Create a combined file for processing demo
    all_file = processed_dir / "amr_training_data.jsonl"
    create_jsonl_file(sample_data, all_file)
    print(f"✅ Created combined file: {all_file}")
    
    print("\n🎯 Next steps:")
    print("1. Test data processing:")
    print("   python main.py process-data --input-dir data/train --output-dir data/processed_local")
    print("\n2. Test training:")
    print("   python main.py train --config config/local_test_config.yaml")
    print("\n3. Test evaluation:")
    print("   python main.py evaluate --model-path models/amr_model_local --test-data data/processed_local/test.jsonl")
    print("\n4. Test prediction:")
    print("   python main.py predict --model-path models/amr_model_local --text 'Tôi yêu Việt Nam' --format")

if __name__ == "__main__":
    main()
