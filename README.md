# AMR Vietnamese Parser

## Cấu trúc thư mục

```
semantic_parsing/
├─ amr_vn/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ data.py
│  ├─ preprocess.py
│  ├─ modeling.py
│  ├─ trainer.py
│  ├─ evaluator.py
│  ├─ inference.py
│  └─ utils/
│     └─ penman.py
├─ scripts/
│  ├─ train.py
│  └─ predict.py
├─ data/
│  ├─ train/
│  │  └─ train_split.txt
│  └─ test/
│     └─ public_test.txt
├─ outputs/
│  ├─ models/
│  ├─ predictions/
│  └─ logs/
├─ requirements.txt
└─ README.md
```

## Đặt dữ liệu

- **Train**: đặt tất cả file train `.txt` theo chuẩn `#::snt …` + AMR vào `data/train/`. Có thể nhiều file.
- **Test**: đặt `public_test.txt` vào `data/test/` (mỗi dòng một câu).

## Cài đặt

```bash
pip install -r requirements.txt
python -c "import nltk; import nltk; nltk.download('punkt', quiet=True)"
```

## 1) Build JSON từ nhiều file train

```bash
python -m amr_vn.cli build-json   --project-root .   --train-files data/train/train_split.txt  # có thể thêm nhiều file theo sau
```

Tạo `outputs/amr_data.json`.

## 2) Train

```bash
python -m amr_vn.cli train   --project-root .   --json-path outputs/amr_data.json   --model-name VietAI/vit5-base   --max-length 512 --batch-size 4 --epochs 10 --lr 5e-5
```

Model và tokenizer sẽ lưu ở `outputs/models/`.

## 3) Đánh giá nhanh (BLEU)

```bash
python -m amr_vn.cli eval   --project-root .   --json-path outputs/amr_data.json   --model-path outputs/models   --limit 100
```

## 4) Dự đoán trên public_test

```bash
python -m amr_vn.cli predict   --project-root .   --test-file data/test/public_test.txt   --model-path outputs/models
```

Kết quả: `outputs/predictions/public_test_result.txt`.

## Ghi chú

- `AMRCorpusLoader` hỗ trợ **nhiều file train** và gom toàn bộ mẫu vào một JSON duy nhất.
- `PublicTestLoader` đọc mỗi **dòng** của `public_test.txt` là một câu input.
- `basic_penman_format` không thay đổi nội dung, chỉ chuẩn hóa xuống dòng và khoảng trắng tối thiểu.

## Chạy kiểu menu đơn giản

Không cần CLI dài. Chạy:

```bash
python run.py
```

Sau đó chọn 1-4 và nhập đường dẫn khi được hỏi.
