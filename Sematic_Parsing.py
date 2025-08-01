# Trước khi chạy code lưu ý:
# Thay thế đường dẫn "/content/drive/MyDrive/Colab Notebooks/Semantic Parsing/" bằng đường dẫn hiện tại trên GG Drive của từng người để không bị báo lỗi File not Found
# Ví dụ: Đường dẫn trên GG Drive của thư mục chứa toàn bộ Project và các file dữ liệu là "/content/drive/OtherDriveName/Project/AMR/"
# Replace "/content/drive/MyDrive/Colab Notebooks/Semantic Parsing/" bằng "/content/drive/OtherDriveName/Project/AMR/"
# File training ban đầu của Ban Tổ Chức là 2 file text. Anh đã gộp lại thành 1 file và đặt tên thành train_amr.txt. Bỏ file này vào đúng đường dẫn /content/drive/OtherDriveName/Project/AMR/ mới chạy được Module 1
# Thứ tự tạo các file như sau:
# Module 1: Input file train_amr.txt để làm sạch dữ liệu và lưu thành file input_amr.txt
# Module 2: Input file input_amr.txt để tạo thành file JSON (amr_data.json)
# Module 3: Input file amr_data,json để Tokenize dữ liệu phục vụ cho mô hình training
# Module 4: Từ dữ liệu đã Tokenize, tiến trình training bắt đầu. Nhúng chức năng wandb để theo dõi tiến trình học. Cần đăng ký tài khoản wandb để lấy API (search GG)
# Các module còn lại không tạo ra file.

# =======================================================
# 📘 MODULE 1: Làm sạch và chuẩn hóa dữ liệu AMR đầu vào
# =======================================================

# =========================
# 📦 IMPORT THƯ VIỆN
# =========================
import os
# 📌 Nội dung: Thư viện hỗ trợ thao tác với đường dẫn file và thư mục
# 🎯 Mục đích: Đảm bảo việc đọc/ghi file ở đúng vị trí với tên chuẩn
# ✅ Kết quả: Có thể trích xuất thư mục của file gốc và lưu output vào cùng thư mục


# =========================
# 🧼 HÀM XỬ LÝ FILE AMR
# =========================
def clean_amr_file(input_file_path):
    """
    📌 Nội dung: Chuẩn hóa file AMR đầu vào để dễ xử lý về sau.
    🎯 Mục đích: Gom nhóm mỗi câu tiếng Việt với sơ đồ AMR tương ứng thành 1 block, phân cách bằng dòng '---'.
    ✅ Kết quả: Tạo file 'input_amr.txt' ở cùng thư mục, đã sẵn sàng để token hóa (Module 2).
    """

    # ------------------------------------
    # Bước 1: Lấy thư mục của file đầu vào
    # ------------------------------------
    dir_path = os.path.dirname(input_file_path)
    # 📌 Nội dung: Tách phần thư mục từ đường dẫn gốc
    # 🎯 Mục đích: Để sau đó có thể tạo file output cùng thư mục
    # ✅ Kết quả: Biến `dir_path` chứa đường dẫn thư mục file gốc

    # -------------------------------------
    # Bước 2: Xác định đường dẫn file output
    # -------------------------------------
    output_file_path = os.path.join(dir_path, "input_amr.txt")
    # 📌 Nội dung: Gộp đường dẫn thư mục với tên file mới
    # 🎯 Mục đích: Đặt tên thống nhất cho file AMR đã chuẩn hóa
    # ✅ Kết quả: File output sẽ nằm ở cùng thư mục với tên rõ ràng

    # ------------------------------------------
    # Bước 3: Đọc toàn bộ nội dung từ file gốc
    # ------------------------------------------
    with open(input_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 📌 Nội dung: Đọc từng dòng từ file
    # 🎯 Mục đích: Chuẩn bị dữ liệu để gom theo từng đoạn AMR
    # ✅ Kết quả: Danh sách `lines` chứa toàn bộ nội dung của file

    # ------------------------------------------
    # Bước 4: Gom nhóm các đoạn AMR theo từng câu
    # ------------------------------------------
    blocks = []          # 📌 Danh sách chứa các block gồm câu và AMR
    current_block = []   # 📌 Danh sách tạm để gom từng block

    for line in lines:
        line = line.rstrip()  # 📌 Xoá ký tự xuống dòng
                              # 🎯 Mục đích: Tránh bị xuống dòng thừa khi ghi lại file

        if line.startswith("#::snt"):  # 📌 Dòng bắt đầu là câu tiếng Việt
            if current_block:
                blocks.append(current_block)
                # 🎯 Mục đích: Nếu đang xử lý một block cũ, thì thêm vào danh sách chính
                # ✅ Kết quả: Mỗi block được phân biệt rõ ràng

                current_block = []  # 📌 Reset block để bắt đầu đoạn mới

        current_block.append(line)  # 📌 Thêm dòng hiện tại vào block hiện tại

    # ------------------------------------------
    # Bước 5: Thêm đoạn cuối nếu chưa được ghi
    # ------------------------------------------
    if current_block:
        blocks.append(current_block)
    # 🎯 Mục đích: Đảm bảo đoạn cuối cùng không bị bỏ sót
    # ✅ Kết quả: Danh sách `blocks` đầy đủ tất cả đoạn câu + AMR

    # ------------------------------------------
    # Bước 6: Ghi các block vào file mới
    # ------------------------------------------
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for block in blocks:
            for line in block:
                f.write(line + '\n')
            f.write('-----------------------------------------------\n')
    # 📌 Nội dung: Ghi từng dòng trong block và phân cách các block bằng dòng gạch ngang
    # 🎯 Mục đích: Dễ dàng phân tích từng đoạn trong các bước tiếp theo
    # ✅ Kết quả: File `input_amr.txt` đã sẵn sàng để dùng tiếp trong pipeline

    # ------------------------------------------
    # Bước 7: In ra đường dẫn file kết quả
    # ------------------------------------------
    print(f"✅ File AMR đã được chuẩn hóa và lưu tại: {output_file_path}")
    # 🎯 Mục đích: Cho người dùng xác nhận quá trình xử lý đã xong
    # ✅ Kết quả: Biết chính xác nơi lưu file để tiếp tục Module 2


# =========================
# ▶️ GỌI HÀM TIỀN XỬ LÝ
# =========================
clean_amr_file("/content/drive/MyDrive/Colab Notebooks/Semantic Parsing/train_amr.txt")
# 📌 Nội dung: Chạy hàm với đường dẫn cụ thể tới file AMR gốc
# 🎯 Mục đích: Chuẩn hóa nội dung để phục vụ Module 2
# ✅ Kết quả: Tạo được file `input_amr.txt` ở thư mục gốc, cấu trúc rõ ràng


# =======================================
# 📦 MODULE 2: CHUYỂN AMR THÀNH JSON
# =======================================

import json  # ✪ Nội dung: Dùng để đọc/ghi file JSON
              # 🌟 Mục đích: Biến danh sách cặp (input/output) thành file training
              # ✅ Kết quả: Cho ra file .json chuẩn huấn luyện

import os    # ✪ Nội dung: Dùng để xử lý đường dẫn file
              # 🌟 Mục đích: Gừi gọn việc chỉnh sửa path file
              # ✅ Kết quả: Biến file đồng bộ với Module 1

# =======================================
# ✅ KHAI BÁO ĐƯỚNG DẪN INPUT/OUTPUT
# =======================================
input_file = "/content/drive/MyDrive/Colab Notebooks/Semantic Parsing/input_amr.txt"  # ✪ File AMR sau khi clean
output_file = "/content/drive/MyDrive/Colab Notebooks/Semantic Parsing/amr_data.json" # ✪ File json huấn luyện

# =======================================
# ✅ BƯỚC 1: ĐỌC FILE VÀ LƯU Dữ LIỆU TỪNG DÒNG
# =======================================
with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()  # ✪ Đọc toàn bộ dữ liệu dòng
                        # 🌟 Cho phép duyệt theo block
                        # ✅ Tạo danh sách lines để xử lý dần

# =======================================
# ✅ BƯỚC 2: TÁCH CẶP INPUT/OUTPUT
# =======================================
data = []                 # ✪ Danh sách dữ liệu output dạng JSON
current_sentence = None   # ✪ Lưu câu đang xử lý
amr_lines = []            # ✪ Gom các dòng AMR tương ứng

for line in lines:
    line = line.strip()  # ✪ Xóa khoảng trắng dư

    if line.startswith("#::snt"):  # ✪ Gặp câu mới
        if current_sentence and amr_lines:
            data.append({  # ✪ Lưu block trước
                "input": current_sentence,
                "output": "\n".join(amr_lines)
            })
        current_sentence = line[7:].strip()  # ✪ Tách câu
        amr_lines = []

    elif line == "-----------------------------------------------":
        if current_sentence and amr_lines:
            data.append({  # ✪ Lưu block cuối của câu
                "input": current_sentence,
                "output": "\n".join(amr_lines)
            })
            current_sentence = None
            amr_lines = []

    elif current_sentence is not None:
        amr_lines.append(line)  # ✪ Gom dòng AMR

# ✪ Nếu câu cuối chưa được lưu
if current_sentence and amr_lines:
    data.append({
        "input": current_sentence,
        "output": "\n".join(amr_lines)
    })

# =======================================
# ✅ BƯỚC 3: LOẠI BỆ TRÙNG LẤP
# =======================================
seen = set()             # ✪ Set để kiểm tra các cặp trùng
filtered_data = []       # ✪ Danh sách đã loại trùng
for entry in data:
    key = (entry["input"], entry["output"])
    if key not in seen:
        filtered_data.append(entry)
        seen.add(key)  # ✪ Đánh dấu đã gặp

# =======================================
# ✅ BƯỚC 4: GHI RA FILE JSON
# =======================================
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)  # ✪ Ghi chuẩn UTF-8, dễ đọc

# =======================================
# ✅ BƯỚC 5: THÔNG BÁO HOÀN THÀNH
# =======================================
print(f"✅ Đã lưu {len(filtered_data)} cặp input/output vào: {output_file}")


# ==============================================================================
# 📦 MODULE 3: TOKENIZE DỮ LIỆU TỪ FILE JSON ĐỂ CHUẨN BỊ CHO MÔ HÌNH HUẤN LUYỆN
# ==============================================================================

# =========================
# 📦 IMPORT CÁC THƯ VIỆN
# =========================
from transformers import T5Tokenizer        # Dùng tokenizer của mô hình T5 hoặc LongT5
import json                                # Đọc/ghi file JSON
import os                                  # Xử lý đường dẫn file

# =========================
# 🛠 CẤU HÌNH ĐƯỜNG DẪN
# =========================
input_json_path = "/content/drive/MyDrive/Colab Notebooks/Semantic Parsing/amr_data.json"
token_output_path = os.path.join(os.path.dirname(input_json_path), "Tokenize_data.txt")

# =========================
# ✂️ TẢI TOKENIZER
# =========================
tokenizer = T5Tokenizer.from_pretrained("google/long-t5-tglobal-base")  # Dùng tokenizer tương thích với LongT5

# =========================
# 🔄 TOKENIZE TOÀN BỘ DỮ LIỆU
# =========================
with open(input_json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)  # Đọc toàn bộ dữ liệu {"input": ..., "output": ...}

with open(token_output_path, 'w', encoding='utf-8') as f_out:
    for item in data:
        input_text = item["input"]
        output_text = item["output"]

        # Tokenize input
        input_tokens = tokenizer.tokenize(input_text)
        # Tokenize output
        output_tokens = tokenizer.tokenize(output_text)

        # Ghi thông tin ra file
        f_out.write("#::input_tokens\n")
        f_out.write(" ".join(input_tokens) + "\n\n")

        f_out.write("#::output_tokens\n")
        f_out.write(" ".join(output_tokens) + "\n")

        # Phân cách giữa các câu
        f_out.write("-" * 40 + "\n")
		
		
# =========================================================================================
# 📦 MODULE 4: HUẤN LUYỆN MÔ HÌNH vit5-base ĐỂ CHUYỂN ĐỔI TỪ CÂU TIẾNG VIỆT SANG SƠ ĐỒ AMR
# =========================================================================================

# =========================
# 📦 IMPORT CÁC THƯ VIỆN
# =========================
from transformers import T5ForConditionalGeneration                 # 📌 Mô hình seq2seq T5
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments   # 📌 Trainer cho mô hình Seq2Seq
from datasets import Dataset                                        # 📌 Dùng để tạo dataset từ JSON list
from transformers import AutoTokenizer                              # 📌 Tokenizer phù hợp với mô hình
import json                                                         # 📌 Dùng đọc file dữ liệu
import os                                                           # 📌 Quản lý file path
import wandb                                                        # 📌 Theo dõi training trên wandb

# =========================
# 🛠 CẤU HÌNH ĐƯỜNG DẪN
# =========================
input_json_path = "/content/drive/MyDrive/Colab Notebooks/Sematic Parsing/amr_data.json"
model_output_dir = "/content/drive/MyDrive/Colab Notebooks/Sematic Parsing/amr_model_vit5"
log_dir = "/content/drive/MyDrive/Colab Notebooks/Sematic Parsing/logs_vit5"

# =========================
# 🧾 KẾT NỐI WANDB
# =========================
wandb.login()  
# 📌 Nội dung: Đăng nhập tài khoản wandb đã cấu hình từ trước
# 🎯 Mục đích: Bắt đầu theo dõi tiến trình training
# ✅ Kết quả: Log mô hình, biểu đồ loss hiển thị trên trang wandb.io

# =========================
# ✂️ LOAD TOKENIZER VIT5
# =========================
tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")  
# 📌 Nội dung: Dùng tokenizer tương ứng với mô hình vit5
# 🎯 Mục đích: Token hóa input/output đúng định dạng mô hình huấn luyện
# ✅ Kết quả: Có thể sử dụng .encode/.decode cho văn bản tiếng Việt

# =========================
# 🚀 HÀM HUẤN LUYỆN
# =========================
def train_model(json_path):
    # Bước 1: Đọc file JSON chứa các cặp (input/output)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 📌 Nội dung: Load toàn bộ dữ liệu huấn luyện
    # 🎯 Mục đích: Chuẩn bị dữ liệu trước khi đưa vào dataset
    # ✅ Kết quả: Danh sách dict với các cặp input/output

    # Bước 2: Tạo Dataset từ danh sách dict
    dataset = Dataset.from_list(data)
    # 📌 Nội dung: Biến list → dataset để Trainer sử dụng
    # 🎯 Mục đích: Chuẩn hóa format theo HuggingFace
    # ✅ Kết quả: Biến `dataset` có thể dùng trực tiếp

    # Bước 3: Tokenize toàn bộ dữ liệu
    def preprocess(example):
        model_input = tokenizer(
            example["input"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
        labels = tokenizer(
            example["output"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
        model_input["labels"] = labels["input_ids"]
        return model_input

    tokenized_dataset = dataset.map(preprocess)
    # 📌 Nội dung: Tokenize toàn bộ câu input/output
    # 🎯 Mục đích: Chuyển văn bản thành ID để mô hình huấn luyện
    # ✅ Kết quả: Dataset đã token hóa đầy đủ

    # Bước 4: Load mô hình vit5-base
    model = T5ForConditionalGeneration.from_pretrained("VietAI/vit5-base")
    # 📌 Nội dung: Sử dụng mô hình tiếng Việt đã pretrain
    # 🎯 Mục đích: Dựa trên kiến thức cũ để fine-tune nhanh hơn
    # ✅ Kết quả: Mô hình đã sẵn sàng huấn luyện

    # Bước 5: Cấu hình huấn luyện
    training_args = Seq2SeqTrainingArguments(
        output_dir=model_output_dir,              
        per_device_train_batch_size=4,             # T4 nên dùng batch 4
        num_train_epochs=10,                       # Có thể điều chỉnh nếu loss ổn định
        logging_dir=log_dir,                       
        logging_steps=500,                         # Ghi log mỗi 500 steps
        save_steps=200,                            # Lưu model định kỳ
        save_total_limit=2,                        # Chỉ lưu tối đa 2 model
        report_to=["wandb"],                       # Kết nối wandb
        run_name="AMR_ViT5_T4GPU",                 # Tên run trong wandb
        fp16=True                                  # Tăng tốc nếu GPU hỗ trợ
    )

    # Bước 6: Tạo Trainer và train
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )

    # Bước 7: Tiến hành huấn luyện
    trainer.train()
    # 📌 Nội dung: Gọi trainer để bắt đầu huấn luyện
    # 🎯 Mục đích: Fine-tune mô hình vit5 trên dữ liệu AMR tiếng Việt
    # ✅ Kết quả: Model được cải thiện theo dữ liệu AMR

    return model

# =========================
# ▶️ GỌI HÀM HUẤN LUYỆN
# =========================
model = train_model(input_json_path)
# 📌 Nội dung: Gọi huấn luyện từ file json
# 🎯 Mục đích: Kích hoạt toàn bộ pipeline training
# ✅ Kết quả: Mô hình vit5 được huấn luyện và lưu


# ====================================================================
# MODULE 5A: ĐÁNH GIÁ CHẤT LƯỢNG MÔ HÌNH T5 BẰNG CÁCH TÍNH BLEU SCORE
# ====================================================================

# =========================
# 📦 IMPORT CẦN THIẾT
# =========================
from nltk.translate.bleu_score import sentence_bleu   # Nội dung: Hàm tính điểm BLEU từ NLTK
                                                      # Mục đích: So sánh độ giống nhau giữa AMR thật và dự đoán
                                                      # Kết quả: Trả về số từ 0.0 đến 1.0

import json  # Nội dung: Dùng để đọc dữ liệu từ file JSON
             # Mục đích: Lấy cặp input/output đã lưu
             # Kết quả: Biến `samples` là list chứa nhiều dict

import torch  # Nội dung: Thư viện tính toán tensor
              # Mục đích: Kiểm tra và chuyển device giữa CPU và GPU
              # Kết quả: Đảm bảo model và dữ liệu nằm trên cùng thiết bị

# =========================
# 📍 ĐƯỜNG DẪN FILE
# =========================
json_path = "/content/drive/MyDrive/Colab Notebooks/Semantic Parsing/amr_data.json"  # Đường dẫn file JSON đầu vào

# =========================
# 🧪 HÀM ĐÁNH GIÁ MÔ HÌNH
# =========================
def evaluate_model(model, tokenizer, json_path):
    # Bước 1: Đọc file chứa cặp input/output từ json
    with open(json_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    total_score = 0  # Bước 2: Khởi tạo biến tổng điểm BLEU

    # Bước 3: Xác định device mô hình đang chạy (GPU nếu có, không thì CPU)
    device = next(model.parameters()).device

    # Bước 4: Lặp 100 mẫu đầu để đánh giá nhanh
    for sample in samples[:100]:
        # Bước 4.1: Token hóa câu đầu vào và chuyển sang đúng device
        input_ids = tokenizer(sample["input"], return_tensors="pt").input_ids.to(device)

        # Bước 4.2: Mô hình sinh sơ đồ AMR dự đoán
        output_ids = model.generate(input_ids)[0]

        # Bước 4.3: Giải mã chuỗi dự đoán thành văn bản
        prediction = tokenizer.decode(output_ids, skip_special_tokens=True)

        # Bước 4.4: Tách chuỗi thành danh sách từ (token) để tính BLEU
        reference = sample["output"].split()      # AMR thật
        hypothesis = prediction.split()           # AMR dự đoán

        # Bước 4.5: Cộng điểm BLEU vào tổng
        total_score += sentence_bleu([reference], hypothesis)

    # Bước 5: Tính điểm BLEU trung bình
    avg_bleu = round(total_score / 100, 4)

    # Bước 6: In kết quả cuối cùng
    print("BLEU score:", avg_bleu)

# =========================
# ▶️ GỌI HÀM ĐÁNH GIÁ
# =========================
evaluate_model(model, tokenizer, json_path)


# ===========================================================================
# MODULE 6: LƯU MÔ HÌNH VÀ TOKENIZER ĐÃ HUẤN LUYỆN VÀO THƯ MỤC ĐỂ SỬ DỤNG LẠI
# ===========================================================================

# Nội dung: Gọi phương thức save_pretrained() của mô hình T5
# Mục đích: Lưu toàn bộ trọng số, kiến trúc và config của mô hình
# Kết quả: Tạo thư mục /content/amr_model chứa các file như config.json, pytorch_model.bin, ...
model.save_pretrained("/content/drive/MyDrive/Colab Notebooks/Semantic Parsing/amr_model")

# Nội dung: Lưu tokenizer kèm theo mô hình
# Mục đích: Đảm bảo khi load lại mô hình vẫn dùng đúng kiểu token hóa
# Kết quả: Thư mục /content/amr_model có thêm tokenizer_config.json, vocab.json, tokenizer.model, ...
tokenizer.save_pretrained("/content/drive/MyDrive/Colab Notebooks/Semantic Parsing/amr_model")


# ========================================================================================
# MODULE 7: CHO NGƯỜI DÙNG NHẬP MỘT CÂU TIẾNG VIỆT, MÔ HÌNH SẼ DỰ ĐOÁN SƠ ĐỒ AMR TƯƠNG ỨNG
# ========================================================================================

from transformers import T5ForConditionalGeneration, T5Tokenizer  # Nội dung: Import mô hình và tokenizer T5
                                                                  # Mục đích: Dùng để load lại mô hình đã huấn luyện và xử lý câu nhập
                                                                  # Kết quả: Có thể sử dụng model.generate để tạo AMR từ input

def predict_amr_from_input():
    # Nội dung: Load lại mô hình đã lưu trước đó
    # Mục đích: Dùng mô hình đã huấn luyện thay vì tạo mới
    # Kết quả: Biến `model` chứa mô hình đã sẵn sàng dự đoán
    model = T5ForConditionalGeneration.from_pretrained("/content/drive/MyDrive/Colab Notebooks/Semantic Parsing/amr_model")

    # Nội dung: Load lại tokenizer tương ứng với mô hình
    # Mục đích: Đảm bảo mô hình và tokenizer đồng bộ
    # Kết quả: Biến `tokenizer` sẵn sàng mã hóa/giải mã văn bản
    tokenizer = T5Tokenizer.from_pretrained("/content/drive/MyDrive/Colab Notebooks/Semantic Parsing/amr_model")

    # Nội dung: Cho phép người dùng nhập một câu tiếng Việt từ bàn phím
    # Mục đích: Làm input cho mô hình dự đoán
    # Kết quả: Câu được lưu vào biến `input_sentence`, giữ nguyên dấu tiếng Việt
    input_sentence = input("Nhập một câu tiếng Việt để chuyển thành sơ đồ AMR: ").strip()

    # Nội dung: Tokenize câu nhập bằng tokenizer của mô hình
    # Mục đích: Chuyển câu thành tensor để mô hình xử lý
    # Kết quả: Tạo input_ids dùng cho mô hình
    input_ids = tokenizer(input_sentence, return_tensors="pt").input_ids

    # Nội dung: Sinh đầu ra từ mô hình dựa trên câu đã token hóa
    # Mục đích: Tạo sơ đồ AMR từ câu tiếng Việt
    # Kết quả: Biến `output_ids` chứa tensor biểu diễn AMR
    output_ids = model.generate(input_ids)[0]

    # Nội dung: Giải mã tensor đầu ra thành văn bản
    # Mục đích: Chuyển từ token ID → chuỗi AMR
    # Kết quả: Biến `prediction` là sơ đồ AMR dưới dạng chuỗi
    prediction = tokenizer.decode(output_ids, skip_special_tokens=True)

    # Nội dung: In kết quả sơ đồ AMR ra màn hình
    # Mục đích: Hiển thị cho người dùng xem trực tiếp
    # Kết quả: Kết quả được hiển thị rõ ràng
    print("\n📌 Sơ đồ AMR dự đoán:\n")
    print(prediction)

# Nội dung: Gọi hàm để bắt đầu dự đoán
# Mục đích: Cho phép nhập câu và chạy mô hình
# Kết quả: Mô hình in ra sơ đồ AMR tương ứng với câu nhập
predict_amr_from_input()