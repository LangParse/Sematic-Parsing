<!-- Banner -->
<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

<!-- Title -->
<h1 align="center"><b>VSLP</b></h1>

<!-- Main -->

# Thành viên nhóm:

| STT |   MSSV   |                Họ và Tên |     Chức Vụ |                                                  Github |                  Email |
| --- | :------: | -----------------------: | ----------: | ------------------------------------------------------: | ---------------------: |
| 1   | 25410098 |          Tran Hai Nguyen | Nhóm trưởng | [nguyentran070397](https://github.com/nguyentran070397) | 25410098@ms.uit.edu.vn |
| 2   | 25410139 |         Nguyen Phuoc Tho |  Thành viên |               [nphuoctho](https://github.com/nphuoctho) | 25410139@ms.uit.edu.vn |
| 3   | 25410124 |             Do Trong Tai |  Thành viên |             [taidotrong](https://github.com/taidotrong) | 25410124@ms.uit.edu.vn |
| 4   | 25410070 |             Do Danh Khoa |  Thành viên |             [dodanhkhoa](https://github.com/dodanhkhoa) | 25410070@ms.uit.edu.vn |
| 5   | 25410145 | Duong Phuong Chuong Toan |  Thành viên |             [ToanIT2004](https://github.com/ToanIT2004) | 25410145@ms.uit.edu.vn |

## GIỚI THIỆU NHÓM

- **Số thứ tự nhóm:** 12
- **Tên nhóm:** LangParse

## ĐỒ ÁN CUỐI KỲ

- **Tên đồ án:** Semantic Parsing
- **Thư mục:** None

### 1. Hiểu bài toán Semantic Parsing
Semantic Parsing là bài toán ánh xạ câu tự nhiên (input: chuỗi tiếng Việt) sang biểu diễn ngữ nghĩa có cấu trúc (ở đây là graph hoặc biểu diễn giống như Abstract Meaning Representation - AMR).

Ví dụ input:
```text
điều lệnh không thay đổi , người thắp đèn nói .
```
Output dạng AMR-like:
```text
(n1 / nói
    :topic(t / thay_đổi
        :theme(đ / điều_lệnh)
        :polarity -)
    :agent(n / người
        :agent-of(t1 / thắp
            :patient(đ1 / đèn))))
```
### 2. Dùng mô hình Seq2Seq như thế nào?
Ta sẽ xử lý như một bài toán dịch máy (machine translation), trong đó:
  - Input: chuỗi tiếng Việt (tokenized nếu cần)
  - Output: biểu diễn ngữ nghĩa tuần tự (linearized semantic graph)
### 3. Các bước thực hiện chi tiết
#### 3.1. Tiền xử lý dữ liệu
##### a. Chuẩn hóa và tách từ
  - Có thể dùng tokenizer như pyvi, underthesea, hoặc BPE/SentencePiece nếu dùng mô hình Transformer.
##### b. Linearize biểu diễn ngữ nghĩa
  - Dạng AMR cần được biến thành chuỗi để đưa vào decoder.
  - Ví dụ:
```text
(n1 / nói :topic(t / thay_đổi :theme(đ / điều_lệnh) :polarity -) :agent(n / người :agent-of(t1 / thắp :patient(đ1 / đèn))))
```
##### c. Ghép lại thành cặp song song:
| Input Câu                                       | Output Semantic |
| ----------------------------------------------- | --------------- |
| điều lệnh không thay đổi , người thắp đèn nói . | (n1 / nói ... ) |
#### 3.2. Huấn luyện mô hình Seq2Seq
**Có thể dùng:**
  - LSTM-based Seq2Seq (với Attention)
  - Transformer
  - mBART, mT5, NLLB (cho tiếng Việt)
  
**Công cụ:**
  - Fairseq
  - Hugging Face Transformers
  - OpenNMT

### 4. Đánh giá
Bạn có thể dùng:
  - Smatch score: phổ biến cho AMR parsing.
  - **BLEU/ROUGE**: nếu chỉ đánh giá dạng chuỗi.
  - Hoặc metric custom nếu bạn có ground truth AMR trees.


<!-- Footer -->
<p align='center'>Copyright © 2025 - LangParse</p>
