# Neural Machine Translation Transformer (En–Vi) – Data Pipeline

Dự án này triển khai **pipeline dữ liệu** cho hệ thống dịch máy Anh–Việt (Neural Machine Translation) dùng kiến trúc Transformer **tự code from scratch**.

> ⚠️ Hiện tại repo mới hoàn thành đến **DataLoader** (chuẩn bị batch cho mô hình).  
> Phần kiến trúc Transformer, huấn luyện và decoding sẽ được bổ sung sau.

---

## 1. Mục tiêu hiện tại

Triển khai trọn vẹn phần **A. Xử lý dữ liệu** trong bài tập lớn NLP 2025:

- Thu thập & chuẩn hóa dữ liệu song ngữ Anh–Việt.
- Tiền xử lý: làm sạch, lọc câu bất thường, thống kê độ dài câu.
- Huấn luyện **subword tokenizer** (SentencePiece).
- Xây dựng `NMTDataset` và `DataLoader` cho PyTorch:
  - Tạo `src_ids`, `tgt_in_ids`, `tgt_out_ids`.
  - Padding, truncation.
  - Tạo padding mask cho src/tgt.

---

## 2. Dataset

### 2.1. Nguồn dữ liệu

Dataset sử dụng: **IWSLT 2015 English–Vietnamese** (bản Stanford NMT), lấy qua mirror (ví dụ: Kaggle).

Các file gốc sau khi tải về và đổi tên:

```text
data/raw/
  train.en
  train.vi
  valid.en      # từ tst2012.en
  valid.vi      # từ tst2012.vi
  test.en       # từ tst2013.en
  test.vi       # từ tst2013.vi
  vocab.en      # (optional, chưa dùng)
  vocab.vi      # (optional, chưa dùng)
  dict.en-vi    # (optional, chưa dùng)
```

### 2.2. Làm sạch & lọc câu bất thường

Các bước xử lý:

- `strip()` và `lower()` cho mỗi câu.
- Bỏ các câu rỗng.
- Lọc outlier: loại bỏ cặp câu nếu **EN hoặc VI > 100 tokens** (token tạm tính bằng `split()` theo khoảng trắng).

Kết quả:

- **train**: kept = 132,406, dropped = 911  
- **valid**: kept = 1,550, dropped = 3  
- **test** : kept = 1,262, dropped = 6  

Dữ liệu sau khi làm sạch được lưu tại:

```text
data/processed/
  train.en
  train.vi
  valid.en
  valid.vi
  test.en
  test.vi
```

### 2.3. Thống kê độ dài câu

Token tạm tính bằng `str.split()` (chưa dùng subword).

**English:**

- min: 1  
- mean: ~20.32  
- median: 16  
- 95% percentile: 47  
- 99% percentile: 71  
- max: 628 (outlier, đã bị lọc nếu > 100)

**Vietnamese:**

- min: 1  
- mean: ~24.86  
- median: 20  
- 95% percentile: 59  
- 99% percentile: 89  
- max: 850 (outlier, đã bị lọc nếu > 100)

Từ thống kê này, hệ thống chọn:

```python
MAX_SRC_LEN = 70
MAX_TGT_LEN = 70
```

để bao phủ ~99% câu mà vẫn giữ mô hình gọn.

---

## 3. Tokenizer & Vocabulary (SentencePiece)

Tokenizer sử dụng **SentencePiece** với mô hình subword (unigram).

### 3.1. Chuẩn bị dữ liệu huấn luyện tokenizer

Tạo file gộp từ train EN + VI:

```text
data/spm/
  train_combined.txt   # train.en + train.vi
```

### 3.2. Huấn luyện SentencePiece

Tham số tiêu biểu:

- `vocab_size = 8000`  
- `model_type = "unigram"`  
- `character_coverage = 0.9995`  
- Special IDs:
  - `pad_id = 0`
  - `unk_id = 1`
  - `bos_id = 2`
  - `eos_id = 3`

Output:

```text
data/spm/
  train_combined.txt
  spm_unigram.model
  spm_unigram.vocab
```

---

## 4. Cấu trúc thư mục

```text
.
├── data/
│   ├── raw/
│   │   ├── train.en
│   │   ├── train.vi
│   │   ├── valid.en
│   │   ├── valid.vi
│   │   ├── test.en
│   │   ├── test.vi
│   │   ├── vocab.en
│   │   ├── vocab.vi
│   │   └── dict.en-vi
│   ├── processed/
│   │   ├── train.en
│   │   ├── train.vi
│   │   ├── valid.en
│   │   ├── valid.vi
│   │   ├── test.en
│   │   └── test.vi
│   └── spm/
│       ├── train_combined.txt
│       ├── spm_unigram.model
│       └── spm_unigram.vocab
├── src/
│   ├── tokenizer.py
│   └── dataset.py
└── notebooks/
    ├── 01_check_raw_data.ipynb
    ├── 02_preprocess_and_stats.ipynb
    └── 03_train_tokenizer.ipynb
```

---

## 5. Code chính đã hoàn thành

### 5.1. `src/tokenizer.py`

Wrapper đơn giản cho SentencePiece:

- Load model từ `data/spm/spm_unigram.model`.
- Định nghĩa các ID đặc biệt:

```python
pad_id = 0
unk_id = 1
bos_id = 2
eos_id = 3
```

- Các hàm chính:

```python
encode_src(text, add_bos=False, add_eos=True)
    # Mã hóa câu nguồn, trả về list[int].
    # Thường dùng add_eos=True cho encoder.

encode_tgt(text)
    # Mã hóa câu đích, trả về:
    #   dec_in_ids: [BOS, ..., token_n]
    #   dec_out_ids: [..., token_n, EOS]
    # Dùng cho decoder input / target output.

decode(ids)
    # Giải mã list[int] về text, bỏ pad nếu cần.
```

### 5.2. `src/dataset.py`

Chứa hai thành phần chính: `NMTDataset` và `collate_fn`.

#### `NMTDataset`

Dataset PyTorch cho bài toán NMT En–Vi.

- Input:
  - `data_dir`: thư mục chứa `data/processed`.
  - `split`: `"train"`, `"valid"` hoặc `"test"`.
  - `tokenizer`: instance của `SubwordTokenizer`.
  - `max_src_len`, `max_tgt_len`: độ dài tối đa cho source/target.

- Với mỗi phần tử `idx`, dataset:
  1. Đọc `src_text` từ `*.en`, `tgt_text` từ `*.vi`.
  2. Encode với SentencePiece:
     - `src_ids = tokenizer.encode_src(src_text, add_eos=True)`
     - `tgt_in_ids, tgt_out_ids = tokenizer.encode_tgt(tgt_text)`
  3. Truncate về `max_src_len` / `max_tgt_len`.
  4. Trả về dict:

```python
{
  "src_ids": List[int],
  "tgt_in_ids": List[int],
  "tgt_out_ids": List[int],
}
```

#### `collate_fn`

Hàm `collate_fn(batch, pad_id=0)`:

- Nhận vào list các sample từ `NMTDataset`.
- Tìm độ dài lớn nhất `max_src_len`, `max_tgt_len` trong batch.
- Padding tất cả sequences đến chiều dài này.
- Tạo thêm padding mask:

```python
{
  "src_ids": LongTensor (B, S),
  "tgt_in_ids": LongTensor (B, T),
  "tgt_out_ids": LongTensor (B, T),
  "src_padding_mask": BoolTensor (B, S),   # True tại vị trí PAD
  "tgt_padding_mask": BoolTensor (B, T),
}
```

---

## 6. Ví dụ sử dụng DataLoader

Ví dụ test nhanh:

```python
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

ROOT = Path(".").resolve()
SRC_DIR = ROOT / "src"
sys.path.append(str(SRC_DIR))

from tokenizer import SubwordTokenizer
from dataset import NMTDataset, collate_fn

tok = SubwordTokenizer(ROOT / "data/spm/spm_unigram.model")
vocab_size = tok.sp.get_vocab_size()

train_dataset = NMTDataset(
    data_dir=str(ROOT / "data/processed"),
    split="train",
    tokenizer=tok,
    max_src_len=70,
    max_tgt_len=70,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, pad_id=tok.pad_id),
)

batch = next(iter(train_loader))
for k, v in batch.items():
    print(k, v.shape, v.dtype)
```

Ví dụ output:

```text
src_ids torch.Size([32, 59]) torch.int64
tgt_in_ids torch.Size([32, 50]) torch.int64
tgt_out_ids torch.Size([32, 50]) torch.int64
src_padding_mask torch.Size([32, 59]) torch.bool
tgt_padding_mask torch.Size([32, 50]) torch.bool
```

---

## 7. Yêu cầu môi trường

- Python 3.9+
- PyTorch
- sentencepiece
- numpy

Cài đặt nhanh:

```bash
pip install torch sentencepiece numpy
```

---

## 8. Kế hoạch tiếp theo (roadmap)

Các phần tiếp theo (sẽ được bổ sung):

1. Cài đặt kiến trúc **Transformer Encoder–Decoder** từ đầu:
   - Scaled Dot-Product Attention, Multi-Head Attention.
   - Positional Encoding (sinusoidal).
   - Encoder/Decoder Layer (self-attention, cross-attention, FFN, LayerNorm, residual).
2. Xây dựng `TransformerNMT`:
   - Embedding + PositionalEncoding.
   - Encoder + Decoder.
   - Output projection sang vocabulary.
3. Training loop:
   - CrossEntropyLoss (ignore padding).
   - Optimizer (Adam / AdamW).
   - Scheduler với warmup.
   - Log loss & perplexity trên train/valid.
4. Decoding:
   - Greedy search.
   - Beam search (theo yêu cầu bài tập).
5. Evaluation:
   - BLEU score trên test set.
   - So sánh greedy vs beam.

---

## 9. Ghi chú

- Pipeline được viết theo hướng **tự triển khai tối đa**, không dùng `torchtext` hay HuggingFace Transformers, để phù hợp với mục tiêu **“Transformer from scratch”**.
- Phần này có thể dùng trực tiếp làm nội dung cho chương **Xử lý dữ liệu** trong báo cáo đồ án (có thể copy lại, thêm hình/biểu đồ minh họa thống kê).

---

## 10. Thêm dữ liệu vào project

Do kích thước dataset lớn, **dữ liệu không được lưu trực tiếp trong GitHub repo**.  
Để chạy được project, bạn cần tự tải dữ liệu về và đặt đúng thư mục như sau:

1. **Tải dữ liệu từ Google Drive**

   - Truy cập link dataset: `https://drive.google.com/drive/folders/1whRgJWkj2gMvpyzlx3Up5nB2mYB51XmM?usp=drive_link`  
   - Tải toàn bộ folder/dataset về máy (ví dụ dạng `.zip`).

2. **Giải nén và đặt vào thư mục `data/`**

   Cấu trúc thư mục sau khi giải nén và copy vào project nên giống:

   nmt-transformer/
   ├── data/
   │   ├── raw/
   │   │   ├── train.en
   │   │   ├── train.vi
   │   │   └── ...
   │   ├── processed/
   │   │   ├── train.en
   │   │   ├── train.vi
   │   │   └── ...
   │   └── spm/
   │       ├── train_combined.txt
   │       └── ...
   └── src/
       └── ...

   Chỉ cần đảm bảo các file dữ liệu (`train.en`, `train.vi`, `train_combined.txt`, v.v.) nằm đúng trong thư mục `data/` như trên.

3. **Lưu ý về Git**

   - Thư mục `data/` đã được thêm vào `.gitignore` để tránh push nhầm dataset lớn lên GitHub.
   - Khi làm việc với git, **không chạy `git add data/`**.

Sau khi hoàn thành các bước trên, bạn có thể chạy các script train/evaluate như hướng dẫn ở các mục trước.
