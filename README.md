# Neural Machine Translation Transformer (En–Vi) và Fine tune trên dữ liệu VLSP Medical

Dự án này triển khai **pipeline dữ liệu** cho hệ thống dịch máy Anh–Việt (Neural Machine Translation) dùng kiến trúc Transformer **tự code from scratch** (70%) và fine tune trên dữ liệu VLSP medical (30%)

---

## 1. Triển khai Transformer From Scratch (En–Vi)

Xây dựng và huấn luyện mô hình dịch máy Seq2Seq dựa trên kiến trúc **Transformer Encoder–Decoder** theo hướng “from scratch” (tự cài đặt Multi-Head Attention, Positional Encoding, Encoder/Decoder blocks bằng PyTorch thuần). Mô hình được huấn luyện cho bài toán dịch **EN→VI** trên dữ liệu tổng hợp từ **IWSLT15 + TED2020**.

### Data

#### Input (raw)
- `data/raw/train.{en,vi}`: tập huấn luyện
- `data/raw/valid.{en,vi}`: tập validation
- `data/raw/test.{en,vi}`: tập test

#### Output (processed)
Sau khi lọc và làm sạch (loại dòng rỗng, chuẩn hoá Unicode, lọc độ dài/độ lệch bất thường), tạo:
- `data/processed/train.{en,vi}`
- `data/processed/valid.{en,vi}`
- `data/processed/test.{en,vi}`

#### Tokenizer (SentencePiece)
Dùng **SentencePiece** với **shared vocabulary** cho cả EN và VI:
- `data/spm/spm.model`
- `data/spm/spm.vocab`

---

### Scripts / Modules liên quan
- `src/models/`: hiện thực kiến trúc Transformer (Attention, Positional Encoding, Encoder/Decoder layers)
- `src/dataset.py`: đọc dữ liệu, encode token, tạo **padding mask** và **causal mask**, `collate_fn`
- `trainning-phan1.ipynb`: training loop, logging, lưu checkpoint


---

### Cấu hình chính 

#### (A) Kiến trúc mô hình
- Transformer theo kiến trúc Encoder–Decoder, mỗi block gồm Multi-Head Attention, Feed-Forward, residual connection và LayerNorm.
- Có cả hai biến thể Pre-Norm và Post-Norm trong kiến trúc để đối chiếu khi huấn luyện.
- Decoder dùng đầy đủ mask để đảm bảo tự hồi quy và không attend vào token PAD.
- Dùng dropout và label smoothing để tăng ổn định và cải thiện tổng quát hoá.

#### (B) Tokenizer & Pipeline dữ liệu
- Dùng SentencePiece với shared vocabulary cho cả EN và VI để giảm OOV và xử lý tốt từ mới.
- Dữ liệu pad động theo batch, khi tính loss thì bỏ qua PAD.
- Target tách thành dec_in và dec_out, dec_in bắt đầu bằng BOS và dec_out kết thúc bằng EOS để train theo teacher forcing.

#### (C) Optimizer & Schedule
- Tối ưu bằng Adam hoặc AdamW.
- Dùng Noam schedule với warmup để học ổn định, learning rate tăng ở giai đoạn đầu và giảm dần về sau.
- Dùng AMP để tăng tốc và tiết kiệm bộ nhớ, khi cần thì kèm gradient clipping để tránh nổ gradient.

---



## 2. Fine-tune dữ liệu medical trên Transformer tự huấn luyện

Fine-tune checkpoint Transformer EN→VI (đã train general domain) để thích nghi miền **y khoa** bằng dữ liệu VLSP Medical.

### Data

#### Input (raw)  
- `data/raw/train.en.txt`
- `data/raw/train.vi.txt`
- `data/raw/public_test.en.txt`
- `data/raw/public_test.vi.txt`

#### Output (processed, align 1-1 theo dòng)
Sau preprocess sẽ tạo:
- `data/processed/train.{en,vi}`
- `data/processed/dev.{en,vi}`
- `data/processed/test.{en,vi}`
- `data/processed/stats.json`

#### Tokenizer (SentencePiece)
Sau khi train SPM sẽ tạo:
- `data/spm/spm.model`
- `data/spm/spm.vocab`
- `data/spm/train.all.txt` (file gộp để train tokenizer)

---

### Scripts / Notebooks liên quan
- `organize_vlsp_data_txt.py`: clean + filter + split `dev` (2%) + export `data/processed/*`
- `train_spm_vlsp.py`: train SentencePiece từ `data/processed/train.{en,vi}` → `data/spm/spm.model`
- `notebook/trainning-nlp-p2.ipynb`: fine-tune Transformer từ checkpoint pretrained
- `notebook/best_transformer_v3.pt`: checkpoint pretrained dùng để khởi tạo fine-tune
- `src/{dataset.py, model.py, tokenizer.py}`: core code (Dataset/Model/Tokenizer) mà notebook import

---

### Cấu hình chính

#### (A) Preprocess (trong `organize_vlsp_data_txt.py`)
- Normalize: NFC + gộp whitespace
- Split dev: `DEV_RATIO=0.02`, `SEED=42`
- Filter:
  - `MAX_LEN=200` (đếm token thô theo whitespace)
  - `MAX_RATIO=9.0` (lọc cặp lệch độ dài)
- Input file name:
  - `train.en.txt`, `train.vi.txt`, `public_test.en.txt`, `public_test.vi.txt`
- Output file name:
  - `train.en/vi`, `dev.en/vi`, `test.en/vi`, `stats.json`

#### (B) Tokenizer (trong `train_spm_vlsp.py` + `src/tokenizer.py`)
- `vocab_size=8000`, `model_type=unigram`, `character_coverage=1.0`
- **Special token IDs cố định** (để khớp tokenizer/model):
  - `pad_id=0`, `unk_id=1`, `bos_id=2`, `eos_id=3`

#### (C) Dataset encode (trong `src/dataset.py`)
- `src`: encode và **thêm EOS** (`add_eos=True`)
- `tgt`: tạo `dec_in=[BOS ...]` và `dec_out=[... EOS]`
- `collate_fn`: pad theo `pad_id=0` + tạo padding mask

---

### Train

#### 1) Preprocess dữ liệu
Chạy tại `project_root/` (nơi có `data/`):
```bash
python organize_vlsp_data_txt.py
```

#### 2) Train tokenizer SentencePiece
Mặc định script đọc `data/processed/train.{en,vi}` và xuất `data/spm/spm.{model,vocab}`:
```bash
python train_spm_vlsp.py
```

(Option) đổi vocab/model_type:
```bash
python train_spm_vlsp.py --vocab_size 12000 --model_type bpe
```

#### 3) Fine-tune Transformer (notebook)
Mở và chạy:
- `notebook/trainning-nlp-p2.ipynb` → **Run All**

Trong notebook, đảm bảo các path trỏ đúng theo cây thư mục này:
- `PROCESSED_DIR = "data/processed"`
- `SPM_PATH = "data/spm/spm.model"`
- `CKPT_PATH = "notebook/best_transformer_v3.pt"`

---

### Output
- Processed data:
  - `data/processed/{train,dev,test}.{en,vi}`
  - `data/processed/stats.json`
- Tokenizer:
  - `data/spm/spm.model`
  - `data/spm/spm.vocab`
- Fine-tuned checkpoint (best):
  - (theo notebook) thường lưu vào `notebook/` hoặc thư mục output trong cell save, ví dụ: `notebook/best_finetune.pt`

---

### Ghi chú
- **Tokenizer phải khớp checkpoint**: nếu checkpoint pretrained của bạn train với `spm.model` khác, hãy dùng đúng `spm.model` đó khi fine-tune (không đổi vocab).
- Preprocess yêu cầu **align 1-1 theo dòng** giữa EN và VI (lệch số dòng sẽ báo lỗi).

## 3. Fine-tune dữ liệu medical trên Qwen2.5-1.5B-Instruct với LoRA

Fine-tune (SFT) `Qwen/Qwen2.5-1.5B-Instruct` bằng **LoRA** cho dịch y khoa 2 chiều **EN→VI** và **VI→EN** (DDP với `torchrun`).

### Data
Input (đã processed, align 1-1 theo dòng):
- `/kaggle/input/data-vlsp/processed/{train,valid,test}.{en,vi}`

### Notebooks liên quan
- `qwen-medical-data-analysis.ipynb`: QC dữ liệu + thống kê độ dài/token + chuẩn bị dữ liệu SFT.
- `qwen-fine-tune-vlsp-en2vi-train.ipynb`: train LoRA EN→VI (tạo `/kaggle/working/train_qwen_en2vi_lora.py`).
- `qwen-fine-tune-vlsp-vi2en-train.ipynb`: train LoRA VI→EN (tạo `/kaggle/working/train_qwen_vi2en_lora.py`).

### Cấu hình chính
- Prompt instruction “professional medical translator”; học theo dạng completion.
- Mask label `-100` cho phần prompt, chỉ học phần target + `eos_token`.
- LoRA target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Hyperparams (theo notebook): `max_seq_length=320`, `bs=4`, `grad_accum=8`, `lr=2e-4`, `max_steps=8000`, `eval/save=800`, `lora_dropout=0.05`
- Lưu ý: giữ tham số trainable (LoRA) ở **fp32** để ổn định AMP/GradScaler.

### Train (DDP)
EN→VI:
```bash
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 /kaggle/working/train_qwen_en2vi_lora.py \
  --model_id "Qwen/Qwen2.5-1.5B-Instruct" \
  --dataset_dir "/kaggle/working/vlsp_en2vi_run/dataset_en2vi_raw" \
  --output_dir "/kaggle/working/vlsp_en2vi_run/lora_en2vi_qwen2.5_1.5b" \
  --max_seq_length 320 \
  --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 --lora_dropout 0.05 \
  --max_steps 8000 \
  --eval_steps 800 --save_steps 800 --logging_steps 50
```
VI→EN:
```bash
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 /kaggle/working/train_qwen_vi2en_lora.py \
  --model_id "Qwen/Qwen2.5-1.5B-Instruct" \
  --dataset_dir "/kaggle/working/vlsp_vi2en_run/dataset_vi2en_raw" \
  --output_dir "/kaggle/working/vlsp_vi2en_run/lora_vi2en_qwen2.5_1.5b" \
  --max_seq_length 320 \
  --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 --lora_dropout 0.05 \
  --max_steps 8000 \
  --eval_steps 800 --save_steps 800 --logging_steps 50
```
### Output
- EN→VI adapter: `/kaggle/working/vlsp_en2vi_run/lora_en2vi_qwen2.5_1.5b`
- VI→EN adapter: `/kaggle/working/vlsp_vi2en_run/lora_vi2en_qwen2.5_1.5b`

(Ghi chú) BLEU evaluation nằm ở các notebook test (`qwen-fine-tune-vlsp-*-test.ipynb`), có bước **cắt prompt khỏi output** trước khi decode để tránh BLEU sai.

## 4. Link data sử dụng cho dự án

- Link GGDrive data dùng cho phần train transformer from scratch: https://drive.google.com/file/d/1Y9MjBf03auNa5M8PqjsIRblN9k1bsyOW/view?usp=sharing
- Link GGDrive data dùng cho phần Fine-tune từ transformer tự phần 1: https://drive.google.com/drive/folders/1PfbaI7k-GGSGXCZAf0nr8b-pIZ94FuT0?usp=sharing
- Link GGDrive data dùng cho phần FIne-tune với Qwen/Qwen2.5-1.5B-Instruct: https://drive.google.com/file/d/1EYK0LBb8KSl3FUuFODRy2JtE3-ZfsTv8/view?usp=sharing
