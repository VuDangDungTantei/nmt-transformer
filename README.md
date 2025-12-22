# Neural Machine Translation Transformer (En–Vi) và Fine tune trên dữ liệu VLSP Medical

Dự án này triển khai **pipeline dữ liệu** cho hệ thống dịch máy Anh–Việt (Neural Machine Translation) dùng kiến trúc Transformer **tự code from scratch** (70%) và fine tune trên dữ liệu VLSP medical (30%)

---

## 1. Triển khai Transformer From Scratch

## 2. Fine-tune dữ liệu medical trên Transformer tự huấn luyện

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

VI→EN:
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

### Output
- EN→VI adapter: `/kaggle/working/vlsp_en2vi_run/lora_en2vi_qwen2.5_1.5b`
- VI→EN adapter: `/kaggle/working/vlsp_vi2en_run/lora_vi2en_qwen2.5_1.5b`

(Ghi chú) BLEU evaluation nằm ở các notebook test (`qwen-fine-tune-vlsp-*-test.ipynb`), có bước **cắt prompt khỏi output** trước khi decode để tránh BLEU sai.

## 4. Link data sử dụng cho dự án