import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast # Tăng tốc độ train
import time
import math
import os
from pathlib import Path
from tqdm import tqdm # Thanh tiến trình đẹp mắt

# Import modules của dự án
from dataset import NMTDataset, collate_fn
from tokenizer import SubwordTokenizer
from model import Transformer

# ==============================================================================
# 1. CẤU HÌNH TRUNG TÂM (CHỈ CẦN CHỈNH SỬA Ở ĐÂY)
# ==============================================================================
class Config:
    # --- Đường dẫn ---
    data_dir = "./data/processed"
    model_path = "./data/spm/spm_unigram.model"
    save_dir = "./checkpoints"
    
    # --- Tham số Model (Tùy chỉnh độ lớn model) ---
    # Cấu hình Small (phù hợp IWSLT/Colab Free): d_model=256, n_layers=3, n_heads=4
    # Cấu hình Base (nếu GPU mạnh): d_model=512, n_layers=6, n_heads=8
    d_model = 256
    n_layers = 3
    n_heads = 4
    d_ff = 1024       # Thường gấp 4 lần d_model
    dropout = 0.1     # Tăng lên 0.3 nếu thấy bị Overfitting
    max_len = 100     # Độ dài câu tối đa
    
    # --- Tham số Huấn luyện ---
    batch_size = 64   # Tăng lên 128 nếu GPU chịu được, giảm xuống 32 nếu lỗi RAM
    num_epochs = 20
    lr = 0.0001       # Learning rate cơ bản (sẽ được Scheduler điều chỉnh)
    label_smoothing = 0.1 # Giúp model không quá tự tin, học tốt hơn
    
    # Tự động chọn thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tạo thư mục lưu model
Path(Config.save_dir).mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 2. CÁC HÀM TIỆN ÍCH (HELPER FUNCTIONS)
# ==============================================================================

def make_masks(src, tgt, pad_id):
    """Tạo mask che padding và mask che tương lai (Look-ahead)"""
    # Source Mask: [Batch, 1, 1, SrcLen]
    src_mask = (src != pad_id).unsqueeze(1).unsqueeze(2)

    # Target Mask: [Batch, 1, TgtLen, TgtLen]
    tgt_pad_mask = (tgt != pad_id).unsqueeze(1).unsqueeze(2)
    tgt_len = tgt.size(1)
    nopeak_mask = torch.tril(torch.ones((1, 1, tgt_len, tgt_len), device=src.device)).bool()
    
    tgt_mask = tgt_pad_mask & nopeak_mask
    return src_mask, tgt_mask

class NoamLR:
    """
    Learning Rate Scheduler chuẩn cho Transformer (Warmup -> Decay).
    Giúp model không bị "sốc" gradient lúc đầu và hội tụ sâu về sau.
    """
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = lr
    
    def rate(self, step=None):
        if step is None: step = self.current_step
        if step == 0: step = 1
        return (self.d_model ** -0.5) * min(step ** -0.5, step * self.warmup_steps ** -1.5)


# ==============================================================================
# 3. HÀM HUẤN LUYỆN & KIỂM THỬ (TRAIN & EVALUATE)
# ==============================================================================

def run_epoch(model, dataloader, optimizer, criterion, device, is_train=True, scheduler=None):
    """Hàm chạy chung cho cả Train và Validation"""
    model.train() if is_train else model.eval()
    
    total_loss = 0
    # Thanh tiến trình (tqdm)
    desc = "Training" if is_train else "Validating"
    pbar = tqdm(dataloader, desc=desc, leave=False)
    
    # Mixed Precision Scaler (Chỉ dùng khi train)
    scaler = GradScaler() if is_train else None

    for batch in pbar:
        src = batch['src_ids'].to(device)
        tgt_in = batch['tgt_in_ids'].to(device)
        tgt_out = batch['tgt_out_ids'].to(device)
        
        # Giả định pad_id = 0 (cần khớp với tokenizer)
        pad_id = 0 
        src_mask, tgt_mask = make_masks(src, tgt_in, pad_id)
        
        # --- FORWARD PASS ---
        with autocast(enabled=is_train): # Tự động chuyển float32 -> float16 để nhanh hơn
            output = model(src, tgt_in, src_mask, tgt_mask)
            
            # Reshape để tính loss: [Batch * Seq, Vocab]
            loss = criterion(output.view(-1, output.size(-1)), tgt_out.view(-1))

        if is_train:
            # --- BACKWARD PASS ---
            optimizer.zero_grad()
            scaler.scale(loss).backward() # Scale loss để tránh underflow
            
            # Gradient Clipping (Tránh bùng nổ gradient)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Optimizer Step
            scaler.step(optimizer)
            scaler.update()
            
            # Cập nhật LR Scheduler
            if scheduler:
                scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


# ==============================================================================
# 4. MAIN PROGRAM
# ==============================================================================
def main():
    print(f"🚀 Device: {Config.device}")
    
    # 1. Load Tokenizer & Vocab
    tokenizer = SubwordTokenizer(Config.model_path)
    vocab_size = tokenizer.sp.get_vocab_size()
    print(f"Dataset Vocab Size: {vocab_size}")

    # 2. Prepare Datasets & DataLoaders
    # Collator giúp padding batch
    collator = lambda b: collate_fn(b, tokenizer.pad_id)
    
    train_dataset = NMTDataset(Config.data_dir, "train", tokenizer, Config.max_len, Config.max_len)
    valid_dataset = NMTDataset(Config.data_dir, "tst2013", tokenizer, Config.max_len, Config.max_len) # Dùng tst2013 làm valid

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, collate_fn=collator, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=Config.batch_size, shuffle=False, collate_fn=collator, num_workers=2)

    print(f"Train batches: {len(train_loader)} | Valid batches: {len(valid_loader)}")

    # 3. Initialize Model
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=Config.d_model,
        n_layers=Config.n_layers,
        n_heads=Config.n_heads,
        d_ff=Config.d_ff,
        dropout=Config.dropout,
        max_len=Config.max_len
    ).to(Config.device)

    # 4. Optimization Setup
    # Label Smoothing: Kỹ thuật giúp model học tốt hơn
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id, label_smoothing=Config.label_smoothing)
    
    # Optimizer chuẩn cho Transformer: Adam với betas=(0.9, 0.98)
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    
    # Scheduler: Warmup trong 4000 bước đầu
    lr_scheduler = NoamLR(optimizer, d_model=Config.d_model, warmup_steps=4000)

    # 5. Training Loop
    best_valid_loss = float('inf')
    
    print("\n🏁 START TRAINING...")
    for epoch in range(1, Config.num_epochs + 1):
        start_time = time.time()
        
        # --- TRAIN ---
        train_loss = run_epoch(model, train_loader, optimizer, criterion, Config.device, is_train=True, scheduler=lr_scheduler)
        
        # --- VALIDATE ---
        # Tắt gradient khi valid để tiết kiệm bộ nhớ
        with torch.no_grad():
            valid_loss = run_epoch(model, valid_loader, optimizer, criterion, Config.device, is_train=False)
            
        epoch_mins = (time.time() - start_time) / 60
        
        # --- LOGGING & SAVING ---
        print(f"Epoch {epoch:02d} | Time: {epoch_mins:.1f}m")
        print(f"\tTrain Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")
        
        # Chỉ lưu model nếu Valid Loss giảm (Tốt nhất từ trước đến nay)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_path = f"{Config.save_dir}/best_model.pt"
            torch.save(model.state_dict(), save_path)
            print(f"\t✅ Saved Best Model to {save_path}")
        
        # Lưu checkpoint định kỳ (để dự phòng)
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"{Config.save_dir}/checkpoint_ep{epoch}.pt")

if __name__ == "__main__":
    main()