# %% 0.IMPORTS
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# %% 1. POSITIONAL ENCODING
# Vì Transformer không có RNN nên cần cộng vector này để biết thứ tự từ.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Tạo ma trận [max_len, d_model] chứa toàn số 0
        pe = torch.zeros(max_len, d_model)
        
        # Tạo vector vị trí [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Tính mẫu số (div_term) cho hàm sin/cos
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Áp dụng công thức: chẵn dùng Sin, lẻ dùng Cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Thêm chiều batch: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # Lưu vào buffer (không train, nhưng vẫn được lưu cùng model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        # Cộng PE vào x (cắt đúng độ dài câu hiện tại)
        return x + self.pe[:, :x.size(1), :]

# %% 2. MULTI-HEAD ATTENTION
# Cơ chế giúp mô hình tập trung vào các phần khác nhau của câu cùng lúc.
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model phải chia hết cho n_heads"
        self.d_model = d_model
        self.d_k = d_model // n_heads  # Kích thước mỗi head
        self.n_heads = n_heads
        
        # Các lớp Linear để chiếu Q, K, V
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Lớp Linear cuối cùng sau khi gộp các heads
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0) # Batch size
        
        # 1. Chiếu Linear và tách thành n_heads
        # Shape: [batch_size, seq_len, n_heads, d_k] -> transpose -> [batch_size, n_heads, seq_len, d_k]
        k = self.k_linear(k).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        q = self.q_linear(q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2. Tính Scaled Dot-Product Attention
        # scores = (Q * K^T) / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 3. Áp dụng Mask (nếu có) - Che đi các vị trí padding hoặc tương lai
        if mask is not None:
            # mask == 0 nghĩa là vị trí đó cần che, gán giá trị rất nhỏ (-1 tỷ)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 4. Softmax để lấy trọng số attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 5. Nhân với V
        output = torch.matmul(attn_weights, v)
        
        # 6. Gộp (Concatenate) các heads lại
        # Shape: [batch_size, seq_len, n_heads * d_k] = [batch_size, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        
        # 7. Đi qua lớp Linear cuối
        return self.out(output)

# %% 3. FEED FORWARD NETWORK
# Mạng nơ-ron đơn giản xử lý từng vị trí riêng biệt
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # Linear -> ReLU -> Dropout -> Linear
        return self.linear_2(self.dropout(F.relu(self.linear_1(x))))
    
# %% 4. ENCODER LAYER
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        # Layer Norm giúp ổn định quá trình huấn luyện
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # --- Khối 1: Self-Attention ---
        # Input x đi vào Attention
        # 3 tham số đầu vào đều là x vì đây là Self-Attention (tự nhìn chính mình)
        attn_output = self.self_attn(x, x, x, mask)
        
        # Residual Connection (Cộng x cũ) + Layer Norm
        # Công thức: Norm(x + Dropout(Sublayer(x)))
        x = self.norm1(x + self.dropout(attn_output))
        
        # --- Khối 2: Feed Forward ---
        # Output trên đi vào FFN
        ffn_output = self.ffn(x)
        
        # Residual Connection + Layer Norm
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x

# %% 5. ENCODER (CONTAINER)
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len):
        """
        Args:
            vocab_size: Kích thước từ điển nguồn (ví dụ 8000)
            d_model: Kích thước vector (ví dụ 512)
            n_layers: Số lượng lớp Encoder chồng lên nhau (ví dụ 6)
        """
        super(Encoder, self).__init__()
        self.d_model = d_model
        
        # 1. Embedding: Chuyển ID từ thành Vector
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # 2. Positional Encoding: Thêm thông tin vị trí
        self.pe = PositionalEncoding(d_model, max_len)
        
        # 3. Stack N lớp EncoderLayer bằng ModuleList
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        
        # 4. Norm cuối cùng trước khi xuất sang Decoder
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src shape: [batch_size, seq_len]
        
        # Bước 1: Embedding
        x = self.embed(src)
        
        # Bước 2: Scaling (Mẹo quan trọng trong paper Transformer)
        # Nhân vector với căn bậc 2 của d_model để giá trị không bị quá nhỏ so với PE
        x = x * math.sqrt(self.d_model)
        
        # Bước 3: Cộng PE
        x = self.pe(x)
        x = self.dropout(x)
        
        # Bước 4: Chạy qua từng lớp Encoder
        for layer in self.layers:
            x = layer(x, src_mask)
            
        # Trả về output đã được chuẩn hóa
        return self.norm(x)
    
# %% 6. DECODER LAYER
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        
        # 1. Masked Self-Attention (Cho chính câu đích)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # 2. Cross-Attention (Quan trọng: Nhìn sang Encoder Output)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # 3. Feed Forward
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        # 3 lớp Norm cho 3 khối trên
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # x: Input của Decoder (hoặc output của layer trước)
        # enc_output: Output từ Encoder (dùng cho Cross-Attention)
        # src_mask: Che padding của Encoder
        # tgt_mask: Che tương lai của Decoder (Look-ahead mask)
        
        # --- Khối 1: Masked Self-Attention ---
        # tgt_mask ở đây rất quan trọng (dạng tam giác) để che các từ tương lai
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # --- Khối 2: Cross-Attention ---
        # Query là x (từ decoder)
        # Key/Value là enc_output (từ encoder) -> Đây là chỗ Decoder "đọc hiểu" câu nguồn
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # --- Khối 3: Feed Forward ---
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        
        return x

# %% 7. DECODER (CONTAINER)
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len):
        super(Decoder, self).__init__()
        self.d_model = d_model
        
        # 1. Embedding & PE
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        
        # 2. Stack N lớp DecoderLayer
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        
        # 3. Norm cuối cùng
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, enc_output, src_mask, tgt_mask):
        # tgt: [batch_size, seq_len]
        
        # Embedding + Scaling + PE
        x = self.embed(tgt) * math.sqrt(self.d_model)
        x = self.pe(x)
        x = self.dropout(x)
        
        # Qua từng lớp Decoder
        for layer in self.layers:
            # Truyền enc_output vào từng lớp để làm Cross-Attention
            x = layer(x, enc_output, src_mask, tgt_mask)
            
        return self.norm(x)


# %% 8. TRANSFORMER (FULL MODEL)
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len=5000):
        super(Transformer, self).__init__()
        
        # Khởi tạo Encoder và Decoder
        self.encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len)
        
        # Lớp chiếu cuối cùng: Chuyển đổi từ d_model sang kích thước từ điển đích
        # Ví dụ: 512 -> 8000 từ
        self.projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Khởi tạo trọng số (Xavier Initialization) giúp model hội tụ nhanh hơn
        self._init_parameters()

    def forward(self, src, tgt, src_mask, tgt_mask):
        # 1. Qua Encoder
        # enc_output shape: [batch_size, src_len, d_model]
        enc_output = self.encoder(src, src_mask)
        
        # 2. Qua Decoder
        # dec_output shape: [batch_size, tgt_len, d_model]
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        
        # 3. Qua lớp chiếu cuối cùng
        # output shape: [batch_size, tgt_len, tgt_vocab_size]
        output = self.projection(dec_output)
        
        return output

    def encode(self, src, src_mask):
        # Hàm phụ dùng khi Inference (dịch thử)
        return self.encoder(src, src_mask)

    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        # Hàm phụ dùng khi Inference
        return self.decoder(tgt, enc_output, src_mask, tgt_mask)

    def _init_parameters(self):
        # Khởi tạo trọng số Xavier Uniform cho các tham số
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


# %% 9. TEST BLOCK (Chạy thử để kiểm tra)
if __name__ == "__main__":
    print("🚀 Đang kiểm tra toàn bộ kiến trúc Transformer...")
    
    # 1. Giả lập tham số
    src_vocab_size = 100
    tgt_vocab_size = 100
    d_model = 512
    n_layers = 2 # Test ít lớp cho nhanh
    n_heads = 8
    d_ff = 2048
    dropout = 0.1
    max_len = 50
    
    # 2. Khởi tạo Model
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len)
    print("✅ Khởi tạo Model thành công!")

    # 3. Tạo dữ liệu giả (Batch size = 2, Seq len = 10)
    src = torch.randint(0, src_vocab_size, (2, 10))
    tgt = torch.randint(0, tgt_vocab_size, (2, 10))
    
    # Tạo Mask giả (Test kỹ thuật Masking sau, giờ test kích thước trước)
    src_mask = torch.ones(2, 1, 1, 10) # Che padding
    tgt_mask = torch.ones(2, 1, 10, 10) # Che tương lai (Look-ahead)

    # 4. Forward Pass
    try:
        out = model(src, tgt, src_mask, tgt_mask)
        print(f"✅ Forward pass thành công! Output shape: {out.shape}")
        
        # Kiểm tra kích thước cuối cùng
        expected_shape = (2, 10, tgt_vocab_size)
        if out.shape == expected_shape:
            print("🎉 CHÚC MỪNG! Kiến trúc Transformer From Scratch đã hoàn thiện chuẩn xác.")
        else:
            print(f"❌ Sai kích thước. Nhận được {out.shape}, kỳ vọng {expected_shape}")
            
    except Exception as e:
        print(f"❌ Lỗi khi chạy Forward: {e}")
        import traceback
        traceback.print_exc()
