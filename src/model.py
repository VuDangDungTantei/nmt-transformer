# %% 0.IMPORTS
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# %% 1. POSITIONAL ENCODING (Giữ nguyên)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# %% 2. MULTI-HEAD ATTENTION (Giữ nguyên)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model phải chia hết cho n_heads"
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        q = self.q_linear(q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = self.dropout(F.softmax(scores, dim=-1))
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        return self.out(output)

# %% 3. FEED FORWARD NETWORK (ĐÃ NÂNG CẤP)
# Hỗ trợ tùy chọn activation: relu hoặc gelu
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1, activation="relu"):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.activation = activation # Lưu cấu hình

    def forward(self, x):
        # Chọn hàm kích hoạt dựa trên tham số
        if self.activation == "relu":
            x = F.relu(self.linear_1(x))
        elif self.activation == "gelu":
            x = F.gelu(self.linear_1(x))
        else:
            raise ValueError(f"Activation {self.activation} không được hỗ trợ.")
            
        return self.linear_2(self.dropout(x))

# %% 4. ENCODER LAYER (ĐÃ NÂNG CẤP)
# Hỗ trợ Pre-Norm (norm_first=True) và Post-Norm (norm_first=False)
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, activation="relu", norm_first=False):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first # Lưu cấu hình

    def forward(self, x, mask):
        # --- Biến thể 1: PRE-NORM (Hiện đại - GPT/Llama) ---
        # Norm trước, rồi mới vào sublayer. Cộng residual sau cùng.
        if self.norm_first:
            x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
            x = x + self.dropout(self.ffn(self.norm2(x)))
            
        # --- Biến thể 2: POST-NORM (Gốc - Vaswani 2017) ---
        # Sublayer trước, cộng residual, rồi mới Norm.
        else:
            attn_output = self.self_attn(x, x, x, mask)
            x = self.norm1(x + self.dropout(attn_output))
            ffn_output = self.ffn(x)
            x = self.norm2(x + self.dropout(ffn_output))
            
        return x

# %% 5. ENCODER (CONTAINER) (Cập nhật truyền tham số)
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len, activation="relu", norm_first=False):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            # Truyền activation và norm_first xuống từng layer
            EncoderLayer(d_model, n_heads, d_ff, dropout, activation, norm_first) 
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        x = self.embed(src) * math.sqrt(self.d_model)
        x = self.pe(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

# %% 6. DECODER LAYER (ĐÃ NÂNG CẤP)
# Tương tự EncoderLayer: Hỗ trợ Pre/Post Norm và Activation
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, activation="relu", norm_first=False):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # --- Biến thể 1: PRE-NORM ---
        if self.norm_first:
            # Block 1: Masked Self-Attention
            x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask))
            # Block 2: Cross-Attention
            x = x + self.dropout(self.cross_attn(self.norm2(x), enc_output, enc_output, src_mask))
            # Block 3: FFN
            x = x + self.dropout(self.ffn(self.norm3(x)))
            
        # --- Biến thể 2: POST-NORM ---
        else:
            # Block 1
            attn_output = self.self_attn(x, x, x, tgt_mask)
            x = self.norm1(x + self.dropout(attn_output))
            # Block 2
            attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
            x = self.norm2(x + self.dropout(attn_output))
            # Block 3
            ffn_output = self.ffn(x)
            x = self.norm3(x + self.dropout(ffn_output))
            
        return x

# %% 7. DECODER (CONTAINER) (Cập nhật truyền tham số)
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len, activation="relu", norm_first=False):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, activation, norm_first) 
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, enc_output, src_mask, tgt_mask):
        x = self.embed(tgt) * math.sqrt(self.d_model)
        x = self.pe(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)

# %% 8. TRANSFORMER (FULL MODEL) (ĐÃ NÂNG CẤP)
# Nhận tham số từ ngoài và truyền sâu vào trong
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len=5000, 
                 activation="relu", norm_first=False): # <--- 2 tham số mới
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len, activation, norm_first)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len, activation, norm_first)
        self.projection = nn.Linear(d_model, tgt_vocab_size)
        self._init_parameters()

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        output = self.projection(dec_output)
        return output

    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)

    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        return self.decoder(tgt, enc_output, src_mask, tgt_mask)

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

# %% 9. TEST CÁC BIẾN THỂ (Thử nghiệm ngay tại đây)
if __name__ == "__main__":
    print("🚀 Đang kiểm tra các biến thể Transformer...")
    
    # Setup chung
    src_vocab = 100; tgt_vocab = 100
    d_model = 64; layers = 2; heads = 4; d_ff = 128; dropout = 0.1
    
    # --- TEST 1: Baseline (Gốc) ---
    print("\n🔹 Test 1: Baseline (Post-Norm + ReLU)")
    model1 = Transformer(src_vocab, tgt_vocab, d_model, layers, heads, d_ff, dropout, 
                         activation="relu", norm_first=False)
    # Chạy thử forward
    src = torch.randint(0, 100, (2, 10))
    tgt = torch.randint(0, 100, (2, 10))
    src_mask = torch.ones(2, 1, 1, 10)
    tgt_mask = torch.ones(2, 1, 10, 10)
    out1 = model1(src, tgt, src_mask, tgt_mask)
    print(f"Output shape: {out1.shape} -> ✅ OK")

    # --- TEST 2: Modern (Hiện đại) ---
    print("\n🔹 Test 2: Modern (Pre-Norm + GeLU)")
    model2 = Transformer(src_vocab, tgt_vocab, d_model, layers, heads, d_ff, dropout, 
                         activation="gelu", norm_first=True)
    out2 = model2(src, tgt, src_mask, tgt_mask)
    print(f"Output shape: {out2.shape} -> ✅ OK")
    
    print("\n🎉 Tất cả các biến thể đã hoạt động ổn định!")