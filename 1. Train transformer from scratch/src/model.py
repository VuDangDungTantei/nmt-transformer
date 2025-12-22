# %% 0. SETUP & IMPORTS
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(42)

# %% 1. CORE UTILITIES
# LayerNormalization + PositionalEncoding

class LayerNormalization(nn.Module):
    """
    Thực hiện chuẩn hóa phân phối dữ liệu trên chiều đặc trưng (feature dimension).
    Áp dụng công thức chuẩn hóa: y = (x - mean) / (std + eps) * gamma + beta.
    """
    def __init__(self, d_model, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # Tính mean và std trên chiều cuối cùng (dim=-1), giữ nguyên số chiều để broadcasting
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        
        # Áp dụng chuẩn hóa và biến đổi affine (Gamma, Beta)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class PositionalEncoding(nn.Module):
    """
    Khởi tạo và bổ sung ma trận mã hóa vị trí vào tensor đầu vào.
    Sử dụng các hàm lượng giác (sin/cos) với tần số biến thiên để biểu diễn thứ tự trong chuỗi.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Tính toán trong không gian log để đảm bảo độ chính xác số học cho các số mũ lớn
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Gán giá trị Sin cho vị trí chẵn, Cos cho vị trí lẻ
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) # Thêm chiều Batch: [1, Max_Len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Cắt ma trận PE theo độ dài thực tế của câu input (x.size(1))
        # Cộng trực tiếp vào Embedding (Broadcasting chiều Batch)
        return x + self.pe[:, :x.size(1), :]

# %% 2. ATTENTION MECHANISM
# MultiHeadAttention

class MultiHeadAttention(nn.Module):
    """
    Cài đặt cơ chế chú ý đa đầu (Multi-Head Attention).
    Thực hiện chiếu tuyến tính Q, K, V, chia tách thành các heads song song và tổng hợp kết quả.
    """
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
        
        # 1. Chiếu tuyến tính & Tách Heads
        # .view(): Biến đổi [Batch, Seq, d_model] -> [Batch, Seq, n_heads, d_k]
        # .transpose(1, 2): Đổi chỗ để dim 'heads' lên trước -> [Batch, n_heads, Seq, d_k]
        # Mục đích: Để phép nhân ma trận (matmul) hoạt động song song trên từng head độc lập
        k = self.k_linear(k).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        q = self.q_linear(q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2. Scaled Dot-Product Attention
        # Nhân ma trận Q với K chuyển vị (transpose 2 chiều cuối)
        # Kết quả: [Batch, n_heads, Seq_Q, Seq_K] - Ma trận tương đồng
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Áp dụng Mask: Gán giá trị cực nhỏ (-1e9) vào các vị trí cần che
        # Khi qua Softmax, e^-1e9 xấp xỉ 0 -> Không có sự chú ý
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Tính trọng số chú ý (Attention Weights)
        attn_weights = self.dropout(F.softmax(scores, dim=-1))
        
        # Tổng hợp thông tin từ V dựa trên trọng số
        output = torch.matmul(attn_weights, v)
        
        # 3. Gộp Heads (Concatenate)
        # .transpose(1, 2): Đưa về [Batch, Seq, n_heads, d_k]
        # .contiguous().view(): Gộp 2 chiều cuối thành d_model -> [Batch, Seq, d_model]
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        
        return self.out(output)

# %% 3. FEED FORWARD NETWORK
# Position-wise FeedForward

class FeedForward(nn.Module):
    """
    Mạng nơ-ron truyền thẳng (Position-wise Feed-Forward).
    Áp dụng biến đổi phi tuyến tính độc lập trên từng vị trí của chuỗi.
    """
    def __init__(self, d_model, d_ff=2048, dropout=0.1, activation="relu"):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # Chiếu lên không gian chiều cao hơn (Expansion)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # Chiếu về không gian gốc (Compression)
        self.activation = activation

    def forward(self, x):
        if self.activation == "relu":
            x = F.relu(self.linear_1(x))
        elif self.activation == "gelu":
            x = F.gelu(self.linear_1(x))
        else:
            raise ValueError(f"Activation {self.activation} not supported")
        return self.linear_2(self.dropout(x))

# %% 4. ENCODER COMPONENTS
# EncoderLayer + Encoder

class EncoderLayer(nn.Module):
    """
    Định nghĩa luồng xử lý của một tầng Encoder.
    Điều phối luồng dữ liệu qua Self-Attention và FeedForward theo cấu trúc Pre-Norm hoặc Post-Norm.
    """
    def __init__(self, d_model, n_heads, d_ff, dropout, activation="relu", norm_first=False):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, x, mask):
        # Pre-Norm: Chuẩn hóa -> Sublayer -> Dropout -> Cộng Residual
        if self.norm_first:
            x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
            x = x + self.dropout(self.ffn(self.norm2(x)))
            
        # Post-Norm: Sublayer -> Dropout -> Cộng Residual -> Chuẩn hóa
        else:
            attn_output = self.self_attn(x, x, x, mask)
            x = self.norm1(x + self.dropout(attn_output))
            ffn_output = self.ffn(x)
            x = self.norm2(x + self.dropout(ffn_output))
        return x


class Encoder(nn.Module):
    """
    Khởi tạo chồng (stack) các lớp Encoder và xử lý embedding đầu vào.
    """
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len, activation="relu", norm_first=False):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, activation, norm_first) 
            for _ in range(n_layers)
        ])
        self.norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # Scale embedding bằng sqrt(d_model) để cân bằng variance với Positional Encoding
        x = self.embed(src) * math.sqrt(self.d_model)
        x = self.pe(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


# %% 5. DECODER COMPONENTS
# DecoderLayer + Decoder

class DecoderLayer(nn.Module):
    """
    Định nghĩa luồng xử lý của một tầng Decoder.
    Bao gồm Masked Self-Attention, Cross-Attention với Encoder và FeedForward.
    """
    def __init__(self, d_model, n_heads, d_ff, dropout, activation="relu", norm_first=False):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout, activation)
        
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.norm3 = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, x, enc_output, src_mask, tgt_mask):
        if self.norm_first:
            # 1. Masked Self-Attention
            x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask))
            # 2. Cross-Attention
            x = x + self.dropout(self.cross_attn(self.norm2(x), enc_output, enc_output, src_mask))
            # 3. Feed Forward
            x = x + self.dropout(self.ffn(self.norm3(x)))
        else:
            attn_output = self.self_attn(x, x, x, tgt_mask)
            x = self.norm1(x + self.dropout(attn_output))
            attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
            x = self.norm2(x + self.dropout(attn_output))
            ffn_output = self.ffn(x)
            x = self.norm3(x + self.dropout(ffn_output))
        return x


class Decoder(nn.Module):
    """
    Khởi tạo chồng (stack) các lớp Decoder và xử lý embedding đích.
    """
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len, activation="relu", norm_first=False):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, activation, norm_first) 
            for _ in range(n_layers)
        ])
        self.norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, enc_output, src_mask, tgt_mask):
        x = self.embed(tgt) * math.sqrt(self.d_model)
        x = self.pe(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)

# %% 6. TRANSFORMER

class Transformer(nn.Module):
    """
    Tổng hợp kiến trúc Transformer (Sequence-to-Sequence).
    Khởi tạo Encoder, Decoder và lớp chiếu đầu ra (Projection).
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len=5000, 
                 activation="relu", norm_first=False):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len, activation, norm_first)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len, activation, norm_first)
        self.projection = nn.Linear(d_model, tgt_vocab_size)
        self._init_parameters()

    def forward(self, src, tgt, src_mask, tgt_mask):
        # Mã hóa câu nguồn -> Context Vectors
        enc_output = self.encoder(src, src_mask)
        
        # Giải mã câu đích dựa trên Context
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        
        # Chiếu về kích thước từ vựng để tính xác suất (Logits)
        output = self.projection(dec_output)
        return output

    def _init_parameters(self):
        # Khởi tạo Xavier Uniform cho các tham số > 1 chiều (Weights)
        # Giúp cân bằng variance của activation giữa các lớp
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

# %% 7. MAIN EXECUTION (KIỂM THỬ)
if __name__ == "__main__":
    print("🚀 Initializing Transformer Model...")
    
    # Khởi tạo mô hình
    model = Transformer(
        src_vocab_size=100, 
        tgt_vocab_size=100, 
        d_model=64, 
        n_layers=2, 
        n_heads=4, 
        d_ff=128, 
        dropout=0.1,
        activation="relu", 
        norm_first=True
    )
    
    # Tạo Batch giả lập: 2 câu, mỗi câu 10 từ
    src = torch.randint(0, 100, (2, 10))
    tgt = torch.randint(0, 100, (2, 10))
    src_mask = torch.ones(2, 1, 1, 10)
    tgt_mask = torch.ones(2, 1, 10, 10)
    
    # Chạy Forward Pass
    output = model(src, tgt, src_mask, tgt_mask)
    print(f"Output shape: {output.shape} -> ✅ Execution Successful")