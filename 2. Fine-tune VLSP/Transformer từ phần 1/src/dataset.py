from pathlib import Path
from typing import List, Tuple, Dict, Any
import torch
from torch.utils.data import Dataset

from tokenizer import SubwordTokenizer  # dùng class bạn đã tạo


class NMTDataset(Dataset):
    """
    Dataset cho bài toán dịch máy En-Vi với Transformer.
    Đọc dữ liệu từ data/processed, encode bằng SentencePiece, truncate theo max_len.
    """
    def __init__(
        self,
        data_dir: str,
        split: str,
        tokenizer: SubwordTokenizer,
        max_src_len: int = 70,
        max_tgt_len: int = 70,
    ) -> None:
        """
        Args:
            data_dir: thư mục chứa processed data (vd: "../data/processed")
            split: "train", "valid", hoặc "test"
            tokenizer: instance của SubwordTokenizer
            max_src_len: độ dài tối đa cho câu nguồn (src)
            max_tgt_len: độ dài tối đa cho câu đích (tgt)
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        src_path = self.data_dir / f"{split}.en"
        tgt_path = self.data_dir / f"{split}.vi"

        # Đọc toàn bộ câu vào list
        with open(src_path, encoding="utf-8") as f:
            self.src_sents = [line.strip() for line in f if line.strip()]
        with open(tgt_path, encoding="utf-8") as f:
            self.tgt_sents = [line.strip() for line in f if line.strip()]

        assert len(self.src_sents) == len(self.tgt_sents), \
            f"src and tgt size mismatch: {len(self.src_sents)} vs {len(self.tgt_sents)}"

    def __len__(self) -> int:
        return len(self.src_sents)

    def _truncate(self, ids: List[int], max_len: int) -> List[int]:
        if len(ids) <= max_len:
            return ids
        # Truncate từ cuối, vẫn giữ BOS/EOS ở đầu/cuối (nếu đã chèn)
        return ids[:max_len]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        src_text = self.src_sents[idx]
        tgt_text = self.tgt_sents[idx]

        # src: thường không cần BOS/EOS, tuỳ thiết kế
        src_ids = self.tokenizer.encode_src(src_text, add_bos=False, add_eos=True)
        src_ids = self._truncate(src_ids, self.max_src_len)

        # tgt: cần cả dec_in (BOS ...), dec_out (... EOS)
        dec_in, dec_out = self.tokenizer.encode_tgt(tgt_text)
        dec_in = self._truncate(dec_in, self.max_tgt_len)
        dec_out = self._truncate(dec_out, self.max_tgt_len)

        return {
            "src_ids": src_ids,
            "tgt_in_ids": dec_in,
            "tgt_out_ids": dec_out,
        }


def collate_fn(batch: List[Dict[str, Any]], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    batch: list các dict {"src_ids": [...], "tgt_in_ids": [...], "tgt_out_ids": [...]}
    Trả về:
        src_ids: (B, S)
        tgt_in_ids: (B, T)
        tgt_out_ids: (B, T)
        src_padding_mask: (B, S)  # True tại vị trí pad
        tgt_padding_mask: (B, T)
    """
    # Lấy độ dài lớn nhất trong batch
    max_src_len = max(len(item["src_ids"]) for item in batch)
    max_tgt_len = max(len(item["tgt_in_ids"]) for item in batch)

    batch_size = len(batch)

    src_batch = torch.full((batch_size, max_src_len), pad_id, dtype=torch.long)
    tgt_in_batch = torch.full((batch_size, max_tgt_len), pad_id, dtype=torch.long)
    tgt_out_batch = torch.full((batch_size, max_tgt_len), pad_id, dtype=torch.long)

    src_padding_mask = torch.ones((batch_size, max_src_len), dtype=torch.bool)
    tgt_padding_mask = torch.ones((batch_size, max_tgt_len), dtype=torch.bool)

    for i, item in enumerate(batch):
        src_ids = item["src_ids"]
        tgt_in_ids = item["tgt_in_ids"]
        tgt_out_ids = item["tgt_out_ids"]

        src_len = len(src_ids)
        tgt_len = len(tgt_in_ids)

        src_batch[i, :src_len] = torch.tensor(src_ids, dtype=torch.long)
        tgt_in_batch[i, :tgt_len] = torch.tensor(tgt_in_ids, dtype=torch.long)
        tgt_out_batch[i, :tgt_len] = torch.tensor(tgt_out_ids, dtype=torch.long)

        src_padding_mask[i, :src_len] = False  # False = not padded
        tgt_padding_mask[i, :tgt_len] = False

    return {
        "src_ids": src_batch,
        "tgt_in_ids": tgt_in_batch,
        "tgt_out_ids": tgt_out_batch,
        "src_padding_mask": src_padding_mask,
        "tgt_padding_mask": tgt_padding_mask,
    }