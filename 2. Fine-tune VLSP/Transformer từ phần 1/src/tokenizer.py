from pathlib import Path
import sentencepiece as spm
from typing import List

class SubwordTokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(model_path))

        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

    def encode_src(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def encode_tgt(self, text: str) -> (List[int], List[int]):
        """
        Trả về:
          - input ids cho decoder: [BOS, ..., token_n]
          - target ids shift-right: [..., token_n, EOS]
        """
        ids = self.sp.encode(text, out_type=int)
        dec_in = [self.bos_id] + ids
        dec_out = ids + [self.eos_id]
        return dec_in, dec_out

    def decode(self, ids: List[int]) -> str:
        # bỏ pad & special nếu có
        ids = [i for i in ids if i not in (self.pad_id,)]
        return self.sp.decode(ids)
