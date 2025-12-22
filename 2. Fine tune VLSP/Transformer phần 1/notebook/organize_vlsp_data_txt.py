import json
import random
import re
import unicodedata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
IN_DIR  = DATA_DIR / "raw"        # chứa train.en.txt, train.vi.txt, public_test.*
OUT_DIR = DATA_DIR / "processed"  # nơi sẽ lưu train.*, dev.*, test.* + stats.json
# =========================================


DEV_RATIO = 0.02     # tách dev 2% từ train
SEED = 42
MAX_LEN = 200        # lọc câu quá dài (đếm token thô theo whitespace)
MAX_RATIO = 9.0      # lọc cặp lệch độ dài

WS_RE = re.compile(r"\s+")

def normalize_line(s: str) -> str:
    s = s.strip()
    s = unicodedata.normalize("NFC", s)
    s = WS_RE.sub(" ", s)
    return s

def read_lines(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\n") for line in f]

def write_lines(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for s in lines:
            f.write(s + "\n")

def align_and_filter(src_lines, tgt_lines, max_len=200, max_ratio=9.0):
    if len(src_lines) != len(tgt_lines):
        raise ValueError(f"Lệch số dòng: src={len(src_lines)} tgt={len(tgt_lines)}")

    kept_src, kept_tgt = [], []
    stats = {
        "pairs_total": len(src_lines),
        "dropped_empty": 0,
        "dropped_max_len": 0,
        "dropped_ratio": 0,
        "pairs_kept": 0,
    }

    for s, t in zip(src_lines, tgt_lines):
        s2 = normalize_line(s)
        t2 = normalize_line(t)

        if not s2 or not t2:
            stats["dropped_empty"] += 1
            continue

        ls = len(s2.split(" "))
        lt = len(t2.split(" "))

        if ls > max_len or lt > max_len:
            stats["dropped_max_len"] += 1
            continue

        ratio = max(ls / max(1, lt), lt / max(1, ls))
        if ratio > max_ratio:
            stats["dropped_ratio"] += 1
            continue

        kept_src.append(s2)
        kept_tgt.append(t2)

    stats["pairs_kept"] = len(kept_src)
    return kept_src, kept_tgt, stats

def split_train_dev(src, tgt, dev_ratio=0.02, seed=42):
    assert len(src) == len(tgt)
    n = len(src)
    n_dev = int(round(n * dev_ratio))
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    dev_set = set(idx[:n_dev])

    tr_s, tr_t, dv_s, dv_t = [], [], [], []
    for i in range(n):
        if i in dev_set:
            dv_s.append(src[i]); dv_t.append(tgt[i])
        else:
            tr_s.append(src[i]); tr_t.append(tgt[i])
    return tr_s, tr_t, dv_s, dv_t

def main():
    # ---- đúng tên file theo ảnh của bạn ----
    train_en = IN_DIR / "train.en.txt"
    train_vi = IN_DIR / "train.vi.txt"
    test_en  = IN_DIR / "public_test.en.txt"
    test_vi  = IN_DIR / "public_test.vi.txt"

    for p in [train_en, train_vi, test_en, test_vi]:
        if not p.exists():
            raise FileNotFoundError(f"Không thấy file: {p}")

    # read
    tr_en = read_lines(train_en)
    tr_vi = read_lines(train_vi)
    te_en = read_lines(test_en)
    te_vi = read_lines(test_vi)

    # filter
    tr_en2, tr_vi2, st_train = align_and_filter(tr_en, tr_vi, MAX_LEN, MAX_RATIO)
    te_en2, te_vi2, st_test  = align_and_filter(te_en, te_vi, MAX_LEN, MAX_RATIO)

    # split dev
    train_en_out, train_vi_out, dev_en_out, dev_vi_out = split_train_dev(
        tr_en2, tr_vi2, DEV_RATIO, SEED
    )

    # write output
    write_lines(OUT_DIR / "train.en", train_en_out)
    write_lines(OUT_DIR / "train.vi", train_vi_out)
    write_lines(OUT_DIR / "dev.en", dev_en_out)
    write_lines(OUT_DIR / "dev.vi", dev_vi_out)
    write_lines(OUT_DIR / "test.en", te_en2)
    write_lines(OUT_DIR / "test.vi", te_vi2)

    stats = {
        "input_dir": str(IN_DIR),
        "output_dir": str(OUT_DIR),
        "dev_ratio": DEV_RATIO,
        "seed": SEED,
        "filter": {"max_len": MAX_LEN, "max_ratio": MAX_RATIO},
        "train_filter_stats": st_train,
        "test_filter_stats": st_test,
        "final_counts": {
            "train_pairs": len(train_en_out),
            "dev_pairs": len(dev_en_out),
            "test_pairs": len(te_en2),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ Done! Saved to:", OUT_DIR.resolve())
    print(json.dumps(stats["final_counts"], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
