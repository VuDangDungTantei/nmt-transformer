from pathlib import Path
import argparse
import sentencepiece as spm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "spm"
# ==========================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="thư mục chứa train.en + train.vi",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="thư mục lưu spm.model/spm.vocab",
    )
    ap.add_argument("--vocab_size", type=int, default=8000)
    ap.add_argument("--model_type", type=str, default="unigram", choices=["unigram", "bpe"])
    ap.add_argument("--character_coverage", type=float, default=1.0)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_en = data_dir / "train.en"
    train_vi = data_dir / "train.vi"
    if not train_en.exists() or not train_vi.exists():
        raise FileNotFoundError(f"Không thấy train.en/train.vi trong data_dir={data_dir}")

    # gộp train.en + train.vi để train tokenizer
    combined = out_dir / "train.all.txt"
    with combined.open("w", encoding="utf-8", newline="\n") as w:
        for p in [train_en, train_vi]:
            with p.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        w.write(line + "\n")

    prefix = str(out_dir / "spm")

    # Khớp special-token IDs với SubwordTokenizer của bạn:
    # pad=0, unk=1, bos=2, eos=3
    spm.SentencePieceTrainer.Train(
        input=str(combined),
        model_prefix=prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )

    print("✅ Done!")
    print("Saved:", prefix + ".model")
    print("Saved:", prefix + ".vocab")

if __name__ == "__main__":
    main()
