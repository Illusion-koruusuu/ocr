#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export tokenizer vocab to id2token TSV for C++ decoder.")
    parser.add_argument(
        "--vocab",
        default="/home/lu/code/ocr/onnx/tjoab_latex_finetuned/vocab.json",
        help="Path to vocab.json",
    )
    parser.add_argument(
        "--out",
        default="/home/lu/code/ocr/onnx/tjoab_latex_finetuned/id2token.txt",
        help="Output TSV path (id<TAB>token)",
    )
    args = parser.parse_args()

    vocab_path = Path(args.vocab)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    size = max(vocab.values()) + 1
    id2token = [""] * size
    for token, idx in vocab.items():
        id2token[idx] = token

    with out_path.open("w", encoding="utf-8") as f:
        for idx, token in enumerate(id2token):
            f.write(f"{idx}\t{token}\n")

    print(f"Exported {len(id2token)} entries -> {out_path}")


if __name__ == "__main__":
    main()
