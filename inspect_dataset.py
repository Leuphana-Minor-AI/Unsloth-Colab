"""Sanity-check a finance dataset before training.

Runs on any machine with `pip install datasets` — no GPU needed.

Usage:
    python inspect_dataset.py                            # default: gbharti/finance-alpaca
    python inspect_dataset.py FinGPT/fingpt-fiqa_qa      # any other HF dataset
"""

import argparse

from datasets import load_dataset


DEFAULT_DATASET = "gbharti/finance-alpaca"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("-n", "--samples", type=int, default=3, help="How many samples to print.")
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.split)

    print(f"Dataset: {args.dataset}")
    print(f"Split:   {args.split}")
    print(f"Rows:    {len(ds):,}")
    print(f"Columns: {ds.column_names}")
    print()

    print(f"--- First {args.samples} samples ---")
    for i, row in enumerate(ds.select(range(args.samples))):
        print(f"\n[{i}]")
        for k, v in row.items():
            v_str = str(v)
            if len(v_str) > 240:
                v_str = v_str[:240] + "..."
            print(f"    {k}: {v_str!r}")


if __name__ == "__main__":
    main()
