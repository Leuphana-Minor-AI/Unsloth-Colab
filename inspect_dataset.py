"""Sanity-check the finance dataset before training.

Runs on any machine with `pip install datasets` — no GPU needed.
Usage: python inspect_dataset.py
"""

from datasets import load_dataset


DATASET_NAME = "gbharti/finance-alpaca"


def main() -> None:
    ds = load_dataset(DATASET_NAME, split="train")

    print(f"Dataset: {DATASET_NAME}")
    print(f"Rows:    {len(ds):,}")
    print(f"Columns: {ds.column_names}")
    print()

    print("--- First 3 samples ---")
    for i, row in enumerate(ds.select(range(3))):
        print(f"\n[{i}] instruction: {row['instruction']!r}")
        print(f"    input:       {row.get('input', '')!r}")
        print(f"    output:      {row['output'][:200]!r}{'...' if len(row['output']) > 200 else ''}")

    print()
    rows_with_input = sum(1 for r in ds if r.get("input"))
    print(f"Rows with non-empty 'input' field: {rows_with_input:,} / {len(ds):,}")


if __name__ == "__main__":
    main()
