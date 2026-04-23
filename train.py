"""Finetune a small LLM on the finance-alpaca dataset with Unsloth (QLoRA).

Runs on an NVIDIA GPU. NOT runnable on macOS / Apple Silicon.

Usage:
    # Smoke test first (fast, ~2k rows, 1 epoch):
    python train.py --smoke

    # Real run (full dataset, 1 epoch):
    python train.py
"""

import argparse

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


# ------------------------------------------------------------
# Model tier — pick one based on `nvidia-smi` VRAM.
# ------------------------------------------------------------
# <=  8 GB  :  unsloth/Qwen2.5-3B-Instruct-bnb-4bit   (default: safest pick)
#    12 GB  :  unsloth/Qwen2.5-7B-Instruct-bnb-4bit
#   16+ GB  :  unsloth/Qwen2.5-7B-Instruct-bnb-4bit   (increase MAX_SEQ_LEN / batch)
MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"

MAX_SEQ_LEN = 2048
DATASET_NAME = "gbharti/finance-alpaca"
OUTPUT_DIR = "lora_finance"

# Alpaca-style prompt: each row has {instruction, input, output}.
# Some rows have an empty "input" — we fold that into a single user turn.
def row_to_messages(row: dict) -> list[dict]:
    instruction = row["instruction"].strip()
    extra = (row.get("input") or "").strip()
    user_content = f"{instruction}\n\n{extra}" if extra else instruction
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": row["output"].strip()},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Use 2000 rows only.")
    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )

    # Qwen2.5 uses the ChatML template. `get_chat_template` configures the
    # tokenizer so `apply_chat_template` emits the right special tokens.
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    dataset = load_dataset(DATASET_NAME, split="train")
    if args.smoke:
        dataset = dataset.select(range(2000))
        print(f"[smoke] training on {len(dataset)} rows")

    def format_row(row: dict) -> dict:
        text = tokenizer.apply_chat_template(
            row_to_messages(row), tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(format_row, remove_columns=dataset.column_names)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=SFTConfig(
            output_dir="outputs",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            warmup_ratio=0.03,
            lr_scheduler_type="linear",
            optim="adamw_8bit",
            logging_steps=10,
            save_strategy="no",
            bf16=True,
            report_to="none",
            seed=3407,
        ),
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nSaved LoRA adapter to ./{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
