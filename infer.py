"""Run a fixed set of finance prompts through the model (finetuned or base).

Side-by-side comparison is the point: run once with --base, once without,
diff the outputs, and include them in your project write-up.

Usage:
    python infer.py             # uses ./lora_finance adapter
    python infer.py --base      # same prompts, base model only
    python infer.py --out results_finetuned.md
"""

import argparse

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


# Must match train.py
BASE_MODEL = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
ADAPTER_DIR = "lora_finance"
MAX_SEQ_LEN = 2048

PROMPTS = [
    "Explain what a P/E ratio is and why it matters to an investor.",
    "What is the difference between a Roth IRA and a Traditional IRA?",
    "How does dollar-cost averaging work, and when is it a bad idea?",
    "What is an ETF, and how is it different from a mutual fund?",
    "Is it better to pay off a low-interest mortgage early or invest the extra money?",
    "Explain bond duration in simple terms.",
    "What are the tax implications of selling a stock I've held for 6 months?",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", action="store_true", help="Use base model only (skip LoRA adapter).")
    parser.add_argument("--out", default=None, help="Write results to this markdown file.")
    args = parser.parse_args()

    model_name = BASE_MODEL if args.base else ADAPTER_DIR
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    FastLanguageModel.for_inference(model)

    lines: list[str] = []
    header = f"# Inference results — {'base model' if args.base else 'finetuned adapter'}\n\nModel: `{model_name}`\n"
    print(header)
    lines.append(header)

    for prompt in PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
        ).to(model.device)
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=400,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)

        block = f"\n---\n\n**Q:** {prompt}\n\n**A:** {response.strip()}\n"
        print(block)
        lines.append(block)

    if args.out:
        with open(args.out, "w") as f:
            f.write("".join(lines))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
