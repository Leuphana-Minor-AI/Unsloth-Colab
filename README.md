# Finance Q&A — Unsloth LoRA Finetune

Semester project: finetune a small LLM into a finance Q&A assistant using [Unsloth](https://unsloth.ai).

- **Base model:** `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` (default — safe for ≥8 GB VRAM). See `MODEL_NAME` in [train.py](train.py) to switch to the 7B tier if VRAM allows.
- **Dataset:** [`gbharti/finance-alpaca`](https://huggingface.co/datasets/gbharti/finance-alpaca) (~68k Alpaca-format rows).
- **Method:** QLoRA (4-bit base + LoRA adapter, r=16).

## Setup (on the GPU machine)

1. Check GPU and CUDA:
   ```
   nvidia-smi
   ```
2. Install Unsloth — pick the line in [requirements.txt](requirements.txt) that matches your CUDA version (12.1 or 12.4). Example:
   ```
   pip install "unsloth[cu124-torch240] @ git+https://github.com/unslothai/unsloth.git"
   ```
3. Install remaining deps:
   ```
   pip install -r requirements.txt
   ```
4. Authenticate with HuggingFace (needed to download the dataset and base model):
   ```
   huggingface-cli login
   ```

## Train

Always do a smoke test first (2000 rows, ~5–15 min on any modern GPU):
```
python train.py --smoke
```
If that completes and the loss decreased, do the real run:
```
python train.py
```
The LoRA adapter is saved to `./lora_finance/`.

## Evaluate (the core deliverable)

Run the same prompt list against both models and save the outputs:
```
python infer.py --base --out results_base.md
python infer.py        --out results_finetuned.md
```
Diff the two files for your report — that comparison *is* the evidence of domain adaptation.

## Export (optional)

Merge LoRA into the base and save a full HF model:
```python
# in a short python session:
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained("lora_finance", load_in_4bit=True)
model.save_pretrained_merged("merged", tok, save_method="merged_16bit")
```
Or export to GGUF for Ollama / llama.cpp:
```python
model.save_pretrained_gguf("gguf", tok, quantization_method="q4_k_m")
```

## File map

- [inspect_dataset.py](inspect_dataset.py) — prints 3 samples + column info. Only file that runs on the Mac (CPU).
- [train.py](train.py) — QLoRA training pipeline.
- [infer.py](infer.py) — runs a fixed finance prompt list; `--base` flag toggles between finetuned adapter and base model.
- [requirements.txt](requirements.txt) — deps + CUDA-specific Unsloth install instructions.

## Troubleshooting

- **OOM during training** → drop to the 3B model, or reduce `per_device_train_batch_size` to 1.
- **Loss doesn't decrease** → chat template mismatch. Verify `MODEL_NAME` and `chat_template="qwen-2.5"` match the family of the base model in [train.py](train.py).
- **Finetuned model is *worse* than base on the eval prompts** → likely overfitting. Try `num_train_epochs=0.5` or regenerate with `do_sample=True, temperature=0.7`.
- **`ImportError: xformers`** on Windows → use a WSL2 Ubuntu environment, or follow Unsloth's Windows install guide (requires Triton-Windows).
