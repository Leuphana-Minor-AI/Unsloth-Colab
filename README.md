# Finance Q&A — Unsloth LoRA Finetune (Colab Edition)

Semester project: finetune a small LLM into a finance Q&A assistant using [Unsloth](https://unsloth.ai).

This is the **Google Colab** fork of the project — use this when you don't have access to a local CUDA GPU. The original CUDA/DGX version lives in [../Unsloth/](../Unsloth/).

- **Base model:** `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`
- **Dataset:** [`gbharti/finance-alpaca`](https://huggingface.co/datasets/gbharti/finance-alpaca) (~68k Alpaca-format rows; notebook defaults to a 10k subset for speed)
- **Method:** QLoRA (4-bit base + LoRA adapter, r=16)
- **Runtime:** Google Colab free tier (T4 GPU, 15 GB VRAM)

## Start here

👉 **[TUTORIAL_COLAB.md](TUTORIAL_COLAB.md)** — step-by-step walkthrough for first-timers (upload notebook, enable GPU, run cells, download results).

## Files

| File | Purpose |
|---|---|
| [finance_finetune_colab.ipynb](finance_finetune_colab.ipynb) | **The main file.** Open this in Colab. Self-contained — has install, training, eval, and export all in one. |
| [TUTORIAL_COLAB.md](TUTORIAL_COLAB.md) | Step-by-step tutorial for running the notebook. Read this first. |
| [inspect_dataset.py](inspect_dataset.py) | Local-only dataset browser. `.venv/bin/python inspect_dataset.py` on Mac if you want to poke at rows before uploading to Colab. |
| [train.py](train.py), [infer.py](infer.py), [requirements.txt](requirements.txt) | Left over from the local-GPU version. **Ignore these for the Colab path.** They're kept so you can pivot back to the DGX Spark later without losing anything. |
| [TUTORIAL.md](TUTORIAL.md) | The DGX/local-GPU tutorial. Also not needed for Colab. |

## What success looks like

After running the notebook end-to-end, you'll have:
- A LoRA adapter (`lora_finance/`) on Colab and in your Google Drive.
- A `results_comparison.md` downloaded to your Mac, showing base-model vs finetuned-model answers side-by-side on 7 finance prompts.

That comparison file is the core deliverable for the project report.
