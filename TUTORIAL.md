# Tomorrow's Tutorial — Run the Finetune on the GPU Machine

This is the step-by-step script for when your tutor hands over the GPU machine. Each step shows the **command**, the **expected output**, and **what to do if it breaks**.

Target: finish with a trained adapter + a `results_finetuned.md` / `results_base.md` comparison you can hand in.

Rough time budget on a mid-range GPU (e.g. RTX 3060 / 3090 / 4090):
- Setup + install: **15–30 min**
- Smoke test training: **5–15 min**
- Full training (1 epoch, 68k rows, 3B model): **30–90 min**
- Evaluation: **5 min**

---

## 0. Move the project to the GPU machine

Pick whichever is easiest:

- **GitHub:** on this Mac, `git remote add origin <repo>` → `git push`. Then `git clone` on the GPU box.
- **scp:** on this Mac, `scp -r /Users/hagen/Downloads/Unsloth user@gpu-host:~/`.
- **USB stick:** copy the folder. Skip `.venv/` (it's Mac-only anyway).

On the GPU machine, `cd` into the project directory. All commands below run from there.

---

## 1. Inspect the GPU

```
nvidia-smi
```

Look at three things in the output:

| What | Where | Why it matters |
|---|---|---|
| **GPU name** (e.g. "NVIDIA GeForce RTX 3060") | top-left header | context for classmates/tutor |
| **Driver Version / CUDA Version** | top-right header | picks the Unsloth install command in step 3 |
| **Memory** (e.g. "12288MiB") | middle section | picks the base model tier in step 4 |

Write these three numbers down — you'll paste them into your report.

---

## 2. Create a Python environment

Unsloth is picky about the exact Python/CUDA/torch combo — an isolated env saves you pain later.

```
python3 -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows PowerShell
python --version                 # should be 3.10 or 3.11 ideally; 3.12 works; 3.13 may not
```

> **If Python is 3.13:** install 3.11 via `pyenv` or the system package manager. Unsloth's pinned torch wheels don't always have 3.13 builds yet.

---

## 3. Install Unsloth

First check your CUDA version from step 1:

```
nvcc --version        # if installed
# or re-read the top right of `nvidia-smi`
```

Then pick ONE of these:

```
# For CUDA 12.1:
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"

# For CUDA 12.4:
pip install "unsloth[cu124-torch240] @ git+https://github.com/unslothai/unsloth.git"
```

> **Other CUDA versions:** check https://docs.unsloth.ai/get-started/installing-+-updating for the matching `cu<version>-torch<version>` extras tag. **Do not guess.**

Then the rest:

```
pip install -r requirements.txt
```

Verify the install:

```
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected: something like `2.4.0`, `True`, `NVIDIA GeForce RTX 3060`. If `cuda.is_available()` is `False`, stop and re-check your CUDA install — do not continue.

---

## 4. Pick the base model tier (maybe edit train.py)

Default in [train.py:30](train.py#L30) is `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`. Change it only if you have headroom:

| VRAM (from `nvidia-smi`) | `MODEL_NAME` value |
|---|---|
| ≤ 8 GB | `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` (leave as-is) |
| 12 GB | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` |
| 16+ GB | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` |

**When in doubt, leave it at 3B.** A finished 3B finetune beats an OOM 7B every time.

---

## 5. Log into HuggingFace

```
huggingface-cli login
```

Paste the access token you created at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). A read token is enough — no write access needed for this project.

---

## 6. Smoke test — train on 2000 rows first

**Do not skip this.** It catches config errors (wrong chat template, OOM on your batch size, missing deps) in 10 minutes instead of 90.

```
python train.py --smoke
```

Expected output — you should see:

```
==((====))==  Unsloth 2024.X.X: Fast Qwen2 patching. Transformers: 4.XX.X.
   \\   /|    GPU: NVIDIA GeForce RTX 3060. Max memory: XX.X GB. ...
O^O/ \_/ \    Torch: 2.4.0. CUDA: 12.1. CUDA Toolkit: 12.1. Triton: ...
\        /    Bfloat16 = TRUE. FA [Flash Attention 2] = True.
 "-____-"     Free Apache license: http://github.com/unslothai/unsloth
...
[smoke] training on 2000 rows
...
{'loss': 2.13, 'learning_rate': 1.9e-04, 'epoch': 0.04}
{'loss': 1.87, 'learning_rate': 1.8e-04, 'epoch': 0.08}
{'loss': 1.71, 'learning_rate': 1.7e-04, 'epoch': 0.12}
...
```

**What to check:**
- Loss starts around 1.5–2.5 and **decreases** over the run. If it stays flat or goes up, something is wrong (most often a chat-template mismatch).
- No `torch.cuda.OutOfMemoryError`.
- Ends with `Saved LoRA adapter to ./lora_finance`.

**If smoke test fails:**

| Symptom | Likely fix |
|---|---|
| `OutOfMemoryError` | In [train.py](train.py), drop `per_device_train_batch_size` from 2 → 1 |
| Very slow training, <0.5 it/s | GPU may be shared; check `nvidia-smi` again. Also confirm `bf16=True` is supported (older GPUs need `fp16=True` instead) |
| `ImportError: xformers` on Windows | Switch to WSL2 Ubuntu. Windows-native Unsloth needs Triton-Windows and is much fussier |
| Loss stays flat | Chat template mismatch. Verify `chat_template="qwen-2.5"` in [train.py:56](train.py#L56) matches your `MODEL_NAME` |

When the smoke test passes cleanly, **delete `./lora_finance/` before the real run** (or it'll be reused instead of re-initialized):
```
rm -rf lora_finance/ outputs/
```

---

## 7. Full training run

```
python train.py 2>&1 | tee training_log.txt
```

`tee` saves the full log to `training_log.txt` — include excerpts in your report.

Monitor in another terminal:
```
watch -n 5 nvidia-smi
```

You should see ~80–95% GPU utilization and stable VRAM. If VRAM is creeping up every step, you have a leak — stop and ask.

**What "good" looks like:**
- Loss trending down from ~2.0 to ~1.0–1.3 over 1 epoch.
- Total time: on 68k rows at batch_size=2 × grad_accum=4, expect ~8,600 optimizer steps. At 0.5–2 s/step, that's 1–5 hours — **if you're short on time**, cut to `num_train_epochs=0.5` or use `dataset.select(range(20000))`.

---

## 8. Evaluate — the core deliverable

Run the same 7 finance prompts against both models:

```
python infer.py --base --out results_base.md
python infer.py         --out results_finetuned.md
```

Then diff them side by side:
```
diff -y results_base.md results_finetuned.md | less
```

Or open both files in VS Code's "Compare" view. **This comparison is the main artifact for your report.** For each prompt, note:

- Is the finetuned answer **more specific** (uses finance terminology correctly)?
- Is it **better structured** (e.g. numbered tradeoffs instead of vague prose)?
- Is it **worse** anywhere (hallucinations, lost instruction-following, repetitive)?

Expected outcome for a healthy finetune: on 3–5 of 7 prompts the finetuned answer is visibly more "finance-flavored." On 1–2 it may be slightly worse (finetuning almost always trades some generality for specialization). If it's worse on everything → see troubleshooting at the bottom.

---

## 9. Save your results

Create a short `REPORT.md` with:

1. **Environment:** GPU model, VRAM, CUDA version, base model, dataset size, LoRA config (r=16, alpha=16).
2. **Training curve:** paste 5–10 loss lines from `training_log.txt` showing the downward trend.
3. **Before / after:** 2–3 prompts from the eval where the finetune clearly won, and 1 where it didn't — with your analysis of why.
4. **What you'd do differently:** longer training? bigger model? different dataset?

---

## 10. (Optional) Export for demo

Run in a quick `python` REPL after training:

```python
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained("lora_finance", load_in_4bit=True)

# Merged HF model (portable, ~6 GB for 3B):
model.save_pretrained_merged("merged", tok, save_method="merged_16bit")

# OR GGUF for Ollama / llama.cpp (~2 GB for 3B q4_k_m):
model.save_pretrained_gguf("gguf", tok, quantization_method="q4_k_m")
```

To run the GGUF in Ollama:
```
ollama create finance-qwen -f ./gguf/Modelfile
ollama run finance-qwen
```

---

## Trying the second dataset (`FinGPT/fingpt-fiqa_qa`)

If the first run goes well and you want to experiment: the FiQA dataset has **different column semantics** than `finance-alpaca`:

| Column | `gbharti/finance-alpaca` | `FinGPT/fingpt-fiqa_qa` |
|---|---|---|
| `instruction` | the actual task/question | a meta-framing phrase ("Offer your insights...") that varies per row |
| `input` | usually empty; optional extra context | **the actual question** |
| `output` | the answer | the answer |

To use it, swap these two constants and the formatter in [train.py](train.py):

```python
DATASET_NAME = "FinGPT/fingpt-fiqa_qa"

def row_to_messages(row: dict) -> list[dict]:
    # For FiQA: `input` is the question, `instruction` is a system-style framing.
    return [
        {"role": "system", "content": row["instruction"].strip()},
        {"role": "user",   "content": row["input"].strip()},
        {"role": "assistant", "content": row["output"].strip()},
    ]
```

Everything else (trainer config, infer.py) stays the same.

---

## Final troubleshooting

- **Finetune is clearly *worse* on every eval prompt** → usually overfitting or wrong chat template. Try: drop `num_train_epochs` to 0.5, re-verify `get_chat_template(tokenizer, "qwen-2.5")` is being called before training.
- **Loss goes to NaN** → almost always an fp16 issue on older GPUs. In [train.py](train.py) swap `bf16=True` for `fp16=True`.
- **Inference gives weird repeating text** → chat template mismatch at inference. Confirm [infer.py](infer.py) also calls `get_chat_template(tokenizer, "qwen-2.5")`.
- **`No module named 'unsloth'`** → activate your venv (`source .venv/bin/activate`).
