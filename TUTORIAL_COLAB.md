# Colab Tutorial — Finance Q&A Finetune

Step-by-step to run the whole finetune on Google Colab's free tier. No local GPU needed.

Total time end-to-end: **~60 minutes** (15 min setup + 30–45 min training + 5 min eval).

---

## What you'll end up with

- A trained LoRA adapter (`lora_finance/`) saved to your Google Drive.
- A markdown file (`results_comparison.md`) showing the **base model** vs **your finetuned model** answering the same 7 finance questions. This is your main project deliverable.

---

## Step 1 — Prerequisites (do these once, on your Mac)

### 1a. Create a Google account (if you don't have one)
Colab runs under your Google account. Any Gmail works.

### 1b. Create a HuggingFace account + access token
1. Go to https://huggingface.co and sign up (free).
2. Go to https://huggingface.co/settings/tokens → **"+ Create new token"**.
3. Name it something like `colab-finetune`, select **"Read"** permission, click Create.
4. **Copy the token** (`hf_...`) somewhere you can paste it later — you won't be able to see it again.

---

## Step 2 — Upload the notebook to Colab

You have three options. Pick one.

### Option A (easiest): Upload directly

1. Go to https://colab.research.google.com
2. On the welcome screen → **"Upload"** tab → drag `finance_finetune_colab.ipynb` (from this folder) onto the page.
3. The notebook opens. Done.

### Option B: Upload via Google Drive

1. Upload `finance_finetune_colab.ipynb` to your Google Drive (any folder).
2. Right-click the file in Drive → **Open with → Google Colaboratory**. (If Colab isn't listed, click "Connect more apps" and add it.)

### Option C: From GitHub

If you've pushed this project to GitHub:
1. Colab welcome → **"GitHub"** tab → paste your repo URL → open the notebook directly.
2. Bonus: changes you make in Colab can be saved back to GitHub via **File → Save a copy in GitHub**.

---

## Step 3 — Enable the GPU runtime

**This is the step everyone forgets.** Without it, you'll be training on CPU and nothing will work.

1. In the Colab menu bar: **Runtime → Change runtime type**.
2. **Hardware accelerator** → select **T4 GPU**.
3. Click **Save**.
4. Colab reconnects to a GPU-equipped machine (takes ~10 sec).

You can verify by running the **first code cell** (`!nvidia-smi`) — you should see something like `Tesla T4` and `~15 GB` memory. If you see nothing GPU-related, go back to step 3 above.

---

## Step 4 — Run the notebook, cell by cell

Click into the first code cell and press **Shift+Enter** to run it. Shift+Enter runs the current cell and moves to the next one.

Don't run all cells at once (`Runtime → Run all`) the first time — go cell by cell so you can see each output and catch any errors early.

Here's what to expect at each section:

### Section 0: `!nvidia-smi`
Output shows `Tesla T4` and ~15 GB. If not, fix the runtime (step 3 above).

### Section 1: Install Unsloth
Takes 2–3 minutes. You'll see nothing because of `%%capture` — just wait. If it errors, remove the `%%capture` line and rerun to see the actual error message.

### Section 2: HuggingFace login
A widget appears with a password field. **Paste the token you created in Step 1b**. Press Enter. It should say "Login successful."

### Section 3: Config
Just defines constants. Zero-output cell. Leave defaults unless you want the full dataset (see "Going further" below).

### Section 4: Load the base model
Downloads ~2 GB. Takes ~45 seconds. You'll see an Unsloth ASCII-art banner and a progress bar. **The first time, you may see a warning about the tokenizer — ignore it.**

### Section 5: Attach LoRA
Instant. Prints something like `trainable params: 29,884,416 || all params: ...`. The `trainable` number should be a small fraction of the total — that's the whole point of LoRA.

### Section 6: Load + format dataset
Takes ~30 seconds. Prints `Training on 10,000 rows` and shows one formatted example. Glance at the example — it should have `<|im_start|>user ... <|im_end|>` tags (Qwen's chat template). If not, something's wrong.

### Section 7: **Train** — this is the long one
30–45 minutes on free T4. You'll see loss values every 10 steps:
```
{'loss': 2.04, 'learning_rate': 0.0002, 'epoch': 0.01}
{'loss': 1.72, 'learning_rate': 0.00019, 'epoch': 0.02}
{'loss': 1.45, 'learning_rate': 0.00018, 'epoch': 0.03}
...
```
**The number you want to watch is `loss`.** It should trend *down*. If it stays flat or goes up, stop and re-read the troubleshooting section.

> ⚠️ **Colab will disconnect if you leave the tab alone for ~90 minutes.** Keep the tab visible (or in a visible window) while training. If it disconnects, you lose progress — you'd need to restart from Section 4.

### Section 8: Save locally
Writes the adapter to `./lora_finance/` and lists the files. You should see `adapter_model.safetensors` at ~50–200 MB.

### Section 8b: Save to Drive (recommended)
Mounts your Drive. A popup asks you to authorize Colab → Drive access — click through. Copies the adapter so it survives after Colab shuts down.

### Section 9: Evaluate the finetuned model
~2 min. Prints the finetuned answer to each of 7 finance prompts. Read a couple — do they sound plausible and on-topic?

### Section 10: Evaluate the base model
~2 min. Same 7 prompts, but run through the un-finetuned base. Compare the outputs yourself as they stream.

### Section 11: Write comparison file
Instant. Downloads `results_comparison.md` to your Mac's Downloads folder automatically.

---

## Step 5 — Write up your project

Open `results_comparison.md` (it downloaded to `~/Downloads/` on your Mac). For each of the 7 prompts it shows the base answer side-by-side with the finetuned answer.

For your report:
1. Copy the training loss trajectory (from Section 7's output) as evidence training worked.
2. Pick 2–3 prompts where the finetune is visibly better, 1 where it isn't. Explain what changed.
3. Include the environment info from Section 0 (`nvidia-smi`).
4. Section on "what I'd try next": bigger model? More data? More epochs?

---

## Common problems

| Problem | What to do |
|---|---|
| "No GPU available" / training is hours-per-step | Runtime isn't set to T4. Runtime → Change runtime type → T4 GPU. |
| `CUDA out of memory` in Section 7 | Edit the training cell: set `per_device_train_batch_size=1`. |
| Loss stays flat at ~2.0 | Chat template didn't apply. Re-run Section 4 and verify the line `tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")` ran without error. |
| Loss goes to `nan` | Rare on T4. Rerun the training cell from scratch (`Runtime → Run after`). |
| "Session disconnected" during training | Free-tier limitation. Shorten `SUBSET_SIZE` to 5000 in Section 3 and retry. Keep the tab visible. |
| Drive mount popup hangs | Check for a blocked popup in your browser; Colab needs to open the auth page. |
| Can't find `results_comparison.md` | Check browser Downloads. On Mac default location: `~/Downloads/`. |

---

## Going further (optional — after the first successful run)

- **Full dataset (68k rows):** in Section 3, change `SUBSET_SIZE = 10000` to `SUBSET_SIZE = None`. On free T4 this takes ~4 hours — you'll almost certainly hit the session timeout. Feasible on Colab Pro (A100).
- **Bigger model (7B):** in Section 3, change `MODEL_NAME` to `"unsloth/Qwen2.5-7B-Instruct-bnb-4bit"`. On free T4 this is tight — set `per_device_train_batch_size=1` in Section 7.
- **More epochs:** in Section 7's `SFTConfig`, change `num_train_epochs=1` to `num_train_epochs=2` or `3`. Watch for overfitting — if the base model becomes *worse* on general prompts, you've overfit.
- **Try the alternative dataset:** in Section 3, change `DATASET_NAME` to `"FinGPT/fingpt-fiqa_qa"`. **Also** change `row_to_messages` in Section 6 — FiQA uses `input` as the question (not as extra context). Snippet in the main `TUTORIAL.md` at the bottom.
- **Export for Ollama:** add a new cell after Section 8: `model.save_pretrained_gguf("gguf", tokenizer, quantization_method="q4_k_m")`. Produces a ~2 GB file you can run locally with `ollama`.

---

## Why Colab vs the DGX Spark

Once the DGX Spark is working, it'll be faster and let you use the full 68k dataset + a larger model in less time. But Colab is:
- **Available right now** (no network setup needed)
- **Free** for the T4 tier
- **x86_64** (standard Unsloth install works out of the box — none of the DGX Spark's ARM64 install friction)

If you get the DGX Spark working later, the code is still here in the [original repo](../Unsloth/) — use [TUTORIAL.md](../Unsloth/TUTORIAL.md) for that path.
