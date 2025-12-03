## Reproduction Guide

This document describes how to reproduce all experiments in this repository on a machine similar to the one used in our project.

---

## 1. Hardware & OS

- **OS**: Windows 11 + WSL2 (Ubuntu)
- **CPU**: Intel(R) Core(TM) i9‑14900K
- **GPU**: NVIDIA GeForce RTX 5090
  - 32 GB dedicated GPU memory (Task Manager shows 31.5 GB)
- **Python**: 3.12 (conda environment `hpc`)

All training and inference are run **inside WSL** on a single RTX 5090 GPU.

---

## 2. Environment Setup (inside WSL)

Open WSL terminal and run the following.

### 2.1 Create and activate conda environment

```bash
# In WSL
conda create -n hpc python=3.12 -y
conda activate hpc
```

### 2.2 Install PyTorch with CUDA 13.0 wheel

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### 2.3 Clone / enter project directory

If you already have the project under `D:\LZU-HPC_2025_AI` on Windows,
it will be visible in WSL as `/mnt/d/LZU-HPC_2025_AI`:

```bash
cd /mnt/d/LZU-HPC_2025_AI
```

### 2.4 Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs:

- `transformers`, `accelerate`, `datasets`
- `peft`, `bitsandbytes`, `sentencepiece`, `protobuf`
- and a few analysis / plotting libraries (`matplotlib`, `seaborn`, etc.)

---

## 3. Download model with ModelScope

We use `Llama-3.2-3B-Instruct` from ModelScope and store it locally.

```bash
cd /mnt/d/LZU-HPC_2025_AI

# Make sure modelscope CLI is available
pip install modelscope

# Download model (this may take some time)
bash download_llm.sh
```

`download_llm.sh` will place the model under:

```text
./Llama-3.2-3B-Instruct/
  - config.json
  - model-00001-of-00002.safetensors
  - model-00002-of-00002.safetensors
  - tokenizer.json
  - ...
```

All later scripts load the model from this local path.

---

## 4. Inference Baseline & INT4 Quantization

### 4.1 Baseline inference (FP16, low-concurrency)

Script: `baseline_inference.py`  
Task: 500 queries on `gsm8k(main)` with `batch_size=1`.

```bash
cd /mnt/d/LZU-HPC_2025_AI

python baseline_inference.py \
  --model_path ./Llama-3.2-3B-Instruct \
  --dataset_name gsm8k \
  --dataset_split test \
  --dataset_config main \
  --num_samples 500 \
  --batch_size 1 \
  --output_dir ./baseline_results
```

Output metrics:

- `baseline_results/baseline_inference_metrics.json`
- Keys: `avg_latency`, `throughput_tokens_per_sec`, `peak_memory_gb`, etc.

### 4.2 INT4 quantized inference

Script: `optimized_inference.py`  
Same data / setting, with 4‑bit quantization enabled.

```bash
cd /mnt/d/LZU-HPC_2025_AI

python optimized_inference.py \
  --model_path ./Llama-3.2-3B-Instruct \
  --dataset_name gsm8k \
  --dataset_split test \
  --dataset_config main \
  --num_samples 500 \
  --batch_size 1 \
  --use_4bit \
  --output_dir ./optimized_results \
  --method_name "INT4"
```

Output metrics:

- `optimized_results/INT4/baseline_inference_metrics.json`

You can then generate comparison plots via:

```bash
python plot_results.py
```

Plots are saved under `./plots/` (e.g. `inference_latency.png`, `inference_memory.png`).

---

## 5. Finetuning Experiments

All finetuning experiments use:

- Model: `./Llama-3.2-3B-Instruct`
- Dataset: `wikitext`, config `wikitext-2-raw-v1`
- Samples: `--num_samples 4000`
- Epochs: `--num_epochs 2`
- Batch size: `--batch_size 4`
- Max length: `--max_length 512`

### 5.1 Full fine-tuning baseline

Script: `baseline_finetune.py`

```bash
cd /mnt/d/LZU-HPC_2025_AI

python baseline_finetune.py \
  --model_path ./Llama-3.2-3B-Instruct \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --num_samples 4000 \
  --num_epochs 2 \
  --batch_size 4 \
  --learning_rate 2e-5 \
  --max_length 512 \
  --output_dir ./baseline_finetune_output
```

Output metrics:

- `baseline_finetune_output/baseline_finetune_metrics.json`

### 5.2 LoRA finetuning

Script: `finetune.py` with `--use_lora`.

```bash
cd /mnt/d/LZU-HPC_2025_AI

python finetune.py \
  --model_name ./Llama-3.2-3B-Instruct \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --use_lora \
  --num_epochs 2 \
  --batch_size 4 \
  --learning_rate 2e-4 \
  --max_length 512 \
  --num_samples 4000 \
  --output_dir ./optimized_finetune_output/lora \
  --device cuda
```

Output metrics:

- `optimized_finetune_output/lora/baseline_finetune_metrics.json`

### 5.3 LoRA + Gradient Checkpointing (LoRA+GC)

```bash
cd /mnt/d/LZU-HPC_2025_AI

python finetune.py \
  --model_name ./Llama-3.2-3B-Instruct \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --use_lora \
  --gradient_checkpointing \
  --num_epochs 2 \
  --batch_size 4 \
  --learning_rate 2e-4 \
  --max_length 512 \
  --num_samples 4000 \
  --output_dir ./optimized_finetune_output/lora_gc \
  --device cuda
```

Output metrics:

- `optimized_finetune_output/lora_gc/baseline_finetune_metrics.json`

### 5.4 QLoRA finetuning

```bash
cd /mnt/d/LZU-HPC_2025_AI

python finetune.py \
  --model_name ./Llama-3.2-3B-Instruct \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --use_qlora \
  --num_epochs 2 \
  --batch_size 4 \
  --learning_rate 2e-4 \
  --max_length 512 \
  --num_samples 4000 \
  --output_dir ./optimized_finetune_output/qlora \
  --device cuda
```

Output metrics:

- `optimized_finetune_output/qlora/baseline_finetune_metrics.json`

### 5.5 QLoRA + Gradient Checkpointing (QLoRA+GC)

```bash
cd /mnt/d/LZU-HPC_2025_AI

python finetune.py \
  --model_name ./Llama-3.2-3B-Instruct \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --use_qlora \
  --gradient_checkpointing \
  --num_epochs 2 \
  --batch_size 4 \
  --learning_rate 2e-4 \
  --max_length 512 \
  --num_samples 4000 \
  --output_dir ./optimized_finetune_output/qlora_gc \
  --device cuda
```

Output metrics:

- `optimized_finetune_output/qlora_gc/baseline_finetune_metrics.json`

### 5.6 Generate finetune comparison plots

After all finetune runs are finished, you can regenerate comparison plots:

```bash
cd /mnt/d/LZU-HPC_2025_AI
python plot_results.py
```

This will (re)create:

- `plots/finetune_time.png`
- `plots/finetune_time_per_epoch.png`
- `plots/finetune_memory.png`

---

## 6. QLoRA Profiler Run (Optional, for analysis)

To reproduce the PyTorch Profiler analysis:

```bash
cd /mnt/d/LZU-HPC_2025_AI

python profile_qlora.py
```

This will:

- Run a short QLoRA training loop (20 steps) on a subset of `wikitext-2-raw-v1`.
- Print the top CUDA kernels by total time.
- Save the table to `qlora_profile_table.txt` for inclusion in the report.

---

## 7. Summary

Following the steps above (environment setup → model download → baseline runs → LoRA/QLoRA variants → plotting scripts) will reproduce all the quantitative results and figures used in our report.  
If hardware differs (e.g., GPU model or memory), absolute numbers will change, but **relative trends** (e.g., LoRA/QLoRA memory savings vs. full fine-tuning) should remain similar. 


