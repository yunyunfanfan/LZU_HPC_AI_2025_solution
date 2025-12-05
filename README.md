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

### 2.3 Install DeepSpeed (optional, for ZeRO-2/3 experiments)

If you want to run DeepSpeed ZeRO-2/3 experiments, install DeepSpeed:

```bash
conda activate hpc
pip install deepspeed
```

**Note**: DeepSpeed requires CUDA Toolkit to be installed on Windows host. See section 6.6 for detailed setup instructions.

### 2.4 Clone / enter project directory

If you already have the project under `D:\LZU-HPC_2025_AI` on Windows,
it will be visible in WSL as `/mnt/d/LZU-HPC_2025_AI`:

```bash
cd /mnt/d/LZU-HPC_2025_AI
```

### 2.5 Install Python dependencies

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

## 5. Inference Baseline & INT4 Quantization

### 5.1 Baseline inference (FP16, low-concurrency)

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

### 5.2 INT4 quantized inference

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

## 6. Finetuning Experiments

All finetuning experiments use:

- Model: `./Llama-3.2-3B-Instruct`
- Dataset: `wikitext`, config `wikitext-2-raw-v1`
- Samples: `--num_samples 4000`
- Epochs: `--num_epochs 2`
- Batch size: `--batch_size 4`
- Max length: `--max_length 512`

### 6.1 Full fine-tuning baseline

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

### 6.2 LoRA finetuning

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

### 6.3 LoRA + Gradient Checkpointing (LoRA+GC)

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

### 6.4 QLoRA finetuning

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

### 6.5 QLoRA + Gradient Checkpointing (QLoRA+GC)

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

### 6.6 QLoRA + DeepSpeed ZeRO-2 finetuning

Script: `finetune_zero2.py` with `--use_qlora` and DeepSpeed ZeRO-2.

**Prerequisites**: DeepSpeed requires CUDA Toolkit to be installed on Windows host and configured in WSL. See setup steps below.

**Setup DeepSpeed environment**:

1. Install CUDA Toolkit 13.0 on Windows (matching PyTorch's CUDA version):
   - Download from: https://developer.nvidia.com/cuda-13-0-0-download-archive
   - Install to default location: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0`

2. Configure CUDA in WSL:
   ```bash
   cd /mnt/d/LZU-HPC_2025_AI
   bash install_cuda_toolkit_wsl.sh
   source ~/.bashrc
   ```

3. Install DeepSpeed:
   ```bash
   conda activate hpc
   pip install deepspeed
   ```

**Run training**:

```bash
cd /mnt/d/LZU-HPC_2025_AI

bash deepspeed_wrapper.sh --num_gpus=1 finetune_zero2.py \
  --model_name ./Llama-3.2-3B-Instruct \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --use_qlora \
  --num_epochs 2 \
  --batch_size 4 \
  --learning_rate 2e-4 \
  --max_length 512 \
  --num_samples 4000 \
  --output_dir ./optimized_finetune_output/qlora_zero2 \
  --device cuda \
  --deepspeed_config ds_zero2.json
```

**Note**: The `deepspeed_wrapper.sh` script automatically:
- Sets `CUDA_HOME` to the correct CUDA Toolkit path
- Creates a symlink to avoid path space issues
- Sets `DS_BUILD_OPS=0` to disable CUDA extension compilation (uses standard PyTorch optimizer)
- Activates the `hpc` conda environment if not already active

Output metrics:

- `optimized_finetune_output/qlora_zero2/baseline_finetune_metrics.json`

### 6.7 QLoRA + DeepSpeed ZeRO-3 finetuning

Script: `finetune_zero3.py` with `--use_qlora` and DeepSpeed ZeRO-3.

**Prerequisites**: Same as ZeRO-2 (see section 6.6). DeepSpeed requires CUDA Toolkit to be installed on Windows host and configured in WSL.

**Run training**:

```bash
cd /mnt/d/LZU-HPC_2025_AI

bash deepspeed_wrapper.sh --num_gpus=1 finetune_zero3.py \
  --model_name ./Llama-3.2-3B-Instruct \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --use_qlora \
  --num_epochs 2 \
  --batch_size 4 \
  --learning_rate 2e-4 \
  --max_length 512 \
  --num_samples 4000 \
  --output_dir ./optimized_finetune_output/qlora_zero3 \
  --device cuda \
  --deepspeed_config ds_zero3.json
```

**Note**: The `deepspeed_wrapper.sh` script automatically:
- Sets `CUDA_HOME` to the correct CUDA Toolkit path
- Creates a symlink to avoid path space issues
- Sets `DS_BUILD_OPS=0` to disable CUDA extension compilation (uses standard PyTorch optimizer)
- Activates the `hpc` conda environment if not already active

**Comparison with ZeRO-2**:
- ZeRO-3 further shards model parameters (in addition to optimizer states and gradients in ZeRO-2).
- In single-GPU environment, ZeRO-3 has similar memory usage (5.48 GB vs 5.45 GB) but longer training time (503.26 s/epoch vs 351.12 s/epoch) due to additional parameter gather/shard overhead.
- ZeRO-3's advantage is more evident in multi-GPU environments where parameters can be truly sharded across devices.

Output metrics:

- `optimized_finetune_output/qlora_zero3/baseline_finetune_metrics.json`

### 6.8 Generate finetune comparison plots

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

## 7. QLoRA Profiler Run (Optional, for analysis)

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

## 8. Summary

Following the steps above (environment setup → model download → baseline runs → LoRA/QLoRA variants → plotting scripts) will reproduce all the quantitative results and figures used in our report.  
If hardware differs (e.g., GPU model or memory), absolute numbers will change, but **relative trends** (e.g., LoRA/QLoRA memory savings vs. full fine-tuning) should remain similar. 


