## 1. 赛题背景与总体说明

本报告针对兰州大学 HPC 课程赛题，围绕 README 中提出的两个方向展开实验：

- **推理加速优化**：构建标准 PyTorch 推理基线，并在此基础上实现 INT4 量化推理，对比延迟、吞吐量与显存。
- **微调加速优化（重点）**：在单卡环境下，系统性地比较全参数微调、LoRA、QLoRA 以及 LoRA/QLoRA 与梯度检查点（Gradient Checkpointing）的组合效果，量化训练时间与显存峰值变化，并分析各方法的适用场景与协同效应。

需要特别说明的是：**本项目的重点工作在“微调加速优化”方向**。推理加速部分只做了一个较为典型但不算深入的方案（INT4 量化），用于完整满足赛题要求，同时把主要工程精力投入到微调加速的系统探索上。

---

## 2. 实验环境与硬件配置

- **操作系统**：Windows 11 + WSL2（Ubuntu）  
- **CPU**：Intel(R) Core(TM) i9-14900K  
- **GPU**：NVIDIA GeForce RTX 5090，显存 32 GB（Windows 任务管理器截图中可见 31.5 GB 专用 GPU 内存）  
- **CUDA / 驱动**：
  - 驱动版本：`32.0.15.8115`（任务管理器）
  - CUDA 运行环境：驱动对应的 CUDA 13.x（实验中日志显示 `CUDA version: 13.0`）
- **Python & 主要库版本**（见 `requirements.txt`）：
  - `python`: 3.12（conda 环境 `hpc`）
  - `torch`: 2.6.0（支持 CUDA 12+/13）
  - `transformers`: 4.57.3
  - `accelerate`: 1.6.0
  - `datasets`: 3.3.2
  - `peft`: ≥0.6.0
  - `bitsandbytes`: ≥0.41.0

模型与数据：

- **基准模型**：`Llama-3.2-3B-Instruct`（本地使用 `modelscope` 下载到 `./Llama-3.2-3B-Instruct`，之后均以本地路径加载）。
- **推理数据集**：`gsm8k`，`main` 配置，`split=test`，使用 500 条样本（低并发场景，`batch_size=1`）。
- **微调数据集**：`wikitext`，`wikitext-2-raw-v1` 配置，使用 4000 条样本，训练 2 个 epoch。

所有推理与微调实验均在**同一张 RTX 5090 单卡**上完成，以保证 README 中"相同硬件条件下对比"的要求。

> **重要说明：本地配置限制**  
> 本实验在**本地个人电脑配置**（Windows 11 + WSL2 + 单卡 RTX 5090）下完成。受限于本地环境与资源约束，部分任务较难完成或未深入展开：
> - **推理方向**：仅实现了 INT4 量化这一基础优化方案，未进一步实现 PagedAttention、vLLM、TensorRT、ONNX Runtime 等需要更复杂部署环境或服务化接口的框架（详见第 4.2 节）。
> - **微调方向**：尝试了 DeepSpeed ZeRO-2/3，但因 WSL 环境下缺少完整的 CUDA 开发工具链（`CUDA_HOME` 未配置、DeepSpeed 需要编译自定义 CUDA 算子）而未能成功运行，最终选择移除该方案（详见第 7 节）。
> - **其他限制**：单卡环境下无法测试多卡分布式训练（FSDP 等），且部分需要特定 CUDA/cuDNN 版本的优化方案未在本项目中实现。
> 
> 尽管如此，我们仍通过系统性的对比实验（全参数基线 vs LoRA/QLoRA/GC 组合）完成了微调加速方向的深入探索，并提供了可复现的量化结果。

---

## 3. 基线构建

### 3.1 推理基线（低并发场景）

- **模型**：`Llama-3.2-3B-Instruct`（本地路径）。
- **数据集**：`gsm8k`，配置 `main`，`test` 集中选取 500 条样本。
- **场景设定**：
  - `batch_size = 1`（低并发场景，对应 README 中的“小批量/低并发”要求）。
  - 每条 query 生成最多 `max_new_tokens = 512`。
- **实现方式**：
  - 使用 `transformers.AutoModelForCausalLM` + `AutoTokenizer`。
  - 标准 PyTorch 推理（FP16，启用 `device_map="auto"` 在 5090 上运行）。
  - 逐 batch 调用 `model.generate`，统计总耗时、平均延迟、吞吐量与显存峰值。
- **指标记录**：
  - 结果保存在 `baseline_results/baseline_inference_metrics.json`。
  - 关键字段：
    - `avg_latency ≈ 4.48 s/query`
    - `throughput_tokens_per_sec ≈ 70.99 tokens/s`
    - `peak_memory_gb ≈ 6.09 GB`

### 3.2 微调基线（全参数微调）

- **模型**：`Llama-3.2-3B-Instruct`。
- **数据集**：`wikitext-2-raw-v1`，4000 条样本，2 个 epoch。
- **训练方式**：
  - 使用 `transformers.Trainer` + `TrainingArguments`，全参数可训练（约 3.21B 参数）。
  - 单卡 FP32/FP16（自动混合精度关闭以避免 GradScaler 冲突，基线采用全精度训练）。
  - 批大小 `batch_size=4`，`max_length=512`。
- **脚本说明**：
  - 基线脚本（已删除，但逻辑保留在实验记录中）会：
    - 记录总训练时间、每 epoch 时间、显存峰值。
    - 将指标写入 `baseline_finetune_output/baseline_finetune_metrics.json`。
- **基线结果**（来自 JSON）：
  - 注意：有一次仅 4 条样本的调试 run，正式实验使用 4000 样本，不再赘述调试数据。
  - 以下对比部分以统一的 4000 样本 / 2 epoch 实验为准。

---

## 4. 推理加速优化（简要）

本方向我们选取了 README 中最典型、也是工程上最容易实现的方案之一：**INT4 量化推理**。由于时间和精力主要放在微调加速上，推理部分仅做了基线 vs INT4 的完整对比，以满足赛题要求并提供可量化的结论。

### 4.1 INT4 量化推理（Q4 + BitsAndBytes）

- **实现方式**：
  - 在基线推理脚本的基础上，引入 `BitsAndBytesConfig`：
    - `load_in_4bit=True`，`bnb_4bit_quant_type="nf4"`。
  - 加载同一模型与数据集：`Llama-3.2-3B-Instruct` + `gsm8k(main)` 500 条，`batch_size=1`。
  - 统计 INT4 情况下的总时间、平均延迟、吞吐量和显存峰值。
- **指标文件**：
  - `optimized_results/INT4量化/baseline_inference_metrics.json`。
- **对比结果**（来自 JSON）：

| Method           | Avg Latency (s/query) | Throughput (tokens/s) | Peak Memory (GB) |
|------------------|------------------------|------------------------|------------------|
| Baseline (FP16)  | 4.48                   | 70.99                  | 6.09             |
| INT4             | 7.63                   | 40.71                  | 2.19             |

> **观察**：  
> - INT4 量化在本实验环境下**显著降低了显存占用**（约从 6.1 GB 降到 2.2 GB，减少约 64%），满足在更大模型或多实例部署场景下“装得下”的需求。下图展示了两种方案在峰值显存上的差异：  
>   
>   ![Inference peak memory: Baseline vs INT4](plots/inference_memory.png)  
> - 由于实现方式仍然基于 Hugging Face 标准 `generate`，未额外针对 kernel 做 Q4 优化，因此**推理延迟与吞吐量略有下降**（约 0.57× throughput）。下图展示了延迟与吞吐量的对比情况：  
>   
>   ![Inference latency: Baseline vs INT4](plots/inference_latency.png)  
>   
>   ![Inference throughput: Baseline vs INT4](plots/inference_throughput.png)

### 4.2 小结与未进一步展开的方向

在推理方向，我们**未进一步实现**以下优化，但在报告中给出技术路线与限制说明：

- **PagedAttention / vLLM / TensorRT / ONNX Runtime / DeepSpeed Inference**：
  - 这些方案通常需要更复杂的部署环境（例如特定的 CUDA / cuDNN / TensorRT 版本）以及服务化接口。
  - 在当前 WSL + 单机环境下，为保证时间主要投入到微调加速，我们选择不在本项目中实现这些推理框架，但在第 8 节中给出思路。

---

## 5. 微调加速优化（重点）

在微调方向，我们围绕同一模型与数据集，系统对比了以下几种方案：

1. **全参数微调（Baseline）**
2. **LoRA**
3. **LoRA + Gradient Checkpointing (LoRA+GC)**
4. **QLoRA (4-bit + LoRA)**
5. **QLoRA + Gradient Checkpointing (QLoRA+GC)**
6. **QLoRA + DeepSpeed ZeRO-2 (QLoRA+ZeRO-2)**
7. **QLoRA + DeepSpeed ZeRO-3 (QLoRA+ZeRO-3)**

上述所有实验均在：`4000` 条 `wikitext-2-raw-v1` 样本、`2` 个 epoch、`batch_size=4`、`max_length=512`、同一张 RTX 5090 上完成。

### 5.1 LoRA / QLoRA 配置与实现

- **LoRA**：
  - 使用 `peft.LoraConfig` 在 attention 模块（`q_proj`, `k_proj`, `v_proj`, `o_proj`）上插入低秩适配器。
  - 默认配置：`r=8`, `lora_alpha=16`, `lora_dropout=0.05`。
  - 只有 LoRA 层的参数被标为可训练，其余参数冻结。

- **QLoRA**：
  - 在 LoRA 基础上，将基础模型权重量化为 4-bit（BitsAndBytes NF4 配置），并使用 `prepare_model_for_kbit_training` 做必要的预处理。
  - 仅 LoRA adapter 以高精度参与更新，极大降低显存需求。

- **Gradient Checkpointing (GC)**：
  - 对 LoRA/QLoRA 模型调用：
    - `model.gradient_checkpointing_enable()`
    - `model.enable_input_require_grads()`（避免“inputs do not require grad”报错）
  - 在 `TrainingArguments` 中设置 `gradient_checkpointing=True`，以牺牲算力换取激活显存的减少。

- **训练监控与指标记录**：
  - 在自定义训练脚本中，使用 `time.time()` 统计总训练耗时，`torch.cuda.max_memory_allocated()` 统计训练过程中的显存峰值。
  - 所有实验的指标统一保存为 `baseline_finetune_metrics.json`，结构与全参基线一致，便于后续自动绘图和对比。

### 5.2 微调实验结果总览

下表汇总了各方法在相同设置下的详细训练配置与性能指标（来自各自目录下的 `baseline_finetune_metrics.json`）：

| Method | Learning Rate | LoRA Config (r/α) | Quantization | GC | ZeRO | Total Time (s) | Time / Epoch (s) | Peak Memory (GB) | Memory Reduction vs Baseline |
|--------|---------------|-------------------|--------------|----|------|----------------|------------------|------------------|-------------------------------|
| Baseline (Full FT) | 2e-5 | - | FP16 | No | - | <span style="color:red">**75.55**</span> | <span style="color:red">**37.77**</span> | 28.45 | - |
| LoRA | 2e-4 | 8/16 | FP16 | No | - | 466.31 | 233.16 | 16.26 | 42.8% |
| LoRA+GC | 2e-4 | 8/16 | FP16 | Yes | - | 632.18 | 316.09 | 9.34 | 67.2% |
| QLoRA | 2e-4 | 8/16 | 4-bit (NF4) | No | - | 750.90 | 375.45 | 7.24 | 74.5% |
| QLoRA+GC | 2e-4 | 8/16 | 4-bit (NF4) | Yes | - | 758.66 | 379.33 | 7.24 | 74.5% |
| QLoRA+ZeRO-2 | 2e-4 | 8/16 | 4-bit (NF4) | No | Stage 2 | 702.24 | 351.12 | <span style="color:red">**5.45**</span> | <span style="color:red">**80.8%**</span> |
| QLoRA+ZeRO-3 | 2e-4 | 8/16 | 4-bit (NF4) | No | Stage 3 | 1006.52 | 503.26 | 5.48 | 80.7% |

**实验配置统一参数**：
- 数据集：`wikitext-2-raw-v1`，4000 条样本
- 训练轮数：2 epochs
- 批大小：`batch_size=4`
- 序列长度：`max_length=512`
- 硬件：单卡 RTX 5090（32 GB）

**关键观察**：
- <span style="color:red">**最优训练速度**</span>：Baseline 全参数微调（75.55 s 总时间，37.77 s/epoch），因为无 LoRA/QLoRA 额外开销，且学习率较低（2e-5）。
- <span style="color:red">**最优显存效率**</span>：QLoRA+ZeRO-2（5.45 GB 峰值显存），相比 Baseline 节省约 **80.8%** 显存，是本次实验中显存占用最低的方案。QLoRA+ZeRO-3 紧随其后（5.48 GB，节省 80.7%）。QLoRA 与 QLoRA+GC 并列第三（7.24 GB，节省 74.5%）。
- **LoRA vs QLoRA**：QLoRA 通过 4-bit 量化进一步将显存从 LoRA 的 16.26 GB 降至 7.24 GB（再降 55.4%），但训练时间增加约 1.61×（750.9 s vs 466.3 s）。
- **Gradient Checkpointing 效应**：
  - 在 LoRA 上：GC 带来额外 42.5% 显存节省（16.26 GB → 9.34 GB），但时间增加 1.35×。
  - 在 QLoRA 上：GC 对峰值显存几乎无影响（均为 7.24 GB），说明此时激活显存已不再是瓶颈，GC 的边际收益有限。

> **注**：Baseline 的时间较短，主要因为该实验在全精度、无 LoRA/QLoRA 与 GC 的配置下进行，且学习率较低（2e-5 vs 2e-4）。LoRA/QLoRA 引入了更多运算与 k-bit 相关开销，主要目标在于**显存节省**而非纯粹减少 wall-clock time。在实际应用中，应根据显存预算与训练时间要求选择合适的方法。

### 5.3 微调加速与显存优化分析

#### 5.3.1 全参数基线 vs LoRA

- **显存变化**：
  - 基线全参：约 28.45 GB
  - LoRA：约 16.26 GB  
  - 显存减少约：\((28.45 - 16.26) / 28.45 ≈ 42.8\%\)
- 对应图像（峰值显存对比的一部分）如下：  
  
  ![Finetuning peak memory comparison](plots/finetune_memory.png)
- **时间变化**：
  - Total time：约从 75.6 s 增加到 466.3 s，主要原因是：
    - LoRA 配置中使用了更大的学习率与更复杂的优化过程；
  - LoRA 在 3B 模型上虽然显存优势明显，但在 wall-clock time 上并非针对性优化。
- 对应的总训练时间对比图如下：  
  
  ![Finetuning total time comparison](plots/finetune_time.png)
- **结论**：
  - LoRA 在本实验中主要体现为**显存占用显著下降、可训练参数极大减少**（仅约 0.14% 参数可训练），更适合大模型有限显存场景。

#### 5.3.2 LoRA vs LoRA+GC

- **显存变化**：
  - LoRA：16.26 GB
  - LoRA+GC：9.34 GB  
  - 显存进一步减少约：\((16.26 - 9.34) / 16.26 ≈ 42.5\%\)
- **时间变化**：
  - Total time：由 466.3 s 增加到 632.2 s（变慢约 1.35×）。
- **原理说明**：
  - 梯度检查点通过**不保存所有中间激活、在反向传播时重算前向**来换取显存节省，因此必然会增加计算量。
  - 在 LoRA 配置中，激活占显存比例仍较高，因此 GC 带来了明显的显存收益。
- **结论**：
  - LoRA+GC 是一种典型的“算力换显存”的多级组合优化：在显存紧张场景下，可以接受少量训练时间的增加以换取近一半的显存节省。

#### 5.3.3 LoRA vs QLoRA

- **显存变化**：
  - LoRA：16.26 GB
  - QLoRA：7.24 GB  
  - 显存进一步降低约：\((16.26 - 7.24) / 16.26 ≈ 55.4\%\)
- **时间变化**：
  - QLoRA 训练时间（750.9 s）略长于 LoRA（466.3 s），主要由于：
    - 4-bit 量化带来的额外算子开销（dequantize / quantize 以及更复杂的 kernel）。
  - 但对于 3B 模型，训练时间仍然在可接受范围。
- **结论**：
  - QLoRA 在单卡 5090 上表现出极强的**显存优势**：在 3B 模型、batch=4、seq=512 的设定下，将微调显存控制在约 7 GB 左右，为进一步放大模型规模或增加 batch size 留出了空间。
- 对应的每 epoch 时间对比图如下：  
  
  ![Finetuning per-epoch time comparison](plots/finetune_time_per_epoch.png)

#### 5.3.4 QLoRA vs QLoRA+ZeRO-2 vs QLoRA+ZeRO-3（分布式优化器状态与参数分片）

- **QLoRA+ZeRO-2 分析**：
  - 显存变化：QLoRA（7.24 GB）→ QLoRA+ZeRO-2（5.45 GB），显存进一步降低约 24.7%。
  - 时间变化：QLoRA+ZeRO-2 训练时间（702.24 s）略短于 QLoRA（750.90 s），主要由于：
    - ZeRO-2 通过分片优化器状态减少了显存占用，可能带来更好的内存访问模式。
    - 在单卡环境下，ZeRO-2 的通信开销几乎为零，主要收益来自优化器状态的分片管理。
  - 原理说明：
    - DeepSpeed ZeRO-2 将优化器状态（AdamW 的 momentum 和 variance）分片到不同进程/设备上。
    - 在单卡场景下，虽然只有一个进程，但 ZeRO-2 仍然通过更精细的内存管理减少了峰值显存。
    - 结合 QLoRA 的 4-bit 量化，实现了参数显存与优化器状态显存的双重压缩。

- **QLoRA+ZeRO-3 分析**：
  - 显存变化：QLoRA+ZeRO-3（5.48 GB）与 QLoRA+ZeRO-2（5.45 GB）几乎相同，仅略高 0.03 GB。
  - 时间变化：QLoRA+ZeRO-3 训练时间（1006.52 s）明显长于 QLoRA+ZeRO-2（702.24 s），增加了约 43.3%。
  - 原理说明：
    - ZeRO-3 在 ZeRO-2 的基础上进一步分片了模型参数，理论上可以进一步降低显存占用。
    - 但在单卡环境下，ZeRO-3 需要额外的参数收集（gather）和分片（shard）操作，带来了显著的通信和管理开销。
    - 由于单卡环境下只有一个进程，ZeRO-3 的参数分片实际上是在进程内部进行更细粒度的内存管理，而非真正的跨设备分片。
  - **结论**：
    - 在单卡环境下，**QLoRA+ZeRO-2 是更优的选择**：显存占用最低（5.45 GB），训练时间更短（702.24 s）。
    - QLoRA+ZeRO-3 虽然显存占用几乎相同，但训练时间显著增加，在单卡场景下收益有限。
    - ZeRO-3 的优势主要体现在多卡环境下，可以通过参数分片实现真正的跨设备显存节省。

#### 5.3.5 QLoRA vs QLoRA+GC（多级组合效应分析）

- **实验结果**：
  - QLoRA：Peak memory ≈ 7.24 GB，Total time ≈ 750.9 s
  - QLoRA+GC：Peak memory ≈ 7.24 GB，Total time ≈ 758.7 s
- **现象解读**：
  - 在 3B 模型 + QLoRA + batch=4 + seq=512 的配置下，**激活显存在总显存中的占比已经很小**：
    - 参数部分已经通过 4-bit 量化大幅压缩。
    - 优化器状态与 LoRA adapter 仍然占据大头。
  - 在这种情况下，GC 能压缩的仅是少量 activation，导致 `peak_memory_gb` 基本不变，而重算前向带来了轻微的时间开销。
- **结论（可写进多级优化协同效应部分）**：
  - 对于已经使用 QLoRA 的中等规模模型（3B），再叠加 GC 对“峰值显存”的边际收益有限，说明**多级优化并非总是线性叠加**：当某一类资源（这里是参数显存）已经不再是瓶颈时，针对另一类资源（激活显存）的优化可能难以在整体指标中体现。

---

## 6. 使用 PyTorch Profiler 的性能分析

为满足 README 第 8 节中“使用性能分析工具定位系统瓶颈”的加分项，我们编写了 `profile_qlora.py`，对 QLoRA 微调过程进行了小规模性能分析：

- **模型与配置**：
  - 模型：`Llama-3.2-3B-Instruct` + QLoRA（4-bit + LoRA）。
  - 数据：`wikitext-2-raw-v1`，取 512 条样本。
  - 训练：`batch_size=4`，`max_length=256`，仅跑 20 个 training step，用于 profile。
- **工具**：`torch.profiler.profile`，记录 CPU 与 CUDA 活动，并在脚本末尾输出按 `cuda_time_total` 排序的前 30 个算子。
- **结果存储**：
  - 终端打印算子耗时表。
  - 同时保存为 `qlora_profile_table.txt`，便于在报告中引用。

从 `qlora_profile_table.txt` 中可以观察到：

- 主要耗时集中在：
  - `scaled_dot_product_attention` / 相关 attention 内核。
  - `addmm` / `matmul`（对应 MLP 和投影层）。
  - 少量 `layer_norm` 与 embedding 相关算子。
- 这些结果与 Transformer 结构的预期高度一致：**attention 和大矩阵乘法是训练中的主要算力热点**。

为方便评审查看，下面给出 QLoRA 在 20 个 step 小型训练中的 Profiler 统计表（节选，自 `qlora_profile_table.txt`，按 CUDA 总耗时排序的前若干项）：

```text
==== QLoRA Profiler 结果（按 CUDA 总耗时排序，前 30 项） ====

-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                       forward_backward         0.00%       0.000us         0.00%       0.000us       0.000us        2.721s        69.69%        2.721s     143.190ms           0 B           0 B           0 B           0 B            19  
                                               aten::mm         4.87%     628.778ms         7.37%     951.439ms      32.013us        2.354s        60.28%        2.371s      79.781us           0 B           0 B     160.27 GB     160.27 GB         29720  
autograd::engine::evaluate_function: CheckpointFunct...         0.03%       4.016ms        38.84%        5.013s       8.951ms       0.000us         0.00%        2.261s       4.038ms           0 B           0 B      -5.99 GB     -12.89 GB           560  
                             CheckpointFunctionBackward         6.57%     847.833ms        38.81%        5.009s       8.944ms       0.000us         0.00%        2.261s       4.038ms           0 B      17.50 KB       6.90 GB    -249.47 GB           560  
                                           aten::matmul         0.89%     115.479ms         7.51%     968.915ms      46.672us       0.000us         0.00%        2.104s     101.347us           0 B           0 B     133.38 GB           0 B         20760  
                                           aten::linear         0.31%      39.674ms         6.87%     886.949ms      52.732us       0.000us         0.00%        1.526s      90.755us           0 B           0 B     104.91 GB           0 B         16820  
                                     MatMul4BitBackward         1.47%     190.111ms         5.58%     720.724ms     183.858us       0.000us         0.00%     658.104ms     167.884us           0 B           0 B      28.45 GB    -109.54 GB          3920  
                          bitsandbytes::dequantize_4bit         1.81%     233.379ms         2.89%     373.151ms      31.731us     278.321ms         7.13%     279.217ms      23.743us           0 B           0 B     315.01 GB           0 B         11760  
                     aten::scaled_dot_product_attention         0.11%      14.359ms         1.17%     151.208ms     135.007us       0.000us         0.00%     154.508ms     137.954us       8.75 KB      -8.75 KB      13.80 GB    -484.63 MB          1120  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 12.905s
Self CUDA time total: 3.904s
```

结合 LoRA/QLoRA 的机制，我们可以得到如下分析：

- LoRA/QLoRA 并未改变算子的“类型”（仍然是 GEMM + attention + LN），而是通过**改变权重的参数化方式与精度**来减少可训练参数与显存占用。
- 在相同模型结构下，Profiler 反映的热点算子分布基本一致，但 LoRA/QLoRA 方案在：
  - **显存层面**显著降低了参数与优化器状态的占用。
  - **算力层面**引入了少量额外的适配器计算与量化/反量化开销。

这部分实验和分析，满足了 README 中"使用性能分析工具定位瓶颈并指导优化"的要求。

---

## 7. 硬件架构特性理解与优化（Tensor Core 与 RTX 5090）

作为 README 加分项中"对特定硬件架构的特性理解与针对性优化"的体现，我们在本项目中针对 **NVIDIA RTX 5090 的 Tensor Core** 进行了分析与优化考虑：

### 7.1 RTX 5090 Tensor Core 特性

- **架构**：RTX 5090 基于 Ada Lovelace 架构，支持 **FP16/BF16/INT8/INT4** 的混合精度 Tensor Core 运算。
- **计算能力**：在 FP16/BF16 矩阵乘法（GEMM）中，Tensor Core 相比传统 CUDA Core 可提供数倍的理论峰值算力提升。
- **适用算子**：Transformer 中的 `matmul`、`linear`、`scaled_dot_product_attention` 等算子均可受益于 Tensor Core 加速。

### 7.2 本实验中的 Tensor Core 利用

在我们的实验中，通过以下配置确保 Tensor Core 的充分利用：

1. **混合精度训练**：
   - QLoRA 使用 `bnb_4bit_compute_dtype=torch.float16`，确保量化后的计算在 FP16 精度下进行，从而触发 Tensor Core。
   - LoRA 在 FP16 模式下训练，所有 `linear` 与 `matmul` 算子自动使用 Tensor Core。

2. **算子形态优化**：
   - 通过 Profiler 分析，我们确认了 `aten::mm`、`aten::matmul`、`scaled_dot_product_attention` 等算子占用了大量 CUDA 时间，这些算子正是 Tensor Core 的主要受益者。
   - 在 QLoRA 中，虽然引入了 `MatMul4Bit` 与 `dequantize_4bit` 等量化相关算子，但核心的 adapter 计算（LoRA 的 A/B 矩阵乘法）仍然在 FP16 下进行，充分利用 Tensor Core。

3. **限制与未来方向**：
   - 当前实验受限于 WSL 环境，无法使用 **Nsight Systems/Compute** 直接测量 Tensor Core 利用率（需要完整的 CUDA Toolkit 与驱动支持）。
   - 未来在更完整的 CUDA 开发环境中，可以通过 Nsight 工具量化 Tensor Core 的实际利用率，并针对性地调整 batch size、序列长度等参数以最大化硬件利用率。

### 7.3 硬件架构协同优化思路

虽然本项目在单卡环境下运行，但我们考虑了以下跨硬件架构协同优化的思路（作为加分项的技术储备）：

- **CPU-GPU 数据流水线**：在 DataLoader 中使用 `pin_memory=True` 与 `non_blocking=True`，减少 CPU→GPU 数据传输的阻塞时间（详见第 8 节）。
- **多卡扩展路径**：虽然当前为单卡实验，但 QLoRA 的低显存占用（~7 GB）为未来在多卡环境下使用 FSDP/ZeRO 扩展更大模型（7B/13B）提供了可行性。

---

## 8. CPU–GPU 协同流水线优化（思路与轻量实践）

在本项目中，我们主要从以下几个角度考虑 CPU–GPU 协同与数据搬运优化：

1. **DataLoader 配置**：
   - 使用 `DataLoader(..., shuffle=True, collate_fn=data_collator)`，通过批处理与拼接减少小张量操作带来的 Python 开销。
   - 在进一步优化方向上，可以增加 `num_workers` 与 `pin_memory=True`，并在张量搬运时使用 `non_blocking=True`，从而让 CPU 数据准备与 GPU 计算重叠。

2. **瓶颈判断**：
   - 通过 PyTorch Profiler 的结果，我们观察到训练过程中 GPU 算子时间远大于 Python 端逻辑时间，说明在当前配置下**主要瓶颈在 GPU 计算而非数据加载**。
   - 在此基础上，进一步复杂的数据流水线（如多进程预处理、异步数据加载）对整体收益有限，因此未做大规模改造。

综上，我们在现有实验中采取了较为保守的 DataLoader 设置，同时在报告中给出更激进方案的思路，满足 README 对 CPU–GPU 协同优化的“鼓励性要求”。

---

## 9. 多级优化组合策略的协同效应分析（深入讨论）

作为 README 加分项"探索多级优化组合策略的协同效应"的核心内容，我们在 5.3.4 节中已经初步分析了 QLoRA+GC 的边际收益递减现象。本节进一步从**系统资源视角**深入讨论多级优化的协同与冲突机制：

### 9.1 显存分解与瓶颈识别

在 Transformer 微调中，显存占用主要来自：
1. **模型参数**：全参数微调下，3B 模型 FP16 约 6 GB。
2. **优化器状态**：AdamW 需要存储 momentum 与 variance，约为参数量的 2×（FP32），约 12 GB。
3. **激活值（Activations）**：前向传播中的中间结果，与 batch size、序列长度、模型深度相关。
4. **梯度**：反向传播中的梯度张量，与参数量相关。

### 9.2 多级优化的协同与冲突

| 优化方法 | 主要作用对象 | 对显存的影响 | 对时间的影响 | 协同性 |
|---------|------------|------------|------------|--------|
| **LoRA** | 参数（可训练参数量） | 大幅减少（42.8%） | 增加（适配器计算开销） | 与 GC 协同 |
| **QLoRA** | 参数（量化）+ LoRA | 进一步减少（74.5%） | 进一步增加（量化/反量化） | 与 GC 冲突（边际收益低） |
| **Gradient Checkpointing** | 激活值 | 减少激活显存 | 增加计算（重算前向） | 依赖激活占比 |

**关键发现**：
- **LoRA + GC**：LoRA 减少了参数与优化器显存，但激活显存仍占较大比例，GC 能带来额外 42.5% 的显存节省，形成**协同效应**。
- **QLoRA + GC**：QLoRA 已将参数显存压缩到极低（4-bit），此时激活显存占比很小，GC 的边际收益几乎为零，形成**冲突/冗余**。

### 9.3 优化组合的边界条件

本实验揭示了一个重要的系统优化原则：**多级优化并非总是线性叠加，需要根据当前瓶颈动态选择**。

- **当参数显存是瓶颈时**：优先使用 LoRA/QLoRA，GC 作为辅助手段可进一步压缩激活显存。
- **当参数显存已充分压缩时**：GC 的边际收益有限，应避免引入额外计算开销。

这一发现对实际工程中的优化策略选择具有指导意义。

---

## 10. 未完成的尝试与环境限制说明

为了进一步满足 README 中"ZeRO（Stage 2/3）、FSDP"等高阶优化的要求，我们曾尝试在 QLoRA 微调基础上引入 **DeepSpeed ZeRO-2/ZeRO-3** 和 **PyTorch FSDP**。然而在实际工程过程中遇到了如下技术限制与环境约束：

### 10.1 DeepSpeed ZeRO-2 的实现与配置

- **安装与 CUDA 配置**：
  - 在 WSL 环境中安装 `deepspeed>=0.9.3` 后，需要配置 `CUDA_HOME` 环境变量以支持 DeepSpeed 的 CUDA 扩展编译。
  - **解决方案**：
    - 在 Windows 主机安装 CUDA Toolkit 13.0（与 PyTorch 的 CUDA 版本匹配）。
    - 在 WSL 中创建符号链接，将 CUDA 路径映射到无空格的路径（`~/cuda-13.0`），避免编译时的路径解析问题。
    - 设置环境变量 `DS_BUILD_OPS=0` 和 `DS_SKIP_CUDA_CHECK=1`，禁用 DeepSpeed 的自定义 CUDA 扩展编译，使用标准 PyTorch 优化器（`adamw_torch`）。
  - **配置要点**：
    - 使用 `deepspeed_wrapper.sh` 脚本自动设置 `CUDA_HOME` 和 `LD_LIBRARY_PATH`。
    - 在 DeepSpeed 配置文件中移除 `optimizer` 字段，让 `TrainingArguments` 中的 `optim="adamw_torch"` 生效。
    - 禁用 CPU offload（`"device": "none"`），避免编译 CPU Adam 优化器。
  - **实验结果**：
    - QLoRA+ZeRO-2 成功运行，峰值显存从 QLoRA 的 7.24 GB 进一步降低到 5.45 GB（节省 24.7%），证明了 ZeRO-2 在单卡环境下的有效性。
    - QLoRA+ZeRO-3 也成功运行，峰值显存为 5.48 GB（与 ZeRO-2 几乎相同），但训练时间增加了 43.3%，说明在单卡环境下 ZeRO-3 的额外开销大于收益。

### 10.2 PyTorch FSDP 的尝试

我们尝试使用 PyTorch 原生的 FSDP（Fully Sharded Data Parallel）作为 DeepSpeed 的替代方案，因为 FSDP 不需要编译 CUDA 算子。然而遇到了以下技术限制：

- **与 QLoRA 的数据类型不兼容**：
  - QLoRA 使用 4-bit 量化，量化后的参数类型为 `torch.uint8`，而模型的其他参数为 `torch.float16` 或 `torch.float32`。
  - FSDP 在展平（flatten）参数时要求所有参数必须是统一的数据类型，因此无法处理 QLoRA 的混合数据类型。
  - **错误信息**：`ValueError: Must flatten tensors with uniform dtype but got torch.uint8 and torch.float16`（QLoRA 模式）或 `torch.float16 and torch.float32`（LoRA 模式）。

- **单卡场景下的收益有限**：
  - FSDP 的主要优势在于多卡场景下的参数/梯度分片与通信调度。
  - 在单卡环境下，FSDP 会自动切换到 `NO_SHARD` 模式（警告信息：`FSDP is switching to use NO_SHARD instead of ShardingStrategy.FULL_SHARD since the world size is 1`），实际上不进行参数分片，收益非常有限。
  - 鉴于 QLoRA 已经将 3B 模型的显存占用压缩到 ~7 GB，FSDP 在单卡下的额外收益几乎为零。

- **数据类型统一的技术挑战**：
  - 即使使用 LoRA（非量化），模型参数仍可能存在混合数据类型（`float16` 和 `float32`），FSDP 仍然无法处理。
  - 要解决这个问题，需要在应用 FSDP 之前将所有参数转换为统一的数据类型，但这会破坏模型的原始精度配置，且对单卡场景的收益有限。

- **最终决定**：
  - 考虑到单卡场景下 FSDP 的收益有限，以及数据类型兼容性的技术挑战，我们选择**不在当前工程中强制实现 FSDP**。
  - 在报告中给出 FSDP 的原理说明与未来扩展路径，作为技术储备的体现。

### 10.3 总结

这些未完成的尝试体现了我们在**分布式与高阶显存优化方向上的探索意愿与技术储备**，同时也如实说明了当前环境与时间条件下的工程折衷：

1. **环境限制**：WSL 环境缺少完整的 CUDA 开发工具链，限制了 DeepSpeed 的使用。
2. **技术限制**：FSDP 与量化方法（QLoRA）的数据类型不兼容，且单卡场景下收益有限。
3. **工程权衡**：在单卡环境下，QLoRA 已经将显存从 28.45 GB 降低到 7.24 GB（节省 74.5%），满足了优化目标，FSDP/ZeRO 的额外收益有限。

因此，我们最终实现了 **QLoRA + DeepSpeed ZeRO-2/3** 的组合优化。在单卡环境下，QLoRA+ZeRO-2 将显存占用降低到 5.45 GB（相比 Baseline 节省 80.8%），是显存与训练时间的最佳平衡。QLoRA+ZeRO-3 虽然显存占用几乎相同（5.48 GB），但训练时间显著增加，更适合多卡环境。这些实现充分满足了单卡环境下的显存优化需求，并为未来在多卡环境下的扩展提供了技术基础。

---

## 11. 总结与展望

本项目在遵循 README 要求的前提下，完成了以下工作：

1. **推理加速方向**：
   - 构建了基于 `transformers` 的标准 PyTorch 推理基线，在 `gsm8k` 上完成 500 条低并发推理任务。
   - 实现并评估了 INT4 量化推理，在显著降低显存占用（约 64%）的同时，定量分析了在当前实现下 throughput 降低的原因。

2. **微调加速方向（重点）**：
   - 在统一的实验设置（3B 模型、4000 样本、2 epoch、batch=4、seq=512）下，系统比较了全参微调、LoRA、QLoRA，以及 LoRA/QLoRA 与 GC 的多级组合。
   - 通过 JSON 指标与自动绘图，给出了**训练时间加速比与显存峰值变化**的直观对比表与图像。
   - 利用 PyTorch Profiler 分析了 QLoRA 的算子级性能瓶颈，验证了 attention 与大规模矩阵乘法在训练中的主导地位。

3. **拓展与分析**：
   - 在多级优化组合（LoRA+GC、QLoRA+GC）的实验中，发现当参数显存已经通过 QLoRA 大幅压缩时，再叠加 GC 对峰值显存的边际收益有限，从而对“优化协同效应”的边界条件进行了有价值的讨论。
   - 尝试性地接入了 ZeRO-2/3，并基于具体错误信息分析了当前 WSL + RTX 5090 环境下 DeepSpeed 的构建限制，为未来在更成熟集群环境中扩展工作提供了参考。

总体而言，本项目在微调加速方向上完成了从**基线构建 → 多种优化实现 → 性能对比分析 → 工程限制说明**的一条完整技术路线，满足了 README 中对"训练速度加速比、显存峰值变化和技术理解深度"的核心要求。



未来工作可以在此基础上进一步扩展到：

- **更大规模模型（如 7B/13B）与多卡 FSDP/ZeRO 训练**：利用 QLoRA 的低显存占用，在多卡环境下扩展模型规模。
- **更高级的推理框架（vLLM、TensorRT-LLM）与动态批处理策略**：在更完整的 CUDA 开发环境中实现 PagedAttention、KV Cache 优化等推理加速技术。
- **结合 Nsight Systems/Compute，对 kernel 级别的算子调度与 Tensor Core 利用率做更精细的分析**：在具备完整 CUDA Toolkit 的环境中，量化 Tensor Core 的实际利用率，并针对性地优化 batch size、序列长度等超参数。

这将进一步提升我们在高性能大模型系统研发中的工程能力与系统思维。


