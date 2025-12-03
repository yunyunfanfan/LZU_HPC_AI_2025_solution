"""
QloRA 小型性能分析脚本

用途：
- 跑少量 step 的 QLoRA 训练（如 20 步）
- 使用 torch.profiler 记录算子耗时（CPU/CUDA）
- 在终端打印 Top-K 耗时算子表，用于写报告分析瓶颈
"""

import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./Llama-3.2-3B-Instruct"
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
NUM_SAMPLES = 512          # 只用少量样本做分析
MAX_LENGTH = 256
BATCH_SIZE = 4
MAX_STEPS = 20             # 只跑 20 个 step 做 profile


def load_qlora_model():
    print(f"加载 QLoRA 模型: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if DEVICE == "cuda" else torch.float32,
    }
    # QLoRA 4-bit 量化配置
    from transformers import BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto" if DEVICE == "cuda" else None,
        **model_kwargs,
    )
    if DEVICE == "cpu":
        model = model.to(DEVICE)

    # 准备模型用于 k-bit 训练
    model = prepare_model_for_kbit_training(model)

    # 应用 LoRA 适配器
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()
    return model, tokenizer


def prepare_dataset(tokenizer):
    print(f"加载数据集: {DATASET_NAME}, 配置: {DATASET_CONFIG}")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    dataset = dataset.select(range(min(NUM_SAMPLES, len(dataset))))

    def tokenize_fn(examples):
        texts = examples.get("text", [])
        if isinstance(texts, str):
            texts = [texts]
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )
    print(f"数据集准备完成，样本数: {len(tokenized)}")
    return tokenized


def main():
    model, tokenizer = load_qlora_model()
    dataset = prepare_dataset(tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    print("\n开始 QLoRA + Profiler 小型训练（仅用于性能分析）")
    activities = [ProfilerActivity.CPU]
    if DEVICE == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
        profile_memory=True,
    ) as prof:
        step = 0
        for batch in dataloader:
            if step >= MAX_STEPS:
                break
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            with record_function("forward_backward"):
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
            optimizer.step()
            step += 1
            if step % 5 == 0:
                print(f"step {step}/{MAX_STEPS}, loss={loss.item():.4f}")

    print("\n==== 按 CUDA 总耗时排序的算子统计（前 30 项） ====\n")
    sort_key = "cuda_time_total" if DEVICE == "cuda" else "cpu_time_total"
    table_str = prof.key_averages().table(sort_by=sort_key, row_limit=30)
    print(table_str)

    # 将表格结果保存到文件，便于在报告中引用
    output_path = "qlora_profile_table.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("==== QLoRA Profiler 结果（按 CUDA 总耗时排序，前 30 项） ====\n\n")
        f.write(table_str)
    print(f"\nProfiler 表格结果已保存到: {output_path}")

    # 如果需要可视化 timeline，可取消下面注释，生成 Chrome trace 文件
    # prof.export_chrome_trace("qlora_profile_trace.json")


if __name__ == "__main__":
    main()

