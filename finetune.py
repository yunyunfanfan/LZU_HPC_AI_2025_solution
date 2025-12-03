"""
Llama 模型微调脚本
支持 LoRA/QLoRA 等参数高效微调方法
"""
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import argparse
from typing import Optional
import time
import json
import os


def load_model_for_training(
    model_name: str,
    device: str = "cuda",
    use_lora: bool = True,
    use_qlora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None,
):
    """
    加载模型用于训练
    
    Args:
        model_name: 模型名称或路径
        device: 设备类型
        use_lora: 是否使用 LoRA
        use_qlora: 是否使用 QLoRA (4-bit 量化 + LoRA)
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        target_modules: LoRA 目标模块（如 ["q_proj", "v_proj"]）
    """
    print(f"正在加载模型用于训练: {model_name}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 模型加载参数
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
    }
    
    # QLoRA: 4-bit 量化
    if use_qlora:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model_kwargs["quantization_config"] = quantization_config
        print("使用 QLoRA (4-bit 量化 + LoRA)")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device == "cuda" else None,
        **model_kwargs
    )
    
    if device == "cpu":
        model = model.to(device)
    
    # 准备模型用于训练（QLoRA 需要）
    if use_qlora:
        model = prepare_model_for_kbit_training(model)
    
    # 应用 LoRA
    if use_lora or use_qlora:
        if target_modules is None:
            # 默认目标模块（适用于 Llama）
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print("LoRA 配置完成")
    
    return model, tokenizer


def prepare_dataset(
    dataset_name: str,
    tokenizer,
    max_length: int = 512,
    num_samples: Optional[int] = None,
    dataset_config: Optional[str] = None,
):
    """
    准备训练数据集
    
    Args:
        dataset_name: 数据集名称或路径
        tokenizer: 分词器
        max_length: 最大序列长度
        num_samples: 使用的样本数量（None 表示使用全部）
        dataset_config: 数据集配置名称（如 wikitext 需要 'wikitext-2-raw-v1'）
    """
    print(f"正在加载数据集: {dataset_name}")
    
    # 加载数据集（处理需要 config_name 的情况）
    try:
        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config, split="train")
        else:
            dataset = load_dataset(dataset_name, split="train")
    except ValueError as e:
        # 自动从报错中解析可用的 config，并选择一个默认值
        error_msg = str(e)
        if "Config name is missing" in error_msg and "available configs" in error_msg:
            import re

            match = re.search(r"available configs: \\[(.*?)\\]", error_msg)
            if match:
                configs = [
                    c.strip().strip("'\\\"") for c in match.group(1).split(",")
                ]
                preferred = None
                # 优先选择较小的数据集配置
                for cand in ["wikitext-2-raw-v1", "wikitext-2-v1", "main", "default"]:
                    if cand in configs:
                        preferred = cand
                        break
                if preferred is None and configs:
                    preferred = configs[0]
                if preferred is None:
                    raise
                print(
                    f"警告: 数据集 {dataset_name} 需要配置名称，自动使用: {preferred}"
                )
                dataset = load_dataset(dataset_name, preferred, split="train")
            else:
                raise
        else:
            raise
    
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    # 预处理函数
    def tokenize_function(examples):
        # 假设数据集有 "text" 字段，根据实际数据集调整
        texts = examples.get("text", examples.get("content", examples.get("instruction", [])))
        if isinstance(texts, str):
            texts = [texts]
        
        # 分词
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    # 应用预处理
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    print(f"数据集准备完成，样本数: {len(tokenized_dataset)}")
    return tokenized_dataset


def main():
    parser = argparse.ArgumentParser(description="Llama 模型微调脚本")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="模型名称或路径"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="数据集名称"
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="wikitext-2-raw-v1",
        help="数据集配置"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="输出目录"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=2,
        help="训练轮数"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="批次大小"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="学习率"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="最大序列长度"
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="使用 LoRA"
    )
    parser.add_argument(
        "--use_qlora",
        action="store_true",
        help="使用 QLoRA (4-bit + LoRA)"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="使用的样本数量（用于快速测试）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备类型"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="启用梯度检查点以进一步节省显存（LoRA/QLoRA 优化时使用）"
    )
    args = parser.parse_args()
    
    # 加载模型
    model, tokenizer = load_model_for_training(
        args.model_name,
        device=args.device,
        use_lora=args.use_lora,
        use_qlora=args.use_qlora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    # 如果启用梯度检查点，需要确保模型和输入允许计算梯度
    if args.gradient_checkpointing:
        print("启用梯度检查点：gradient_checkpointing_enable() + enable_input_require_grads()")
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    
    # 准备数据集
    dataset = prepare_dataset(
        args.dataset_name,
        tokenizer,
        max_length=args.max_length,
        num_samples=args.num_samples,
        dataset_config=args.dataset_config,
    )
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 因果语言模型，不使用 MLM
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fp16=args.device == "cuda",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        report_to="none",  # 可以改为 "tensorboard" 使用 TensorBoard
        gradient_checkpointing=args.gradient_checkpointing,
    )
    
    # 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # 开始训练并计时
    print("\n开始训练...")
    start_time = time.time()
    trainer.train()
    total_time = time.time() - start_time
    
    # 保存模型
    print(f"\n保存模型到: {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # 显存使用情况
    peak_memory_gb = None
    if args.device == "cuda":
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"峰值显存使用: {peak_memory_gb:.2f} GB")

    # 组装并保存指标，格式尽量与 baseline_finetune 对齐，便于对比
    metrics = {
        "total_time_seconds": total_time,
        "time_per_epoch_seconds": total_time / args.num_epochs if args.num_epochs else None,
        "num_samples": len(dataset),
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    if peak_memory_gb is not None:
        metrics["peak_memory_gb"] = peak_memory_gb

    metrics_payload = {
        "config": vars(args),
        "metrics": metrics,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "baseline_finetune_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    print(f"\n微调指标已保存到: {metrics_path}")


if __name__ == "__main__":
    main()

