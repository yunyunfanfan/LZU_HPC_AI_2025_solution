"""
标准 PyTorch 基线微调脚本
这是未优化的全参数微调实现，用于作为后续优化的对比基线
"""
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import argparse
import json
import os
import time
from tqdm import tqdm
from typing import Optional

def load_baseline_model_for_training(model_path: str, device: str = "cuda"):
    """
    加载基线模型用于训练（全参数微调，无优化）
    
    Args:
        model_path: 模型路径
        device: 设备类型
    """
    # 检查 CUDA 可用性和兼容性
    if device == "cuda":
        if not torch.cuda.is_available():
            print("警告: CUDA 不可用，自动切换到 CPU")
            device = "cpu"
        else:
            try:
                # 尝试创建一个简单的 tensor 来检查兼容性
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
                torch.cuda.empty_cache()
            except RuntimeError as e:
                print(f"警告: CUDA 兼容性问题 ({e})，自动切换到 CPU")
                device = "cpu"
    
    # 转换为绝对路径
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)
    
    print(f"正在加载基线模型用于训练: {model_path}")
    print(f"使用设备: {device}")
    
    # 检查路径是否存在
    if not os.path.exists(model_path):
        # 尝试查找可能的模型目录
        current_dir = os.path.dirname(model_path)
        base_name = os.path.basename(model_path)
        possible_paths = [
            model_path,
            os.path.join(current_dir, base_name),
            os.path.join(os.getcwd(), base_name),
        ]
        print(f"尝试查找模型路径...")
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                print(f"找到模型路径: {model_path}")
                break
        else:
            raise ValueError(f"模型路径不存在: {model_path}\n请检查模型是否已下载到正确位置")
    
    # 加载分词器（使用 local_files_only 确保从本地加载）
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception as e:
        # 如果 local_files_only 失败，尝试不使用它
        print(f"使用 local_files_only 失败，尝试不使用该选项: {e}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型（标准方式，全参数微调）
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None,
            local_files_only=True,
        )
    except Exception as e:
        # 如果 local_files_only 失败，尝试不使用它
        print(f"使用 local_files_only 失败，尝试不使用该选项: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None,
        )
    
    if device == "cpu":
        model = model.to(device)
    
    # 启用训练模式
    model.train()
    
    # 打印可训练参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params / 1e9:.2f}B")
    print(f"可训练参数量: {trainable_params / 1e9:.2f}B (全参数微调)")
    
    return model, tokenizer


def prepare_dataset(
    dataset_name: str,
    tokenizer,
    num_samples: int,
    max_length: int = 512,
    dataset_config: Optional[str] = None,
):
    """
    准备训练数据集
    
    Args:
        dataset_name: 数据集名称
        tokenizer: 分词器
        num_samples: 样本数量（至少4000）
        max_length: 最大序列长度
    """
    print(f"正在加载数据集: {dataset_name}")
    
    # 正确处理需要 config_name 的数据集（例如 wikitext、gsm8k 等）
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

            match = re.search(r"available configs: \[(.*?)\]", error_msg)
            if match:
                configs = [
                    c.strip().strip("'\"") for c in match.group(1).split(",")
                ]
                # 优先选用较小的数据集配置（例如 wikitext-2-raw-v1）
                preferred = None
                for cand in ["wikitext-2-raw-v1", "wikitext-2-v1", "main", "default"]:
                    if cand in configs:
                        preferred = cand
                        break
                if preferred is None:
                    preferred = configs[0]
                print(
                    f"警告: 数据集 {dataset_name} 需要配置名称，自动使用: {preferred}"
                )
                dataset = load_dataset(dataset_name, preferred, split="train")
            else:
                raise
        else:
            # 回退：尝试不带 split 的形式
            try:
                if dataset_config:
                    dataset = load_dataset(dataset_name, dataset_config)["train"]
                else:
                    dataset = load_dataset(dataset_name)["train"]
            except Exception as e2:
                raise ValueError(f"无法加载数据集 {dataset_name}: {e2}")
    
    # 选择指定数量的样本
    if len(dataset) > num_samples:
        dataset = dataset.select(range(num_samples))
    
    # 预处理函数
    def tokenize_function(examples):
        # 提取文本字段
        texts = []
        for item in examples:
            if isinstance(item, dict):
                # 尝试不同的字段名
                if "text" in item:
                    texts.append(item["text"])
                elif "content" in item:
                    texts.append(item["content"])
                elif "instruction" in item:
                    texts.append(item["instruction"])
                else:
                    # 使用第一个字符串字段
                    for key, value in item.items():
                        if isinstance(value, str) and len(value) > 10:
                            texts.append(value)
                            break
            elif isinstance(item, str):
                texts.append(item)
        
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
    parser = argparse.ArgumentParser(description="基线微调脚本")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./Llama-3.2-3B-Instruct",
        help="模型路径"
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
        "--num_samples",
        type=int,
        default=4000,
        help="样本数量（至少4000）"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=2,
        help="训练轮数（至少2）"
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
        default=2e-5,
        help="学习率"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="最大序列长度"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./baseline_finetune_output",
        help="输出目录"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备类型 (cuda/cpu)，如果不指定会自动检测"
    )
    
    args = parser.parse_args()
    
    # 自动检测设备
    if args.device is None:
        if torch.cuda.is_available():
            try:
                # 测试 CUDA 兼容性
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                args.device = "cuda"
            except RuntimeError:
                print("警告: CUDA 不可用或不兼容，使用 CPU")
                args.device = "cpu"
        else:
            args.device = "cpu"
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    model, tokenizer = load_baseline_model_for_training(args.model_path, args.device)
    
    # 准备数据集
    dataset = prepare_dataset(
        args.dataset_name,
        tokenizer,
        args.num_samples,
        args.max_length,
        args.dataset_config,
    )
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 因果语言模型
    )
    
    # 清空显存统计
    if args.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # 训练参数
    # 说明：这里关闭 fp16 混合精度，避免 Accelerate 的 GradScaler 与当前环境冲突
    # 对于基线，全精度训练即可，后续可以在优化方案里单独开启混合精度进行对比
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fp16=False,  # 关闭 fp16，避免 \"Attempting to unscale FP16 gradients\" 错误
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        # 基线不使用梯度检查点
        gradient_checkpointing=False,
    )
    
    # 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    print("\n" + "=" * 50)
    print("开始基线微调（全参数微调）")
    print("=" * 50)
    print(f"模型: {args.model_path}")
    print(f"数据集: {args.dataset_name}")
    print(f"样本数: {len(dataset)}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"设备: {args.device}")
    print("-" * 50)
    
    start_time = time.time()
    trainer.train()
    total_time = time.time() - start_time
    
    # 保存模型
    print(f"\n保存模型到: {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # 收集性能指标
    metrics = {
        "total_time_seconds": total_time,
        "time_per_epoch_seconds": total_time / args.num_epochs,
        "num_samples": len(dataset),
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    }
    
    # 显存使用情况
    if args.device == "cuda":
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        metrics["peak_memory_gb"] = memory_used
    
    # 打印结果
    print("\n" + "=" * 50)
    print("基线微调性能指标")
    print("=" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # 保存结果
    output_file = os.path.join(args.output_dir, "baseline_finetune_metrics.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args),
            "metrics": metrics,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    main()

