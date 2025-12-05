"""
使用 DeepSpeed ZeRO-3 进行微调的脚本
基于 finetune_zero2.py，使用 ZeRO-3 配置
"""
import os

# 在导入 deepspeed 之前禁用 CUDA 扩展编译（WSL 环境无法编译）
os.environ['DS_BUILD_OPS'] = '0'
os.environ['DS_SKIP_CUDA_CHECK'] = '1'

# 在导入 deepspeed 之前设置 CUDA_HOME（如果未设置）
# 优先使用符号链接路径（避免空格问题），如果没有则使用原始路径
if 'CUDA_HOME' not in os.environ:
    import os.path as osp
    home_dir = os.path.expanduser("~")
    
    # 优先使用符号链接路径（避免空格）
    cuda_link_v13_0 = osp.join(home_dir, "cuda-13.0")
    cuda_link_v13_1 = osp.join(home_dir, "cuda-13.1")
    
    # 原始路径
    cuda_source_v13_0 = "/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0"
    cuda_source_v13_1 = "/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1"
    
    cuda_home = None
    
    # 优先使用符号链接
    if osp.exists(cuda_link_v13_0) or osp.islink(cuda_link_v13_0):
        cuda_home = cuda_link_v13_0
    elif osp.exists(cuda_link_v13_1) or osp.islink(cuda_link_v13_1):
        cuda_home = cuda_link_v13_1
    # 如果没有符号链接，检查原始路径并创建符号链接
    elif osp.exists(cuda_source_v13_0):
        try:
            if not osp.exists(cuda_link_v13_0):
                os.symlink(cuda_source_v13_0, cuda_link_v13_0)
            cuda_home = cuda_link_v13_0
        except:
            cuda_home = cuda_source_v13_0
    elif osp.exists(cuda_source_v13_1):
        try:
            if not osp.exists(cuda_link_v13_1):
                os.symlink(cuda_source_v13_1, cuda_link_v13_1)
            cuda_home = cuda_link_v13_1
        except:
            cuda_home = cuda_source_v13_1
    
    if cuda_home:
        os.environ['CUDA_HOME'] = cuda_home
        os.environ['PATH'] = f"{cuda_home}/bin:" + os.environ.get('PATH', '')
        # 创建 nvcc 符号链接（如果不存在）
        nvcc_path = osp.join(cuda_home, "bin", "nvcc")
        nvcc_exe_path = osp.join(cuda_home, "bin", "nvcc.exe")
        if osp.exists(nvcc_exe_path) and not osp.exists(nvcc_path):
            try:
                os.symlink(nvcc_exe_path, nvcc_path)
            except:
                pass

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
    """加载模型用于训练（与 finetune.py 相同）"""
    print(f"正在加载模型用于训练: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
    }

    if use_qlora:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device == "cuda" else None,
        **model_kwargs
    )
    if device == "cpu":
        model = model.to(device)

    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    if use_lora or use_qlora:
        if target_modules is None:
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

    model.train()
    return model, tokenizer


def prepare_dataset(
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: Optional[str] = None,
    num_samples: int = 4000,
    max_length: int = 512,
):
    """准备数据集（与 finetune.py 相同）"""
    print(f"加载数据集: {dataset_name}, 配置: {dataset_config}")
    
    if dataset_config is None:
        try:
            from datasets import get_dataset_config_names
            available_configs = get_dataset_config_names(dataset_name)
            if available_configs:
                dataset_config = available_configs[0]
                print(f"自动选择配置: {dataset_config}")
        except:
            pass
    
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split="train")
    else:
        dataset = load_dataset(dataset_name, split="train")
    
    dataset = dataset.select(range(min(num_samples, len(dataset))))

    def tokenize_fn(examples):
        texts = examples.get("text", [])
        if isinstance(texts, str):
            texts = [texts]
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="使用 DeepSpeed ZeRO-3 进行微调")
    parser.add_argument("--model_name", type=str, default="./Llama-3.2-3B-Instruct",
                        help="模型路径")
    parser.add_argument("--dataset_name", type=str, default="wikitext",
                        help="数据集名称")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1",
                        help="数据集配置")
    parser.add_argument("--output_dir", type=str, default="./optimized_finetune_output/qlora_zero3",
                        help="输出目录")
    parser.add_argument("--num_epochs", type=int, default=2,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="批大小")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="学习率")
    parser.add_argument("--max_length", type=int, default=512,
                        help="最大序列长度")
    parser.add_argument("--use_lora", action="store_true",
                        help="使用 LoRA")
    parser.add_argument("--use_qlora", action="store_true",
                        help="使用 QLoRA")
    parser.add_argument("--num_samples", type=int, default=4000,
                        help="样本数量")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备类型")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="启用梯度检查点")
    parser.add_argument("--deepspeed_config", type=str, default="ds_zero3.json",
                        help="DeepSpeed 配置文件路径")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="本地 rank（由 DeepSpeed launcher 自动设置）")
    
    args = parser.parse_args()

    # 检查并设置 CUDA_HOME（在导入 deepspeed 之前）
    if 'CUDA_HOME' not in os.environ:
        cuda_home = "/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1"
        if os.path.exists(cuda_home):
            os.environ['CUDA_HOME'] = cuda_home
            os.environ['PATH'] = f"{cuda_home}/bin:" + os.environ.get('PATH', '')
            print(f"已设置 CUDA_HOME: {cuda_home}")
    
    # 检查 DeepSpeed 是否可用
    try:
        import deepspeed
        print(f"DeepSpeed version: {deepspeed.__version__}")
    except ImportError:
        print("错误: DeepSpeed 未安装。请运行: pip install deepspeed")
        return
    except Exception as e:
        if "CUDA_HOME" in str(e):
            print(f"错误: DeepSpeed 需要 CUDA_HOME。当前 CUDA_HOME: {os.environ.get('CUDA_HOME', '未设置')}")
            print("请确保 CUDA Toolkit 已安装，并设置 CUDA_HOME 环境变量")
        else:
            print(f"错误: DeepSpeed 导入失败: {e}")
        return

    # 检查 CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA 不可用，将使用 CPU")
        args.device = "cpu"

    # 加载模型和分词器
    model, tokenizer = load_model_for_training(
        model_name=args.model_name,
        device=args.device,
        use_lora=args.use_lora,
        use_qlora=args.use_qlora,
    )

    # 梯度检查点
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # 准备数据集
    tokenized_dataset = prepare_dataset(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        num_samples=args.num_samples,
        max_length=args.max_length,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 训练参数（DeepSpeed 会自动检测 deepspeed 参数）
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fp16=True,  # DeepSpeed ZeRO-3 需要 FP16
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
        # DeepSpeed 配置
        deepspeed=args.deepspeed_config,  # 指定配置文件
        # 使用标准 PyTorch 优化器，避免编译 DeepSpeed 自定义扩展
        optim="adamw_torch",
    )

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # 开始训练并计时
    print("\n开始训练...")
    start_time = time.time()
    trainer.train()
    total_time = time.time() - start_time

    # 保存模型（与 finetune.py 保持一致）
    print(f"\n保存模型到: {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    # 显存使用情况
    peak_memory_gb = None
    if args.device == "cuda":
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"峰值显存使用: {peak_memory_gb:.2f} GB")

    # 组装并保存指标，格式与 baseline_finetune 对齐，便于对比
    metrics = {
        "total_time_seconds": total_time,
        "time_per_epoch_seconds": total_time / args.num_epochs if args.num_epochs else None,
        "num_samples": len(tokenized_dataset),
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

