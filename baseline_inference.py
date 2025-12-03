"""
标准 PyTorch 基线推理脚本
这是未优化的标准实现，用于作为后续优化的对比基线
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json
import argparse
from datasets import load_dataset
from tqdm import tqdm
import os


def load_baseline_model(model_path: str, device: str = "cuda"):
    """
    加载基线模型（标准方式，无任何优化）
    
    Args:
        model_path: 模型路径
        device: 设备类型
    """
    # 检查 CUDA 可用性
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，请检查 GPU 驱动和 CUDA 安装")
        else:
            print(f"检测到 GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA 版本: {torch.version.cuda}")
            # 检查计算能力
            if torch.cuda.is_available():
                capability = torch.cuda.get_device_capability(0)
                print(f"计算能力: {capability}")
                if capability[0] >= 12:
                    print("警告: RTX 5090 (sm_120) 需要 PyTorch nightly 版本支持")
                    print("如果遇到 CUDA 错误，请升级 PyTorch:")
                    print("  pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124")
    
    # 转换为绝对路径
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)
    
    print(f"正在加载基线模型: {model_path}")
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
    
    # 加载模型（标准方式，无优化）
    # 如果使用 CUDA，先检查兼容性
    if device == "cuda":
        try:
            # 测试 CUDA 是否真的可用
            test_tensor = torch.zeros(1, device="cuda")
            del test_tensor
            torch.cuda.empty_cache()
        except RuntimeError as e:
            error_msg = str(e)
            if "no kernel image" in error_msg or "sm_120" in error_msg:
                raise RuntimeError(
                    f"\n{'='*60}\n"
                    f"错误: PyTorch 不支持 RTX 5090 (sm_120)\n"
                    f"{'='*60}\n"
                    f"当前 PyTorch 版本: {torch.__version__}\n"
                    f"需要: PyTorch nightly 版本（支持 sm_120）\n\n"
                    f"解决方案:\n"
                    f"1. 卸载旧版本:\n"
                    f"   pip uninstall torch torchvision torchaudio -y\n\n"
                    f"2. 安装支持 RTX 5090 的版本:\n"
                    f"   pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124\n\n"
                    f"3. 验证安装:\n"
                    f"   python -c \"import torch; print(torch.__version__); print(torch.cuda.is_available())\"\n"
                    f"{'='*60}\n"
                ) from e
            else:
                raise
    
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
        error_msg = str(e)
        if "CUDA error" in error_msg or "no kernel image" in error_msg:
            raise RuntimeError(
                f"\n{'='*60}\n"
                f"错误: PyTorch 不支持 RTX 5090\n"
                f"{'='*60}\n"
                f"请升级 PyTorch 到支持 sm_120 的版本\n"
                f"详细步骤请查看: UPGRADE_PYTORCH_FOR_RTX5090.md\n"
                f"{'='*60}\n"
            ) from e
        print(f"使用 local_files_only 失败，尝试不使用该选项: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None,
        )
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    print("基线模型加载完成")
    
    return model, tokenizer


def inference_batch_baseline(
    model,
    tokenizer,
    prompts: list,
    batch_size: int = 1,
    max_new_tokens: int = 512,
    device: str = "cuda",
):
    """
    基线批量推理（标准实现，无优化）
    
    Returns:
        results: 生成结果列表
        metrics: 性能指标字典
    """
    results = []
    total_tokens = 0
    total_time = 0
    
    # 清空显存缓存
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="推理中"):
        batch_prompts = prompts[i:i + batch_size]
        
        # 编码输入
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        
        # 推理（标准方式）
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # 使用贪心解码，更稳定
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # 解码输出
        for j, output in enumerate(outputs):
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            results.append(generated_text)
            total_tokens += len(output)
    
    total_time = time.time() - start_time
    
    # 计算指标
    metrics = {
        "total_queries": len(prompts),
        "total_time": total_time,
        "avg_latency": total_time / len(prompts),
        "total_tokens": total_tokens,
        "throughput_tokens_per_sec": total_tokens / total_time if total_time > 0 else 0,
        "throughput_queries_per_sec": len(prompts) / total_time if total_time > 0 else 0,
    }
    
    # 显存使用情况
    if device == "cuda":
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        metrics["peak_memory_gb"] = memory_used
    
    return results, metrics


def load_inference_dataset(dataset_name: str, num_samples: int, split: str = "test", dataset_config: str = None):
    """
    加载推理数据集
    
    Args:
        dataset_name: 数据集名称
        num_samples: 样本数量
        split: 数据集分割
        dataset_config: 数据集配置名称（如 gsm8k 需要 'main' 或 'socratic'）
    """
    print(f"正在加载数据集: {dataset_name}")
    
    try:
        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
    except ValueError as e:
        # 如果缺少配置，尝试自动检测
        error_msg = str(e)
        if "Config name is missing" in error_msg or "Please pick one" in error_msg:
            # 提取可用的配置
            import re
            configs_match = re.search(r"available configs: \[(.*?)\]", error_msg)
            if configs_match:
                available_configs = [c.strip().strip("'\"") for c in configs_match.group(1).split(",")]
                # 使用第一个可用配置
                default_config = available_configs[0]
                print(f"警告: 数据集需要配置名称，自动使用: {default_config}")
                dataset = load_dataset(dataset_name, default_config, split=split)
            else:
                # 尝试常见配置名称
                for config in ["main", "default", "train", "test"]:
                    try:
                        dataset = load_dataset(dataset_name, config, split=split)
                        print(f"使用配置: {config}")
                        break
                    except:
                        continue
                else:
                    raise ValueError(f"无法自动确定数据集配置。错误: {error_msg}")
        else:
            raise
    except Exception as e:
        # 如果还是失败，尝试不使用 split
        try:
            if dataset_config:
                dataset = load_dataset(dataset_name, dataset_config)[split]
            else:
                dataset = load_dataset(dataset_name)[split]
        except:
            raise ValueError(f"无法加载数据集 {dataset_name}: {e}")
    
    # 选择指定数量的样本
    if len(dataset) > num_samples:
        dataset = dataset.select(range(num_samples))
    
    # 提取文本（根据数据集结构调整）
    prompts = []
    for item in dataset:
        # 尝试不同的字段名
        if "question" in item:
            prompts.append(item["question"])
        elif "instruction" in item:
            prompts.append(item["instruction"])
        elif "text" in item:
            prompts.append(item["text"])
        elif "input" in item:
            prompts.append(item["input"])
        else:
            # 如果都没有，使用第一个字符串字段
            for key, value in item.items():
                if isinstance(value, str) and len(value) > 10:
                    prompts.append(value)
                    break
    
    print(f"加载了 {len(prompts)} 条查询")
    return prompts


def main():
    parser = argparse.ArgumentParser(description="基线推理脚本")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./Llama-3.2-3B-Instruct",
        help="模型路径"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="gsm8k",
        help="数据集名称"
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test",
        help="数据集分割"
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="数据集配置名称（如 gsm8k 需要 'main'）"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="样本数量（低并发至少500，高并发至少2000）"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="批次大小（低并发场景使用1）"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="最大生成token数"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备类型 (cuda/cpu)，如果不指定会自动检测"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./baseline_results",
        help="结果输出目录"
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
    model, tokenizer = load_baseline_model(args.model_path, args.device)
    
    # 加载数据集
    prompts = load_inference_dataset(
        args.dataset_name,
        args.num_samples,
        args.dataset_split,
        args.dataset_config
    )
    
    print(f"\n开始基线推理测试...")
    print(f"模型: {args.model_path}")
    print(f"数据集: {args.dataset_name}")
    print(f"查询数量: {len(prompts)}")
    print(f"批次大小: {args.batch_size}")
    print(f"设备: {args.device}")
    print("-" * 50)
    
    # 运行推理
    results, metrics = inference_batch_baseline(
        model,
        tokenizer,
        prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    
    # 打印结果
    print("\n" + "=" * 50)
    print("基线推理性能指标")
    print("=" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # 保存结果
    output_file = os.path.join(args.output_dir, "baseline_inference_metrics.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args),
            "metrics": metrics,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_file}")
    
    # 保存部分生成结果（用于检查质量）
    results_file = os.path.join(args.output_dir, "baseline_inference_results.json")
    sample_results = []
    for i, (prompt, result) in enumerate(zip(prompts[:10], results[:10])):
        sample_results.append({
            "prompt": prompt,
            "generated": result,
        })
    
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(sample_results, f, indent=2, ensure_ascii=False)
    
    print(f"示例结果已保存到: {results_file}")


if __name__ == "__main__":
    main()

