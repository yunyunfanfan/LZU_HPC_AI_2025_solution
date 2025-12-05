"""
Generate comparison plots from existing JSON metric files for the report.

Two main parts:
1. Inference speedup: Baseline vs INT4 quantization
2. Finetuning speedup: Full fine-tuning vs LoRA / LoRA+GC / QLoRA / QLoRA+GC / ZeRO combinations

Outputs (saved under ./plots/):
- inference_latency.png
- inference_memory.png
- inference_throughput.png
- finetune_time.png
- finetune_time_per_epoch.png
- finetune_memory.png
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = Path(__file__).parent
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def load_json(path: Path):
    if not path.exists():
        print(f"[WARN] file not found, skip: {path}")
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_inference():
    """Inference baseline vs INT4 quantization comparison plots."""
    print("\n=== Generating inference comparison plots ===")

    baseline_path = BASE_DIR / "baseline_results" / "baseline_inference_metrics.json"
    int4_path = BASE_DIR / "optimized_results" / "INT4量化" / "baseline_inference_metrics.json"

    baseline = load_json(baseline_path)
    int4 = load_json(int4_path)
    if baseline is None or int4 is None:
        print("[WARN] Inference metric files missing, skip inference plots")
        return

    methods = ["Baseline", "INT4"]
    latencies = [
        baseline["metrics"]["avg_latency"],
        int4["metrics"]["avg_latency"],
    ]
    throughputs = [
        baseline["metrics"].get("throughput_tokens_per_sec", 0.0),
        int4["metrics"].get("throughput_tokens_per_sec", 0.0),
    ]
    mems = [
        baseline["metrics"].get("peak_memory_gb", 0.0),
        int4["metrics"].get("peak_memory_gb", 0.0),
    ]

    sns.set_style("whitegrid")

    # Latency comparison
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=methods, y=latencies, palette="Blues")
    ax.set_ylabel("Avg latency (s / query)")
    ax.set_title("Inference latency: Baseline vs INT4")
    for i, v in enumerate(latencies):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    out_path = PLOTS_DIR / "inference_latency.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] saved inference latency plot: {out_path}")

    # Memory comparison
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=methods, y=mems, palette="Greens")
    ax.set_ylabel("Peak memory (GB)")
    ax.set_title("Inference peak memory: Baseline vs INT4")
    for i, v in enumerate(mems):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    out_path = PLOTS_DIR / "inference_memory.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] saved inference memory plot: {out_path}")

    # Throughput comparison (tokens/s)
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=methods, y=throughputs, palette="Oranges")
    ax.set_ylabel("Throughput (tokens / s)")
    ax.set_title("Inference throughput: Baseline vs INT4")
    for i, v in enumerate(throughputs):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    plt.tight_layout()
    out_path = PLOTS_DIR / "inference_throughput.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] saved inference throughput plot: {out_path}")


def plot_finetune():
    """Finetuning baseline vs LoRA / LoRA+GC / QLoRA / QLoRA+GC / ZeRO comparison plots."""
    print("\n=== Generating finetuning comparison plots ===")

    # 各结果文件路径（按显存占用从高到低排序，便于对比）
    paths = {
        "Baseline": BASE_DIR / "baseline_finetune_output" / "baseline_finetune_metrics.json",
        "LoRA": BASE_DIR / "optimized_finetune_output" / "lora" / "baseline_finetune_metrics.json",
        "LoRA+GC": BASE_DIR / "optimized_finetune_output" / "lora_gc" / "baseline_finetune_metrics.json",
        "LoRA+ZeRO-2": BASE_DIR / "optimized_finetune_output" / "lora_zero2" / "baseline_finetune_metrics.json",
        "LoRA+ZeRO-3": BASE_DIR / "optimized_finetune_output" / "lora_zero3" / "baseline_finetune_metrics.json",
        "LoRA+ZeRO-2+GC": BASE_DIR / "optimized_finetune_output" / "lora_zero2_gc" / "baseline_finetune_metrics.json",
        "LoRA+ZeRO-3+GC": BASE_DIR / "optimized_finetune_output" / "lora_zero3_gc" / "baseline_finetune_metrics.json",
        "QLoRA": BASE_DIR / "optimized_finetune_output" / "qlora" / "baseline_finetune_metrics.json",
        "QLoRA+GC": BASE_DIR / "optimized_finetune_output" / "qlora_gc" / "baseline_finetune_metrics.json",
        "QLoRA+ZeRO-2": BASE_DIR / "optimized_finetune_output" / "qlora_zero2" / "baseline_finetune_metrics.json",
        "QLoRA+ZeRO-3": BASE_DIR / "optimized_finetune_output" / "qlora_zero3" / "baseline_finetune_metrics.json",
        "QLoRA+ZeRO-2+GC": BASE_DIR / "optimized_finetune_output" / "qlora_zero2_gc" / "baseline_finetune_metrics.json",
        "QLoRA+ZeRO-3+GC": BASE_DIR / "optimized_finetune_output" / "qlora_zero3_gc" / "baseline_finetune_metrics.json",
    }

    methods = []
    times = []
    mems = []
    time_per_epoch_list = []

    for name, path in paths.items():
        data = load_json(path)
        if data is None:
            continue
        metrics = data["metrics"]
        methods.append(name)
        times.append(metrics.get("total_time_seconds", 0.0))
        mems.append(metrics.get("peak_memory_gb", 0.0))
        # Calculate time per epoch
        num_epochs = metrics.get("num_epochs", 2)
        total_time = metrics.get("total_time_seconds", 0.0)
        time_per_epoch_list.append(total_time / num_epochs if total_time > 0 else 0.0)

    if not methods:
        print("[WARN] No finetune metric files found, skip finetune plots")
        return

    # Use baseline time as reference
    baseline_time = times[0] if methods[0] == "Baseline" else None

    sns.set_style("whitegrid")

    # Total training time comparison
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(x=methods, y=times, palette="Oranges")
    ax.set_ylabel("Total training time (s)", fontsize=12)
    ax.set_xlabel("Method", fontsize=12)
    ax.set_title("Finetuning total time comparison", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    for i, v in enumerate(times):
        label = f"{v:.1f}"
        if baseline_time and i > 0 and v > 0:
            speed = baseline_time / v
            label += f"\n({speed:.2f}x)"
        ax.text(i, v, label, ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out_path = PLOTS_DIR / "finetune_time.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] saved finetune total time plot: {out_path}")

    # Time per epoch comparison
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(x=methods, y=time_per_epoch_list, palette="Reds")
    ax.set_ylabel("Time per epoch (s)", fontsize=12)
    ax.set_xlabel("Method", fontsize=12)
    ax.set_title("Finetuning time per epoch comparison", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    baseline_time_per_epoch = time_per_epoch_list[0] if methods[0] == "Baseline" else None
    for i, v in enumerate(time_per_epoch_list):
        label = f"{v:.1f}"
        if baseline_time_per_epoch and i > 0 and v > 0:
            speed = baseline_time_per_epoch / v
            label += f"\n({speed:.2f}x)"
        ax.text(i, v, label, ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out_path = PLOTS_DIR / "finetune_time_per_epoch.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] saved finetune per-epoch time plot: {out_path}")

    # Peak memory comparison
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(x=methods, y=mems, palette="Purples")
    ax.set_ylabel("Peak memory (GB)", fontsize=12)
    ax.set_xlabel("Method", fontsize=12)
    ax.set_title("Finetuning peak memory comparison", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    base_mem = mems[0] if methods[0] == "Baseline" else None
    for i, v in enumerate(mems):
        label = f"{v:.2f}"
        if base_mem and i > 0 and base_mem > 0:
            reduction = (base_mem - v) / base_mem * 100
            label += f"\n(-{reduction:.1f}%)"
        ax.text(i, v, label, ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out_path = PLOTS_DIR / "finetune_memory.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] saved finetune memory plot: {out_path}")


def main():
    plot_inference()
    plot_finetune()
    print("\nAll plots generated. Check the ./plots/ directory for PNG files.")


if __name__ == "__main__":
    main()


