#!/bin/bash
# DeepSpeed 包装脚本，自动设置 CUDA_HOME
# 优先使用 v13.0（与 PyTorch 匹配）
# 使用符号链接避免路径中的空格问题

# 激活 conda 环境（如果未激活）
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "hpc" ]; then
    # 初始化 conda
    eval "$(/home/yunfan/miniconda3/bin/conda shell.bash hook)" 2>/dev/null
    # 激活 hpc 环境
    conda activate hpc 2>/dev/null || {
        echo "警告: 无法激活 hpc 环境，请手动运行: conda activate hpc"
    }
fi

# 优先使用符号链接路径（如果存在）
CUDA_LINK_V13_0="$HOME/cuda-13.0"
CUDA_LINK_V13_1="$HOME/cuda-13.1"

# 原始路径（用于创建符号链接）
CUDA_SOURCE_V13_0="/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0"
CUDA_SOURCE_V13_1="/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1"

# 优先使用 v13.0 的符号链接
if [ -L "$CUDA_LINK_V13_0" ] || [ -d "$CUDA_LINK_V13_0" ]; then
    export CUDA_HOME="$CUDA_LINK_V13_0"
elif [ -L "$CUDA_LINK_V13_1" ] || [ -d "$CUDA_LINK_V13_1" ]; then
    export CUDA_HOME="$CUDA_LINK_V13_1"
# 如果没有符号链接，检查原始路径并创建符号链接
elif [ -d "$CUDA_SOURCE_V13_0" ]; then
    echo "创建 CUDA 符号链接以避免路径空格问题..."
    ln -sf "$CUDA_SOURCE_V13_0" "$CUDA_LINK_V13_0" 2>/dev/null
    if [ -L "$CUDA_LINK_V13_0" ]; then
        export CUDA_HOME="$CUDA_LINK_V13_0"
    else
        export CUDA_HOME="$CUDA_SOURCE_V13_0"
    fi
elif [ -d "$CUDA_SOURCE_V13_1" ]; then
    echo "创建 CUDA 符号链接以避免路径空格问题..."
    ln -sf "$CUDA_SOURCE_V13_1" "$CUDA_LINK_V13_1" 2>/dev/null
    if [ -L "$CUDA_LINK_V13_1" ]; then
        export CUDA_HOME="$CUDA_LINK_V13_1"
    else
        export CUDA_HOME="$CUDA_SOURCE_V13_1"
    fi
else
    echo "错误: 未找到 CUDA Toolkit (v13.0 或 v13.1)"
    exit 1
fi

export PATH="$CUDA_HOME/bin:$PATH"

# 确保 nvcc 符号链接存在
if [ ! -f "$CUDA_HOME/bin/nvcc" ] && [ -f "$CUDA_HOME/bin/nvcc.exe" ]; then
    cd "$CUDA_HOME/bin"
    ln -sf nvcc.exe nvcc 2>/dev/null
    cd - > /dev/null
fi

# 添加 PyTorch 的 CUDA 库路径到 LD_LIBRARY_PATH（用于链接器）
# 这样链接器可以找到 cudart、curand 等库
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    PYTORCH_LIB="$HOME/miniconda3/envs/$CONDA_DEFAULT_ENV/lib/python3.12/site-packages/torch/lib"
    if [ -d "$PYTORCH_LIB" ]; then
        export LD_LIBRARY_PATH="$PYTORCH_LIB:${LD_LIBRARY_PATH}"
    fi
fi

# 禁用 DeepSpeed CUDA 扩展编译（WSL 环境无法编译）
# 使用标准 PyTorch 优化器代替
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1

# 调用原始的 deepspeed 命令
exec deepspeed "$@"

