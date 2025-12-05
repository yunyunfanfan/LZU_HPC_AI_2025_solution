#!/bin/bash
# 在 WSL 中配置 CUDA Toolkit 的脚本
# 用于支持 DeepSpeed 等需要编译 CUDA 算子的库

set -e

echo "=========================================="
echo "WSL CUDA Toolkit 配置脚本"
echo "=========================================="

# 步骤 1：检查 Windows 主机上的 CUDA Toolkit
echo ""
echo "步骤 1: 检查 Windows 主机上的 CUDA Toolkit"
echo "----------------------------------------"

# 常见的 CUDA Toolkit 安装路径
WINDOWS_CUDA_PATHS=(
    "/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0"
    "/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1"
    "/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4"
    "/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3"
    "/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2"
    "/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1"
    "/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0"
    "/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8"
)

CUDA_PATH=""
for path in "${WINDOWS_CUDA_PATHS[@]}"; do
    # 检查 nvcc.exe（Windows）或 nvcc（Linux）
    if [ -d "$path" ] && ([ -f "$path/bin/nvcc.exe" ] || [ -f "$path/bin/nvcc" ]); then
        CUDA_PATH="$path"
        echo "✓ 找到 CUDA Toolkit: $CUDA_PATH"
        # 尝试读取版本信息
        if [ -f "$CUDA_PATH/version.json" ]; then
            VERSION=$(grep -o '"cuda_version":"[^"]*"' "$CUDA_PATH/version.json" | cut -d'"' -f4 || echo "unknown")
            echo "  CUDA 版本: $VERSION"
        elif [ -f "$CUDA_PATH/version.txt" ]; then
            echo "  CUDA 版本: $(cat "$CUDA_PATH/version.txt")"
        else
            echo "  CUDA 版本: unknown"
        fi
        break
    fi
done

if [ -z "$CUDA_PATH" ]; then
    echo "✗ 未在 Windows 主机上找到 CUDA Toolkit"
    echo ""
    echo "请先在 Windows 主机上安装 CUDA Toolkit:"
    echo "1. 访问: https://developer.nvidia.com/cuda-downloads"
    echo "2. 选择: Windows → x86_64 → 10/11 → exe (local)"
    echo "3. 下载并安装 CUDA Toolkit（建议版本 12.4 或 13.0，匹配你的驱动）"
    echo "4. 安装完成后，重新运行此脚本"
    exit 1
fi

# 步骤 2：设置环境变量
echo ""
echo "步骤 2: 配置 CUDA_HOME 环境变量"
echo "----------------------------------------"

# 转换 Windows 路径为 WSL 路径（如果需要）
CUDA_HOME="$CUDA_PATH"

# 检查 nvcc 是否可访问（支持 .exe 扩展名）
if [ ! -f "$CUDA_HOME/bin/nvcc.exe" ] && [ ! -f "$CUDA_HOME/bin/nvcc" ]; then
    echo "✗ 错误: 无法访问 nvcc: $CUDA_HOME/bin/nvcc(.exe)"
    exit 1
fi

echo "✓ CUDA_HOME 将设置为: $CUDA_HOME"
if [ -f "$CUDA_HOME/bin/nvcc.exe" ]; then
    echo "✓ nvcc 路径: $CUDA_HOME/bin/nvcc.exe"
else
    echo "✓ nvcc 路径: $CUDA_HOME/bin/nvcc"
fi

# 步骤 3：添加到 ~/.bashrc
echo ""
echo "步骤 3: 添加到 ~/.bashrc（永久生效）"
echo "----------------------------------------"

BASHRC="$HOME/.bashrc"
BACKUP="$HOME/.bashrc.backup.$(date +%Y%m%d_%H%M%S)"

# 备份
if [ -f "$BASHRC" ]; then
    cp "$BASHRC" "$BACKUP"
    echo "✓ 已备份 ~/.bashrc 到: $BACKUP"
fi

# 移除旧的 CUDA_HOME 设置（如果存在）
if grep -q "CUDA_HOME" "$BASHRC" 2>/dev/null; then
    echo "⚠ 发现旧的 CUDA_HOME 设置，将更新"
    sed -i '/^export CUDA_HOME=/d' "$BASHRC"
    sed -i '/^# CUDA_HOME/d' "$BASHRC"
fi

# 添加新的设置
cat >> "$BASHRC" << EOF

# CUDA Toolkit for DeepSpeed (added by install script)
export CUDA_HOME="$CUDA_HOME"
export PATH="\$CUDA_HOME/bin:\$PATH"
export LD_LIBRARY_PATH="\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
EOF

echo "✓ 已添加到 ~/.bashrc"

# 步骤 4：立即生效
echo ""
echo "步骤 4: 使环境变量立即生效"
echo "----------------------------------------"
export CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

echo "✓ 环境变量已设置（当前会话）"

# 步骤 5：验证
echo ""
echo "步骤 5: 验证安装"
echo "----------------------------------------"

# 尝试运行 nvcc（支持 .exe）
if [ -f "$CUDA_HOME/bin/nvcc.exe" ]; then
    NVCC_CMD="$CUDA_HOME/bin/nvcc.exe"
elif [ -f "$CUDA_HOME/bin/nvcc" ]; then
    NVCC_CMD="$CUDA_HOME/bin/nvcc"
else
    NVCC_CMD=""
fi

if [ -n "$NVCC_CMD" ] && $NVCC_CMD --version &> /dev/null; then
    NVCC_VERSION=$($NVCC_CMD --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
    echo "✓ nvcc 可用，版本: $NVCC_VERSION"
elif command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
    echo "✓ nvcc 可用（通过 PATH），版本: $NVCC_VERSION"
else
    echo "⚠ 警告: nvcc 不可用，可能需要重新加载 shell 或检查 PATH"
fi

if [ -n "$CUDA_HOME" ]; then
    echo "✓ CUDA_HOME: $CUDA_HOME"
else
    echo "✗ 警告: CUDA_HOME 未设置"
fi

# 步骤 6：测试 DeepSpeed 安装
echo ""
echo "步骤 6: 测试 DeepSpeed 安装（可选）"
echo "----------------------------------------"
echo "现在可以尝试重新安装 DeepSpeed:"
echo "  pip install deepspeed --force-reinstall --no-cache-dir"
echo ""
echo "或者如果已经安装，测试导入:"
echo "  python -c 'import deepspeed; print(f\"DeepSpeed {deepspeed.__version__}\")'"

echo ""
echo "=========================================="
echo "配置完成！"
echo "=========================================="
echo ""
echo "重要提示:"
echo "1. 请重新打开终端或运行: source ~/.bashrc"
echo "2. 然后尝试安装/测试 DeepSpeed"
echo "3. 如果遇到问题，检查 CUDA_HOME 是否正确: echo \$CUDA_HOME"
echo ""

