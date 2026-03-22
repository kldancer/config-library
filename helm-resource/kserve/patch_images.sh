#!/bin/bash

# ============================================================================
# KServe Infrastructure Image Registry Patcher
# ============================================================================
# 该脚本用于批量修改 package 目录下的 Helm Values 和 YAML 文件，
# 将公开镜像域名和短路径镜像统一替换为私有镜像库地址。
# ============================================================================

set -e

# 配置私有仓库前缀 (按需修改)
PRIVATE_REG="${1:-jusuan.io:8080}"
PACKAGE_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && echo "$(pwd)/package")"

if [ ! -d "$PACKAGE_DIR" ]; then
    echo "错误: 未找到 package 目录 ($PACKAGE_DIR)"
    exit 1
fi

echo ">>> 开始私有化镜像替换 (目标仓库: $PRIVATE_REG)..."

# 1. 执行全局域名替换 (处理带有完整域名的镜像)
# 涵盖常用的公开镜像托管服务
DOMAINS=("docker.io/" "quay.io/" "ghcr.io/" "registry.k8s.io/" "nvcr.io/")

for domain in "${DOMAINS[@]}"; do
    echo "  - 正在替换域名: $domain"
    find "$PACKAGE_DIR" -type f \( -name "*.yaml" -o -name "*.tpl" -o -name "*.json" \) \
        -exec sed -i "s|$domain|$PRIVATE_REG/|g" {} +
done

# 2. 处理“无域名”短路径镜像 (针对 image: xxx/yyy 格式)
# 常见的 KServe 及其推理引擎短路径
SHORT_PATHS=("kserve/" "tensorflow/" "pytorch/" "seldonio/" "nvidia/")

for path in "${SHORT_PATHS[@]}"; do
    echo "  - 正在处理短路径镜像: $path"
    # 替换 image: 或 repository: 后面紧跟短路径的情况 (避免重复替换已有的私有域名)
    find "$PACKAGE_DIR" -type f \( -name "*.yaml" -o -name "*.tpl" \) \
        -exec sed -i "s|image: $path|image: $PRIVATE_REG/$path|g" {} +
    find "$PACKAGE_DIR" -type f \( -name "*.yaml" -o -name "*.tpl" \) \
        -exec sed -i "s|repository: $path|repository: $PRIVATE_REG/$path|g" {} +
done

# 3. 特殊补丁：处理某些硬编码在模板中的镜像
echo "  - 正在执行特定组件补丁..."
# 处理某些 Chart 中可能存在的裸镜像名替换逻辑 (按需扩展)

echo ">>> 镜像替换完成！"

# 4. 验证结果
echo ">>> 抽样验证 (kserve-runtime-configs):"
if [ -f "$PACKAGE_DIR/08-kserve/kserve-runtime-configs/values.yaml" ]; then
    grep "image:" "$PACKAGE_DIR/08-kserve/kserve-runtime-configs/values.yaml" | head -n 5
fi
