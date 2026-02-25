# vLLM Helm Chart

支持多实例、多部署模式的 vLLM 推理服务 Helm Chart。

## 功能特性

- ✅ 支持多个 vLLM 服务实例
- ✅ 支持两种部署模式：
  - **聚合模式 (aggregated)**: 使用 Tensor Parallel
  - **PD 模式 (pd)**: 使用 Pipeline Parallel + Data Parallel
- ✅ 灵活的资源配置
- ✅ 支持 Ray 集成（PD 模式）

## 部署模式说明

### 聚合模式 (aggregated)

使用 Tensor Parallel 进行模型并行，适合单节点多 GPU 场景。

**配置示例：**
```yaml
- name: vllm-aggregated-tp2
  enabled: true
  mode: aggregated
  tensorParallelSize: 2  # Tensor Parallel 大小
  model:
    name: "meta-llama/Llama-2-7b-chat-hf"
```

### PD 模式 (pd)

使用 Pipeline Parallel + Data Parallel，适合多节点、大规模模型场景。

**配置示例：**
```yaml
- name: vllm-pd-1p2d
  enabled: true
  mode: pd
  dataParallelSize: 2       # Data Parallel 大小
  ray:
    enabled: true           # 启用 Ray 支持
  model:
    name: "meta-llama/Llama-2-7b-chat-hf"
```

## 配置说明

### 全局配置

```yaml
global:
  image:
    repository: vllm/vllm-openai
    pullPolicy: IfNotPresent
    tag: "latest"
  resources:
    requests:
      cpu: "4"
      memory: "16Gi"
    limits:
      cpu: "8"
      memory: "32Gi"
```

### 服务实例配置

每个服务实例可以独立配置：

```yaml
services:
  - name: vllm-service-1        # 服务名称（必填）
    enabled: true                # 是否启用（必填）
    mode: aggregated            # 部署模式：aggregated 或 pd（必填）
    
    # 聚合模式配置
    tensorParallelSize: 2       # 仅聚合模式需要
    
    # PD 模式配置
    dataParallelSize: 2         # 仅 PD 模式需要
    ray:
      enabled: true
      headResources:            # Ray head 资源（可选）
        cpu: "2"
        memory: "4Gi"
      workerResources:          # Ray worker 资源（可选）
        cpu: "4"
        memory: "16Gi"
    
    # 服务配置
    service:
      type: ClusterIP
      port: 8000
    
    # 模型配置
    model:
      name: "meta-llama/Llama-2-7b-chat-hf"
      path: ""                  # 可选：本地模型路径
    
    # 镜像配置（可选，覆盖全局配置）
    image:
      repository: vllm/vllm-openai
      tag: "latest"
    
    # 资源限制（可选，覆盖全局配置）
    resources:
      requests:
        cpu: "8"
        memory: "32Gi"
      limits:
        cpu: "16"
        memory: "64Gi"
    
    # 环境变量（可选）
    env:
      - name: VLLM_USE_RAY
        value: "false"
    
    # 节点选择器（可选）
    nodeSelector: {}
    
    # 容忍度（可选）
    tolerations: []
    
    # 亲和性（可选）
    affinity: {}
```

## 使用示例

### 示例 1: 单个聚合模式服务

```yaml
services:
  - name: vllm-tp2
    enabled: true
    mode: aggregated
    tensorParallelSize: 2
    service:
      type: ClusterIP
      port: 8000
    model:
      name: "meta-llama/Llama-2-7b-chat-hf"
```

### 示例 2: 多个不同配置的服务

```yaml
services:
  # 聚合架构，tp=2
  - name: vllm-aggregated-tp2
    enabled: true
    mode: aggregated
    tensorParallelSize: 2
    model:
      name: "meta-llama/Llama-2-7b-chat-hf"
  
  # 聚合架构，tp=1
  - name: vllm-aggregated-tp1
    enabled: true
    mode: aggregated
    tensorParallelSize: 1
    model:
      name: "meta-llama/Llama-2-7b-chat-hf"
  
  # PD 架构，1p2d
  - name: vllm-pd-1p2d
    enabled: true
    mode: pd
    dataParallelSize: 2
    ray:
      enabled: true
    model:
      name: "meta-llama/Llama-2-7b-chat-hf"
  
  # PD 架构，2p1d
  - name: vllm-pd-2p1d
    enabled: true
    mode: pd
    dataParallelSize: 1
    ray:
      enabled: true
    model:
      name: "meta-llama/Llama-2-70b-chat-hf"
```

## 部署

### 作为独立 Chart 部署

```bash
helm install vllm ./charts/vllm -n vllm --create-namespace
```

### 作为 llm-stack 的一部分部署

在 `llm-stack/values.yaml` 中配置：

```yaml
vllm:
  services:
    - name: vllm-service-1
      enabled: true
      mode: aggregated
      tensorParallelSize: 2
      # ... 其他配置
```

然后部署：

```bash
helm install llm-stack ./llm-stack -n llm --create-namespace
```

## 注意事项

1. **聚合模式**：需要确保节点有足够的 GPU 资源（至少 `tensorParallelSize` 个 GPU）
2. **PD 模式**：需要启用 Ray，确保集群有足够的资源运行 Ray head 和 workers
3. **资源限制**：根据模型大小和并行度合理配置资源请求和限制
4. **模型路径**：如果使用本地模型，需要配置相应的 volume 挂载

## 验证部署

```bash
# 查看所有 vLLM 服务
kubectl get svc -l app.kubernetes.io/component=vllm

# 查看部署状态
kubectl get deployments -l app.kubernetes.io/component=vllm

# 测试服务
curl http://<service-name>:8000/health
```

