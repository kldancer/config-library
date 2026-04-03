#!/bin/bash

# ============================================================================
# KServe Infrastructure Management Script (Standalone Unified - Offline Ready)
# ============================================================================
# 
# 步骤与组件映射关系：
# 0: metallb               - 轻量级负载均衡器 (裸金属环境必备)
# 1: cert-manager           - 管理 TLS 证书
# 2: gateway-api-crd        - Kubernetes Gateway API 标准 CRD
# 3: envoy-gateway          - Envoy Gateway 控制器 (基于 Gateway API)
# 4: envoy-ai-gateway       - Envoy AI Gateway 扩展
# 5: lws-operator           - LeaderWorkerSet 算力调度算子
# 6: gateway-api-gwclass    - 定义 Envoy GatewayClass
# 7: gateway-api-gw         - 定义 KServe 专用的 Ingress Gateway
# 8: kserve-helm            - 安装 KServe 核心组件 (KServe, LLMIsvc, LocalModel)
# ============================================================================

set -o errexit
set -o nounset
set -o pipefail

# ============================================================================
# 1. 环境变量与配置
# ============================================================================

# 命名空间
KSERVE_NAMESPACE="${KSERVE_NAMESPACE:-kserve}"
GATEWAY_NAMESPACE="${GATEWAY_NAMESPACE:-kserve}"

# 功能开关
ENABLE_KSERVE="${ENABLE_KSERVE:-true}"
ENABLE_LLMISVC="${ENABLE_LLMISVC:-true}"
ENABLE_LOCALMODEL="${ENABLE_LOCALMODEL:-false}"

# 本地资源路径 (相对于脚本所在目录)
PACKAGE_DIR="package"

# ============================================================================
# 2. 工具函数
# ============================================================================

if [[ -z "${NO_COLOR:-}" ]] && [[ -t 1 ]]; then
    BLUE='\033[94m' ; GREEN='\033[92m' ; RED='\033[91m' ; YELLOW='\033[93m' ; RESET='\033[0m'
else
    BLUE='' ; GREEN='' ; RED='' ; YELLOW='' ; RESET=''
fi

log_info()    { echo -e "${BLUE}[INFO]${RESET} $*" >&2 ; }
log_success() { echo -e "${GREEN}[SUCCESS]${RESET} $*" >&2 ; }
log_error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2 ; }
log_warning() { echo -e "${YELLOW}[WARNING]${RESET} $*" >&2 ; }

is_positive() {
    case "${1:-no}" in
        0|1|true|True|yes|Yes|y|Y) return 0 ;;
        *) return 1 ;;
    esac
}

check_cli_exist() {
    for cmd in "$@"; do
        if ! command -v "$cmd" &>/dev/null; then
            log_error "未找到必要工具: $cmd" ; exit 1
        fi
    done
}

create_or_skip_namespace() {
    local ns="$1"
    if kubectl get namespace "$ns" &>/dev/null; then
        log_info "命名空间 '$ns' 已存在"
    else
        log_info "正在创建命名空间 '$ns'..."
        kubectl create namespace "$ns"
    fi
}

wait_for_pods() {
    local ns="$1" ; local label="$2" ; local timeout="${3:-180s}"
    log_info "等待命名空间 $ns 中标签为 $label 的 Pod 就绪..."
    local elapsed=0
    while [ $(kubectl get pods -n "$ns" -l "$label" --no-headers 2>/dev/null | grep -v "Terminating" | wc -l) -eq 0 ]; do
        if [ $elapsed -ge 60 ]; then log_error "等待 Pod 创建超时" ; return 1 ; fi
        sleep 2 ; elapsed=$((elapsed + 2))
    done
    kubectl wait --for=condition=Ready pod -l "$label" -n "$ns" --timeout="$timeout"
}

wait_for_crds() {
    local timeout="$1" ; shift
    for crd in "$@"; do
        log_info "等待 CRD '$crd' 建立..."
        kubectl wait --for=condition=Established --timeout="$timeout" crd/"$crd" || return 1
    done
}

wait_for_deployment() {
    local ns="$1" ; local name="$2" ; local timeout="${3:-180s}"
    log_info "等待 Deployment '$name' 在 $ns 中可用..."
    kubectl wait --timeout="$timeout" -n "$ns" deployment/"$name" --for=condition=Available
}

determine_shared_resources_config() {
    local install_mode="$1"
    local enable_kserve="$2"
    local enable_llmisvc="$3"

    # 如果在本次执行中两者都启用了，则让第二个 (LLM) 跳过创建共享资源
    if is_positive "${enable_kserve}" && is_positive "${enable_llmisvc}"; then
        LLMISVC_EXTRA_ARGS="${LLMISVC_EXTRA_ARGS:-} --set kserve.createSharedResources=false"
    # 或者是检查集群中是否已经存在其中之一
    elif helm list -n "${KSERVE_NAMESPACE}" -q 2>/dev/null | grep -q "kserve-resources"; then
        LLMISVC_EXTRA_ARGS="${LLMISVC_EXTRA_ARGS:-} --set kserve.createSharedResources=false"
    elif helm list -n "${KSERVE_NAMESPACE}" -q 2>/dev/null | grep -q "kserve-llmisvc-resources"; then
        KSERVE_EXTRA_ARGS="${KSERVE_EXTRA_ARGS:-} --set kserve.createSharedResources=false"
    fi
}

# ============================================================================
# 3. 业务步骤实现 (使用本地 package)
# ============================================================================

# Step 0: metallb
# Pull Command: helm pull metallb/metallb --version v0.14.9 --untar --untardir package/00-metallb
manage_step_0() {
    local action=$1 ; check_cli_exist helm
    local namespace="metallb-system"
    local chart="${PACKAGE_DIR}/00-metallb/metallb"

    apply_metallb_config() {
        cat <<EOF | kubectl apply -f -
apiVersion: metallb.io/v1beta1
kind: IPAddressPool
metadata:
  name: default-pool
  namespace: metallb-system
spec:
  addresses:
  - 192.168.1.200-192.168.1.210
---
apiVersion: metallb.io/v1beta1
kind: L2Advertisement
metadata:
  name: default-advertisement
  namespace: metallb-system
EOF
    }

    uninstall_0() {
        log_info "正在卸载 MetalLB..."
        # 尝试删除配置
        cat <<EOF | kubectl delete --ignore-not-found=true -f - 2>/dev/null || true
apiVersion: metallb.io/v1beta1
kind: IPAddressPool
metadata:
  name: default-pool
  namespace: metallb-system
---
apiVersion: metallb.io/v1beta1
kind: L2Advertisement
metadata:
  name: default-advertisement
  namespace: metallb-system
EOF
        helm uninstall metallb -n "$namespace" 2>/dev/null || true
        kubectl delete namespace "$namespace" --wait=true --timeout=60s 2>/dev/null || true
    }

    install_0() {
        [[ "$action" == "reinstall" ]] && uninstall_0
        if helm list -n "$namespace" 2>/dev/null | grep -q "metallb"; then
            log_info "MetalLB 已安装" ; return 0
        fi

        log_info "==== [1/3] 配置 kube-proxy (strictARP: true) ===="
        kubectl get configmap kube-proxy -n kube-system -o yaml | \
            sed 's/strictARP: false/strictARP: true/' | \
            kubectl apply -f - -n kube-system

        log_info "==== [2/3] Helm 安装 MetalLB ===="
        helm upgrade -i metallb "$chart" --namespace "$namespace" --create-namespace --wait
        
        log_info "==== [3/3] 应用 IP 池配置 (MetalLB) ===="
        # 增加一点等待时间确保 CRD 注册完成
        sleep 5
        wait_for_pods "$namespace" "component=controller" "180s"

        # 重试逻辑，防止 CRD 延迟
        for i in {1..5}; do
            log_info "尝试应用 IPAddressPool 配置 (第 $i 次)..."
            if apply_metallb_config; then
                log_success "MetalLB 配置应用成功！"
                break
            else
                log_warning "CRD 可能尚未就绪，等待 10 秒后重试..."
                sleep 10
            fi
        done
    }
    [[ "$action" == "uninstall" ]] && uninstall_0 || install_0
}

# Step 1: cert-manager
# Pull Command: helm pull jetstack/cert-manager --version v1.17.0 --untar --untardir package/01-cert-manager
manage_step_1() {
    local action=$1 ; check_cli_exist helm
    local chart="${PACKAGE_DIR}/01-cert-manager/cert-manager"
    uninstall_1() {
        log_info "正在卸载 cert-manager..."
        helm uninstall cert-manager -n cert-manager 2>/dev/null || true
        kubectl delete namespace cert-manager --wait=true --timeout=60s 2>/dev/null || true
    }
    install_1() {
        [[ "$action" == "reinstall" ]] && uninstall_1
        if helm list -n cert-manager 2>/dev/null | grep -q "cert-manager"; then
            log_info "cert-manager 已安装" ; return 0
        fi
        helm install cert-manager "$chart" --namespace cert-manager --create-namespace --set crds.enabled=true --wait
        wait_for_pods "cert-manager" "app in (cert-manager,webhook,cainjector)"
    }
    [[ "$action" == "uninstall" ]] && uninstall_1 || install_1
}

# Step 2: gateway-api-crd
# Pull Command: curl -L https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.4.1/standard-install.yaml -o package/02-gateway-api-crd/standard-install.yaml
manage_step_2() {
    local action=$1 ; local yaml="${PACKAGE_DIR}/02-gateway-api-crd/standard-install.yaml"
    uninstall_2() {
        log_info "正在卸载 Gateway API CRDs..."
        kubectl delete -f "$yaml" --ignore-not-found=true 2>/dev/null || true
    }
    install_2() {
        [[ "$action" == "reinstall" ]] && uninstall_2
        if kubectl get crd gateways.gateway.networking.k8s.io &>/dev/null; then
            log_info "Gateway API CRDs 已存在" ; return 0
        fi
        kubectl apply -f "$yaml"
        wait_for_crds "60s" "gateways.gateway.networking.k8s.io" "gatewayclasses.gateway.networking.k8s.io"
    }
    [[ "$action" == "uninstall" ]] && uninstall_2 || install_2
}

# Step 3: envoy-gateway
# Pull Command: helm pull oci://docker.io/envoyproxy/gateway-helm --version v1.6.3 --untar --untardir package/03-envoy-gateway
manage_step_3() {
    local action=$1 ; check_cli_exist helm
    local chart="${PACKAGE_DIR}/03-envoy-gateway/gateway-helm"
    uninstall_3() {
        log_info "正在卸载 Envoy Gateway..."
        kubectl delete gatewayclass envoy --ignore-not-found=true 2>/dev/null || true
        helm uninstall eg -n envoy-gateway-system 2>/dev/null || true
        kubectl delete namespace envoy-gateway-system --wait=true --timeout=60s 2>/dev/null || true
    }
    install_3() {
        [[ "$action" == "reinstall" ]] && uninstall_3
        if helm list -n envoy-gateway-system 2>/dev/null | grep -q "eg"; then
            log_info "Envoy Gateway 已安装" ; return 0
        fi
        helm upgrade -i eg "$chart" -n envoy-gateway-system --create-namespace --wait
        wait_for_pods "envoy-gateway-system" "control-plane=envoy-gateway"
    }
    [[ "$action" == "uninstall" ]] && uninstall_3 || install_3
}

# Step 4: envoy-ai-gateway
# Pull Commands:
#   helm pull oci://docker.io/envoyproxy/ai-gateway-crds-helm --version v0.5.0 --untar --untardir package/04-envoy-ai-gateway
#   helm pull oci://docker.io/envoyproxy/ai-gateway-helm --version v0.5.0 --untar --untardir package/04-envoy-ai-gateway
#   curl -L https://raw.githubusercontent.com/envoyproxy/ai-gateway/v0.5.0/manifests/envoy-gateway-values.yaml -o package/04-envoy-ai-gateway/envoy-gateway-values.yaml
#   curl -L https://raw.githubusercontent.com/envoyproxy/ai-gateway/v0.5.0/examples/inference-pool/envoy-gateway-values-addon.yaml -o package/04-envoy-ai-gateway/envoy-gateway-values-addon.yaml
manage_step_4() {
    local action=$1 ; check_cli_exist helm
    local eg_chart="${PACKAGE_DIR}/03-envoy-gateway/gateway-helm" # 复用 Step 3
    local aieg_crd_chart="${PACKAGE_DIR}/04-envoy-ai-gateway/ai-gateway-crds-helm"
    local aieg_chart="${PACKAGE_DIR}/04-envoy-ai-gateway/ai-gateway-helm"
    local val_yaml="${PACKAGE_DIR}/04-envoy-ai-gateway/envoy-gateway-values.yaml"
    local addon_yaml="${PACKAGE_DIR}/04-envoy-ai-gateway/envoy-gateway-values-addon.yaml"

    uninstall_4() {
        log_info "正在卸载 Envoy AI Gateway..."
        helm uninstall aieg -n envoy-ai-gateway-system 2>/dev/null || true
        helm uninstall aieg-crd -n envoy-ai-gateway-system 2>/dev/null || true
        kubectl delete namespace envoy-ai-gateway-system redis-system --wait=true 2>/dev/null || true
    }
    install_4() {
        [[ "$action" == "reinstall" ]] && uninstall_4
        # 更新 Envoy Gateway 增加 AI 扩展支持
        helm upgrade -i eg "$eg_chart" -n envoy-gateway-system -f "$val_yaml" -f "$addon_yaml" --wait
        
        helm upgrade -i aieg-crd "$aieg_crd_chart" -n envoy-ai-gateway-system --create-namespace
        helm upgrade -i aieg "$aieg_chart" -n envoy-ai-gateway-system --create-namespace
        wait_for_deployment "envoy-ai-gateway-system" "ai-gateway-controller"
    }
    [[ "$action" == "uninstall" ]] && uninstall_4 || install_4
}

# Step 5: lws-operator
# Pull Command: curl -L https://github.com/kubernetes-sigs/lws/releases/download/v0.7.0/manifests.yaml -o package/05-lws-operator/manifests.yaml
manage_step_5() {
    local action=$1 ; local yaml="${PACKAGE_DIR}/05-lws-operator/manifests.yaml"
    uninstall_5() {
        log_info "正在卸载 LWS..."
        kubectl delete -f "$yaml" --ignore-not-found=true 2>/dev/null || true
    }
    install_5() {
        [[ "$action" == "reinstall" ]] && uninstall_5
        if kubectl get deployment lws-controller-manager -n lws-system &>/dev/null; then
            log_info "LWS 已安装" ; return 0
        fi
        kubectl apply --server-side -f "$yaml"
        wait_for_pods "lws-system" "control-plane=controller-manager"
    }
    [[ "$action" == "uninstall" ]] && uninstall_5 || install_5
}

# Step 6: gateway-api-gwclass (YAML 模板无需下载)
manage_step_6() {
    local action=$1 ; local name="${GATEWAYCLASS_NAME:-envoy}"
    local proxy_name="kserve-proxy-private"
    local ns="${KSERVE_NAMESPACE:-kserve}"

    uninstall_6() {
        log_info "正在删除 GatewayClass 和 EnvoyProxy..."
        kubectl delete gatewayclass "${name}" --ignore-not-found=true 2>/dev/null || true
        kubectl delete envoyproxy "${proxy_name}" -n "${ns}" --ignore-not-found=true 2>/dev/null || true
    }
    install_6() {
        create_or_skip_namespace "${ns}"
        [[ "$action" == "reinstall" ]] && uninstall_6

        log_info "正在创建自定义 EnvoyProxy '${proxy_name}'..."
        cat <<EOF | kubectl apply -f -

apiVersion: gateway.envoyproxy.io/v1alpha1
kind: EnvoyProxy
metadata:
  name: ${proxy_name}
  namespace: ${ns}
spec:
  provider:
    type: Kubernetes
    kubernetes:
      envoyDeployment:
        container:
          image: jusuan.io:8080/envoyproxy/envoy:distroless-v1.36.4

EOF

        log_info "正在创建引用 EnvoyProxy 的 GatewayClass '${name}'..."
        cat <<EOF | kubectl apply -f -
apiVersion: gateway.networking.k8s.io/v1
kind: GatewayClass
metadata:
  name: ${name}
spec:
  controllerName: ${CONTROLLER_NAME:-gateway.envoyproxy.io/gatewayclass-controller}
  parametersRef:
    group: gateway.envoyproxy.io
    kind: EnvoyProxy
    name: ${proxy_name}
    namespace: ${ns}
EOF
        log_success "GatewayClass 和 EnvoyProxy 配置完成！"
    }
    [[ "$action" == "uninstall" ]] && uninstall_6 || install_6
}

# Step 7: gateway-api-gw (YAML 模板无需下载)
manage_step_7() {
    local action=$1 ; local name="kserve-ingress-gateway" ; local ns="${GATEWAY_NAMESPACE:-kserve}"
    uninstall_7() {
        kubectl delete gateway "${name}" -n "${ns}" --ignore-not-found=true 2>/dev/null || true
    }
    install_7() {
        create_or_skip_namespace "${ns}"
        [[ "$action" == "reinstall" ]] && uninstall_7
        cat <<EOF | kubectl apply -f -
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: ${name}
  namespace: ${ns}
spec:
  gatewayClassName: ${GATEWAYCLASS_NAME:-envoy}
  listeners:
    - {name: http, protocol: HTTP, port: 80, allowedRoutes: {namespaces: {from: All}}}
  infrastructure:
    labels: {serving.kserve.io/gateway: ${name}}
EOF
    }
    [[ "$action" == "uninstall" ]] && uninstall_7 || install_7
}

# Step 8: kserve-helm
# Pull Command: for chart in kserve-crd kserve-llmisvc-crd kserve-resources kserve-llmisvc-resources kserve-runtime-configs; do helm pull oci://ghcr.io/kserve/charts/$chart --version v0.17.0 --untar --untardir package/08-kserve ; done
manage_step_8() {
    local action=$1 ; check_cli_exist helm
    local KSERVE_NS="${KSERVE_NAMESPACE:-kserve}"
    local KSERVE_EXTRA_ARGS="${KSERVE_EXTRA_ARGS:-}"
    local LLMISVC_EXTRA_ARGS="${LLMISVC_EXTRA_ARGS:-}"
    
    local pkg_path="${PACKAGE_DIR}/08-kserve"

    uninstall_8() {
        log_info "正在卸载 KServe..."
        local charts=("kserve-runtime-configs" "kserve-localmodel-resources" "kserve-llmisvc-resources" "kserve-resources" "kserve-localmodel-crd" "kserve-llmisvc-crd" "kserve-crd")
        for c in "${charts[@]}"; do
            helm uninstall "$c" -n "${KSERVE_NS}" 2>/dev/null || true
        done
    }

    install_8() {
        [[ "$action" == "reinstall" ]] && uninstall_8
        determine_shared_resources_config "helm" "${ENABLE_KSERVE}" "${ENABLE_LLMISVC}"
        
        # CRDs
        if is_positive "${ENABLE_KSERVE}"; then helm upgrade -i kserve-crd "${pkg_path}/kserve-crd" -n "${KSERVE_NS}" --create-namespace --wait ; fi
        if is_positive "${ENABLE_LLMISVC}"; then helm upgrade -i kserve-llmisvc-crd "${pkg_path}/kserve-llmisvc-crd" -n "${KSERVE_NS}" --create-namespace --wait ; fi
        
        # Resources
        if is_positive "${ENABLE_KSERVE}"; then
            helm upgrade -i kserve-resources "${pkg_path}/kserve-resources" -n "${KSERVE_NS}" --wait \
                --set kserve.controller.imagePullPolicy=IfNotPresent ${KSERVE_EXTRA_ARGS}
            wait_for_deployment "${KSERVE_NS}" "kserve-controller-manager"
        fi

        if is_positive "${ENABLE_LLMISVC}"; then
            helm upgrade -i kserve-llmisvc-resources "${pkg_path}/kserve-llmisvc-resources" -n "${KSERVE_NS}" --wait \
                --set kserve.controller.imagePullPolicy=IfNotPresent ${LLMISVC_EXTRA_ARGS}
            wait_for_deployment "${KSERVE_NS}" "llmisvc-controller-manager"
        fi

        # Runtime Configs
        helm upgrade -i kserve-runtime-configs "${pkg_path}/kserve-runtime-configs" -n "${KSERVE_NS}" --wait \
            --set kserve.servingruntime.enabled="${ENABLE_KSERVE}" \
            --set kserve.llmisvcConfigs.enabled="${ENABLE_LLMISVC}"
    }

    [[ "$action" == "uninstall" ]] && uninstall_8 || install_8
}

# ============================================================================
# 4. 参数解析与执行引擎
# ============================================================================

STEP="" ; ACTION="install" ; ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --step) STEP="$2"; shift 2 ;;
        --uninstall) ACTION="uninstall"; shift ;;
        --reinstall) ACTION="reinstall"; shift ;;
        --force-upgrade) ACTION="force-upgrade"; shift ;;
        --all) ALL=true; shift ;;
        -h|--help) 
            echo "用法: $0 [--step 0-8] [--uninstall|--reinstall|--all]"
            exit 0 ;;
        *) log_error "未知参数: $1" ; exit 1 ;;
    esac
done

execute_step() {
    local num=$1 ; local act=$2
    log_info ">>> 执行步骤 ${num}: $(get_step_name $num) [${act}]"
    case $num in
        0) manage_step_0 "$act" ;;
        1) manage_step_1 "$act" ;;
        2) manage_step_2 "$act" ;;
        3) manage_step_3 "$act" ;;
        4) manage_step_4 "$act" ;;
        5) manage_step_5 "$act" ;;
        6) manage_step_6 "$act" ;;
        7) manage_step_7 "$act" ;;
        8) manage_step_8 "$act" ;;
    esac
}

get_step_name() {
    case $1 in
        0) echo "metallb" ;;
        1) echo "cert-manager" ;;
        2) echo "gateway-api-crd" ;;
        3) echo "envoy-gateway" ;;
        4) echo "envoy-ai-gateway" ;;
        5) echo "lws-operator" ;;
        6) echo "gateway-api-gwclass" ;;
        7) echo "gateway-api-gw" ;;
        8) echo "kserve-helm" ;;
    esac
}

if [ "$ALL" = true ]; then
    if [[ "$ACTION" == "uninstall" ]]; then
        for i in {8..0}; do execute_step $i "$ACTION"; done
    else
        for i in {0..8}; do execute_step $i "$ACTION"; done
    fi
elif [ -n "$STEP" ]; then
    execute_step "$STEP" "$ACTION"
else
    $0 --help
fi
