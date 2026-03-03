#!/bin/bash

set -eo pipefail
############################################################
# Help                                                     #
############################################################
Help() {
   # Display Help
   echo "LLMISvc quick install script."
   echo
   echo "Syntax: [-u|-d]"
   echo "options:"
   echo "u Uninstall."
   echo "d Install only dependencies."
   echo
}

export GATEWAY_API_VERSION=v1.2.1
export KSERVE_VERSION=v0.16.0
export LLMISVC_VERSION=v0.16.0
export LWS_VERSION=0.7.0
export ENVOY_GATEWAY_VERSION=v1.5.0
export ENVOY_AI_GATEWAY_VERSION=v0.3.0
SCRIPT_DIR="$(dirname -- "${BASH_SOURCE[0]}")"
export CERT_MANAGER_VERSION=v1.16.1
export SCRIPT_DIR

uninstall() {
   # Uninstall Cert Manager
   helm uninstall --ignore-not-found cert-manager -n cert-manager
   echo "😀 Successfully uninstalled Cert Manager"
   
    # Uninstall Envoy Gateway
   helm uninstall --ignore-not-found eg -n envoy-gateway-system
   echo "😀 Successfully uninstalled Envoy Gateway"

   # Uninstall Envoy AI Gateway
   helm uninstall --ignore-not-found aieg -n envoy-ai-gateway-system
#   helm uninstall --ignore-not-found aieg-crd -n envoy-ai-gateway-system
   echo "😀 Successfully uninstalled Envoy AI Gateway"

   # Uninstall Leader Worker Set
   helm uninstall --ignore-not-found lws -n lws-system
   echo "😀 Successfully uninstalled Leader Worker Set"

   # Delete Gateway resources
   kubectl delete --ignore-not-found=true gateway kserve-ingress-gateway -n kserve
   kubectl delete --ignore-not-found=true gatewayclass envoy

   # Uninstall LLMISvc
   helm uninstall --ignore-not-found llmisvc -n kserve
   helm uninstall --ignore-not-found llmisvc-crd -n kserve
   echo "😀 Successfully uninstalled LLMISvc"


   # Delete namespaces
   kubectl delete --ignore-not-found=true namespace kserve
   kubectl delete --ignore-not-found=true namespace lws-system
   kubectl delete --ignore-not-found=true namespace envoy-ai-gateway-system
   kubectl delete --ignore-not-found=true namespace envoy-gateway-system
}

# Check if helm command is available
if ! command -v helm &>/dev/null; then
   echo "😱 Helm command not found. Please install Helm."
   exit 1
fi

installLLMISvc=true
while getopts ":hud" option; do
   case $option in
   h) # display Help
      Help
      exit
      ;;
   u) # uninstall
      uninstall
      exit
      ;;
   d) # install only dependencies
      installLLMISvc=false ;;
   \?) # Invalid option
      echo "Error: Invalid option"
      exit
      ;;
   esac
done

get_kube_version() {
   kubectl version --short=true 2>/dev/null || kubectl version | awk -F '.' '/Server Version/ {print $2}'
}

if [ "$(get_kube_version)" -lt 24 ]; then
   echo "😱 install requires at least Kubernetes 1.24"
   exit 1
fi

# Install Cert Manager
#helm repo add jetstack https://charts.jetstack.io --force-update
#helm install \
#   cert-manager jetstack/cert-manager \
#   --namespace cert-manager \
#   --create-namespace \
#   --version ${CERT_MANAGER_VERSION} \
#   --set crds.enabled=true
helm install cert-manager ./01-cert-manager/cert-manager -n cert-manager --create-namespace -f ./01-cert-manager/cert-manager/values.yaml
echo "😀 Successfully installed Cert Manager"

echo "Installing Gateway API CRDs ..."
#kubectl apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/${GATEWAY_API_VERSION}/standard-install.yaml
kubectl apply -f 02-gateway-api-crd/standard-install.yaml

# Need to install before Envoy Gateway
#kubectl apply -f ${SCRIPT_DIR}/../config/llmisvc/gateway-inference-extension.yaml
kubectl apply -f 02-gateway-api-crd/gateway-inference-extension.yaml

# Install Envoy Gateway
echo "Installing Envoy Gateway ..."
#helm upgrade -i eg oci://docker.io/envoyproxy/gateway-helm \
#  --version ${ENVOY_GATEWAY_VERSION} \
#  --namespace envoy-gateway-system \
#  --create-namespace
helm install eg ./03-envoy-gateway/gateway-helm --namespace envoy-gateway-system --create-namespace -f ./03-envoy-gateway/gateway-helm/values.yaml
echo "😀 Successfully installed Envoy Gateway"
kubectl wait --timeout=2m -n envoy-gateway-system deployment/envoy-gateway --for=condition=Available

# Install Envoy AI Gateway
echo "Installing Envoy AI Gateway ..."
#helm upgrade -i aieg-crd oci://docker.io/envoyproxy/ai-gateway-crds-helm \
#  --version ${ENVOY_AI_GATEWAY_VERSION} \
#  --namespace envoy-ai-gateway-system \
#  --create-namespace
#
#
#helm upgrade -i aieg oci://docker.io/envoyproxy/ai-gateway-helm \
#  --version ${ENVOY_AI_GATEWAY_VERSION} \
#  --namespace envoy-ai-gateway-system \
#  --create-namespace

helm install aieg ./04-envoy-ai-gateway/ai-gateway-helm --namespace envoy-ai-gateway-system --create-namespace -f ./04-envoy-ai-gateway/ai-gateway-helm/values.yaml
echo "😀 Successfully installed Envoy AI Gateway"
kubectl wait --timeout=2m -n envoy-ai-gateway-system deployment/ai-gateway-controller --for=condition=Available

# Configure Envoy Gateway for AI Gateway integration
echo "Configuring Envoy Gateway for AI Gateway integration ..."
#kubectl apply -f https://raw.githubusercontent.com/envoyproxy/ai-gateway/v${ENVOY_AI_GATEWAY_VERSION#v}/manifests/envoy-gateway-config/redis.yaml
#kubectl apply -f https://raw.githubusercontent.com/envoyproxy/ai-gateway/v${ENVOY_AI_GATEWAY_VERSION#v}/manifests/envoy-gateway-config/config.yaml
#kubectl apply -f https://raw.githubusercontent.com/envoyproxy/ai-gateway/v${ENVOY_AI_GATEWAY_VERSION#v}/manifests/envoy-gateway-config/rbac.yaml
kubectl apply -f 04-envoy-ai-gateway/envoy-gateway-config-redis.yaml
kubectl apply -f 04-envoy-ai-gateway/envoy-gateway-config-config.yaml
kubectl apply -f 04-envoy-ai-gateway/envoy-gateway-config-rbac.yaml

# Enable Gateway API Inference Extension support for Envoy Gateway
echo "Enabling Gateway API Inference Extension support for Envoy Gateway ..."
#kubectl apply -f https://raw.githubusercontent.com/envoyproxy/ai-gateway/v${ENVOY_AI_GATEWAY_VERSION#v}/examples/inference-pool/config.yaml
kubectl apply -f 04-envoy-ai-gateway/inference-pool-config.yaml
kubectl rollout restart -n envoy-gateway-system deployment/envoy-gateway
kubectl wait --timeout=2m -n envoy-gateway-system deployment/envoy-gateway --for=condition=Available
echo "😀 Successfully enabled Gateway API Inference Extension support for Envoy Gateway"

# Create kserve namespace if it doesn't exist
kubectl create namespace kserve --dry-run=client -o yaml | kubectl apply -f -

# Install Leader Worker Set (LWS)
echo "Installing Leader Worker Set ..."
#helm install lws oci://registry.k8s.io/lws/charts/lws \
#  --version=${LWS_VERSION} \
#  --namespace lws-system \
#  --create-namespace \
#  --wait --timeout 300s

helm install lws ./05-leaderWorkerSet-operator/lws --namespace kserve --create-namespace --wait --timeout 300s -f ./05-leaderWorkerSet-operator/lws/values.yaml
echo "😀 Successfully installed Leader Worker Set"


if [ "${installLLMISvc}" = false ]; then
   exit
fi


echo "Installing LLMISvc using local charts..."
# Install LLMISvc CRDs from local chart
helm install llmisvc-crd ./06-llmisvc/llmisvc-crd --namespace kserve --create-namespace --wait -f ./06-llmisvc/llmisvc-crd/values.yaml
# Install LLMISvc resources from local chart
helm install llmisvc ./06-llmisvc/llmisvc-resources --namespace kserve --create-namespace --wait -f ./06-llmisvc/llmisvc-resources/values.yaml
echo "😀 Successfully installed LLMISvc using local charts"


#if [ "${USE_LOCAL_CHARTS}" = true ]; then
#   # Install LLMISvc using local charts (to avoid template function errors in published charts)
#   echo "Installing LLMISvc using local charts..."
#   echo "📍 Using local charts from $(pwd)/charts/"
#   # Install LLMISvc CRDs from local chart
#   helm install llmisvc-crd ./charts/llmisvc-crd --namespace kserve --create-namespace --wait
#
#   # Install LLMISvc resources from local chart
#   helm install llmisvc ./charts/llmisvc-resources --namespace kserve --create-namespace --wait --set kserve.llmisvc.controller.tag=local-test --set kserve.llmisvc.controller.imagePullPolicy=Never
#   echo "😀 Successfully installed LLMISvc using local charts"
#
#else
#   echo "Installing LLMISvc ..."
#   helm install llmisvc-crd oci://ghcr.io/kserve/charts/llmisvc-crd --version ${LLMISVC_VERSION} --namespace kserve --create-namespace --wait
#   helm install llmisvc oci://ghcr.io/kserve/charts/llmisvc-resources --version ${LLMISVC_VERSION} --namespace kserve --create-namespace --wait
#
#fi
echo "😀 Successfully installed LLMISvc"


# 自定义EnvoyProxy镜像
kubectl apply -f - <<'EOF'
apiVersion: gateway.envoyproxy.io/v1alpha1
kind: EnvoyProxy
metadata:
  name: kserve-proxy-private
  namespace: kserve
spec:
  provider:
    type: Kubernetes
    kubernetes:
      envoyService:
        type: NodePort
      envoyDeployment:
        patch:
          type: StrategicMerge
          value:
            spec:
              template:
                spec:
                  containers:
                  - name: envoy
                    image: jusuan.io:8080/envoyproxy/envoy:distroless-v1.35.0
                  - name: shutdown-manager
                    image: jusuan.io:8080/envoyproxy/gateway-dev:latest
EOF

# Create GatewayClass for Envoy
echo "Creating GatewayClass for Envoy ..."
kubectl apply -f - <<EOF
apiVersion: gateway.networking.k8s.io/v1
kind: GatewayClass
metadata:
  name: envoy
spec:
  controllerName: gateway.envoyproxy.io/gatewayclass-controller
  parametersRef:
    group: gateway.envoyproxy.io
    kind: EnvoyProxy
    name: kserve-proxy-private
    namespace: kserve
EOF
echo "😀 Successfully created GatewayClass for Envoy"




# Create Gateway resource
echo "Creating kserve-ingress-gateway ..."
kubectl apply -f - <<EOF
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: kserve-ingress-gateway
  namespace: kserve
spec:
  gatewayClassName: envoy
  listeners:
    - name: http
      protocol: HTTP
      port: 80
      allowedRoutes:
        namespaces:
          from: All
  infrastructure:
    labels:
      serving.kserve.io/gateway: kserve-ingress-gateway
EOF
echo "😀 Successfully created kserve-ingress-gateway"