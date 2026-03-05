
bash 01.manage.cert-manager-helm.sh
./02.manage.gateway-api-crd.sh
./03.manage.envoy-gateway-helm.sh
./04.manage.envoy-ai-gateway-helm.sh
./05.manage.lws-operator.sh
./06.manage.gateway-api-gwclass.sh
./07.manage.gateway-api-gw.sh
./08.manage.kserve-helm.sh






./08.manage.kserve-helm.sh --uninstall
./07.manage.gateway-api-gw.sh --uninstall
./06.manage.gateway-api-gwclass.sh --uninstall
./05.manage.lws-operator.sh --uninstall
./04.manage.envoy-ai-gateway-helm.sh --uninstall
./03.manage.envoy-gateway-helm.sh --uninstall
./02.manage.gateway-api-crd.sh --uninstall
./01.manage.cert-manager-helm.sh --uninstall









#helm pull jetstack/cert-manager  --version v1.17.0 --untar -d .
#helm pull oci://docker.io/envoyproxy/gateway-helm --version v1.6.3 --untar -d .
#helm pull oci://docker.io/envoyproxy/ai-gateway-crds-helm --version v0.5.0 --untar -d .
#helm pull  oci://docker.io/envoyproxy/ai-gateway-helm  --version v0.5.0 --untar -d .
#helm pull  oci://ghcr.io/kserve/charts/kserve-crd  --version v0.16.0 --untar -d .
#helm pull oci://ghcr.io/kserve/charts/kserve --version v0.16.0 --untar -d .
#helm pull  oci://ghcr.io/kserve/charts/kserve-llmisvc-crd  --version v0.16.0 --untar -d .
#helm pull  oci://ghcr.io/kserve/charts/kserve-llmisvc-resources  --version v0.16.0 --untar -d .