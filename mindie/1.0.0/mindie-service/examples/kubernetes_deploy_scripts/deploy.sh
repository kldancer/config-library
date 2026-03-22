#!/bin/bash
kubectl create configmap common-env --from-literal=MINDIE_USER_HOME_PATH=/usr/local -n mindie
kubectl create configmap python-script-get-group-id --from-file=./boot_helper/get_group_id.py -n mindie;
kubectl create configmap python-script-update-server-conf --from-file=./boot_helper/update_mindie_server_config.py -n mindie;
kubectl create configmap boot-bash-script --from-file=./boot_helper/boot.sh -n mindie;

kubectl create configmap global-ranktable --from-file=./gen_ranktable_helper/global_ranktable.json -n mindie;
kubectl create configmap mindie-server-config --from-file=./conf/config.json -n mindie;
kubectl create configmap mindie-ms-coordinator-config --from-file=./conf/ms_coordinator.json -n mindie;
kubectl create configmap mindie-ms-controller-config --from-file=./conf/ms_controller.json -n mindie;
kubectl create configmap mindie-http-client-ctl-config --from-file=./conf/http_client_ctl.json -n mindie;

kubectl apply -f ./deployment/mindie_ms_coordinator.yaml;
kubectl apply -f ./deployment/mindie_ms_controller.yaml;
kubectl apply -f ./deployment/mindie_server.yaml;
if [ $# -eq 1 ]; then
    if [[ $1 == "heter" ]]; then
        kubectl apply -f ./deployment/mindie_server_heterogeneous.yaml;
        python3 ./gen_ranktable_helper/gen_global_ranktable.py --heter True
        exit 0
    fi
fi

python3 ./gen_ranktable_helper/gen_global_ranktable.py