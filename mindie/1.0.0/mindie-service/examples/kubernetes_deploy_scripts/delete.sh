#!/bin/bash
rm -rf ./logs/*
kubectl delete cm global-ranktable -n mindie;
kubectl delete cm python-script-get-group-id -n mindie;
kubectl delete cm python-script-update-server-conf -n mindie;
kubectl delete cm boot-bash-script -n mindie;
kubectl delete cm mindie-server-config -n mindie;
kubectl delete cm mindie-ms-controller-config -n mindie;
kubectl delete cm mindie-ms-coordinator-config -n mindie;
kubectl delete cm mindie-http-client-ctl-config -n mindie;
kubectl delete cm common-env -n mindie;
YAML_DIR=./deployment
for yaml_file in $YAML_DIR/*.yaml; do
  # 检查文件是否存在
  if [ -f "$yaml_file" ]; then
    # 使用kubectl delete命令删除每个YAML文件中定义的部署
    kubectl delete -f "$yaml_file"
  fi
done