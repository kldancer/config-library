#!/bin/bash

# 检查是否传入探针类型
if [ -z "$1" ]; then
  echo "Error: Missing probe type. Please provide one of 'startup', 'readiness', or 'liveness'."
  exit 1
fi
PROBE_TYPE=$1
PWD=$(cd "$(dirname "$0")"; pwd)

export HSECEASY_PATH=$MIES_INSTALL_PATH/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MIES_INSTALL_PATH/lib
export MINDIE_UTILS_HTTP_CLIENT_CTL_CONFIG_FILE_PATH=$MIES_INSTALL_PATH/conf/http_client_ctl.json

group_id=-1
if [[ $MINDIE_SERVER_PROBE_ONLY -ne 1 ]]; then
  PYTHONUNBUFFERED=1 python3 $PWD/get_group_id.py
  group_id=$?
  if [ $group_id -eq 255 ]; then
    echo "Error: Getting group id from the global rank table failed."
    exit 1
  fi
fi

if [[ $MINDIE_SERVER_DISTRIBUTE -eq 1 ]]; then
  PYTHONUNBUFFERED=1 python3 $PWD/get_distribute_role.py
  role=$?
  if [ $role -eq 255 ]; then
    echo "Error: Getting group id from the global rank table failed."
    exit 1
  fi
  if [ $role -eq 1 ]; then
    echo "Distributed slave mindie-server skip probe."
    exit 0
  fi
fi

# MindIE-Server
SERVICE_TYPE=
if [ $group_id -eq 2 ] || [[ $MINDIE_SERVER_PROBE_ONLY -eq 1 ]]; then
    STARTUP_URL=/v2/health/ready
    READINESS_URL=/v2/health/ready
    LIVENESS_URL=/v2/health/ready
    PORT=$(python3 $PWD/get_mies_mgmt_port.py)
    if [ $PORT -eq -1 ]; then
      echo "get PORT fail"
      exit 1
    fi
    SERVICE_TYPE="mies"
fi

# MindIE-MS-Controller
if [ $group_id -eq 1 ]; then
    STARTUP_URL=/v1/startup
    READINESS_URL=/v1/health
    LIVENESS_URL=/v1/health
    PORT=1026
fi

# MindIE-MS-Coordinator
if [ $group_id -eq 0 ]; then
    STARTUP_URL=/v1/startup
    READINESS_URL=/v1/readiness
    LIVENESS_URL=/v1/health
    PORT=1026
fi


# 根据不同的探针类型执行不同的逻辑
case "$PROBE_TYPE" in
  startup)
    echo "Executing startup probe..."
    # 在这里放置你需要的启动探针逻辑
    # 比如检查某个服务是否已经成功启动
    $MIES_INSTALL_PATH/bin/http_client_ctl $POD_IP $PORT $STARTUP_URL 600 0
    if [ $? -eq 0 ]; then
      echo "Service is running."
      exit 0
    else
      echo "Service is not running."
      exit 1
    fi
    ;;

  readiness)
    echo "Executing readiness probe..."
    # 在这里放置你的就绪探针逻辑
    # 比如检查某个API端点是否可用
    $MIES_INSTALL_PATH/bin/http_client_ctl $POD_IP $PORT $READINESS_URL 600 0
    if [ $? -ne 0 ]; then
      echo "Service is not ready."
      exit 1
    fi
    if [ "${SERVICE_TYPE}X" = "miesX" ]; then
      python3 $PWD/check_npu_status.py
      if [ $? -ne 0 ]; then
        echo "Service is not ready."
        exit 1
      fi
    fi
    echo "Service is ready."
    exit 0
    ;;

  liveness)
    echo "Executing liveness probe..."
    # 在这里放置你的存活探针逻辑
    # 比如检查进程是否还在运行
    $MIES_INSTALL_PATH/bin/http_client_ctl $POD_IP $PORT $LIVENESS_URL 600 0
    if [ $? -ne 0 ]; then
      echo "Service is not alive."
      exit 1
    fi
    if [ "${SERVICE_TYPE}X" = "miesX" ]; then
      python3 $PWD/check_npu_status.py
      if [ $? -ne 0 ]; then
        echo "Service is not alive."
        exit 1
      fi
    fi
    echo "Service is alive."
    exit 0
    ;;

  *)
    echo "Error: Invalid probe type. Please use 'startup', 'readiness', or 'liveness'."
    exit 1
    ;;
esac