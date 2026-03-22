#!/bin/bash
source $MINDIE_USER_HOME_PATH/Ascend/mindie/set_env.sh
PWD=$(cd "$(dirname "$0")"; pwd)
BOOT_SCRIPT_DIR=$PWD
PYTHONUNBUFFERED=1 python3 $BOOT_SCRIPT_DIR/get_group_id.py
exit_code=$?
export HSECEASY_PATH=$MIES_INSTALL_PATH/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MIES_INSTALL_PATH/lib
export MINDIE_LOG_TO_FILE=1
export MINDIE_LOG_TO_STDOUT=1
cd $MIES_INSTALL_PATH
if [ $exit_code -eq 2 ]; then
        if [ -n "$CONFIG_FROM_CONFIGMAP_PATH" ]; then
            cp $CONFIG_FROM_CONFIGMAP_PATH/config.json $MIES_INSTALL_PATH/conf/config.json
            cp $CONFIG_FROM_CONFIGMAP_PATH/http_client_ctl.json $MIES_INSTALL_PATH/conf/http_client_ctl.json
        fi
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common
        source $MINDIE_USER_HOME_PATH/Ascend/ascend-toolkit/set_env.sh
        source $MINDIE_USER_HOME_PATH/Ascend/llm_model/set_env.sh
        source $MINDIE_USER_HOME_PATH/Ascend/nnal/atb/set_env.sh
        export MIES_CONTAINER_IP=$POD_IP
        export MIES_CONTAINER_MANAGEMENT_IP=$POD_IP
        export MINDIE_LLM_PYTHON_LOG_TO_FILE=1
        export MINDIE_LLM_PYTHON_LOG_TO_STDOUT=1
        export MINDIE_LLM_PYTHON_LOG_LEVEL=INFO
        export MINDIE_LLM_PYTHON_LOG_PATH=${MIES_INSTALL_PATH}/logs/pythonlog.log
        export MINDIE_LLM_LOG_TO_FILE=1
        export MINDIE_LLM_LOG_TO_STDOUT=1
        export MINDIE_LLM_LOG_LEVEL=INFO
        PYTHONUNBUFFERED=1 python3 $BOOT_SCRIPT_DIR/update_mindie_server_config.py $MIES_INSTALL_PATH/conf/config.json
        ./bin/mindieservice_daemon
fi

if [ $exit_code -eq 1 ]; then
  if [ -n "$CONFIG_FROM_CONFIGMAP_PATH" ]; then
      cp $CONFIG_FROM_CONFIGMAP_PATH/ms_controller.json $MIES_INSTALL_PATH/conf/ms_controller.json
      cp $CONFIG_FROM_CONFIGMAP_PATH/http_client_ctl.json $MIES_INSTALL_PATH/conf/http_client_ctl.json
  fi
  cp $GLOBAL_RANK_TABLE_FILE_PATH $MIES_INSTALL_PATH
  export GLOBAL_RANK_TABLE_FILE_PATH=$MIES_INSTALL_PATH/global_ranktable.json
  export MINDIE_MS_CONTROLLER_CONFIG_FILE_PATH=$MIES_INSTALL_PATH/conf/ms_controller.json
  ./bin/ms_controller
fi

if [ $exit_code -eq 0 ]; then
  if [ -n "$CONFIG_FROM_CONFIGMAP_PATH" ]; then
      cp $CONFIG_FROM_CONFIGMAP_PATH/ms_coordinator.json $MIES_INSTALL_PATH/conf/ms_coordinator.json
      cp $CONFIG_FROM_CONFIGMAP_PATH/http_client_ctl.json $MIES_INSTALL_PATH/conf/http_client_ctl.json
  fi
  export MINDIE_MS_COORDINATOR_CONFIG_FILE_PATH=$MIES_INSTALL_PATH/conf/ms_coordinator.json
  ./bin/ms_coordinator $POD_IP 1025 $POD_IP 1026
fi