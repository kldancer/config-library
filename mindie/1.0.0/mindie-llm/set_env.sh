#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

path="${BASH_SOURCE[0]}"

if [[ -f "$path" ]] && [[ "$path" =~ 'set_env.sh' ]];then
    mindie_llm_path=$(cd $(dirname $path); pwd)
    chmod u+w "${mindie_llm_path}"
    py_logs_path=${mindie_llm_path}/logs
    if [ ! -d "${py_logs_path}" ]; then
        mkdir "${py_logs_path}"
    fi
    chmod 750 "${py_logs_path}"

    export MINDIE_LLM_HOME_PATH="${mindie_llm_path}"
    export MINDIE_LLM_PYTHON_LOG_LEVEL=INFO
    export MINDIE_LLM_PYTHON_LOG_TO_STDOUT=1
    export MINDIE_LLM_PYTHON_LOG_TO_FILE=0
    export MINDIE_LLM_PYTHON_LOG_PATH="${py_logs_path}"

    export MINDIE_LLM_CONTINUOUS_BATCHING=1
    export MINDIE_LLM_RECOMPUTE_THRESHOLD=0.5
    export LD_LIBRARY_PATH=$MINDIE_LLM_HOME_PATH/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$MINDIE_LLM_HOME_PATH/lib/grpc:$LD_LIBRARY_PATH
    export PYTHONPATH=$MINDIE_LLM_HOME_PATH:$PYTHONPATH
    export PYTHONPATH=$MINDIE_LLM_HOME_PATH/lib:$PYTHONPATH

    export MINDIE_LLM_LOG_LEVEL=INFO
    export MINDIE_LLM_LOG_TO_STDOUT=0
    export MINDIE_LLM_LOG_TO_FILE=1

    if [[ -z "$1" ]]; then
        MINDIE_LLM_BACKEND=("atb" "pt" "ms")
    else
        if [[ "$1" == "--backend="* ]]; then
            MINDIE_LLM_BACKEND="${1#*=}"
        else
            echo "Usage: source set_env.sh --backend=<backend>"
        fi
    fi

    for backend_opt in "${MINDIE_LLM_BACKEND[@]}"; do
        case "$backend_opt" in
            atb)
                ATB_SET_ENV_PATH=$MINDIE_LLM_HOME_PATH/../examples/atb_models/output/atb_models/set_env.sh
                if [ ! -f "$ATB_SET_ENV_PATH" ]; then
                    ATB_SET_ENV_PATH=latest/atb_models/set_env.sh
                fi
                if [ -f "$ATB_SET_ENV_PATH" ]; then
                    source $ATB_SET_ENV_PATH
                fi
                ;;
            pt)
                ;;
            ms)
                ;;
            *)
                echo "Inner Error: unknown option'$backend_opt'"
                ;;
        esac
    done
else
	echo "There is no 'set_env.sh' to import"
fi
