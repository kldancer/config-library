#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

path="${BASH_SOURCE[0]}"

if [[ -f "$path" ]] && [[ "$path" =~ 'set_env.sh' ]];then
    mindie_path=$(cd $(dirname $path); pwd )

    if [[ -f "${mindie_path}/latest/version.info" ]];then
    	if [ -f "${mindie_path}/latest/mindie-rt/set_env.sh" ] && \
            [ -f "${mindie_path}/latest/mindie-rt/version.info" ];then
			source ${mindie_path}/latest/mindie-rt/set_env.sh
		else
			echo "mindie-rt package is incomplete, please check it."
		fi

    	if [ -f "${mindie_path}/latest/mindie-torch/set_env.sh" ] && \
            [ -f "${mindie_path}/latest/mindie-torch/version.info" ];then
			source ${mindie_path}/latest/mindie-torch/set_env.sh
		else
			echo "mindie-torch package is incomplete, please check it."
		fi

    	if [[ -f "${mindie_path}/latest/mindie-service/set_env.sh" ]];then
			source ${mindie_path}/latest/mindie-service/set_env.sh
		else
			echo "mindie-service package is incomplete please check it."
		fi

    	if [[ -f "${mindie_path}/latest/mindie-llm/set_env.sh" ]];then
			source ${mindie_path}/latest/mindie-llm/set_env.sh
		else
			echo "mindie-llm package is incomplete please check it."
		fi

    else
        echo "The package of mindie is incomplete, please check it."
    fi
else
    echo "There is no 'set_env.sh' to import"
fi