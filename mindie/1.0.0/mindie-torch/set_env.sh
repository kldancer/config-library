#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
path="${BASH_SOURCE[0]}"

if [[ -f "$path" ]] && [[ "$path" =~ 'set_env.sh' ]];then
    torch_mindie_path=$(cd $(dirname $path); pwd )

    if [[ -f "$torch_mindie_path"/version.info ]];then
        export MINDIE_TORCH_HOME="$torch_mindie_path"
        export LD_LIBRARY_PATH="${MINDIE_TORCH_HOME}/lib":${LD_LIBRARY_PATH}
    else
        echo "The package is incomplete, please check it."
        fi
else
    echo "There is no 'set_env.sh' to import"
fi