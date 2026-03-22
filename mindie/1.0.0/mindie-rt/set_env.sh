#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.

path="${BASH_SOURCE[0]}"

if [[ -f "$path" ]] && [[ "$path" =~ 'set_env.sh' ]];then
    aie_path=$(cd $(dirname $path); pwd )

    if [[ -f "${aie_path}/version.info" ]];then
        export ASCENDIE_HOME="${aie_path}"
        export TUNE_BANK_PATH="${ASCENDIE_HOME}/aoe"
        export LD_LIBRARY_PATH="${ASCENDIE_HOME}/lib":${LD_LIBRARY_PATH}
        export ASCEND_CUSTOM_OPP_PATH="${ASCENDIE_HOME}/ops/vendors/customize":${ASCEND_CUSTOM_OPP_PATH}
        export ASCEND_CUSTOM_OPP_PATH="${ASCENDIE_HOME}/ops/vendors/aie_ascendc":${ASCEND_CUSTOM_OPP_PATH}
    else
        echo "The package is incomplete, please check it."
    fi
else
    echo "There is no 'set_env.sh' to import"
fi