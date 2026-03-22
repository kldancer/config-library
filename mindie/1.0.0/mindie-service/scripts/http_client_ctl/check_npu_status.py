#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import logging
import os
import sys
import yaml

from file_util import FileUtils

logger = logging.getLogger('check_logger')
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
log_formatter = logging.Formatter('%(message)s')
handler.setFormatter(log_formatter)
logger.addHandler(handler)


def __is_npu_health():
    # 该文件为MA平台特有，如果没有则忽略
    yaml_file = "/opt/cloud/node/npu_status.yaml"
    if (os.path.exists(yaml_file) is False):
        return True

    check_file_flag, err_msg = FileUtils.check_file_size(yaml_file)
    if not check_file_flag:
        logger.error(f"[__is_npu_health] file size check failed %s", err_msg)
        return False
    try:
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)
        resources = data.get("resources")
        if resources is None:
            return True
    except Exception:
        # 如果文件非法，则不检查这个文件，认为健康
        return True

    if not isinstance(resources, list):
        return True

    for resource in resources:
        if not isinstance(resource, dict):
            continue
        if ("type" not in resource) or (not resource["type"] == "NPU"):
            continue
        if ("status" in resource):
            for item in resource["status"]:
                if (item["health"] is not None) and (item["health"] is False):
                    return False

    return True

if __name__ == "__main__":
    return_value = __is_npu_health()
    if return_value:
        sys.exit(0)
    else:
        sys.exit(1)
