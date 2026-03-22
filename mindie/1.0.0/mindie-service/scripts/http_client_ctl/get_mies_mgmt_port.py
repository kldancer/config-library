#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import json
import logging
import sys
from file_util import FileUtils

logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)

formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

KEY_SERVER_CONFIG = "ServerConfig"
KEY_MGMT_PORT = "managementPort"


def __check_file_path(file_path, mode, check_permission):
    check_path_flag, err_msg, real_path = FileUtils.regular_file_path(file_path)
    if not check_path_flag:
        logger.error(f"check file path failed: %s", err_msg)
        return False
    check_file_flag, err_msg = FileUtils.is_file_valid(real_path, mode=mode, check_permission=check_permission)
    if not check_file_flag:
        logger.error(f"check file path is invalid: %s", err_msg)
        return False
    return True


def __get_config_file_path() -> (str, bool):
    user_defined_config_file_path = os.getenv("MIES_CONFIG_JSON_PATH")
    if user_defined_config_file_path is not None and user_defined_config_file_path != "":
        return user_defined_config_file_path, os.getenv("MINDIE_CHECK_INPUTFILES_PERMISSION") != "0"
    root_path = os.getenv('MIES_INSTALL_PATH')
    if root_path is None or root_path == "":
        logging.error("env MIES_INSTALL_PATH not found.")
        return "", True
    return os.path.join(root_path, "conf/config.json"), True


def __get_port():
    json_file_path, check_permission = __get_config_file_path()

    if not __check_file_path(json_file_path, 0o640, check_permission):
        return -1
    if os.path.exists(json_file_path) is False:
        return -1
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    if not isinstance(data, dict) or KEY_SERVER_CONFIG not in data:
        return -1
    if not isinstance(data[KEY_SERVER_CONFIG], dict) or KEY_MGMT_PORT not in data[KEY_SERVER_CONFIG]:
        return -1

    port = data[KEY_SERVER_CONFIG][KEY_MGMT_PORT]
    if not isinstance(port, int):
        return -1
    if port < 1 or port > 65535:
        return -1
    return port


if __name__ == "__main__":
    return_value = __get_port()
    logger.info(return_value)
