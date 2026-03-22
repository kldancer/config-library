#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import json
import sys
import os
import time
import stat
import argparse
from update_log import UpdateLogger
from config_adapter import Config, ConfigAdapter
from server_file_util import FileUtils


def __read_latest_version():
    return "1.1.0"


def __check_file_path(file_path, logger, mode):
    check_path_flag, err_msg, real_path = FileUtils.regular_file_path(file_path)
    if not check_path_flag:
        logger.error(f"check file path failed: %s", err_msg)
        return False
    check_file_flag, err_msg = FileUtils.is_file_valid(real_path, mode=mode)
    if not check_file_flag:
        logger.error(f"check file path is invalid: %s", err_msg)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        prog='python3 update.py',
        description='update config to target version.',
    )

    parser.add_argument('--old_config_path', type=str, required=True, help='Path of the config file to be updated')
    parser.add_argument('--new_config_path', default='', type=str, required=False,
                        help='Path of the target version config file, default value {timestamp}.json')
    parser.add_argument('--target_version', default=__read_latest_version(), type=str, required=False,
                        choices=["0.2.0", "1.0.0", "1.1.0"], help=f'Version of the target config file, default '
                                                                    f'value is latest '
                                                                    f'version {__read_latest_version()}')
    parser.add_argument('--enable_log', action='store_true', help=f'enable log or not')
    args = parser.parse_args()

    old_config_path = args.old_config_path
    new_config_path = args.new_config_path
    enable_log = args.enable_log
    logger = UpdateLogger(enable_log).logger
    if not __check_file_path(old_config_path, logger, 0o640):
        return False
    timestamp = 0
    if new_config_path == '':
        timestamp = int(time.time())
        new_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), str(timestamp) + ".json")
    if not os.path.basename(new_config_path) == str(timestamp) + ".json" and not (
        FileUtils.is_json_file(new_config_path) and os.path.dirname(new_config_path)):
        if not __check_file_path(new_config_path, logger, 0o640):
            return False
    if os.path.isfile(new_config_path):
        logger.info(f"new config file already exists")
        return False

    target_version = args.target_version
    old_config = Config()
    old_config.parse_py_config_path(old_config_path)
    logger.info(f"Start to update old config version: {old_config.version} to new config version: {target_version}")
    new_config = ConfigAdapter(old_config, target_version).update()
    if new_config is None:
        logger.info(f"config can not be upgrade, please check old or target version")
        return False

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    modes = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP
    try:
        with os.fdopen(os.open(new_config_path, flags, modes), 'w') as f:
            json.dump(new_config, f, indent=4)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return False
    logger.info(f"Success to update old config version {old_config.version} to new config version {target_version}")
    return True


if __name__ == '__main__':
    ret = main()
    if not ret:
        sys.exit(1)