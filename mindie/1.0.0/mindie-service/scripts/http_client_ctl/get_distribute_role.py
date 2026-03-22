#!/usr/bin/env python3
# coding=utf-8

# Copyright (c) 2024 Huawei Technologies Co., Ltd
# All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import json
import os
import time
import logging
from file_util import FileUtils
logging.basicConfig(level=logging.INFO)

RANK_TABLE_ENV = 'RANKTABLEFILE'
PARSE_ERROR = 255
MASTER_LABEL = 0
SLAVE_LABEL = 1


def get_distribute_role():
    rank_table_path = os.getenv(RANK_TABLE_ENV)
    if not rank_table_path:
        logging.error('read env \"%s\" failed', RANK_TABLE_ENV)
        return PARSE_ERROR
    try:
        check_path_flag, err_msg, real_path = FileUtils.regular_file_path(rank_table_path)
        if not check_path_flag:
            logger.error(f"check file path failed: %s", err_msg)
            return PARSE_ERROR
        with open(real_path, 'r') as file:
            buf = file.read()
        rank_table = json.loads(buf)
        if rank_table["status"] != "completed":
            logging.error("status of ranktable is not completed!")
            return PARSE_ERROR
        master_server = rank_table["server_list"][0]
        pod_ip = os.getenv('MIES_CONTAINER_IP')
        if master_server["container_ip"] == pod_ip:
            return MASTER_LABEL
        else:
            return SLAVE_LABEL
    except Exception as e:
        logging.error(e)
        return PARSE_ERROR

if __name__ == "__main__":
    sys.exit(get_distribute_role())