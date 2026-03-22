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

logging.basicConfig(level=logging.INFO)

GLOBAL_RANK_TABLE_ENV = 'GLOBAL_RANK_TABLE_FILE_PATH'


def wait_global_ranktable_completed(argv):
    global_rank_table_path = os.getenv(GLOBAL_RANK_TABLE_ENV)
    if not global_rank_table_path:
        logging.error('read env \"%s\" failed', GLOBAL_RANK_TABLE_ENV)
        return 255
    try:
        while True:
            with open(global_rank_table_path, 'r') as file:
                buf = file.read()
            rank_table = json.loads(buf)
            if rank_table["status"] == "completed":
                break
            else:
                logging.error("status of ranktable is not completed!")
                time.sleep(1)
        server_group_list = rank_table['server_group_list']
        pod_ip = os.getenv('POD_IP')
        for group in server_group_list:
            group_id = "-1"
            server_list = group["server_list"]
            for server in server_list:
                if server["server_ip"] == pod_ip:
                    group_id = group["group_id"]
            if group_id == '2': # 启动MindIE-Server
                return 2
            elif group_id == '1': # 启动MindIE-MS mindie-ms-controller
                return 1
            elif group_id == '0': # 启动MindIE-MS coordinator
                return 0
            else:
                continue
        return 255
    except Exception as e:
        logging.error(e)
        return 255

if __name__ == "__main__":
    sys.exit(wait_global_ranktable_completed(sys.argv[1:]))