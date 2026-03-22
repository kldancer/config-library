#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import psutil

from .log_util import logger


class ServerUtil:

    @classmethod
    def is_server_running(cls, process_name: str) -> bool:
        for proc in psutil.process_iter():
            try:
                if process_name in proc.name() and proc.status() in (psutil.STATUS_RUNNING, psutil.STATUS_SLEEPING):
                    return True
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                return False
            except psutil.AccessDenied:
                logger.error(f"Get process: {process_name}  status failed by access denied.")
                return True
        return False

