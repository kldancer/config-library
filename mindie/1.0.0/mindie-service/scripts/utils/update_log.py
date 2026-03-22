#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2012-2024. All rights reserved.

import logging


def setup_stdout_logging(log_name, log_level, log_format, enable_log):
    custom_logger = logging.getLogger(log_name)
    custom_logger.propagate = False
    custom_logger.setLevel(log_level)

    if enable_log:
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter(log_format)
        stream_handler.setFormatter(formatter)
        custom_logger.addHandler(stream_handler)
    return custom_logger


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class UpdateLogger(metaclass=Singleton):
    def __init__(self, enable):
        self._format = '%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s'
        self._logger = setup_stdout_logging("update_config_date", logging.INFO, self._format, enable)

    @property
    def logger(self):
        return self._logger

