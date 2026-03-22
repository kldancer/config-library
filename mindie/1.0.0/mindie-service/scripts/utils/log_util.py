#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import os
import time
import logging
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler

log_to_file: bool = os.getenv("MIES_CERTS_LOG_TO_FILE", "0") == "1"
log_to_stdout: bool = os.getenv("MIES_CERTS_LOG_TO_STDOUT", "1") == "1"
log_level_env = os.getenv("MIES_CERTS_LOG_LEVEL", "INFO")
log_path = os.getenv("MIES_CERTS_LOG_PATH", "/workspace/log/certs.log")

LOG_MAX_BYTES = (500 << 20)  # 500MB
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.FATAL
}


class NoNewlineFormatter(logging.Formatter):
    def format(self, record):
        special_chars = [
            '\n', '\r', '\f', '\t', '\v', '\b',
            '\u000A', '\u000D', '\u000C',
            '\u000B', '\u0008', '\u007F',
            '\u0009', '    ',
        ]
        for c in special_chars:
            record.msg = str(record.msg).replace(c, ' ')
        if record.levelname == "WARNING":
            record.levelname = "WARN"
        return super(NoNewlineFormatter, self).format(record)

    def formatTime(self, record, datefmt=None):
        timezone_offset = time.timezone
        offset_hours = -timezone_offset // 3600
        dt = datetime.fromtimestamp(record.created, timezone(timedelta(hours=offset_hours)))
        timestamp = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        offset = dt.strftime("%z")
        offset = f"{offset[:3]}:{offset[3:]}"
        return f"{timestamp}{offset} DST" if time.daylight else f"{timestamp}{offset}"


LOG_FORMATTER = NoNewlineFormatter(
    '[%(asctime)s] [%(process)d] [%(thread)d] [%(name)s] [%(levelname)s] '
    '[%(filename)s:%(lineno)s] %(message)s'
)


def check_log_dir():
    log_dir = os.path.dirname(log_path)
    try:
        os.makedirs(log_dir, exist_ok=True)
        os.chmod(log_dir, 0o700)
    except OSError as e:
        raise Exception("OSError when creating python log dir") from e


def setup_logging():
    pid = os.getpid()
    custom_logger = logging.getLogger(f"cert")
    custom_logger.propagate = False
    custom_logger.setLevel(LOG_LEVEL_MAP.get(log_level_env, logging.INFO))

    if log_to_file:
        check_log_dir()
        certs_logger_path = f"{log_path}"
        clean_path = os.path.normpath(certs_logger_path)
        if os.path.islink(clean_path):
            raise ValueError(f"Path of log file is a soft link")
        file_handle = RotatingFileHandler(
            filename=certs_logger_path,
            maxBytes=LOG_MAX_BYTES,
            backupCount=10)
        os.chmod(certs_logger_path, 0o600)
        file_handle.setFormatter(LOG_FORMATTER)
        custom_logger.addHandler(file_handle)
    if log_to_stdout:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(LOG_FORMATTER)
        custom_logger.addHandler(stream_handler)
    return custom_logger


logger = setup_logging()


def is_log_enable():
    return log_to_file or log_to_stdout

