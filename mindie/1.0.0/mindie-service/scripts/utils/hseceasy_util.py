#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import gc
import os
import re
import subprocess

from .log_util import logger
from .file_util import FileUtils

MIN_PASSWORD_LENGTH = 8
PASSWORD_REQUIREMENT = 2


class HseceasyUtil:

    @classmethod
    def store_key_pass_file(cls, project_path: str, key_pwd_path: str, password: str):
        stdout, stderr = None, None
        try:
            child = subprocess.Popen(
                ['bin/seceasy_encrypt', '--encrypt', '1', '2'],
                cwd=project_path, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, stdin=subprocess.PIPE, universal_newlines=True)
            stdout, stderr = child.communicate(f'{password}\n{password}\n', timeout=10)
            code = child.returncode
            if code != 0:
                logger.error(f'Error: encrypt password failed: {code}\n')
                logger.error(f'Error output: {stderr}\n')
                return False
            encrypted_password = None
            outputs = stdout.split('\n')
            for line in outputs:
                matches = re.match(r'encrypted: ([A-Za-z0-9/+=]*)', line, re.M | re.I)
                if matches:
                    encrypted_password = matches.group(1)
                    break

            if not encrypted_password:
                logger.error(f'Error: encrypt password failed.\n')
                return False
        except Exception as e:
            logger.error(f"Internal Error by {e}")
            return False

        try:
            if FileUtils.check_file_exists(key_pwd_path) and FileUtils.check_file_size(key_pwd_path):
                os.chmod(path=key_pwd_path, mode=0o600)
            flags = os.O_WRONLY | os.O_CREAT
            with os.fdopen(os.open(key_pwd_path, flags, 0o600), "w") as fd:
                fd.write(encrypted_password)
            return True
        finally:
            os.chmod(path=key_pwd_path, mode=0o400)

    @classmethod
    def check_password(cls, plain_text) -> bool:
        # Initialize flags for character types
        has_lower = has_upper = has_digits = has_symbol = False

        # Iterate through each character in the plain text
        special_characters = set("~!@#$%^&*()-_=+\\|[{}];:'\",<.>/? ")
        for char in plain_text:
            if char.islower():
                has_lower = True
            elif char.isupper():
                has_upper = True
            elif char.isdigit():
                has_digits = True
            elif char in special_characters:
                has_symbol = True
            # raise error if encounter invalid character
            else:
                logger.error("Password Check Failed")
                raise ValueError("Password Check Failed")

        # Check if password meets requirements
        if len(plain_text) >= MIN_PASSWORD_LENGTH and \
                (has_lower + has_upper + has_digits + has_symbol) >= PASSWORD_REQUIREMENT:
            return True
        else:
            logger.error("[MIE00E000210] The password is too weak. It should contain at least two of the following:"
                            " lowercase characters, uppercase characters, numbers, and symbols,"
                            " and the password must contain at least %d characters. ", MIN_PASSWORD_LENGTH)
            return False


    @classmethod
    def check_secret_file(cls, file_path: str) -> (bool, str):
        file_path_name = os.path.basename(file_path)
        check_flag, err_msg, dst = FileUtils.regular_file_path(file_path=file_path)
        if not check_flag:
            logger.error(f"Checking secret file path: {file_path_name} failed by {err_msg}")
            return False, ''

        if FileUtils.check_file_exists(dst):
            check_flag, err_msg = FileUtils.is_file_valid(file_path=dst, mode=0o400)
            if not check_flag:
                logger.error(f"Checking secret file {file_path_name} failed by {err_msg}")
                return False, ''
        return True, dst

    @classmethod
    def clear_secret_file(cls, file_path: str):
        if FileUtils.check_file_exists(file_path) and FileUtils.check_file_size(file_path):
            try:
                os.chmod(path=file_path, mode=0o600)
                file_size = os.path.getsize(file_path)
                with open(file_path, 'rb+') as file:
                    file.seek(0)
                    file.truncate()
                    file.write(bytearray([0] * file_size))
                    file.seek(0)
                    file.truncate()
                    file.write(bytearray([1] * file_size))
                    file.seek(0)
                    file.truncate()
                    file.write(os.urandom(file_size))
            except Exception as e:
                logger.error(f"Clear secret file {os.path.basename(file_path)} failed: {e}")
                # 异常抛出，统一处理
                raise ValueError("File open or file chmod filed when clear secret file") from e
            finally:
                os.chmod(path=file_path, mode=0o400)

