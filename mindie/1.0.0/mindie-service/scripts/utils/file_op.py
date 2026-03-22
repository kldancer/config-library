#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import os
import shutil

from .log_util import logger
from .file_util import FileUtils


class FileOps:

    @classmethod
    def delete(cls, path):
        """Delete a file.

        :param * path: list of str path to joined as a
        new base directory to make.

        """
        check_flag, err_msg, path = FileUtils.regular_file_path(path, base_dir="/", allow_symlink=False)
        if not check_flag:
            raise ValueError(err_msg)
        check_flag, err_msg = FileUtils.is_file_valid(path, mode=0o777, check_owner=False, check_permission=False)
        if not check_flag:
            raise ValueError(err_msg)
        os.remove(path)
        return

    @classmethod
    def copy_file(cls, src, dst, mode=0o400):
        """
            Copy a file from src to dst.
        """
        if os.path.isdir(dst):
            basename = os.path.basename(src)
            dst = os.path.join(dst, basename)

        check_flag, err_msg, src = FileUtils.regular_file_path(src, base_dir="/", allow_symlink=False)
        if not check_flag:
            raise ValueError(err_msg)
        check_flag, err_msg = FileUtils.is_file_valid(src, mode=0o777, check_owner=False, check_permission=False)
        if not check_flag:
            raise ValueError(err_msg)

        check_flag, err_msg, dst = FileUtils.regular_file_path(dst, base_dir="/", allow_symlink=False)
        if not check_flag:
            raise ValueError(err_msg)
        if FileUtils.check_file_exists(dst):
            check_flag, err_msg = FileUtils.is_file_valid(dst, mode=0o777, check_owner=False, check_permission=False)
            if not check_flag:
                raise ValueError(err_msg)

        try:
            if FileUtils.check_file_exists(dst):
                os.chmod(dst, mode=0o600)
            shutil.copy(src, dst)
        except shutil.SameFileError:
            logger.warning("Copying same file, ignored")
        finally:
            os.chmod(dst, mode=mode)

    @classmethod
    def list_file_name(cls, dir_path: str) -> set:
        """
            List dir path file.
        """
        file_list = set()
        check_flag, err_msg, dir_path = FileUtils.regular_file_path(dir_path, base_dir="/", allow_symlink=False)
        if not check_flag:
            raise ValueError(err_msg)
        for filename in os.listdir(dir_path):
            if os.path.isfile(os.path.join(dir_path, filename)):
                file_list.add(os.path.basename(filename))

        return file_list

