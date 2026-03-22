#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import os
from enum import Enum

from .file_util import FileUtils
from .log_util import logger


class CommonFlag(Enum):
    GEN_CERT = 'gen_cert'
    IMPORT_CA = 'import_ca'
    DELETE_CA = 'delete_ca'
    IMPORT_CERT = 'import_cert'
    DELETE_CERT = 'delete_cert'
    IMPORT_CRL = 'import_crl'
    QUERY_CERT = 'query'
    RESTORE_CA = 'restore_ca'
    RESTORE_CERT = 'restore_cert'
    ERROR = 'parameter_error'


class Parameter:
    def __init__(self, args, flag: CommonFlag):
        security = 'security'
        # 软件安装路径--文件夹
        self.base_dir = args.project_path
        logger.info(f"Install path is {self.base_dir}")
        # 业务名称
        self.business = args.business
        logger.info("Business type is " + self.business)
        # 命令模式
        self.flag = flag
        # ca 文件列表--外部文件
        self.ca_files = args.ca_files if flag in (CommonFlag.IMPORT_CA, CommonFlag.DELETE_CA) else []
        # 服务证书文件--外部文件
        self.cert_file = args.cert_file if flag in (CommonFlag.IMPORT_CERT, CommonFlag.DELETE_CERT,
            CommonFlag.QUERY_CERT) else None
        # 服务证书私钥文件--外部文件
        self.key_file = args.key_file if flag in (CommonFlag.IMPORT_CERT, CommonFlag.DELETE_CERT) else None
        # 吊销列表文件--外部文件
        self.crl_file = args.crl_file if flag in (CommonFlag.IMPORT_CRL, CommonFlag.QUERY_CERT) else None
        # 生成证书的配置文件--外部文件
        self.conf_file = args.conf_file if flag is CommonFlag.GEN_CERT else None
        self.ip = args.ip if flag is CommonFlag.GEN_CERT else None
        # 备份 ca 证书的路径
        self.ca_backup_file = args.ca_backup_file if flag in (CommonFlag.RESTORE_CA,) else None
        # 恢复 ca 证书的路径
        self.ca_dst_file = args.ca_dst_file if flag in (CommonFlag.RESTORE_CA,) else None
        # 备份服务证书文件的路径
        self.cert_backup_file = args.cert_backup_file if flag in (CommonFlag.RESTORE_CERT,) else None
        # 恢复服务证书文件的路径
        self.cert_dst_file = args.cert_dst_file if flag in (CommonFlag.RESTORE_CERT,) else None
        # 备份服务证书私钥文件的路径
        self.key_backup_file = args.key_backup_file if flag in (CommonFlag.RESTORE_CERT,) else None
        # 恢复服务证书私钥文件的路径
        self.key_dst_file = args.key_dst_file if flag in (CommonFlag.RESTORE_CERT,) else None

        # 组合路径
        # 文件
        self.hseceasy_bin_path = os.path.join(self.base_dir, 'bin/seceasy_encrypt')
        self.gen_cert_bin_path = os.path.join(self.base_dir, 'bin/gen_cert')
        # 文件
        self.key_pwd_path = os.path.join(self.base_dir, security, self.business, 'pass/key_pwd.txt')

        # 文件夹
        self.ca_dir_path = os.path.join(self.base_dir, security, self.business, 'ca/')
        # 文件夹
        self.certs_dir_path = os.path.join(self.base_dir, security, self.business, 'certs/')
        # 文件夹
        self.keys_dir_path = os.path.join(self.base_dir, security, self.business, 'keys/')
        # 文件夹
        self.pass_dir_path = os.path.join(self.base_dir, security, self.business, 'pass/')
        # 文件夹
        self.crl_dir_path = os.path.join(self.base_dir, security, self.business, 'certs/')

    def check_path(self) -> bool:
        """
        外部直接、间接路径入参校验

        1. 路径校验【regular_file_path】
        3. 若文件预期存在 【is_file_valid】
        :return: True/False
        """
        if not self.__base_check():
            logger.error("base check failed")
            return False
        if CommonFlag.IMPORT_CA == self.flag:
            if not self.__check_ca_dir() or not self.__check_ca_files():
                return False
            return True

        if CommonFlag.DELETE_CA == self.flag:
            if not self.__check_ca_dir():
                return False
            return True

        if CommonFlag.IMPORT_CERT == self.flag:
            is_cert_dir_valid = self.__check_cert_dir()
            is_key_dir_valid = self.__check_key_dir()
            is_pass_valid = self.__check_pass()
            is_certs_files_valis = self.__check_certs_files()
            return is_cert_dir_valid and is_key_dir_valid and is_pass_valid and is_certs_files_valis

        if CommonFlag.DELETE_CERT == self.flag:
            if not self.__check_cert_dir() or not self.__check_key_dir() or not self.__check_pass():
                return False
            return True

        if CommonFlag.IMPORT_CRL == self.flag:
            if not self.__check_cert_dir() or not self.__check_crl_files():
                return False
            return True

        if CommonFlag.ERROR == self.flag:
            return False
        return True

    def __base_check(self):
        # base dir
        check_path_flag, err_msg, real_path = FileUtils.regular_file_path(self.base_dir, '/', allow_symlink=True)
        if not check_path_flag:
            logger.error(f"Checking base directory failed, because {err_msg}")
            return False
        self.base_dir = real_path

        # seceasy_encrypt path
        check_path_flag, err_msg, real_path = FileUtils.regular_file_path(self.hseceasy_bin_path, self.base_dir)
        if not check_path_flag:
            logger.error(f"Checking path of binary file named hseceasy failed {err_msg}")
            return False
        self.hseceasy_bin_path = real_path
        check_file_flag, err_msg = FileUtils.is_file_valid(self.hseceasy_bin_path, mode=0o500, check_owner=True,
                                                           check_permission=True)
        if not check_file_flag:
            logger.error(f"[MIE00E000216] Checking binary file named hseceasy failed {err_msg}")
            return False

        # gen_cert path
        check_path_flag, err_msg, real_path = FileUtils.regular_file_path(self.gen_cert_bin_path, self.base_dir)
        if not check_path_flag:
            logger.error(f"Checking path of binary file named gen_cert failed {err_msg}")
            return False
        self.gen_cert_bin_path = real_path
        check_file_flag, err_msg = FileUtils.is_file_valid(self.gen_cert_bin_path, mode=0o500, check_owner=True,
                                                           check_permission=True)
        if not check_file_flag:
            logger.error(f"Checking binary file named gen_cert failed {err_msg}")
            return False

        return True

    def __check_ca_dir(self):
        # security/ca/
        check_path_flag, err_msg, real_path = FileUtils.regular_file_path(self.ca_dir_path, self.base_dir)
        if not check_path_flag:
            logger.error(f"Checking directory path of CA failed. {err_msg}")
            return False
        self.ca_dir_path = real_path
        check_flag, err_msg = FileUtils.is_dir_valid(self.ca_dir_path, mode=0o700)
        if not check_flag:
            logger.error(f"[MIE00E000211] Checking directory path of CA failed. {err_msg}")
            return False
        return True

    def __check_cert_dir(self):
        # security/certs/
        check_path_flag, err_msg, real_path = FileUtils.regular_file_path(self.certs_dir_path, self.base_dir)
        if not check_path_flag:
            logger.error(f"Checking directory path of cert failed {err_msg}")
            return False
        self.certs_dir_path = real_path
        check_flag, err_msg = FileUtils.is_dir_valid(self.certs_dir_path, mode=0o700)
        if not check_flag:
            logger.error(f"[MIE00E000213] Checking directory path of cert failed: {err_msg}")
            return False
        return True

    def __check_key_dir(self):
        # security/keys/
        check_path_flag, err_msg, real_path = FileUtils.regular_file_path(self.keys_dir_path, self.base_dir)
        if not check_path_flag:
            logger.error(f"Checking directory path of key file failed. {err_msg}")
            return False
        self.keys_dir_path = real_path
        check_flag, err_msg = FileUtils.is_dir_valid(self.keys_dir_path, mode=0o700)
        if not check_flag:
            logger.error(f"[MIE00E000214] Checking directory path of key file failed. {err_msg}")
            return False
        return True

    def __check_pass(self):
        # security/pass/
        check_path_flag, err_msg, real_path = FileUtils.regular_file_path(self.pass_dir_path, self.base_dir)
        if not check_path_flag:
            logger.error(f"Checking directory path of pass file failed. {err_msg}")
            return False
        self.pass_dir_path = real_path
        check_flag, err_msg = FileUtils.is_dir_valid(self.pass_dir_path, mode=0o700)
        if not check_flag:
            logger.error(f"[MIE00E000215] Checking directory path of pass file failed. {err_msg}")
            return False
        # security/pass/key_pwd.txt
        check_flag, err_msg, real_path = FileUtils.regular_file_path(self.key_pwd_path, self.pass_dir_path)
        if not check_flag:
            logger.error(f"Checking directory path of pass file failed. {err_msg}")
            return False
        self.key_pwd_path = real_path
        if FileUtils.check_file_exists(self.key_pwd_path):
            check_flag, err_msg = FileUtils.is_file_valid(self.key_pwd_path, mode=0o400)
            if not check_flag:
                logger.error(f"Checking pass file failed. {err_msg}")
                return False
        return True

    def __check_ca_files(self):
        real_ca_files = []
        for item in self.ca_files:
            check_path_flag, err_msg, real_path = FileUtils.regular_file_path(item, '/')
            if not check_path_flag:
                logger.error(f"Checking path of CA file failed. {err_msg}")
                return False
            check_file_flag, err_msg = FileUtils.is_file_valid(real_path, mode=0o600)
            if not check_file_flag:
                logger.error(f"[MIE00E000212] Checking CA file failed. {err_msg}")
                return False
            real_ca_files.append(real_path)
        self.ca_files = real_ca_files
        return True

    def __check_certs_files(self):
        check_path_flag, err_msg, real_path = FileUtils.regular_file_path(self.cert_file, '/')
        if not check_path_flag:
            logger.error(f"Checking path of cert file failed. {err_msg}")
            return False
        self.cert_file = real_path
        check_file_flag, err_msg = FileUtils.is_file_valid(self.cert_file, mode=0o600)
        if not check_file_flag:
            logger.error(f"Checking cert file failed. {err_msg}")
            return False

        check_path_flag, err_msg, real_path = FileUtils.regular_file_path(self.key_file, '/')
        if not check_path_flag:
            logger.error(f"Checking path of key file failed. {err_msg}")
            return False
        self.key_file = real_path
        check_file_flag, err_msg = FileUtils.is_file_valid(self.key_file, mode=0o400)
        if not check_file_flag:
            logger.error(f"Checking key file failed. {err_msg}")
            return False
        return True

    def __check_crl_files(self):
        check_path_flag, err_msg, real_path = FileUtils.regular_file_path(self.crl_file, '/')
        if not check_path_flag:
            logger.error(f"Checking path of crl file failed. {err_msg}")
            return False
        self.crl_file = real_path
        check_file_flag, err_msg = FileUtils.is_file_valid(self.crl_file, mode=0o600)
        if not check_file_flag:
            logger.error(f"Checking crl file failed {err_msg}")
            return False
        return True
