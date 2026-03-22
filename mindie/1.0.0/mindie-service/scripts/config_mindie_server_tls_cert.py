#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import argparse
import getpass
import os
import gc
import sys
import ctypes

import json
import subprocess
import shutil
import psutil

from utils.hseceasy_util import HseceasyUtil
from utils.cert_util import CertUtil
from utils.file_op import FileOps
from utils.cert_store_path_util import CertStorePathUtil
from utils.parameter import CommonFlag, Parameter
from utils.log_util import logger
from utils.file_util import FileUtils


def check_arguments(args: argparse.Namespace) -> CommonFlag:
    """
    Check if the command line arguments is valid.

    Args:
        args (argparse.Namespace): Command line arguments that input by user.

    Returns:
        CommonFlag: Status code, to show operation result.
    """
    if not args.project_path:
        logger.error(f'[MIE00E000200] The mindie server install path required.\n')
        return CommonFlag.ERROR
    if args.command == CommonFlag.GEN_CERT.value:
        if not args.conf_file:
            logger.error(f'[MIE00E000201] The conf_file required.\n')
        return CommonFlag.GEN_CERT
    if args.command == CommonFlag.IMPORT_CA.value:
        if len(args.ca_files) > 5:
            logger.error(f'[MIE00E000202] Import CA files required no more than 5.\n')
            return CommonFlag.ERROR
        return CommonFlag.IMPORT_CA
    if args.command == CommonFlag.DELETE_CA.value:
        if len(args.ca_files) > 5:
            logger.error(f'[MIE00E000203] Delete CA files required no more than 5.\n')
            return CommonFlag.ERROR
        return CommonFlag.DELETE_CA
    if args.command == CommonFlag.IMPORT_CERT.value:
        if not args.cert_file or not args.key_file:
            logger.error(f'[MIE00E000204] The cert_file and key_file required.\n')
        return CommonFlag.IMPORT_CERT
    if args.command == CommonFlag.DELETE_CERT.value:
        return CommonFlag.DELETE_CERT
    if args.command == CommonFlag.IMPORT_CRL.value:
        if not args.crl_file:
            logger.error(f'[MIE00E000205] The crl_file is required.\n')
        return CommonFlag.IMPORT_CRL
    if args.command == CommonFlag.QUERY_CERT.value:
        return CommonFlag.QUERY_CERT
    if args.command == CommonFlag.RESTORE_CA.value:
        return CommonFlag.RESTORE_CA
    if args.command == CommonFlag.RESTORE_CERT.value:
        return CommonFlag.RESTORE_CERT
    return CommonFlag.ERROR


def parse_arguments():
    """
    Parses command-line arguments for configuring TLS on a local node.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    def check_str_len(input_str):
        if not isinstance(input_str, str):
            input_str = str(input_str)
        if len(input_str) > 1024:
            raise ValueError(f"The length of input string should be less than 1024, but got {len(input_str)}")
        return input_str

    # 获取工程路径
    parser = argparse.ArgumentParser(description='configure TLS on local node.')

    # 获取业务名称
    parser.add_argument('--business', type=check_str_len, default="", help='the business need tls')

    parser.add_argument('project_path', type=check_str_len, default=None, help='the project install path')

    subparsers = parser.add_subparsers(dest='command')

    # gen_cert command
    gen_cert_parser = subparsers.add_parser(CommonFlag.GEN_CERT.value)
    gen_cert_parser.add_argument('conf_file', type=check_str_len, help='config file for generating cert')
    gen_cert_parser.add_argument('--ip', type=check_str_len, default='', help='ip address list')

    # import_ca command
    import_ca_parser = subparsers.add_parser(CommonFlag.IMPORT_CA.value)
    import_ca_parser.add_argument('ca_files', type=check_str_len, nargs='+', help='list of CA files')

    # delete_ca command
    delete_ca_parser = subparsers.add_parser(CommonFlag.DELETE_CA.value)
    delete_ca_parser.add_argument('ca_files', type=check_str_len, nargs='+', help='list of CA files')

    # import_certs command
    import_certs_parser = subparsers.add_parser(CommonFlag.IMPORT_CERT.value)
    import_certs_parser.add_argument('cert_file', type=check_str_len, help='file containing certificates')
    import_certs_parser.add_argument('key_file', type=check_str_len, help='file containing private key')

    # delete_cert command
    delete_certs_parser = subparsers.add_parser(CommonFlag.DELETE_CERT.value)
    delete_certs_parser.add_argument('--cert_file', type=check_str_len, help='file containing certificate')
    delete_certs_parser.add_argument('--key_file', type=check_str_len, default='server.key.pem',
        help='file containing private key')

    # import_crl command
    import_crl_parser = subparsers.add_parser(CommonFlag.IMPORT_CRL.value)
    import_crl_parser.add_argument('crl_file', type=check_str_len, help='file containing certificate revocation list')

    # query cert info command
    query_parser = subparsers.add_parser(CommonFlag.QUERY_CERT.value)
    query_parser.add_argument('--cert_file', type=check_str_len, help='file containing certificate')
    query_parser.add_argument('--crl_file', type=check_str_len, default='server_crl.pem',
        help='file containing certificate revocation list')

    # restore ca info command
    restore_ca_parser = subparsers.add_parser(CommonFlag.RESTORE_CA.value)
    restore_ca_parser.add_argument('--ca_backup_file', type=check_str_len,
                                   help='file containing backup CA certificate')
    restore_ca_parser.add_argument('--ca_dst_file', type=check_str_len,
                                   help='file containing restore CA certificate')

    # restore ca info command
    restore_cert_parser = subparsers.add_parser(CommonFlag.RESTORE_CERT.value)
    restore_cert_parser.add_argument('--cert_backup_file', type=check_str_len,
                                     help='file containing backup cert certificate')
    restore_cert_parser.add_argument('--cert_dst_file', type=check_str_len,
                                     help='file containing restore cert certificate')
    restore_cert_parser.add_argument('--key_backup_file', type=check_str_len,
                                     help='file containing backup cert key')
    restore_cert_parser.add_argument('--key_dst_file', type=check_str_len,
                                     help='file containing restore cert key')

    args = parser.parse_args()
    return args


def check_gen_cert_file(file_path, mode):
    check_path_flag, err_msg, real_path = FileUtils.regular_file_path(file_path)
    if not check_path_flag:
        logger.error(f"[MIE00E000206] gen_cert path check failed {err_msg}")
        return False
    check_file_flag, err_msg = FileUtils.is_file_valid(real_path, mode=mode)
    if not check_file_flag:
        logger.error(f"[MIE00E000207] gen_cert file exist check failed {err_msg}")
        return False
    return True


def handler_gen_cert(config: Parameter) -> bool:
    """
    Handles the generation of certificates based on the provided configuration.

    Args:
        config (Parameter): An object containing configuration details, including paths to required files,
                            base directory, and optional IP address.

    Returns:
        bool: True if certificate generation is successful, False otherwise.
    """
    if not check_gen_cert_file(config.conf_file, 0o640):
        return False

    # load gen_cert conf
    try:
        with open(config.conf_file, 'r') as reader:
            data = json.load(reader)
            ca_cert_path = data['ca_cert']
            ca_key_path = data['ca_key']
            ca_key_pwd_path = data['ca_key_pwd']
            cert_config_path = data['cert_config']
            kmc_master_path = data['kmc_ksf_master']
            kmc_standby_path = data['kmc_ksf_standby']
    except Exception as e:
        logger.error(f'An error occurred when loading conf file : {str(e)}')
        return False

    if not check_gen_cert_file(ca_cert_path, 0o400):
        return False

    if not check_gen_cert_file(ca_key_path, 0o400):
        return False

    if not check_gen_cert_file(ca_key_pwd_path, 0o400):
        return False

    if not check_gen_cert_file(cert_config_path, 0o640):
        return False

    if not check_gen_cert_file(kmc_master_path, 0o600):
        return False

    if not check_gen_cert_file(kmc_standby_path, 0o600):
        return False

    if not CertUtil.validate_ca_certs(ca_cert_path):
        logger.error("CA cert validate failed")
        return False

    if config.ip is None:
        config.ip = 'None'

    try:
        child = subprocess.Popen(
            ['bin/gen_cert', config.conf_file, config.ip],
            cwd=config.base_dir, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, stdin=subprocess.PIPE, universal_newlines=True
        )

        stdout, stderr = child.communicate(timeout=60)
        code = child.returncode
        if code != 0:
            logger.error(f'Error: generate cert failed: {code}\n')
            logger.error(f'Log output: {stdout}\n')
            logger.error(f'Error output: {stderr}\n')
            return False
    except subprocess.TimeoutExpired:
        logger.error(f'Error: Subprocess timed out.')
        child.kill()
        return False
    except Exception as e:
        logger.error(f'An error occurred when generate cert : {str(e)}')
        return False
    logger.info("Generate cert successful!")
    return True


def handler_import_ca(config: Parameter) -> bool:
    """
    Handles the import or update of CA (Certificate Authority) certificates.

    Args:
        config (Parameter): Configuration object containing CA file paths,
                            the target CA directory path, and other settings.

    Returns:
        bool: True if the import or update succeeds, False otherwise.
    """
    new_files = set()
    try:
        for file_path in config.ca_files:
            file_name = os.path.basename(file_path)
            new_files.add(file_name)

        # CA 文件夹下的文件
        dir_list = list(FileOps.list_file_name(config.ca_dir_path))
        backup_folder = os.path.join(config.ca_dir_path, 'backup')

        # 并集为导入后总文件
        union_set = new_files.union(dir_list)
        if len(union_set) > 20:
            logger.error("CA cert files required no more than 20.")
            return False
        # 证书校验
        for item in config.ca_files:
            if not CertUtil.validate_ca_certs(item):
                logger.error("[MIE00E00020F] CA cert validate failed")
                return False
        # 交集为已导入的ca文件
        intersection = set(new_files).intersection(set(dir_list))
        logger.info(f"Intersection is {intersection}")
        # 拷贝覆盖部分确认
        logger.info(f"Repeat CA list {intersection}")
        if len(intersection) > 0:
            sig = input("Do you want to continue with override this CA file? [y/n]: ")
            if sig.lower() != 'y' and sig.lower() != 'yes':
                logger.error("Program exits.")
                logger.error("[MIE00E00020E] import CA failed by Program exits.")
                return False

        if len(dir_list) == 0:
            # 首次导入，直接对导入的证书做备份
            logger.info(f"There is no former certificate, backup directly.")
            os.makedirs(backup_folder, exist_ok=True)
            os.chmod(backup_folder, 0o700)
            for file_path in config.ca_files:
                FileOps.copy_file(file_path, backup_folder)
                logger.info(f"New import certificate [{file_path}] is directly backup in [{backup_folder}].")
        else:
            # 非首次导入，backup 文件夹清空，会把当前已导入的所有 CA 证书做备份
            logger.info(f"There is former certificate {dir_list} in [{config.ca_dir_path}].")
            logger.info(f"Delete old backup certificate and backup former certificate")
            check_flag, err_msg, real_backup_folder = FileUtils.regular_file_path(backup_folder,
                                                                                  base_dir="/",
                                                                                  allow_symlink=False)
            if not check_flag:
                logger.error(f"[MIE00E000208] You have to create backup folder")
                raise ValueError(err_msg)
            old_backup_list = list(FileOps.list_file_name(real_backup_folder))
            for old_backup_file in old_backup_list:
                logger.info(f"Delete old backup certificate [{old_backup_file}] in [{real_backup_folder}].")
                remove_file = os.path.join(f"{real_backup_folder}", f"{old_backup_file}")
                FileOps.delete(remove_file)
            for former_ca_item in dir_list:
                former_ca_path = os.path.join(f"{config.ca_dir_path}", f"{former_ca_item}")
                FileOps.copy_file(former_ca_path, backup_folder)
                logger.info(f"Former certificate [{former_ca_path}] is backup in [{backup_folder}]")

        # 拷贝/覆盖 文件
        for item in config.ca_files:
            FileOps.copy_file(item, config.ca_dir_path)

    except ValueError as e:
        logger.error(f"Import or update CA failed by {str(e)}")
        return False
    logger.info("Import or update CA successful!")
    return True


def handler_delete_ca(config: Parameter) -> bool:
    """
    Handles the deletion of specified CA (Certificate Authority) certificates.

    Args:
        config (Parameter): Configuration object containing:
            - base_dir: Base directory for certificate operations.
            - business: Business name associated with the certificates.
            - ca_files: List of CA certificate file paths to delete.
            - ca_dir_path: Directory containing CA certificates.
            - certs_dir_path: Directory containing CRL files (optional).

    Returns:
        bool: True if the deletion succeeds, False otherwise.
    """
    new_files = set()
    try:
        cert_store_path_util = CertStorePathUtil(config.base_dir, config.business)
        for file_path in config.ca_files:
            file_name = os.path.basename(file_path)
            new_files.add(file_name)
        old_files = list(cert_store_path_util.get_tls_ca_file_list())
        intersection = new_files.intersection(old_files)
        if len(intersection) != len(new_files):
            logger.error(f"[MIE00E000209] Some input files don't exist: {new_files - intersection}")
            return False

        # 如果吊销列表存在一起删除
        delete_crl = False

        # 删除确认
        sig = input("Do you want to continue with delete this CA files? [y/n]: ")
        if sig.lower() != 'y' and sig.lower() != 'yes':
            logger.error("Delete CA failed by Program exits.")
            logger.error("Program exits.")
            return False

        # 删除文件
        for item in new_files:
            file_path = os.path.join(config.ca_dir_path, item)
            FileOps.delete(file_path)
        if delete_crl:
            FileOps.delete(os.path.join(config.certs_dir_path, cert_store_path_util.get_tls_crl_file_name()))

    except ValueError as e:
        logger.error(f"Delete CA file failed by {str(e)}")
        return False
    logger.info("Delete CA file successful!")
    return True


def handler_import_cert(config: Parameter) -> bool:
    """
    Handles the import or update of a certificate and private key, including password verification,
    file validation, and backup management.

    Args:
        config (Parameter): Configuration object containing:
            - cert_file: Path to the certificate file to import.
            - key_file: Path to the private key file to import.
            - base_dir: Base directory for operations.
            - business: Associated business identifier.
            - certs_dir_path: Directory where certificates are stored.
            - keys_dir_path: Directory where private keys are stored.
            - key_pwd_path: Path for storing encrypted private key passwords.

    Returns:
        bool: True if the import or update operation succeeds, False otherwise.
    """
    password = ""
    try:
        # 输入口令
        password = getpass.getpass("Password for private key file: ")
        password_again = getpass.getpass("Retype password for private key file: ")

        if password != password_again:
            logger.error(f'Error: passwords do not match.\n')
            return False

        if not HseceasyUtil.check_password(password):
            logger.error("[MIE00E00020B] Check password for key file failed")
            return False

        # 校验服务证书
        if not CertUtil.validate_cert_and_key(config.cert_file, config.key_file, plain_text=bytes(password, 'utf-8')):
            return False
        cert_store_path_util = CertStorePathUtil(config.base_dir, config.business)
        # 获取配置文件中的文件和文件名
        cert_file_name = os.path.basename(config.cert_file)
        key_file_name = os.path.basename(config.key_file)
        certs_files_list = list(FileOps.list_file_name(config.certs_dir_path))
        key_files_list = list(FileOps.list_file_name(config.keys_dir_path))
        check_flag = cert_file_name in certs_files_list and key_file_name in key_files_list
        logger.info(f"Intersection overwrite: {check_flag}")
        certs_backup_folder = os.path.join(config.certs_dir_path, 'backup')
        keys_backup_folder = os.path.join(config.keys_dir_path, 'backup')

        # 是否覆盖文件
        if check_flag:
            sig = input("Do you want to continue with override this cert file and key file? [y/n]: ")
            if sig.lower() != 'y' and sig.lower() != 'yes':
                logger.error("[MIE00E00020C] Import cert failed. exit.")
                return False

        if (len(certs_files_list) == 0) and (len(key_files_list) == 0):
            # 首次导入，直接对导入的证书做备份
            logger.info(f"There is no former certificate, backup directly.")
            os.makedirs(certs_backup_folder, exist_ok=True)
            os.makedirs(keys_backup_folder, exist_ok=True)
            os.chmod(certs_backup_folder, 0o700)
            os.chmod(keys_backup_folder, 0o700)
            cert_file = os.path.join(f"{config.certs_dir_path}", f"{cert_file_name}")
            key_file = os.path.join(f"{config.keys_dir_path}", f"{key_file_name}")
            FileOps.copy_file(config.cert_file, certs_backup_folder)
            FileOps.copy_file(config.key_file, keys_backup_folder)
        elif (len(certs_files_list) != 0) and (len(key_files_list) != 0):
            # 非首次导入，backup 文件夹清空，会把当前已导入的所有 CERT 证书做备份
            logger.info(f"There is former certificate {certs_files_list} in [{config.certs_dir_path}].")
            logger.info(f"There is former keys {key_files_list} in [{config.keys_dir_path}].")
            logger.info(f"Delete old backup certificate and backup former certificate")
            check_flag, err_msg, real_certs_backup_folder = FileUtils.regular_file_path(certs_backup_folder,
                                                                                        base_dir="/",
                                                                                        allow_symlink=False)
            if not check_flag:
                logger.error(f"You have to create backup folder")
                raise ValueError(err_msg)
            check_flag, err_msg, real_keys_backup_folder = FileUtils.regular_file_path(keys_backup_folder,
                                                                                        base_dir="/",
                                                                                        allow_symlink=False)
            if not check_flag:
                logger.error(f"You have to create backup folder")
                raise ValueError(err_msg)

            old_certs_backup_list = list(FileOps.list_file_name(real_certs_backup_folder))
            old_keys_backup_list = list(FileOps.list_file_name(real_keys_backup_folder))

            for old_backup_file in old_certs_backup_list:
                logger.info(f"Delete old backup certificate [{old_backup_file}] in [{real_certs_backup_folder}].")
                remove_file = os.path.join(f"{real_certs_backup_folder}", f"{old_backup_file}")
                FileOps.delete(remove_file)

            for old_backup_file in old_keys_backup_list:
                logger.info(f"Delete old backup keys [{old_backup_file}] in [{real_keys_backup_folder}].")
                remove_file = os.path.join(f"{real_keys_backup_folder}", f"{old_backup_file}")
                FileOps.delete(remove_file)

            for former_cert_item in certs_files_list:
                former_cert_path = os.path.join(f"{config.certs_dir_path}", f"{former_cert_item}")
                FileOps.copy_file(former_cert_path, real_certs_backup_folder)
                logger.info(f"Former certificate [{former_cert_path}] is backup in [{real_certs_backup_folder}]")

            for former_key_item in key_files_list:
                former_key_path = os.path.join(f"{config.keys_dir_path}", f"{former_key_item}")
                FileOps.copy_file(former_key_path, real_keys_backup_folder)
                logger.info(f"Former certificate [{former_key_path}] is backup in [{real_keys_backup_folder}]")

        # 拷贝文件
        FileOps.copy_file(config.cert_file, config.certs_dir_path)
        FileOps.copy_file(config.key_file, config.keys_dir_path)

        # 生成加密文件
        if not HseceasyUtil.store_key_pass_file(config.base_dir, config.key_pwd_path, password):
            return False
    except ValueError as e:
        logger.error(f"Import or update cert failed by {str(e)}")
        return False
    finally:
        password_len = len(password)
        password_offset = sys.getsizeof(password) - password_len - 1
        ctypes.memset(id(password) + password_offset, 0, password_len)

        password_len = len(password_again)
        password_offset = sys.getsizeof(password_again) - password_len - 1
        ctypes.memset(id(password_again) + password_offset, 0, password_len)

    logger.info("Import or update cert successful!")
    return True


def handler_delete_cert(config: Parameter) -> bool:
    """
    Deletes a certificate and its associated files after verifying their existence
    and obtaining user confirmation.

    Parameters:
        config (Parameter): The configuration object containing details about the
                            certificate and its file paths.

    Returns:
        bool: True if the certificate and associated files are successfully deleted;
              False otherwise.
    """
    try:
        cert_store_path_util = CertStorePathUtil(config.base_dir, config.business)
        cert_file_name = os.path.basename(cert_store_path_util.get_tls_cert_file_name())
        key_file_name = os.path.basename(cert_store_path_util.get_tls_pk_file_name())
        cert_file_name = config.cert_file
        key_file_name = config.key_file
        certs_files_name = FileOps.list_file_name(config.certs_dir_path)
        key_files_name = FileOps.list_file_name(config.keys_dir_path)
        check_flag = cert_file_name in certs_files_name and key_file_name in key_files_name

        # 判断服务证书是否已存在
        if not check_flag:
            logger.error("Cert not exit, can not delete")
            return False

        # 删除确认
        sig = input("Do you want to continue with delete certs files? [y/n]: ")
        if sig.lower() != 'y' and sig.lower() != 'yes':
            logger.error("Delete cert failed by Program exits.")
            logger.error("Program exits.")
            return False

        cert_file_path = os.path.join(config.certs_dir_path, cert_file_name)
        key_file_path = os.path.join(config.keys_dir_path, key_file_name)

        if not HseceasyUtil.check_secret_file(key_file_path):
            logger.error("Import cert failed by clear secret file failed.")
            return False
        if not HseceasyUtil.check_secret_file(config.key_pwd_path):
            logger.error("Import cert failed by clear secret file failed.")
            return False
        HseceasyUtil.clear_secret_file(key_file_path)
        HseceasyUtil.clear_secret_file(config.key_pwd_path)

        # 删除文件
        FileOps.delete(cert_file_path)
        FileOps.delete(key_file_path)
        FileOps.delete(config.key_pwd_path)

    except ValueError as e:
        logger.error(f"Delete cert failed by {str(e)}")
        return False
    logger.info("Delete cert successful!")
    return True


def handler_import_crl(config: Parameter) -> bool:
    """
    Imports a Certificate Revocation List (CRL) file after validating it
    and checking for any conflicts with existing files.

    Parameters:
        config (Parameter): The configuration object containing details about the CRL file and its paths.

    Returns:
        bool: True if the CRL file is successfully imported; False otherwise.
    """
    try:
        cert_store_path_util = CertStorePathUtil(config.base_dir, config.business)
        if not CertUtil.validate_revoke_list(config.crl_file):
            return False

        # 获取ca[0]
        ca_files = list(cert_store_path_util.get_tls_ca_file_list())
        if len(ca_files) > 0:
            ca_crl_crt = ca_files[0]
            if not CertUtil.validate_ca_crl(os.path.join(config.ca_dir_path, ca_crl_crt), config.crl_file):
                logger.info(f"Import or update crl file failed")
                return False

        crl_file_config_name = os.path.basename(cert_store_path_util.get_tls_crl_file_name())
        crl_dir_files_name = FileOps.list_file_name(config.certs_dir_path)
        check_flag = crl_file_config_name in crl_dir_files_name
        if check_flag:
            sig = input("Do you want to continue with override this crl file? [y/n]: ")
            if sig.lower() != 'y' and sig.lower() != 'yes':
                logger.error("Import crl failed by Program exits.")
                logger.error("Program exits.")
                return False
            FileOps.delete(os.path.join(config.certs_dir_path, crl_file_config_name))
        FileOps.copy_file(config.crl_file, config.certs_dir_path)
    except ValueError as e:
        logger.error(f"Import crl failed by {str(e)}")
        return False
    logger.info("Import crl successful!")
    return True


def handler_query(config: Parameter) -> bool:
    """
    Queries and logs information about the CA certificates, service certificate,
    and CRL (Certificate Revocation List) files.

    Parameters:
        config (Parameter): The configuration object containing paths for certificates and CRL files.

    Returns:
        bool: True if the query is successful; False otherwise.
    """
    try:
        cert_store_path_util = CertStorePathUtil(config.base_dir, config.business)
        ca_file_name_list = cert_store_path_util.get_tls_ca_file_list()
        ca_dir_file_name_list = FileOps.list_file_name(config.ca_dir_path)
        for item in ca_file_name_list:
            if item in ca_dir_file_name_list:
                ca_info = CertUtil.query_cert_info(os.path.join(config.ca_dir_path, item))
                logger.info('-' * 100)
                logger.info(f"{item} CA info : \n {ca_info}.")
                logger.info('-' * 100)
            else:
                logger.info('-' * 100)
                logger.info(f"{item} CA info : file not exit.")
                logger.info('-' * 100)
        # 查询服务证书详情
        cert_file_name = os.path.basename(cert_store_path_util.get_tls_cert_file_name())
        cert_file_name = config.cert_file
        cert_dir_file_name_list = FileOps.list_file_name(config.certs_dir_path)
        logger.info(f"cert_dir_file_name_list is {cert_dir_file_name_list}")
        if cert_file_name in cert_dir_file_name_list:
            cert_info = CertUtil.query_cert_info(os.path.join(config.certs_dir_path, cert_file_name))
            logger.info('-' * 100)
            logger.info(f"{cert_file_name} cert info : \n {cert_info}.")
            logger.info('-' * 100)
        else:
            logger.info('-' * 100)
            logger.info(f"{cert_file_name} cert info : file not exit.")
            logger.info('-' * 100)
        # 查询吊销列表详情
        crl_file_name = os.path.basename(cert_store_path_util.get_tls_crl_file_name())
        crl_file_name = config.crl_file
        crl_dir_file_name_list = FileOps.list_file_name(config.crl_dir_path)
        if crl_file_name in crl_dir_file_name_list:
            crl_info = CertUtil.query_crl_info(os.path.join(config.certs_dir_path, crl_file_name))
            logger.info('-' * 100)
            logger.info(f"{crl_file_name} crl info : \n {crl_info}.")
            logger.info('-' * 100)
        else:
            logger.info('-' * 100)
            logger.info(f"{crl_file_name} crl info : file not exit.")
            logger.info('-' * 100)
    except ValueError as e:
        logger.error(f"Query cert and crl info failed by {str(e)}")
        return False
    return True


def handler_restore_ca(config: Parameter) -> bool:
    """
    Restores a CA (Certificate Authority) backup file to its destination
    and imports the restored CA file.

    Parameters:
        config (Parameter): The configuration object containing the paths
        for the CA backup file and destination file.

    Returns:
        bool: True if the restoration and import are successful; False otherwise.
    """
    try:
        logger.info(f"Backup Certificate [{config.ca_backup_file}] is restore to {config.ca_dst_file}")
        shutil.copy(config.ca_backup_file, config.ca_dst_file)
        config.ca_files = [config.ca_dst_file]
        return handler_import_ca(config)
    except ValueError as e:
        logger.error(f"Restore CA failed by {str(e)}")
        return False
    return True


def handler_restore_cert(config: Parameter) -> bool:
    """
    Restores a certificate and its associated key from backup files to their destination
    and imports the restored certificate.

    Parameters:
        config (Parameter): The configuration object containing the paths for the certificate
        and key backup files and their destination files.

    Returns:
        bool: True if the restoration and import are successful; False otherwise.
    """
    try:
        logger.info(f"Backup Certificate [{config.cert_backup_file}] is restore to {config.cert_dst_file}")
        logger.info(f"Backup Key [{config.key_backup_file}] is restore to {config.key_dst_file}")
        shutil.copy(config.cert_backup_file, config.cert_dst_file)
        shutil.copy(config.key_backup_file, config.key_dst_file)
        config.cert_file = config.cert_dst_file
        config.key_file = config.key_dst_file
        return handler_import_cert(config)
    except ValueError as e:
        logger.error(f"Restore cert failed by {str(e)}")
        return False
    return True


def switch(config: Parameter, flag: CommonFlag) -> bool:
    if CommonFlag.GEN_CERT == flag:
        return handler_gen_cert(config)
    if CommonFlag.IMPORT_CA == flag:
        return handler_import_ca(config)
    if CommonFlag.DELETE_CA == flag:
        return handler_delete_ca(config)
    if CommonFlag.IMPORT_CERT == flag:
        return handler_import_cert(config)
    if CommonFlag.DELETE_CERT == flag:
        return handler_delete_cert(config)
    if CommonFlag.IMPORT_CRL == flag:
        return handler_import_crl(config)
    if CommonFlag.QUERY_CERT == flag:
        return handler_query(config)
    if CommonFlag.RESTORE_CA == flag:
        return handler_restore_ca(config)
    if CommonFlag.RESTORE_CERT == flag:
        return handler_restore_cert(config)
    return False


def main_process() -> bool:
    args = parse_arguments()
    common_flag = check_arguments(args)
    if CommonFlag.ERROR == common_flag:
        return False
    config = Parameter(args, common_flag)
    if not config.check_path():
        logger.error("Config path check failed")
        return False
    return switch(config, common_flag)

if __name__ == "__main__":
    p = psutil.Process(os.getpid())
    username = p.username()
    terminal = p.terminal()
    message = f"[{username}] [{terminal}] Operator result: {main_process()}"
    logger.info(message)
