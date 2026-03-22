#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import os
from utils.file_op import FileOps
from .log_util import logger


class CertStorePathUtil:
    def __init__(self, base_dir: str, business: str):
        logger.info(f"The business is {business}")
        self.business = business
        self.base_dir = base_dir

    @staticmethod
    def get_kmc_ksf_master_file_name():
        return "tools/pmt/master/ksfa"

    @staticmethod
    def get_kmc_ksf_standby_file_name():
        return "tools/pmt/standby/ksfb"

    def get_tls_ca_file_list(self):
        ca_path = os.path.join(self.base_dir, "security", self.business, "ca/")
        logger.info(f"The ca_path is {ca_path}")
        return FileOps.list_file_name(ca_path)

    def get_tls_cert_file_name(self):
        return os.path.join("security", "certs", self.business + "server.pem")

    def get_tls_pk_file_name(self):
        return os.path.join("security", "keys", self.business + "server.key.pem")

    def get_tls_pk_pwd_file_name(self):
        return os.path.join("security", "pass", self.business + "mindie_server_key_pwd.txt")

    def get_tls_crl_file_name(self):
        if len(self.business) == 0:
            return os.path.join("security", "certs", "server_crl.pem")
        return os.path.join("security", "certs", self.business + "server_crl.pem")

