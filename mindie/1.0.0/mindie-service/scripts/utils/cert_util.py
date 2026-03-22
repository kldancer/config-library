#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import datetime
import json
import os.path
import gc

from OpenSSL import crypto
from .log_util import logger

MIN_RSA_LENGTH = 3072
RSA_SHA_256 = "sha256WithRSAEncryption"
RSA_SHA_512 = "sha512WithRSAEncryption"


def validate_certs_and_keys_modulus(server_crt: crypto.x509, server_key: crypto.x509) -> bool:
    # 格式是否为x509_v3
    cert_pub_key = cert_rsa_key = cert_modulus = key_rsa_key = key_modulus = None
    cert_pub_key = server_crt.get_pubkey()
    cert_rsa_key = cert_pub_key.to_cryptography_key()
    cert_modulus = cert_rsa_key.public_numbers().n

    key_rsa_key = server_key.to_cryptography_key()
    key_modulus = key_rsa_key.public_key().public_numbers().n
    return cert_modulus == key_modulus


def has_expired(cert: crypto.x509) -> bool:
    # 格式是否为x509_v3
    before_time_str = cert.get_notBefore().decode("utf-8")
    before_time = datetime.datetime.strptime(before_time_str, "%Y%m%d%H%M%SZ")
    after_time_str = cert.get_notAfter().decode("utf-8")
    after_time = datetime.datetime.strptime(after_time_str, "%Y%m%d%H%M%SZ")
    current_time = datetime.datetime.utcnow()
    return before_time > current_time or current_time > after_time


def validate_server_certs(server_cert: crypto.x509) -> bool:
    # 1. 格式是否为x509_v3[错误退出]

    if server_cert.get_version() != 2:
        logger.error(f"The cert do not use X509v3")
        return False

    decode_format = 'utf-8'
    check_yes = 'yes'
    tip_info = 'Program exits.'
    # 2. Retrieve the signature algorithm OID [检查签名算法--警告]
    pkey_algorithm = server_cert.get_signature_algorithm()
    pkey_algorithm = pkey_algorithm.decode(decode_format)
    if pkey_algorithm not in [RSA_SHA_256, RSA_SHA_512]:
        logger.warning("Insecure encryption algorithm.")
        sig = input(f"Server cert use insecure encryption algorithm, Do you want to "
                    f"continue with this cert certificate? [y/n]: ")
        if sig.lower() != 'y' and sig.lower() != check_yes:
            logger.info(tip_info)
            return False

    # 3.Check RSA key length [检查RSA密钥算法长度--警告]
    pkey = server_cert.get_pubkey()
    key_algorithm_id = pkey.type()
    if key_algorithm_id == crypto.TYPE_RSA:
        rsa_key = pkey.to_cryptography_key()
        rsa_length = rsa_key.key_size
        if rsa_length < MIN_RSA_LENGTH:
            logger.error(f"Insecure RSA key length, required no less than: {MIN_RSA_LENGTH}")
            return False
    else:
        logger.error("Cert pkey please use a RSA key.")
        return False

    # 4. 不包含 Certificate Signature 和 cRLSign [警告]
    check_key_cert_sign = False
    check_crl_sign = False
    check_ca_true = False
    num_extensions = server_cert.get_extension_count()
    for index in range(num_extensions):
        ext = server_cert.get_extension(index)
        ext_name = ext.get_short_name()
        if ext_name.decode(decode_format) == "basicConstraints":
            ext_content = str(ext)
            if "CA:TRUE" in ext_content.upper():
                check_ca_true = True
        if ext_name.decode(decode_format) == "keyUsage":
            ext_content = str(ext)
            if "Certificate Sign" in ext_content:
                check_key_cert_sign = True
            if "CRL Sign" in ext_content:
                check_crl_sign = True
    if check_key_cert_sign or check_crl_sign or check_ca_true:
        logger.warning(f"The cert is not End Entity cert with check_certificate_sign: {check_key_cert_sign}, \
                    check_crl_sign:"
                    f" {check_crl_sign}, check_ca_true: {check_ca_true}")
        sig = input(f"Server cert is not End Entity cert. Do you want to "
                    f"continue with this cert certificate? [y/n]: ")
        if sig.lower() != 'y' and sig.lower() != check_yes:
            logger.info(tip_info)
            return False

    # 5. Is Expired [错误退出]
    if has_expired(server_cert):
        logger.error("Server cert expired.")
        return False

    return True


class CertUtil:
    @classmethod
    def validate_revoke_list(cls, crl_file_path: str) -> bool:
        try:
            with open(crl_file_path, 'rb') as ca_crl_file:
                ca_crl = crypto.load_crl(crypto.FILETYPE_PEM, ca_crl_file.read())

            next_update_time = ca_crl.to_cryptography().next_update
            current_time = datetime.datetime.utcnow()
            if current_time >= next_update_time:
                logger.error(f"Current time is later than next update time of crl")
                return False

            # crl 列表是否未空
            revoked_certs = ca_crl.get_revoked()
            if not revoked_certs:
                logger.error("Crl list is empty")
                return False
            return True

        except Exception as e:
            # crl 或 crt 格式是否正确
            logger.error(f"Internal Error: {str(e)}, please check crl or crt")
            return False

    @classmethod
    def validate_ca_certs(cls, ca_crt_path: str) -> bool:
        decode_format = 'utf-8'
        check_yes = 'yes'
        tip_info = 'Program exits.'
        # Load the X509 Certificate
        try:
            with open(ca_crt_path, "rb") as ca_crt_file:
                ca_cert = crypto.load_certificate(crypto.FILETYPE_PEM, ca_crt_file.read())

            # 1. 格式是否为x509_v3[错误退出]
            if ca_cert.get_version() != 2:
                logger.error(f"The CA: {os.path.basename(ca_crt_path)} do not use X509v3")
                return False

            # 2. ca flag  digital_signature Certificate Signature cRLSign
            check_ca_flag = False
            check_digital_signature_flag = False
            check_key_cert_sign = False
            check_crl_sign = False
            num_extensions = ca_cert.get_extension_count()
            for index in range(num_extensions):
                ext = ca_cert.get_extension(index)
                ext_name = ext.get_short_name()
                if ext_name.decode(decode_format) == "basicConstraints":
                    ext_content = str(ext)
                    if "CA:TRUE" in ext_content.upper():
                        check_ca_flag = True
                if ext_name.decode(decode_format) == "keyUsage":
                    ext_content = str(ext)
                    if "Digital Signature" in ext_content:
                        check_digital_signature_flag = True
                    if "Certificate Sign" in ext_content:
                        check_key_cert_sign = True
                    if "CRL Sign" in ext_content:
                        check_crl_sign = True

            # Validate Basic Constraints (CA Flag)
            if not check_ca_flag:
                logger.error(f"The cafile {os.path.basename(ca_crt_path)} CA flag is not found in basic constraints.")
                return False

            # Validate Key Usage (Digital Signature)
            if not check_digital_signature_flag:
                logger.error(f"The cafile {os.path.basename(ca_crt_path)} Digital Signature is "
                             f"not found in key usage.")
                return False

            if not check_key_cert_sign:
                logger.error(f"The cafile {os.path.basename(ca_crt_path)} Certificate Sign is "
                             f"not found in key usage.")
                return False

            if not check_crl_sign:
                logger.error(f"The cafile {os.path.basename(ca_crt_path)} cRLSign is not found in key usage.")
                return False

            # Validate Algorithm and Encryption [检查签名算法类型--警告]
            pkey_algorithm = ca_cert.get_signature_algorithm().decode(decode_format)
            if pkey_algorithm not in [RSA_SHA_256, RSA_SHA_512]:
                logger.warning(f"CA {os.path.basename(ca_crt_path)} use Insecure encryption algorithm.")
                sig = input(f"file: {os.path.basename(ca_crt_path)} use insecure encryption algorithm, Do you want to "
                            f"continue with this CA certificate? [y/n]: ")
                if sig.lower() != 'y' and sig.lower() != check_yes:
                    logger.info(tip_info)
                    return False
            # Check RSA key length [检查RSA密钥算法长度--警告]
            pkey = ca_cert.get_pubkey()
            key_algorithm_id = pkey.type()
            if key_algorithm_id == crypto.TYPE_RSA:
                rsa_key = pkey.to_cryptography_key()
                rsa_length = rsa_key.key_size
                if rsa_length < MIN_RSA_LENGTH:
                    logger.error(f"{os.path.basename(ca_crt_path)} Insecure RSA key length,"
                                f" required no less than: {MIN_RSA_LENGTH}")
                    return False

            # Calculate Fingerprint
            fingerprint = ca_cert.digest(pkey_algorithm).decode(decode_format)
            logger.info(f"CA path: {os.path.basename(ca_crt_path)} pKeyAlgorithm: {pkey_algorithm}"
                        f" Fingerprint: {fingerprint}")
            sig = input("Do you want to continue with this CA certificate? [y/n]: ")
            if sig.lower() != 'y' and sig.lower() != check_yes:
                logger.info(tip_info)
                return False

            # Is Expired
            if has_expired(ca_cert):
                logger.error(f"CA path: {os.path.basename(ca_crt_path)} CA cert expired.")
                return False
            return True
        except Exception as e:
            logger.error(f"Internal Error: {str(e)}")
            return False

    @classmethod
    def query_cert_info(cls, crt_path: str) -> json:
        try:
            with open(crt_path, "rb") as file:
                cert_data = file.read()
            cert = crypto.load_certificate(crypto.FILETYPE_PEM, cert_data)
            not_before = cert.get_notBefore().decode('utf-8')
            not_after = cert.get_notAfter().decode('utf-8')
            issuer = cert.get_issuer()
            issuer_msg = f'{issuer.CN}, {issuer.O}, {issuer.OU}, {issuer.L}, {issuer.ST}, {issuer.C}'
            subject = cert.get_subject()
            subject_msg = f'{subject.CN}, {subject.O}, {subject.OU}, {subject.L}, {subject.ST}, {subject.C}'
            serial_number = cert.get_serial_number()
            version = cert.get_version()
            return {
                'Not Before': not_before,
                'Not After': not_after,
                'Issuer': issuer_msg,
                'Subject': subject_msg,
                'Serial Number': serial_number,
                'Version': version
            }
        except Exception as e:
            logger.error(f"Internal Error: {str(e)}")
            return {}

    @classmethod
    def query_crl_info(cls, crl_file_path: str) -> json:
        try:
            with open(crl_file_path, "r") as file:
                crl_data = file.read()
            crl = crypto.load_crl(crypto.FILETYPE_PEM, crl_data)
            # Get revoked certificates
            revoked_certs = crl.get_revoked()
            data = []
            if revoked_certs:
                for cert in revoked_certs:
                    serial_number = cert.get_serial().decode('utf-8')
                    revoked_reason = cert.get_reason()
                    revocation_date = cert.get_rev_date().decode('utf-8')
                    item = {
                        'Serial Number': serial_number,
                        'Revoked Reason': revoked_reason,
                        'Revocation Date': revocation_date
                    }
                    data.append(item)
            else:
                logger.error("No revoked certificates found in the CRL")
            return data
        except Exception as e:
            logger.error(f"Internal Error: {str(e)}")
            return []

    @classmethod
    def validate_cert_and_key(cls, server_crt_path: str, server_key_path: str, plain_text=None):
        server_key = None
        try:
            # Load the X509 Certificate
            with open(server_crt_path, 'rb') as f:
                cert_data = f.read()
                server_cert = crypto.load_certificate(crypto.FILETYPE_PEM, cert_data)
            with open(server_key_path, 'rb') as f:
                key_data = f.read()
                if "ENCRYPTED" not in str(key_data):
                    logger.error(f"Checking server key failed. Because private key is not encrypted.")
                    return False
                if not plain_text:
                    server_key = crypto.load_privatekey(crypto.FILETYPE_PEM, key_data)
                else:
                    server_key = crypto.load_privatekey(crypto.FILETYPE_PEM, key_data, passphrase=plain_text)

            # 校验私钥文件
            if server_key is None:
                logger.error(f"Checking server key failed")
                return False

            # 校验服务证书
            if not validate_server_certs(server_cert):
                logger.error(f"Checking server crt failed")
                return False

            # 校验服务和私钥的文件是否匹配
            if not validate_certs_and_keys_modulus(server_cert, server_key):
                logger.error(f"Checking server crt & server key failed")
                return False
            return True
        except Exception as e:
            # crl 或 crt 格式是否正确
            logger.error(f"Internal Error: {str(e)}")
            return False

    @classmethod
    def validate_ca_crl(cls, ca_path: str, crl_path: str):
        try:
            with open(crl_path, 'rb') as crl_path_file:
                ca_crl = crypto.load_crl(crypto.FILETYPE_PEM, crl_path_file.read())
            with open(ca_path, "rb") as ca_crt_file:
                ca_cert = crypto.load_certificate(crypto.FILETYPE_PEM, ca_crt_file.read())

            ca_pub_key = ca_cert.get_pubkey().to_cryptography_key()
            crl_crypto = ca_crl.to_cryptography()
            valid_signature = crl_crypto.is_signature_valid(ca_pub_key)
            if valid_signature:
                return True
            else:
                logger.error(f'CRL {os.path.basename(crl_path)} is not valid for CA {os.path.basename(ca_path)}')
                return False
        except Exception as e:
            # crl 或 ca 格式是否正确
            logger.error(f"Internal Error: {str(e)}")
            return False
