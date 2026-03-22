#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import json
from enum import Enum
from collections import OrderedDict
from template.v1_0_rc1_template import V1_RC1_CONFIG_TEMPLATE
from template.v1_0_rc2_template import V1_RC2_CONFIG_TEMPLATE
from template.v1_0_rc3_template import V1_RC3_CONFIG_TEMPLATE
from template.v1_0_0_template import V1_RC4_CONFIG_TEMPLATE


ConfigTemplateMap = {"0.1.0": V1_RC1_CONFIG_TEMPLATE,
                     "0.2.0": V1_RC2_CONFIG_TEMPLATE,
                     "1.0.0": V1_RC3_CONFIG_TEMPLATE,
                     "1.1.0": V1_RC4_CONFIG_TEMPLATE}


class Version(Enum):
    V1_RC1 = 1
    V1_RC2 = 2
    V1_RC3 = 3
    V1_RC4 = 4


class CustomerPath(Enum):
    V1_RC1ToRC3 = 11,
    V1_RC1ToRC4 = 12


VersionMap = {"0.1.0": Version.V1_RC1,
              "0.2.0": Version.V1_RC2,
              "1.0.0": Version.V1_RC3,
              "1.1.0": Version.V1_RC4}


class Config:
    def __init__(self):
        self.config_path = ""
        self.config_data = {}
        self.version = ""

    def parse_py_config_path(self, config_path):
        self.config_path = config_path
        with open(self.config_path, 'r') as f:
            self.config_data = json.load(f, object_pairs_hook=OrderedDict)
        self.__parse_version()

    def set_config_data(self, config_data):
        self.config_data = config_data
        self.__parse_version()

    def __find_key_in_json(self, json_obj, key):
        not_found = None
        if not isinstance(json_obj, dict):
            return not_found

        if key in json_obj.keys():
            return json_obj[key]

        for value in json_obj.values():
            if isinstance(value, dict):
                result = self.__find_key_in_json(value, key)
                if result is not None:
                    return result
            elif isinstance(value, list):
                result = list(map(self.__find_key_in_json, value, key))
                if len(result) != 0 and None not in result:
                    return result
        return not_found

    def __parse_version(self):
        self.version = self.__find_key_in_json(self.config_data, "Version")
        pre_alloc_blocks = self.__find_key_in_json(self.config_data, "preAllocBlocks")

        if self.version is None and pre_alloc_blocks is not None:
            self.version = '0.1.0'
        elif self.version is None and pre_alloc_blocks is None:
            self.version = "0.2.0"