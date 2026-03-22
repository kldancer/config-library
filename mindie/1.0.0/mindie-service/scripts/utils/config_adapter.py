#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from config_upgrade_converters import ConfigConverters
from config_template import VersionMap, Config, CustomerPath


def is_standard(old_version, target_version):
    ret = True
    if old_version == "0.1.0":
        if target_version == "1.0.0" or target_version == "1.1.0":
            ret = False
    return ret


def get_update_path(old_version, target_version):
    if old_version == "0.1.0":
        if target_version == "1.0.0":
            return CustomerPath.V1_RC1ToRC3
        if target_version == "1.1.0":
            return CustomerPath.V1_RC1ToRC4
    return None


class ConfigAdapter:
    def __init__(self, old_config: Config, version):
        self.old_config = old_config
        self.update_to_version = VersionMap[version]
        self.target_version = version

    def update(self):
        target_config = None
        standard = is_standard(self.old_config.version, self.target_version)
        if self.__can_update() and standard:
            target_config = ConfigConverters(self.old_config).standard_transform(VersionMap[self.old_config.version],
                                                                                 self.update_to_version)
        if self.__can_update() and not standard:
            target_config = ConfigConverters(self.old_config).custom_transform(\
                get_update_path(self.old_config.version, self.target_version))
        return target_config
    
    def __is_old_version(self):
        return VersionMap[self.old_config.version].value < self.update_to_version.value

    def __is_in_current_range(self):
        return (self.update_to_version.value - VersionMap[self.old_config.version].value) <= 3

    def __can_update(self):
        if self.__is_old_version() and self.__is_in_current_range():
            return True
        return False