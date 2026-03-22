#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import json
import os
from config_template import ConfigTemplateMap, Version, Config, CustomerPath

MODEL_DEPLOY_PARAM_KEY = "ModelDeployParam"
MODEL_DEPLOY_CONFIG_KEY = "ModelDeployConfig"
MODEL_PARAM_KEY = "ModelParam"
MODEL_CONFIG_KEY = "ModelConfig"
PLUGIN_PARAMS_KEY = "pluginParams"
PLUGIN_PARAMS_VALUE = ""
TRUSTREMOTECODE_KEY = "trustRemoteCode"
TRUSTREMOTECODE_VALUE = False
TLSCRL = "tlsCrl"
TLSCRL_PATH = "tlsCrlPath"
TLSCRL_FILES = "tlsCrlFiles"

MANAGEMENT_TLSCRL = "managementTlsCrl"
MANAGEMENT_TLSCRL_FILES = "managementTlsCrlFiles"
MANAGEMENT_TLSCRL_PATH = "managementTlsCrlPath"

INTERCOMMON_TLSCA = "interCommTlsCaFile"
INTERCOMMON_TLSCA_FILES = "interCommTlsCaFiles"
INTERCOMMON_TLSCA_PATH = "interCommTlsCaPath"

INTERCOMMON_TLSCRL = "interCommTlsCrl"
INTERCOMMON_TLSCRL_FILES = "interCommTlsCrlFiles"
INTERCOMMON_TLSCRL_PATH = "interCommTlsCrlPath"

INTERNODE_TLSCA = "interNodeTlsCaFile"
INTERNODE_TLSCA_FILES = "interNodeTlsCaFiles"
INTERNODE_TLSCA_PATH = "interNodeTlsCaPath"

INTERNODE_TLSCRL = "interNodeTlsCrl"
INTERNODE_TLSCRL_FILE = "interNodeTlsCrlFile"
INTERNODE_TLSCRL_FILES = "interNodeTlsCrlFiles"
INTERNODE_TLSCRL_PATH = "interNodeTlsCrlPath"

V1_RC2_VERSION = "0.2.0"
V1_RC3_VERSION = "1.0.0"
V1_RC4_VERSION = "1.1.0"
OTHER_PARAM_KEY = "OtherParam"
LOG_PARAM_KEY = "LogParam"
LOG_CONFIG_KEY = "LogConfig"
SERVER_PARAM_KEY = "ServeParam"
SERVER_CONFIG_KEY = "ServerConfig"
BACKEND_CONFIG_KEY = "BackendConfig"
BACKEND_NAME_KEY = "backendName"
ENGINE_NAME_KEY = "engineName"
WORK_FLOW_PARAM_KEY = "WorkFlowParam"
TEMPLATE_PARAM_KEY = "TemplateParam"
SCHEDULE_CONFIG_KEY = "ScheduleConfig"
MULTI_NODES_INFER_PORT_KEY = "multiNodesInferPort"
TEMPLATE_NAME_KEY = "templateName"
TEMPLATE_NAME_VALUE = "Standard_LLM"


class Converter:
    def __init__(self):
        pass

    def convert(self, json_config):
        pass


class ConfigConverters:
    def __init__(self, json_config: Config):
        self.standard_converters = {}
        self.customization_converters = {}
        self.json_config = json_config
        self.target_config_data = json_config.config_data

        self.__register_standard_converter(Version.V1_RC1, V1RC1ToRC2Convert())
        self.__register_standard_converter(Version.V1_RC2, V1RC2ToRC3Convert())
        self.__register_standard_converter(Version.V1_RC3, V1RC3ToRC4Convert())
        self.__register_customization_converter(CustomerPath.V1_RC1ToRC3, V1RC1ToRC3Convert())
        self.__register_customization_converter(CustomerPath.V1_RC1ToRC4, V1RC1ToRC4Convert())

    def standard_transform(self, start_version: Version, end_version: Version):
        start = start_version.value
        end = end_version.value

        for version in range(start, end):
            if version in self.standard_converters:
                self.target_config_data = self.standard_converters[version].convert(self.target_config_data)
            else:
                self.target_config_data = None
        return self.target_config_data

    def custom_transform(self, version_path: CustomerPath):
        self.target_config_data = self.customization_converters[version_path.value].convert(self.target_config_data)
        return self.target_config_data

    def __register_standard_converter(self, version: Version, converter: Converter):
        self.standard_converters[version.value] = converter

    def __register_customization_converter(self, version: Version, converter: Converter):
        self.customization_converters[version.value] = converter


def leaf_node_value_convert(source_config, target_config):
    for key, value in target_config.items():
        if isinstance(value, dict) and key in source_config:
            leaf_node_value_convert(source_config[key], target_config[key])
        else:
            if key in source_config:
                target_config[key] = source_config[key]


def list_ele_add_param(obj_list: list, add_key, add_value):
    for ele in obj_list:
        ele[add_key] = add_value


def list_ele_delete_param(obj_list: list, add_key):
    for ele in obj_list:
        if add_key in ele:
            del ele[add_key]


def __path_parse(key: str):
    ret = key.split('-')
    return ret


def __set_config_val(target_config, path, val):
    if len(path) != 1:
        __set_config_val(target_config[path[0]], path[1:], val)
    else:
        target_config[path[0]] = val
        return


def __set_config_dict_val(target_config, path, val):
    if len(path) != 1:
        __set_config_dict_val(target_config[path[0]], path[1:], val)
    else:
        leaf_node_value_convert(val, target_config[path[0]])
        return


def __get_config_val(source_config, path):
    curr = source_config
    for node in path:
        curr = curr[node]
    return curr  


def node_mapping(source_config, target_config, mapping_data: dict):
    for ele in mapping_data:
        src_dist = __src_dist_parse(ele)
        __set_config_val(target_config, __path_parse(src_dist[1]),
            __get_config_val(source_config, __path_parse(src_dist[0])))


def __src_dist_parse(key):
    ret = key.split('_')
    return ret


def dict_mapping(source_config, target_config, mapping_data: dict):
    for ele in mapping_data:
        src_dist = __src_dist_parse(ele)
        __set_config_dict_val(target_config, __path_parse(src_dist[1]),
                            __get_config_val(source_config, __path_parse(src_dist[0])))


def split_path_to_dirt_and_files(path):
    parts = path.rsplit(os.sep, 1)
    return parts[0], [parts[1]]


def combine_directory_and_files_to_path(directory, files):
    if directory.endswith(os.sep):
        return directory + files[0]
    return directory + os.sep + files[0]


class V1RC1ToRC2Convert(Converter):
    def convert(self, json_config):
        rc1_config = Config()
        rc1_config.set_config_data(json_config)
        rc2_json = json.loads(ConfigTemplateMap[V1_RC2_VERSION], object_pairs_hook=dict)
        rc2_config = Config()
        rc2_config.set_config_data(rc2_json)

        leaf_node_value_convert(rc1_config.config_data, rc2_config.config_data)
        list_ele_add_param(rc2_config.config_data[MODEL_DEPLOY_PARAM_KEY][MODEL_PARAM_KEY], \
                             PLUGIN_PARAMS_KEY, PLUGIN_PARAMS_VALUE)
        dir1, files1 = split_path_to_dirt_and_files(rc1_config.config_data[OTHER_PARAM_KEY][SERVER_PARAM_KEY][TLSCRL])
        rc2_config.config_data[OTHER_PARAM_KEY][SERVER_PARAM_KEY][TLSCRL_PATH] = dir1
        rc2_config.config_data[OTHER_PARAM_KEY][SERVER_PARAM_KEY][TLSCRL_FILES] = files1
        return rc2_config.config_data


class V1RC2ToRC3Convert(Converter):
    def convert(self, json_config):
        rc2_config = Config()
        rc2_config.set_config_data(json_config)
        rc3_json = json.loads(ConfigTemplateMap[V1_RC3_VERSION], object_pairs_hook=dict)

        rc3_config = Config()
        rc3_config.set_config_data(rc3_json)

        dict_mapping_list = [ 
            "OtherParam-LogParam_LogConfig",
            "OtherParam-ServeParam_ServerConfig",
            "OtherParam-ServeParam_BackendConfig",
            "ModelDeployParam_BackendConfig",
            "WorkFlowParam-TemplateParam_BackendConfig-ScheduleConfig",
            "ModelDeployParam_BackendConfig-ModelDeployConfig",
            "ScheduleParam_BackendConfig-ScheduleConfig"
        ]

        dict_mapping(rc2_config.config_data, rc3_config.config_data, dict_mapping_list)
        
        node_mapping_list = [ 
            "ModelDeployParam-engineName_BackendConfig-backendName", 
            "ModelDeployParam-ModelParam_BackendConfig-ModelDeployConfig-ModelConfig",
            "OtherParam-ServeParam-multiNodesInferPort_BackendConfig-multiNodesInferPort",
            "ModelDeployParam-modelInstanceNumber_BackendConfig-modelInstanceNumber",
            "ModelDeployParam-npuDeviceIds_BackendConfig-npuDeviceIds",
            "OtherParam-ResourceParam-cacheBlockSize_BackendConfig-ScheduleConfig-cacheBlockSize"
        ]

        node_mapping(rc2_config.config_data, rc3_config.config_data, node_mapping_list)

        list_ele_delete_param(rc3_config.config_data[BACKEND_CONFIG_KEY][MODEL_DEPLOY_CONFIG_KEY][MODEL_CONFIG_KEY], \
                                PLUGIN_PARAMS_KEY)
        
        rc3_config.config_data[SERVER_CONFIG_KEY][TLSCRL] = combine_directory_and_files_to_path(\
            rc2_config.config_data[OTHER_PARAM_KEY][SERVER_PARAM_KEY][TLSCRL_PATH],\
            rc2_config.config_data[OTHER_PARAM_KEY][SERVER_PARAM_KEY][TLSCRL_FILES])
        
        rc3_config.config_data[SERVER_CONFIG_KEY][MANAGEMENT_TLSCRL] = combine_directory_and_files_to_path(\
            rc2_config.config_data[OTHER_PARAM_KEY][SERVER_PARAM_KEY][MANAGEMENT_TLSCRL_PATH],\
            rc2_config.config_data[OTHER_PARAM_KEY][SERVER_PARAM_KEY][MANAGEMENT_TLSCRL_FILES])

        rc3_config.config_data[BACKEND_CONFIG_KEY][INTERNODE_TLSCA] = combine_directory_and_files_to_path(\
            rc2_config.config_data[OTHER_PARAM_KEY][SERVER_PARAM_KEY][INTERNODE_TLSCA_PATH],\
            rc2_config.config_data[OTHER_PARAM_KEY][SERVER_PARAM_KEY][INTERNODE_TLSCA])
        
        rc3_config.config_data[BACKEND_CONFIG_KEY][INTERNODE_TLSCRL] = combine_directory_and_files_to_path(\
            rc2_config.config_data[OTHER_PARAM_KEY][SERVER_PARAM_KEY][INTERNODE_TLSCRL_PATH],\
            rc2_config.config_data[OTHER_PARAM_KEY][SERVER_PARAM_KEY][INTERNODE_TLSCRL_FILE])

        rc3_config.config_data[BACKEND_CONFIG_KEY][SCHEDULE_CONFIG_KEY][TEMPLATE_NAME_KEY] = TEMPLATE_NAME_VALUE

        return rc3_config.config_data


class V1RC3ToRC4Convert(Converter):
    def convert(self, json_config):
        rc3_config = Config()
        rc3_config.set_config_data(json_config)
        rc4_json = json.loads(ConfigTemplateMap[V1_RC4_VERSION], object_pairs_hook=dict)

        rc4_config = Config()
        rc4_config.set_config_data(rc4_json)

        dict_mapping_list = [ 
            "LogConfig_LogConfig",
            "ServerConfig_ServerConfig",
            "BackendConfig_BackendConfig",
            "BackendConfig-ScheduleConfig_BackendConfig-ScheduleConfig",
            "BackendConfig-ModelDeployConfig_BackendConfig-ModelDeployConfig"
        ]

        dict_mapping(rc3_config.config_data, rc4_config.config_data, dict_mapping_list)

        dir1, files1 = split_path_to_dirt_and_files(rc3_config.config_data[SERVER_CONFIG_KEY][TLSCRL])
        rc4_config.config_data[SERVER_CONFIG_KEY][TLSCRL_PATH] = dir1
        rc4_config.config_data[SERVER_CONFIG_KEY][TLSCRL_FILES] = files1

        dir2, files2 = split_path_to_dirt_and_files(rc3_config.config_data[SERVER_CONFIG_KEY][MANAGEMENT_TLSCRL])
        rc4_config.config_data[SERVER_CONFIG_KEY][MANAGEMENT_TLSCRL_PATH] = dir2
        rc4_config.config_data[SERVER_CONFIG_KEY][MANAGEMENT_TLSCRL_FILES] = files2

        dir3, files3 = split_path_to_dirt_and_files(rc3_config.config_data[SERVER_CONFIG_KEY][INTERCOMMON_TLSCA])
        rc4_config.config_data[SERVER_CONFIG_KEY][INTERCOMMON_TLSCA_PATH] = dir3
        rc4_config.config_data[SERVER_CONFIG_KEY][INTERCOMMON_TLSCA_FILES] = files3

        dir4, files4 = split_path_to_dirt_and_files(rc3_config.config_data[SERVER_CONFIG_KEY][INTERCOMMON_TLSCRL])
        rc4_config.config_data[SERVER_CONFIG_KEY][INTERCOMMON_TLSCRL_PATH] = dir4
        rc4_config.config_data[SERVER_CONFIG_KEY][INTERCOMMON_TLSCRL_FILES] = files4

        dir5, files5 = split_path_to_dirt_and_files(rc3_config.config_data[BACKEND_CONFIG_KEY][INTERNODE_TLSCA])
        rc4_config.config_data[BACKEND_CONFIG_KEY][INTERNODE_TLSCA_PATH] = dir5
        rc4_config.config_data[BACKEND_CONFIG_KEY][INTERNODE_TLSCA_FILES] = files5

        dir6, files6 = split_path_to_dirt_and_files(rc3_config.config_data[BACKEND_CONFIG_KEY][INTERNODE_TLSCRL])
        rc4_config.config_data[BACKEND_CONFIG_KEY][INTERNODE_TLSCRL_PATH] = dir6
        rc4_config.config_data[BACKEND_CONFIG_KEY][INTERNODE_TLSCRL_FILES] = files6
        list_ele_add_param(rc4_config.config_data[BACKEND_CONFIG_KEY][MODEL_DEPLOY_CONFIG_KEY][MODEL_CONFIG_KEY],\
            TRUSTREMOTECODE_KEY, TRUSTREMOTECODE_VALUE)

        return rc4_config.config_data


class V1RC1ToRC3Convert(Converter):
    def convert(self, json_config):
        rc1_config = Config()
        rc1_config.set_config_data(json_config)
        rc3_json = json.loads(ConfigTemplateMap[V1_RC3_VERSION], object_pairs_hook=dict)

        rc3_config = Config()
        rc3_config.set_config_data(rc3_json)

        dict_mapping_list = [ 
            "OtherParam-LogParam_LogConfig",
            "OtherParam-ServeParam_ServerConfig",
            "ModelDeployParam_BackendConfig",
            "WorkFlowParam-TemplateParam_BackendConfig-ScheduleConfig",
            "ModelDeployParam_BackendConfig-ModelDeployConfig",
            "ScheduleParam_BackendConfig-ScheduleConfig"
        ]

        dict_mapping(rc1_config.config_data, rc3_config.config_data, dict_mapping_list)

        node_mapping_list = [ 
            "ModelDeployParam-ModelParam_BackendConfig-ModelDeployConfig-ModelConfig",
            "ModelDeployParam-npuDeviceIds_BackendConfig-npuDeviceIds",
            "OtherParam-ResourceParam-cacheBlockSize_BackendConfig-ScheduleConfig-cacheBlockSize"
        ]

        node_mapping(rc1_config.config_data, rc3_config.config_data, node_mapping_list)
        rc3_config.config_data[BACKEND_CONFIG_KEY][SCHEDULE_CONFIG_KEY][TEMPLATE_NAME_KEY] = TEMPLATE_NAME_VALUE

        return rc3_config.config_data
    

class V1RC1ToRC4Convert(Converter):
    def convert(self, json_config):
        rc1_config = Config()
        rc1_config.set_config_data(json_config)
        rc4_json = json.loads(ConfigTemplateMap[V1_RC4_VERSION], object_pairs_hook=dict)

        rc4_config = Config()
        rc4_config.set_config_data(rc4_json)

        dict_mapping_list = [ 
            "OtherParam-LogParam_LogConfig",
            "OtherParam-ServeParam_ServerConfig",
            "ModelDeployParam_BackendConfig",
            "WorkFlowParam-TemplateParam_BackendConfig-ScheduleConfig",
            "ModelDeployParam_BackendConfig-ModelDeployConfig",
            "ScheduleParam_BackendConfig-ScheduleConfig"
        ]

        dict_mapping(rc1_config.config_data, rc4_config.config_data, dict_mapping_list)

        node_mapping_list = [ 
            "ModelDeployParam-ModelParam_BackendConfig-ModelDeployConfig-ModelConfig",
            "ModelDeployParam-npuDeviceIds_BackendConfig-npuDeviceIds",
            "OtherParam-ResourceParam-cacheBlockSize_BackendConfig-ScheduleConfig-cacheBlockSize"
        ]

        node_mapping(rc1_config.config_data, rc4_config.config_data, node_mapping_list)
        list_ele_add_param(rc4_config.config_data[BACKEND_CONFIG_KEY][MODEL_DEPLOY_CONFIG_KEY][MODEL_CONFIG_KEY],\
            TRUSTREMOTECODE_KEY, TRUSTREMOTECODE_VALUE)
        rc4_config.config_data[BACKEND_CONFIG_KEY][SCHEDULE_CONFIG_KEY][TEMPLATE_NAME_KEY] = TEMPLATE_NAME_VALUE
        return rc4_config.config_data