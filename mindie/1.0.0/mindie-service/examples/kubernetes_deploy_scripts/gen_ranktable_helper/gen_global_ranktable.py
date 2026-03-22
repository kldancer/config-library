#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import json
import time
import logging
import sys
import argparse
logging.basicConfig(level=logging.INFO, format='%(asctime)s |%(levelname)s| %(message)s', stream=sys.stdout)


def wait_hccl_ranktable_finish(configmap_name):
    hccl_ranktable_str = output_from_kubectl(
        "kubectl get configmap %s -n mindie -o jsonpath='{.data}'" % configmap_name)
    try:
        hccl_ranktable_obj = json.loads(json.loads(hccl_ranktable_str)['hccl.json'])
        hardware_type = None
        if 'hardware_type' in json.loads(hccl_ranktable_str):
            hardware_type = json.loads(hccl_ranktable_str)['hardware_type']
        if hccl_ranktable_obj["status"] == 'completed':
            return True, hccl_ranktable_obj, hardware_type
        else:
            return False, None, None
    except Exception:
        logging.error("failed to get ranktable configmap")
        return False, None, None


def output_from_kubectl(command, print_log=True):
    import subprocess
    import shlex

    cmd_args = shlex.split(command)
    child = subprocess.Popen(cmd_args, stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)

    stdout, stderr = child.communicate(timeout=60)
    
    if print_log:
        logging.info(f"Output from command:\n {command} \n is: {stdout}, {stderr} .")
    return stdout


def generate_server_ranktable_for_300i():
    deployment_label = output_from_kubectl(
       "kubectl get deployment mindie-server -n mindie -o jsonpath='{.metadata.labels}'")
    try:
        deployment_label_json_obj = json.loads(deployment_label)
        if deployment_label_json_obj["ring-controller.atlas"] == "ascend-310p":
            replicas = output_from_kubectl(
                "kubectl get deployment mindie-server -n mindie -o jsonpath='{.spec.replicas}'")
            while True:
                pod_ips = output_from_kubectl(
                    "kubectl get pods -l ring-controller.atlas=ascend-310p,"
                    "app=mindie-server -o jsonpath='{.items[*].status.podIP}' -n mindie").split(" ")
                if len(pod_ips) != int(replicas):
                    logging.info("Getting pod ips with label ring-controller.atlas=ascend-310p")
                    time.sleep(2)
                    continue
                else:
                    break
            hccl_rank_table = {}
            
            hccl_rank_table['version'] = "1.0"
            hccl_rank_table['server_list'] = []
            hccl_rank_table['server_count'] = int(replicas)
            hccl_rank_table['status'] = "completed"
            for i in range(int(replicas)):
                hccl_rank_table['server_list'].append({"server_id":pod_ips[i], "container_ip":pod_ips[i]})
            patch_data = f'{{"data":{{"hccl.json":{json.dumps(json.dumps(hccl_rank_table, indent=4))}}}}}'
            ret = output_from_kubectl(
                f"kubectl patch configmap rings-config-mindie-server -n mindie --type merge -p '{patch_data}'", False)
            if ret is None:
                logging.info('return is None')
    except Exception:
        logging.error("failed to get get deployment info")
        return -1
    return 0


def add_mindie_server_group(heter):
    ret = generate_server_ranktable_for_300i()
    if ret != 0:
        logging.info('generate_server_ranktable_for_300i not ok')
    while True:
        finish, hccl_ranktable, hardware_type = wait_hccl_ranktable_finish("rings-config-mindie-server")
        if finish:
            break
        else:
            time.sleep(2)
            continue
    server_group = {}
    server_group['group_id'] = '2'
    server_group['server_count'] = len(hccl_ranktable['server_list'])
    server_group['server_list'] = []
    for item in hccl_ranktable['server_list']:
        server = {}
        server['server_id'] = item['server_id']
        server['server_ip'] = item['container_ip']
        if 'device' in item:
            server['device'] = item['device']
            if hardware_type:
                server['hardware_type'] = hardware_type
            for i in range(len(server['device'])):
                server['device'][i]["device_logical_id"] = str(i)
        server_group['server_list'].append(server)
    if heter:
        while True:
            finish, hccl_ranktable_heter, hardware_type = wait_hccl_ranktable_finish(
                "rings-config-mindie-server-heterogeneous")
            if finish:
                break
            else:
                time.sleep(2)
                continue
        server_group['server_count'] += len(hccl_ranktable_heter['server_list'])
        for item in hccl_ranktable_heter['server_list']:
            server = {}
            server['server_id'] = item['server_id']
            server['server_ip'] = item['container_ip']
            if 'device' in item:
                server['device'] = item['device']
                if hardware_type:
                    server['hardware_type'] = hardware_type
                for i in range(len(server['device'])):
                    server['device'][i]["device_logical_id"] = str(i)
                server_group['server_list'].append(server)
    return server_group


def add_mindie_ms_controller_group():
    while True:
        controller_pod_ip = output_from_kubectl(
            "kubectl get pods -l app=mindie-ms-controller -n mindie -o jsonpath='{.items[*].status.podIP}'")
        if controller_pod_ip == "":
            logging.info("MindIE MS controller is not running")
            time.sleep(1)
            continue
        else:
            break
    server_group = {}
    server_group['group_id'] = '1'
    server_group['server_count'] = 1
    server_group['server_list'] = [{"server_ip":controller_pod_ip}]
    return server_group


def add_mindie_ms_coordinator_group():
    while True:
        coordinator_pod_ip = output_from_kubectl(
            "kubectl get pods -l app=mindie-ms-coordinator -n mindie -o jsonpath='{.items[*].status.podIP}'")
        if coordinator_pod_ip == "":
            logging.info("MindIE MS coordinator is not running")
            time.sleep(1)
            continue
        else:
            break
    server_group = {}
    server_group['group_id'] = '0'
    server_group['server_count'] = 1
    server_group['server_list'] = [{"server_ip":coordinator_pod_ip}]
    return server_group


def generate_global_ranktable(heter=False):
    global_rank_table = {}
    global_rank_table['version'] = "1.0"
    global_rank_table['server_group_list'] = []
    global_rank_table['server_group_list'].append(add_mindie_server_group(heter))
    global_rank_table['server_group_list'].append(add_mindie_ms_controller_group())
    global_rank_table['server_group_list'].append(add_mindie_ms_coordinator_group())
    global_rank_table['status'] = "completed"
    logging.info("global ranktable:\n")
    logging.info(json.dumps(global_rank_table, indent=4))
    patch_data = f'{{"data":{{"global_ranktable.json":{json.dumps(json.dumps(global_rank_table, indent=4))}}}}}'
    ret = output_from_kubectl(f"kubectl patch configmap global-ranktable -n mindie --type merge -p '{patch_data}'",
        False)
    if ret is None:
        logging.info('return is None')
    return global_rank_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--heter', type=bool, default=False, help='Enable or disable heterogeneous deployment')
    args = parser.parse_args()
    ret_val = generate_global_ranktable(args.heter)
    if ret_val is None:
        logging.info('return is None')