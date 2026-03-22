#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.


V1_RC1_CONFIG_TEMPLATE = """{
    "OtherParam":
    {
        "ResourceParam" :
        {
            "cacheBlockSize" : 128,
            "preAllocBlocks" : 4
        },
        "LogParam" :
        {
            "logLevel" : "Info",
            "logPath" : "/logs/mindservice.log"
        },
        "ServeParam" :
        {
            "ipAddress" : "127.0.0.1",
            "port" : 1025,
            "maxLinkNum" : 300,
            "httpsEnabled" : true,
            "tlsCaPath" : "security/ca/",
            "tlsCaFile" : ["ca.pem"],
            "tlsCert" : "security/certs/server.pem",
            "tlsPk" : "security/keys/server.key.pem",
            "tlsPkPwd" : "security/pass/mindie_server_key_pwd.txt",
            "kmcKsfMaster" : "tools/pmt/master/ksfa",
            "kmcKsfStandby" : "tools/pmt/standby/ksfb",
            "tlsCrl" : "security/certs/server_crl.pem"
        }
    },
    "WorkFlowParam":
    {
        "TemplateParam" :
        {
            "templateType": "Standard",
            "templateName" : "Standard_llama",
            "pipelineNumber" : 1
        }
    },
    "ModelDeployParam":
    {
        "maxSeqLen" : 2560,
        "npuDeviceIds" : [[0,1,2,3]],
        "ModelParam" : [
            {
                "modelInstanceType": "Standard",
                "modelName" : "llama_65b",
                "modelWeightPath" : "/data/atb_testdata/weights/llama1-65b-safetensors",
                "worldSize" : 4,
                "cpuMemSize" : 5,
                "npuMemSize" : 8,
                "backendType": "atb"
            }
        ]
    },
    "ScheduleParam":
    {
        "maxPrefillBatchSize" : 50,
        "maxPrefillTokens" : 8192,
        "prefillTimeMsPerReq" : 150,
        "prefillPolicyType" : 0,

        "decodeTimeMsPerReq" : 50,
        "decodePolicyType" : 0,

        "maxBatchSize" : 200,
        "maxIterTimes" : 512,
        "maxPreemptCount" : 200,
        "supportSelectBatch" : false,
        "maxQueueDelayMicroseconds" : 5000
    }
}"""