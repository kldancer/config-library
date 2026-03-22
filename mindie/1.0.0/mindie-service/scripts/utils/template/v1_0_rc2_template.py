#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.


V1_RC2_CONFIG_TEMPLATE = """{
    "OtherParam" :
    {
        "ResourceParam" :
        {
            "cacheBlockSize" : 128
        },
        "LogParam" :
        {
            "logLevel" : "Info",
            "logPath" : "logs/mindservice.log"
        },
        "ServeParam" :
        {
            "ipAddress" : "127.0.0.1",
            "managementIpAddress" : "127.0.0.2",
            "port" : 1025,
            "managementPort" : 1026,
            "maxLinkNum" : 1000,
            "httpsEnabled" : true,
            "tlsCaPath" : "security/ca/",
            "tlsCaFile" : ["ca.pem"],
            "tlsCert" : "security/certs/server.pem",
            "tlsPk" : "security/keys/server.key.pem",
            "tlsPkPwd" : "security/pass/key_pwd.txt",
            "tlsCrlPath" : "security/certs/",
            "tlsCrlFiles" : ["server_crl.pem"],
            "managementTlsCaFile" : ["management_ca.pem"],
            "managementTlsCert" : "security/management/certs/server.pem",
            "managementTlsPk" : "security/management/keys/server.key.pem",
            "managementTlsPkPwd" : "security/management/pass/key_pwd.txt",
            "managementTlsCrlPath" : "security/management/certs/",
            "managementTlsCrlFiles" : ["server_crl.pem"],
            "kmcKsfMaster" : "tools/pmt/master/ksfa",
            "kmcKsfStandby" : "tools/pmt/standby/ksfb",
            "multiNodesInferPort" : 1120,
            "interNodeTLSEnabled" : true,
            "interNodeTlsCaPath" : "security/grpc/ca/",
            "interNodeTlsCaFile" : ["ca.pem"],
            "interNodeTlsCert" : "security/grpc/certs/server.pem",
            "interNodeTlsPk" : "security/grpc/keys/server.key.pem",
            "interNodeTlsPkPwd" : "security/grpc/pass/key_pwd.txt",
            "interNodeKmcKsfMaster" : "tools/pmt/master/ksfa",
            "interNodeKmcKsfStandby" : "tools/pmt/standby/ksfb",
            "interNodeTlsCrlPath" : "security/grpc/certs/",
            "interNodeTlsCrlFile" : ["server_crl.pem"]
        }
    },
    "WorkFlowParam" : 
    {
        "TemplateParam" :
        {
            "templateType" : "Standard",
            "templateName" : "Standard_llama"
        }
    },
    "ModelDeployParam" : 
    {
        "engineName" : "mindieservice_llm_engine",
        "modelInstanceNumber" : 1,
        "tokenizerProcessNumber" : 8,
        "maxSeqLen" : 2560,
        "npuDeviceIds" : [[0,1,2,3]],
        "multiNodesInferEnabled" : false,
        "ModelParam" : [
            {
                "modelInstanceType" : "Standard",
                "modelName" : "llama_65b",
                "modelWeightPath" : "/data/atb_testdata/weights/llama1-65b-safetensors",
                "worldSize" : 4,
                "cpuMemSize" : 5,
                "npuMemSize" : 8,
                "backendType": "atb",
                "pluginParams" : ""
            }
        ]
    },
    "ScheduleParam" : 
    {
        "maxPrefillBatchSize" : 50,
        "maxPrefillTokens" : 8192,
        "prefillTimeMsPerReq" : 150,
        "prefillPolicyType" : 0,

        "decodeTimeMsPerReq" : 50,
        "decodePolicyType" : 0,

        "maxBatchSize" : 200,
        "maxIterTimes" : 512,
        "maxPreemptCount" : 0,
        "supportSelectBatch" : false,
        "maxQueueDelayMicroseconds" : 5000
    }
}"""