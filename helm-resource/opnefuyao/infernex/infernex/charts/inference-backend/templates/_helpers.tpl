{{/*
Expand the name of the chart.
*/}}
{{- define "vllm.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "vllm.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "vllm.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels for a vllm service instance
Used for Deployment metadata.labels
*/}}
{{- define "vllm.instance.labels" -}}
helm.sh/chart: {{ include "vllm.chart" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/instance: {{ printf "%s-%s" .Release.Name .service.name }}
app.kubernetes.io/name: {{ include "vllm.name" . }}
{{- end }}

{{/*
Service-specific labels (for Service metadata only)
Distinguishes Service from Pod/Deployment
*/}}
{{- define "vllm.service.labels" -}}
{{- include "vllm.instance.labels" . }}
app.kubernetes.io/component: service
{{- end }}

{{/*
Selector labels for a vllm service instance
*/}}
{{- define "vllm.instance.selectorLabels" -}}
app.kubernetes.io/name: {{ include "vllm.name" . }}
app.kubernetes.io/instance: {{ printf "%s-%s" .Release.Name .service.name }}
{{- end }}

{{/*
Generate service name for a vllm instance
*/}}
{{- define "vllm.instance.serviceName" -}}
{{- printf "%s-%s" (include "vllm.fullname" .) .service.name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Generate deployment name for a vllm instance
*/}}
{{- define "vllm.instance.deploymentName" -}}
{{- printf "%s-%s" (include "vllm.fullname" .) .service.name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Get image repository
*/}}
{{- define "vllm.image.repository" -}}
{{- if and .service.image .service.image.repository }}
{{- .service.image.repository }}
{{- else if and .Values.global.image .Values.global.image.repository }}
{{- .Values.global.image.repository }}
{{- else }}
{{- "" }}
{{- end }}
{{- end }}

{{/*
Get image tag
*/}}
{{- define "vllm.image.tag" -}}
{{- if and .service.image .service.image.tag }}
{{- .service.image.tag }}
{{- else if and .Values.global.image .Values.global.image.tag }}
{{- .Values.global.image.tag }}
{{- else }}
{{- .Chart.AppVersion }}
{{- end }}
{{- end }}

{{/*
Get image pull policy
*/}}
{{- define "vllm.image.pullPolicy" -}}
{{- if and .service.image .service.image.pullPolicy }}
{{- .service.image.pullPolicy }}
{{- else if and .Values.global.image .Values.global.image.pullPolicy }}
{{- .Values.global.image.pullPolicy }}
{{- else }}
{{- "" }}
{{- end }}
{{- end }}

{{/*
Merge global and service-specific resources
*/}}
{{- define "vllm.resources" -}}
{{- if .service.resources }}
{{- .service.resources | toYaml }}
{{- else }}
{{- .Values.global.resources | toYaml }}
{{- end }}
{{- end }}

{{/*
Calculate resources for PD mode (prefill or decode)
*/}}
{{- define "vllm.pd.resources" -}}
{{- $tensorParallelSize := .tensorParallelSize | default 1 | int }}
{{- $dataParallelSize := .dataParallelSize | default 1 | int }}
{{- $cardCount := mul $tensorParallelSize $dataParallelSize }}
{{- $baseResources := .baseResources | default (dict "requests" dict "limits" dict) }}
{{- $cardResourceName := "huawei.com/Ascend910" }}
{{- $requests := $baseResources.requests | default dict }}
{{- $limits := $baseResources.limits | default dict }}
{{- $requests = set $requests $cardResourceName (toString $cardCount) }}
{{- $limits = set $limits $cardResourceName (toString $cardCount) }}
{{- $result := dict "requests" $requests "limits" $limits }}
{{- $result | toYaml }}
{{- end }}

{{/*
Merge global and service-specific env
*/}}
{{- define "vllm.env" -}}
{{- $globalEnv := .Values.global.env | default list }}
{{- $serviceEnv := .service.env | default list }}
{{- concat $globalEnv $serviceEnv | toYaml }}
{{- end }}

{{/*
Common Pod environment variables (POD_NAME, POD_IP)
*/}}
{{- define "vllm.podEnv" -}}
- name: POD_NAME
  valueFrom:
    fieldRef:
      fieldPath: metadata.name
- name: POD_IP
  valueFrom:
    fieldRef:
      fieldPath: status.podIP
{{- end }}

{{/*
Common volumeMounts shared by all vLLM deployments
*/}}
{{- define "vllm.defaultVolumeMounts" -}}
- name: shm
  mountPath: /dev/shm
- name: dcmi
  mountPath: /usr/local/dcmi
- name: npusmi
  mountPath: /usr/local/bin/npu-smi
- name: lib64
  mountPath: /usr/local/Ascend/driver/lib64
- name: version
  mountPath: /usr/local/Ascend/driver/version.info
- name: installinfo
  mountPath: /etc/ascend_install.info
- name: rootcache
  mountPath: /root/.cache
- name: hccntool
  mountPath: /usr/bin/hccn_tool
- name: hccnconf
  mountPath: /etc/hccn.conf
{{- end }}

{{/*
Common volumes shared by all vLLM deployments
*/}}
{{- define "vllm.defaultVolumes" -}}
- name: shm
  emptyDir:
    medium: Memory
    sizeLimit: "24Gi"
- name: dcmi
  hostPath:
    path: /usr/local/dcmi
- name: npusmi
  hostPath:
    path: /usr/local/bin/npu-smi
    type: File
- name: lib64
  hostPath:
    path: /usr/local/Ascend/driver/lib64
- name: version
  hostPath:
    path: /usr/local/Ascend/driver/version.info
    type: File
- name: installinfo
  hostPath:
    path: /etc/ascend_install.info
    type: File
- name: rootcache
  hostPath:
    path: {{ (.Values.global.cachePath | default "/home/llm_cache") }}
- name: hccntool
  hostPath:
    path: /usr/bin/hccn_tool
- name: hccnconf
  hostPath:
    path: /etc/hccn.conf
{{- end }}

