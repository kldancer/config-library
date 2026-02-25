{{/*
Expand the name of the chart.
*/}}
{{- define "gaie.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "gaie.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- $releaseName := .Values.global.name | default .Release.Name }}
{{- if contains $name $releaseName }}
{{- $releaseName | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" $releaseName $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "gaie.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "gaie.labels" -}}
helm.sh/chart: {{ include "gaie.chart" . }}
{{ include "gaie.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "gaie.selectorLabels" -}}
app.kubernetes.io/name: {{ include "gaie.name" . }}
app.kubernetes.io/instance: {{ .Values.global.name | default .Release.Name }}
{{- end }}

{{/*
Generate InferencePool name
InferencePool is created by the inferencepool subchart, typically named as <release-name>-inferencepool
*/}}
{{- define "gaie.inferencePoolName" -}}
{{- if .Values.inferencepool.inferencePool.name }}
{{- .Values.inferencepool.inferencePool.name }}
{{- else }}
{{- $releaseName := .Values.global.name | default .Release.Name }}
{{- printf "%s-inferencepool" $releaseName | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
