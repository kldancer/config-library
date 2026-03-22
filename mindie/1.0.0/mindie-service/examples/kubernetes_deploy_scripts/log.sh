#!/bin/bash
echo "==================================================== Stdout Log From MindIE-MS Coordinator ===================================================="
kubectl logs -l app=mindie-ms-coordinator -n mindie
echo

echo "==================================================== Stdout Log From MindIE-MS Controller ======================================================"
kubectl logs -l app=mindie-ms-controller -n mindie
echo

echo "==================================================== Stdout Log From MindIE-Server ============================================================="
kubectl logs -l app=mindie-server -n mindie
echo

if [ $# -eq 1 ]; then
    if [[ $1 == "heter" ]]; then
        echo "==================================================== Stdout Log From MindIE-Server-Heterogeneous ============================================================="
        kubectl logs -l app=mindie-server-heterogeneous -n mindie
        echo
    fi
fi