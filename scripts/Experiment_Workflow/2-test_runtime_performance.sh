#!/bin/bash -e

PROJECT_ROOT=$(git rev-parse --show-toplevel)
cd ${PROJECT_ROOT}

source scripts/hl

MODEL=

while [[ $# -gt 0 ]]; do
        case $1 in
        --model)
                MODEL=$2
                shift
                shift
                ;;
        --model=*)
                MODEL=${1#*=}
                shift
                ;;
        *)
                negative "Usage: $(basename ${BASH_SOURCE[0]}) [--model xxx]"
                exit -1
                ;;
        esac
done

echo "Running the end-to-end runtime performance tests ..."
cd experiments

echo "[$(emph 1/3)] Baseline"
pytest test_runtime_performance_${MODEL}.py::test_baseline

sleep 15 # wait for the GPU to cool down

echo "[$(emph 2/3)] PtGraph"
pytest test_runtime_performance_${MODEL}.py::test_ptgraph

sleep 15 # wait for the GPU to cool down

echo "[$(emph 3/3)] Grape"
pytest test_runtime_performance_${MODEL}.py::test_grape

echo "Visualizing the runtime performance results ..."
csvtool readable speedometer.csv
