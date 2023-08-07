#!/bin/bash -e

PROJECT_ROOT=$(git rev-parse --show-toplevel)
cd ${PROJECT_ROOT}

source scripts/hl

echo "Profiling the runtime breakdown ..."
cd experiments

echo "[$(emph 1/2)] PtGraph"
pytest test_runtime_performance_gpt2.py::test_ptgraph

sleep 15 # wait for the GPU to cool down

echo "[$(emph 2/2)] Grape"
pytest test_runtime_performance_gpt2.py::test_grape

echo "Visualizing the runtime breakdown ..."
csvtool readable gpt2_generate_profile.csv
