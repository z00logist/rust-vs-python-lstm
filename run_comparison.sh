#!/usr/bin/env bash

set -e

DATA_PATH="data/en-ru.tsv"
BATCH_SIZE=256
EPOCHS=10
MAX_LEN=12
EMB_DIM=32
HID_DIM=64
N_LAYERS=1
LEARNING_RATE=0.001
SEED=42

if [ -d "python_lstm/.venv" ]; then
    source python_lstm/.venv/bin/activate
    python_location=$(which python)
else
    python_location=$(which python3)
fi

echo "Python location: $python_location"

if [ -d "python_lstm/.venv" ]; then
    venv_root=$(dirname "$(dirname "$python_location")")
    echo "Virtual environment root: $venv_root"
    CUDA_LOCATION="$venv_root/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so"
else
    CUDA_LOCATION=$(find /usr -path "*/site-packages/torch/lib/libtorch_cuda.so" 2>/dev/null | head -1)
    if [ -z "$CUDA_LOCATION" ]; then
        CUDA_LOCATION="/usr/local/lib/python3/dist-packages/torch/lib/libtorch_cuda.so"
    fi
fi

echo "CUDA library location: $CUDA_LOCATION"

if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    DEVICE="cuda"
    echo "CUDA is available, using GPU"
else
    DEVICE="cpu"
    echo "CUDA is not available, using CPU"
fi

echo "====================================="
echo "Starting benchmark with these settings:"
echo "Data path: $DATA_PATH"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Device: $DEVICE"
echo "====================================="

echo "[INFO] Running Rust LSTM..."
cd rust_lstm
/usr/bin/time -v \
  bash -c "export RUST_BACKTRACE=1 && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH && cargo run --release -- \
  --data-path \"../$DATA_PATH\" \
  --batch-size $BATCH_SIZE \
  --epochs $EPOCHS \
  --max-len $MAX_LEN \
  --emb-dim $EMB_DIM \
  --hid-dim $HID_DIM \
  --n-layers $N_LAYERS \
  --learning-rate $LEARNING_RATE \
  --seed $SEED \
  --cuda-location $CUDA_LOCATION" \
  2> ../rust_metrics.txt

echo "[INFO] Rust time & mem usage:"
grep "Elapsed (wall clock) time" ../rust_metrics.txt
grep "Maximum resident set size" ../rust_metrics.txt

cd ..

echo "[INFO] Running Python LSTM..."
cd python_lstm

if command -v uv &> /dev/null; then
    PYTHON_CMD="uv run python"
else
    PYTHON_CMD="python"
fi

/usr/bin/time -v \
  $PYTHON_CMD src/train.py \
  --data-path "../$DATA_PATH" \
  --batch-size $BATCH_SIZE \
  --epochs $EPOCHS \
  --max-len $MAX_LEN \
  --emb-dim $EMB_DIM \
  --hid-dim $HID_DIM \
  --n-layers $N_LAYERS \
  --learning-rate $LEARNING_RATE \
  --device "$DEVICE" \
  --seed $SEED 2> ../python_metrics.txt

echo "[INFO] Python time & mem usage:"
grep "Elapsed (wall clock) time" ../python_metrics.txt
grep "Maximum resident set size" ../python_metrics.txt

cd ..

echo "====================================="
echo "Benchmark Summary:"
echo "====================================="
echo "Rust Execution Time:"
grep "Elapsed (wall clock) time" rust_metrics.txt
echo "Rust Memory Usage:"
grep "Maximum resident set size" rust_metrics.txt

echo "Python Execution Time:"
grep "Elapsed (wall clock) time" python_metrics.txt
echo "Python Memory Usage:"
grep "Maximum resident set size" python_metrics.txt

