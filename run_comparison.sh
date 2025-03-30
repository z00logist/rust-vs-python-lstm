#!/usr/bin/env bash

DATA_PATH="data/en-ru-25k.tsv"
BATCH_SIZE=8
EPOCHS=10
MAX_LEN=12
EMB_DIM=32
HID_DIM=64
N_LAYERS=1
LEARNING_RATE=0.001
SEED=42
LD_LIBRARY_PATH="$LD_LIBRARY_PATH"


echo "[INFO] Running Rust LSTM..."
cd rust_lstm
/usr/bin/time -v \
  bash -c "export RUST_BACKTRACE=1 && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH && cargo run -- \
  --data-path \"../$DATA_PATH\" \
  --batch-size $BATCH_SIZE \
  --epochs $EPOCHS \
  --max-len $MAX_LEN \
  --emb-dim $EMB_DIM \
  --hid-dim $HID_DIM \
  --n-layers $N_LAYERS \
  --learning-rate $LEARNING_RATE \
  --seed $SEED" \
  2> ../rust_metrics.txt

echo "[INFO] Rust time & mem usage:"
grep "Elapsed (wall clock) time" ../rust_metrics.txt
grep "Maximum resident set size" ../rust_metrics.txt

cd ..

sleep 5s

echo "[INFO] Running Python LSTM..."
cd python_lstm

/usr/bin/time -v \
  uv run python src/train.py \
  --data-path "../$DATA_PATH" \
  --batch-size $BATCH_SIZE \
  --epochs $EPOCHS \
  --max-len $MAX_LEN \
  --emb-dim $EMB_DIM \
  --hid-dim $HID_DIM \
  --n-layers $N_LAYERS \
  --learning-rate $LEARNING_RATE \
  --seed $SEED 2> ../python_metrics.txt

echo "[INFO] Python time & mem usage:"
grep "Elapsed (wall clock) time" ../python_metrics.txt
grep "Maximum resident set size" ../python_metrics.txt

cd ..

