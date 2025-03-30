# Rust vs Python LSTM

This project compares the performance of LSTM (Long Short-Term Memory) neural networks implemented in Python and Rust for sequence-to-sequence translation tasks.

## Project Overview

This repository contains:
- A Python implementation using PyTorch
- A Rust implementation using tch-rs (Rust bindings for PyTorch)
- Infrastructure for comparing the performance of both implementations

Both implementations train a sequence-to-sequence LSTM neural network for language translation (English to Russian), and provide metrics on performance, memory usage, and accuracy.

## Dataset

The project uses English-Russian translation datasets from:
- `data/en-ru.tsv` - Full dataset (30MB)
- `data/en-ru-25k.tsv` - A smaller subset (2MB) for quicker testing

## Project Structure

```
.
├── data/                     # Translation datasets
│   ├── en-ru.tsv             # Full dataset
│   ├── en-ru-25k.tsv         # Smaller dataset
│   ├── _about.txt            # Dataset information
├── python_lstm/              # Python implementation
│   ├── src/
│   │   ├── model.py          # LSTM model implementation
│   │   ├── preprocess.py     # Data preprocessing
│   │   ├── train.py          # Training and evaluation
│   ├── requirements.txt      # Python dependencies
│   ├── pyproject.toml        # Python project configuration
├── rust_lstm/                # Rust implementation
│   ├── src/
│   │   ├── main.rs           # Main entry point
│   │   ├── model.rs          # LSTM model implementation
│   │   ├── preprocess.rs     # Data preprocessing
│   ├── Cargo.toml            # Rust dependencies
├── Dockerfile                # Docker configuration
├── run_comparison.sh         # Script to run benchmarks
├── python_metrics.txt        # Python performance metrics
├── rust_metrics.txt          # Rust performance metrics
├── README.md                 # This file
```

## Running Locally

### Python Implementation

1. Navigate to the Python directory:
```bash
cd python_lstm
```

2. Install dependencies:
```bash
uv sync
```

3. Run the model:
```bash
uv run python src/train.py --data-path "../data/en-ru.tsv" --batch-size 256 --epochs 10
```

### Rust Implementation

1. Navigate to the Rust directory:
```bash
cd rust_lstm
```

2. Build and run:
```bash
cargo run --release -- --data-path "../data/en-ru.tsv" --batch-size 256 --epochs 10
```

## Command Line Options

Both implementations support the same command line options:

- `--data-path`: Path to the TSV data file
- `--batch-size`: Number of samples per batch (default: 256)
- `--epochs`: Number of training epochs (default: 10)
- `--max-len`: Maximum sequence length (default: 12)
- `--emb-dim`: Embedding dimension (default: 32)
- `--hid-dim`: Hidden dimension (default: 64)
- `--n-layers`: Number of LSTM layers (default: 1)
- `--learning-rate`: Learning rate for optimizer (default: 0.001)
- `--seed`: Random seed (default: 42)
- `--device` (Python only): Device to run on ("cpu" or "cuda")
- `--cuda-location` (Rust only): Path to CUDA libraries

## Performance Comparison

### Python Implementation
```
Elapsed (wall clock) time (h:mm:ss or m:ss): 32:56.24
Maximum resident set size (kbytes): 2363588
```

### Rust Implementation
```
Elapsed (wall clock) time (h:mm:ss or m:ss): 22:24.48
Maximum resident set size (kbytes): 1296908
```

### Summary

| Metric             | Python    | Rust      | Improvement |
|--------------------|-----------|-----------|-------------|
| Execution Time     | 32:56.24  | 22:24.48  | ~32% faster |
| Memory Usage (KB)  | 2,363,588 | 1,296,908 | ~45% less   |

## Implementation Details

Both implementations include:
- Encoder-Decoder LSTM architecture
- Attention mechanism
- Teacher forcing during training
- Bidirectional encoder
- Evaluation using perplexity and sample translations

## License

This project is licensed under the terms of the license included in the repository.