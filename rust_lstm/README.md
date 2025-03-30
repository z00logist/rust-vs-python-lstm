# Rust LSTM Implementation

This directory contains a minimal seq2seq LSTM model in Rust using [tch-rs](https://github.com/LaurentMazare/tch-rs).

- `preprocess.rs`: Reads and processes the parallel data.
- `model.rs`: Defines the encoder/decoder using LSTM.
- `main.rs`: Trains the model and prints sample predictions.

## How to Run

1. Install Rust and Cargo (https://www.rust-lang.org).
2. `cd rust_lstm`
3. `cargo run --release`
