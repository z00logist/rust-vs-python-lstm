import math
import typing as t
import pathlib as pth
import random

import torch
import torch.nn as nn
import torch.optim as optim
import typer
import numpy as np

from preprocess import get_train_test_loaders
from model import Encoder, Decoder, Seq2Seq

app = typer.Typer()


def decode_tokens(token_ids: t.Sequence[int], vocab: t.Mapping[str, int]) -> t.Sequence[str]:
    id_to_token = {idx: token for token, idx in vocab.items()}
    special = {vocab["<sos>"], vocab["<eos>"], vocab["<pad>"]}
    return [id_to_token[i] for i in token_ids if i not in special]


@app.command()
def train(
    data_path: str = typer.Option(..., help="Path to the TSV data file."),
    batch_size: int = typer.Option(2, help="Batch size."),
    epochs: int = typer.Option(5, help="Number of training epochs."),
    max_len: int = typer.Option(12, help="Maximum sequence length."),
    emb_dim: int = typer.Option(32, help="Embedding dimension."),
    hid_dim: int = typer.Option(64, help="Hidden dimension."),
    n_layers: int = typer.Option(1, help="Number of LSTM layers."),
    learning_rate: float = typer.Option(0.001, help="Learning rate."),
    seed: int = typer.Option(42, help="Random seed."),
    device: str = typer.Option("cpu", help="Device to use."),
) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    device = torch.device(device)
    typer.echo(f"Using device: {device}")

    data_path = pth.Path(data_path)
    train_loader, test_loader, src_vocab, trg_vocab = get_train_test_loaders(
        data_path, batch_size=batch_size, split_ratio=0.8
    )
    typer.echo(f"Data loaded. Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")
    typer.echo(f"Vocab sizes - Source: {len(src_vocab)}, Target: {len(trg_vocab)}")

    input_dim = len(src_vocab)
    output_dim = len(trg_vocab)

    encoder = Encoder(input_dim, emb_dim, hid_dim, n_layers, bidirectional=True)
    decoder = Decoder(output_dim, emb_dim, hid_dim, n_layers)
    model = Seq2Seq(encoder, decoder, device).to(device)
    typer.echo("Model initialized.")

    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab["<pad>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        for src, trg in train_loader:
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            output = model(src, trg, teacher_forcing_ratio=0.5)
            output_dim_cur = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim_cur)
            trg = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / len(train_loader)
        typer.echo(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f}")

    typer.echo("Training finished.")

    model.eval()
    test_loss = 0
    test_batches = 0
    with torch.no_grad():
        for src, trg in test_loader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0.0)
            output_dim_cur = output.shape[-1]
            output_flat = output[:, 1:].reshape(-1, output_dim_cur)
            trg_flat = trg[:, 1:].reshape(-1)
            loss = criterion(output_flat, trg_flat)
            test_loss += loss.item()
            test_batches += 1

    avg_test_loss = test_loss / test_batches
    perplexity = math.exp(avg_test_loss)
    typer.echo(f"Test Loss: {avg_test_loss:.4f}, Perplexity: {perplexity:.4f}")

    typer.echo("\nGenerated Sample Translations:")

    model.eval()
    sample_count = 0
    with torch.no_grad():
        for src, trg in test_loader:
            src = src.to(device)
            trg = trg.to(device)
            for j in range(src.size(0)):
                src_sample = src[j:j+1]
                trg_sample = trg[j:j+1]
                
                output = model(src_sample, trg_sample, teacher_forcing_ratio=0.0)
                pred_tokens = output.argmax(dim=-1)
                
                src_tokens = src_sample[0].tolist()
                trg_tokens = trg_sample[0].tolist()
                pred_tokens_list = pred_tokens[0].tolist()
                
                src_sentence = " ".join(decode_tokens(src_tokens, src_vocab))
                trg_sentence = " ".join(decode_tokens(trg_tokens, trg_vocab))
                pred_sentence = " ".join(decode_tokens(pred_tokens_list, trg_vocab))
                
                typer.echo(f"Source:    {src_sentence}")
                typer.echo(f"Target:    {trg_sentence}")
                typer.echo(f"Predicted: {pred_sentence}\n")
                
                sample_count += 1
                if sample_count >= 3:
                    break
            if sample_count >= 3:
                break
        
if __name__ == "__main__":
    app()
