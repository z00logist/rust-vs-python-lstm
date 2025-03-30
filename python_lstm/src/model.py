import math
import typing as t

import torch
import torch.nn as nn
import torch.optim as optim
import pathlib as pth

from preprocess import get_train_test_loaders


class Encoder(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int, hid_dim: int, n_layers: int = 1, bidirectional: bool = True) -> None:
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        
        if self.bidirectional:
            def combine(states: torch.Tensor) -> torch.Tensor:
                n = states.size(0) // 2
                combined = [states[2 * i] + states[2 * i + 1] for i in range(n)]
                return torch.stack(combined)
                
            hidden = combine(hidden)
            cell = combine(cell)
            
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim: int, emb_dim: int, hid_dim: int, n_layers: int = 1) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input_step: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_step = input_step.unsqueeze(1)
        embedded = self.embedding(input_step)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device) -> None:
        super().__init__()
        self.__encoder = encoder
        self.__decoder = decoder
        self.__device = device

    def forward(self, src: torch.Tensor, trg: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.__decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.__device)
        hidden, cell = self.__encoder(src)
        input_step = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.__decoder(input_step, hidden, cell)
            outputs[:, t, :] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_step = trg[:, t] if teacher_force else top1
            
        return outputs
