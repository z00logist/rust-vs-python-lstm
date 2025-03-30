import typing as t

import torch
from torch.utils.data import Dataset, DataLoader, random_split

SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"


class TranslationDataset(Dataset):
    def __init__(self, data_path: str, max_len: int = 10) -> None:
        self.data_path = data_path
        self.max_len = max_len
        self.src_sentences: t.Sequence[t.Sequence[str]] = []
        self.trg_sentences: t.Sequence[t.Sequence[str]] = []
        self.src_vocab: t.Mapping[str, int] = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2}
        self.trg_vocab: t.Mapping[str, int] = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2}
        self.encoded_src: torch.Tensor = None
        self.encoded_trg: torch.Tensor = None

        self._read_data()
        self._build_vocab()
        self._encode_sentences()

    def _read_data(self) -> None:
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split("\t")
                if len(parts) != 2:
                    print(f"Skipping malformed line: {line}")
                    continue

                en, ru = parts
                self.src_sentences.append(en.strip().split())
                self.trg_sentences.append(ru.strip().split())

    def _build_vocab(self) -> None:
        src_index = len(self.src_vocab)
        for sentence in self.src_sentences:
            for token in sentence:
                if token not in self.src_vocab:
                    self.src_vocab[token] = src_index
                    src_index += 1

        trg_index = len(self.trg_vocab)
        for sentence in self.trg_sentences:
            for token in sentence:
                if token not in self.trg_vocab:
                    self.trg_vocab[token] = trg_index
                    trg_index += 1

    def _encode_sentences(self) -> None:
        self.encoded_src = []
        self.encoded_trg = []
        
        for src, trg in zip(self.src_sentences, self.trg_sentences):
            src_ids = [self.src_vocab[SOS_TOKEN]] + \
                      [self.src_vocab[t] for t in src if t in self.src_vocab] + \
                      [self.src_vocab[EOS_TOKEN]]
            trg_ids = [self.trg_vocab[SOS_TOKEN]] + \
                      [self.trg_vocab[t] for t in trg if t in self.trg_vocab] + \
                      [self.trg_vocab[EOS_TOKEN]]

            src_ids = src_ids[:self.max_len]
            trg_ids = trg_ids[:self.max_len]
            src_ids += [self.src_vocab[PAD_TOKEN]] * (self.max_len - len(src_ids))
            trg_ids += [self.trg_vocab[PAD_TOKEN]] * (self.max_len - len(trg_ids))

            self.encoded_src.append(src_ids)
            self.encoded_trg.append(trg_ids)

        self.encoded_src = torch.tensor(self.encoded_src, dtype=torch.long)
        self.encoded_trg = torch.tensor(self.encoded_trg, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.encoded_src)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoded_src[idx], self.encoded_trg[idx]


def get_train_test_loaders(
    data_path: str, 
    batch_size: int = 2, 
    split_ratio: float = 0.8
) -> tuple[DataLoader, DataLoader, t.Mapping[str, int], t.Mapping[str, int]]:
    dataset = TranslationDataset(data_path)
    total_samples = len(dataset)
    train_size = int(split_ratio * total_samples)
    test_size = total_samples - train_size
    
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, dataset.src_vocab, dataset.trg_vocab

