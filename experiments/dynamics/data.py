
from __future__ import annotations

import os

import numpy as np
import torch

EOT = 50256          
VOCAB_SIZE = 50257   

def get_tokenizer():
    import tiktoken

    return tiktoken.get_encoding("gpt2")

def build_tinystories_tokens(
    out_dir: str,
    n_train_tokens: int = 8_000_000,
    n_val_tokens: int = 500_000,
) -> tuple[np.ndarray, np.ndarray]:
    os.makedirs(out_dir, exist_ok=True)
    tr_path = os.path.join(out_dir, f"train_{n_train_tokens}.npy")
    va_path = os.path.join(out_dir, f"val_{n_val_tokens}.npy")
    if os.path.exists(tr_path) and os.path.exists(va_path):
        return np.load(tr_path), np.load(va_path)

    from datasets import load_dataset

    enc = get_tokenizer()

    def collect(split: str, n_target: int) -> np.ndarray:
        ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        chunks, total = [], 0
        for ex in ds:
            ids = enc.encode_ordinary(ex["text"])
            ids.append(EOT)
            chunks.append(np.asarray(ids, dtype=np.uint16))
            total += len(ids)
            if total >= n_target:
                break
        arr = np.concatenate(chunks)[:n_target]
        if arr.shape[0] < n_target:
            raise RuntimeError(f"{split}: only collected {arr.shape[0]} < {n_target} tokens")
        return arr

    train = collect("train", n_train_tokens)
    val = collect("validation", n_val_tokens)
    np.save(tr_path, train)
    np.save(va_path, val)
    return train, val

def build_wikitext_tokens(
    out_dir: str,
    n_train_tokens: int = 8_000_000,
    n_val_tokens: int = 500_000,
) -> tuple[np.ndarray, np.ndarray]:
    os.makedirs(out_dir, exist_ok=True)
    tr_path = os.path.join(out_dir, f"wikitext_train_{n_train_tokens}.npy")
    va_path = os.path.join(out_dir, f"wikitext_val_{n_val_tokens}.npy")
    if os.path.exists(tr_path) and os.path.exists(va_path):
        return np.load(tr_path), np.load(va_path)

    from datasets import load_dataset

    enc = get_tokenizer()

    def collect(split: str, n_target: int) -> np.ndarray:
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split, streaming=True)
        chunks, total = [], 0
        for ex in ds:
            t = ex["text"].strip()
            if not t:
                continue
            ids = enc.encode_ordinary(t)
            ids.append(EOT)
            chunks.append(np.asarray(ids, dtype=np.uint16))
            total += len(ids)
            if total >= n_target:
                break
        arr = np.concatenate(chunks)[:n_target]
        if arr.shape[0] < n_target:
            raise RuntimeError(f"wikitext {split}: only collected {arr.shape[0]} < {n_target} tokens")
        return arr

    train = collect("train", n_train_tokens)
    val = collect("validation", n_val_tokens)
    np.save(tr_path, train)
    np.save(va_path, val)
    return train, val

class TokenLoader:

    def __init__(self, tokens: np.ndarray, seq_len: int, device: str):
        self.data = torch.from_numpy(tokens.astype(np.int64))
        self.seq_len = seq_len
        self.device = device

    def batch(self, batch_size: int, generator: torch.Generator) -> torch.Tensor:
        hi = self.data.shape[0] - self.seq_len - 1
        ix = torch.randint(hi, (batch_size,), generator=generator)
        x = torch.stack([self.data[i : i + self.seq_len] for i in ix])
        return x.to(self.device, non_blocking=True)

    def iter_eval(self, batch_size: int, n_batches: int, seed: int = 1234):
        g = torch.Generator().manual_seed(seed)
        for _ in range(n_batches):
            yield self.batch(batch_size, g)
