# data_loader.py
# -*- coding: utf-8 -*-
"""
Data loader for CTRA-Net style Knowledge Tracing.

Supported features:
1. Read KT interaction data from csv/tsv.
2. Group by student and sort by timestamp.
3. Build fixed-length subsequences for next-step prediction.
4. Return padded mini-batches.
5. Build question graph adjacency matrix from co-occurrence.

Expected raw columns (configurable):
- user_id
- question_id
- correct
- timestamp

Author: OpenAI
"""

from __future__ import annotations

import os
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# =========================================================
# Utilities
# =========================================================

def set_random_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _safe_read_table(file_path: str, sep: Optional[str] = None) -> pd.DataFrame:
    """
    Read csv/tsv/txt automatically.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    if sep is not None:
        return pd.read_csv(file_path, sep=sep)

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(file_path)
    if ext in [".tsv", ".txt"]:
        return pd.read_csv(file_path, sep="\t")

    # fallback
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.read_csv(file_path, sep="\t")


# =========================================================
# Config-like dataclass for standalone use
# =========================================================

@dataclass
class LoaderConfig:
    data_path: str
    batch_size: int = 64
    max_seq_len: int = 100
    train_ratio: float = 0.8
    num_workers: int = 0
    seed: int = 42

    # column names in raw file
    user_col: str = "user_id"
    question_col: str = "question_id"
    correct_col: str = "correct"
    time_col: str = "timestamp"

    # preprocessing
    min_seq_len: int = 3
    drop_last: bool = False
    pad_val: int = 0

    # graph construction
    graph_window: Optional[int] = None     # None = full-sequence co-occurrence
    self_loop: bool = True
    normalize_adj: bool = True

    # file format
    sep: Optional[str] = None


# =========================================================
# Dataset
# =========================================================

class KnowledgeTracingDataset(Dataset):
    """
    Each sample is a subsequence for next-step prediction.

    Input sequence:
        x_t = q_t + a_t * num_questions
    Prediction target:
        predict correctness of q_{t+1}

    Returned dict fields:
        input_ids      : [L]
        question_ids   : [L]
        responses      : [L]
        target_qids    : [L]
        target_labels  : [L]
        mask           : [L]
        seq_len        : int
    """

    def __init__(
        self,
        sequences: List[Dict[str, List[int]]],
        num_questions: int,
        max_seq_len: int,
        pad_val: int = 0
    ) -> None:
        super().__init__()
        self.samples = []
        self.num_questions = num_questions
        self.max_seq_len = max_seq_len
        self.pad_val = pad_val

        for seq in sequences:
            q_seq = seq["questions"]
            a_seq = seq["answers"]

            if len(q_seq) < 2:
                continue

            # split long sequence into chunks
            start = 0
            while start < len(q_seq) - 1:
                end = min(start + max_seq_len, len(q_seq))
                sub_q = q_seq[start:end]
                sub_a = a_seq[start:end]

                if len(sub_q) >= 2:
                    self.samples.append({
                        "questions": sub_q,
                        "answers": sub_a
                    })
                start += max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        q_seq = item["questions"]
        a_seq = item["answers"]

        # current step input
        input_q = q_seq[:-1]
        input_a = a_seq[:-1]

        # next-step target
        target_q = q_seq[1:]
        target_a = a_seq[1:]

        # interaction encoding:
        # [1, num_questions] -> incorrect
        # [num_questions+1, 2*num_questions] -> correct
        input_ids = []
        for q, a in zip(input_q, input_a):
            token = q + a * self.num_questions
            input_ids.append(token)

        seq_len = len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "question_ids": torch.tensor(input_q, dtype=torch.long),
            "responses": torch.tensor(input_a, dtype=torch.long),
            "target_qids": torch.tensor(target_q, dtype=torch.long),
            "target_labels": torch.tensor(target_a, dtype=torch.float32),
            "mask": torch.ones(seq_len, dtype=torch.float32),
            "seq_len": torch.tensor(seq_len, dtype=torch.long),
        }


# =========================================================
# Collate Function
# =========================================================

def kt_collate_fn(batch: List[Dict[str, torch.Tensor]], pad_val: int = 0) -> Dict[str, torch.Tensor]:
    """
    Pad variable-length sequences in a batch.
    """
    max_len = max(x["seq_len"].item() for x in batch)

    def pad_1d(x: torch.Tensor, pad_value: float) -> torch.Tensor:
        if x.size(0) == max_len:
            return x
        pad_size = max_len - x.size(0)
        pad_tensor = torch.full((pad_size,), pad_value, dtype=x.dtype)
        return torch.cat([x, pad_tensor], dim=0)

    out = {
        "input_ids": [],
        "question_ids": [],
        "responses": [],
        "target_qids": [],
        "target_labels": [],
        "mask": [],
        "seq_len": [],
    }

    for item in batch:
        out["input_ids"].append(pad_1d(item["input_ids"], pad_val))
        out["question_ids"].append(pad_1d(item["question_ids"], pad_val))
        out["responses"].append(pad_1d(item["responses"], 0))
        out["target_qids"].append(pad_1d(item["target_qids"], pad_val))
        out["target_labels"].append(pad_1d(item["target_labels"], -1.0))  # ignored by masked BCE
        out["mask"].append(pad_1d(item["mask"], 0.0))
        out["seq_len"].append(item["seq_len"])

    for k in out:
        out[k] = torch.stack(out[k], dim=0)

    return out


# =========================================================
# Preprocessing
# =========================================================

def preprocess_dataframe(df: pd.DataFrame, cfg: LoaderConfig) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int]]:
    """
    Clean data and remap IDs to contiguous indices starting from 1.
    0 is reserved for padding.
    """
    required_cols = [cfg.user_col, cfg.question_col, cfg.correct_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in data file.")

    # keep required columns
    used_cols = [cfg.user_col, cfg.question_col, cfg.correct_col]
    if cfg.time_col in df.columns:
        used_cols.append(cfg.time_col)

    df = df[used_cols].copy()
    df = df.dropna(subset=[cfg.user_col, cfg.question_col, cfg.correct_col])

    # standardize labels
    df[cfg.correct_col] = df[cfg.correct_col].astype(int)
    df = df[df[cfg.correct_col].isin([0, 1])]

    # sort
    if cfg.time_col in df.columns:
        df = df.sort_values([cfg.user_col, cfg.time_col]).reset_index(drop=True)
    else:
        df = df.sort_values([cfg.user_col]).reset_index(drop=True)

    # remap user/question ids
    user_values = df[cfg.user_col].unique().tolist()
    ques_values = df[cfg.question_col].unique().tolist()

    user2idx = {u: i + 1 for i, u in enumerate(user_values)}
    q2idx = {q: i + 1 for i, q in enumerate(ques_values)}

    df[cfg.user_col] = df[cfg.user_col].map(user2idx)
    df[cfg.question_col] = df[cfg.question_col].map(q2idx)

    return df, user2idx, q2idx


def build_student_sequences(df: pd.DataFrame, cfg: LoaderConfig) -> List[Dict[str, List[int]]]:
    """
    Convert interaction table into per-student sequences.
    """
    sequences = []

    grouped = df.groupby(cfg.user_col)
    for user_id, group in grouped:
        q_seq = group[cfg.question_col].astype(int).tolist()
        a_seq = group[cfg.correct_col].astype(int).tolist()

        if len(q_seq) < cfg.min_seq_len:
            continue

        sequences.append({
            "user_id": int(user_id),
            "questions": q_seq,
            "answers": a_seq
        })

    return sequences


def train_test_split_by_student(
    sequences: List[Dict[str, List[int]]],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[Dict[str, List[int]]], List[Dict[str, List[int]]]]:
    """
    Student-level split.
    """
    rng = random.Random(seed)
    seqs = sequences.copy()
    rng.shuffle(seqs)

    split_idx = int(len(seqs) * train_ratio)
    train_seqs = seqs[:split_idx]
    test_seqs = seqs[split_idx:]

    return train_seqs, test_seqs


# =========================================================
# Graph Construction
# =========================================================

def build_question_adjacency(
    sequences: List[Dict[str, List[int]]],
    num_questions: int,
    window_size: Optional[int] = None,
    self_loop: bool = True,
    normalize: bool = True
) -> torch.Tensor:
    """
    Build question co-occurrence graph from sequences.

    Args:
        sequences: student sequences
        num_questions: number of unique questions
        window_size:
            None -> all questions within a sequence co-occur
            int  -> sliding local co-occurrence
    Returns:
        adj: [num_questions + 1, num_questions + 1]
             index 0 is reserved for padding
    """
    adj = np.zeros((num_questions + 1, num_questions + 1), dtype=np.float32)

    for seq in sequences:
        questions = seq["questions"]

        if window_size is None:
            unique_q = list(dict.fromkeys(questions))
            for i in range(len(unique_q)):
                qi = unique_q[i]
                for j in range(i, len(unique_q)):
                    qj = unique_q[j]
                    adj[qi, qj] += 1.0
                    if qi != qj:
                        adj[qj, qi] += 1.0
        else:
            n = len(questions)
            for i in range(n):
                qi = questions[i]
                left = max(0, i - window_size)
                right = min(n, i + window_size + 1)
                for j in range(left, right):
                    qj = questions[j]
                    adj[qi, qj] += 1.0

    if self_loop:
        for i in range(1, num_questions + 1):
            adj[i, i] += 1.0

    # binarize or keep weighted; here keep weighted
    if normalize:
        deg = adj.sum(axis=1)
        deg[deg == 0] = 1.0
        d_inv_sqrt = np.power(deg, -0.5)
        d_mat = np.diag(d_inv_sqrt)
        adj = d_mat @ adj @ d_mat

    return torch.tensor(adj, dtype=torch.float32)


# =========================================================
# Main API
# =========================================================

def load_kt_data(cfg: LoaderConfig) -> Dict[str, object]:
    """
    Main entrance:
    1. read raw file
    2. preprocess
    3. split by student
    4. build datasets and dataloaders
    5. build question graph

    Returns:
        {
            "train_loader": ...,
            "test_loader": ...,
            "train_dataset": ...,
            "test_dataset": ...,
            "adj_matrix": ...,
            "num_questions": ...,
            "num_users": ...,
            "train_sequences": ...,
            "test_sequences": ...,
        }
    """
    set_random_seed(cfg.seed)

    df = _safe_read_table(cfg.data_path, sep=cfg.sep)
    df, user2idx, q2idx = preprocess_dataframe(df, cfg)

    sequences = build_student_sequences(df, cfg)
    train_sequences, test_sequences = train_test_split_by_student(
        sequences=sequences,
        train_ratio=cfg.train_ratio,
        seed=cfg.seed
    )

    num_questions = len(q2idx)
    num_users = len(user2idx)

    train_dataset = KnowledgeTracingDataset(
        sequences=train_sequences,
        num_questions=num_questions,
        max_seq_len=cfg.max_seq_len,
        pad_val=cfg.pad_val
    )
    test_dataset = KnowledgeTracingDataset(
        sequences=test_sequences,
        num_questions=num_questions,
        max_seq_len=cfg.max_seq_len,
        pad_val=cfg.pad_val
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=cfg.drop_last,
        collate_fn=lambda x: kt_collate_fn(x, pad_val=cfg.pad_val)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
        collate_fn=lambda x: kt_collate_fn(x, pad_val=cfg.pad_val)
    )

    # graph usually built from training sequences only
    adj_matrix = build_question_adjacency(
        sequences=train_sequences,
        num_questions=num_questions,
        window_size=cfg.graph_window,
        self_loop=cfg.self_loop,
        normalize=cfg.normalize_adj
    )

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "adj_matrix": adj_matrix,
        "num_questions": num_questions,
        "num_users": num_users,
        "train_sequences": train_sequences,
        "test_sequences": test_sequences,
        "user2idx": user2idx,
        "q2idx": q2idx,
    }


# =========================================================
# Optional helper for external config.py
# =========================================================

def build_loaders_from_config(config) -> Dict[str, object]:
    """
    Compatible with a typical config.py object/class.
    The config object is expected to have the same attributes as LoaderConfig.
    """
    cfg = LoaderConfig(
        data_path=config.data_path,
        batch_size=getattr(config, "batch_size", 64),
        max_seq_len=getattr(config, "max_seq_len", 100),
        train_ratio=getattr(config, "train_ratio", 0.8),
        num_workers=getattr(config, "num_workers", 0),
        seed=getattr(config, "seed", 42),
        user_col=getattr(config, "user_col", "user_id"),
        question_col=getattr(config, "question_col", "question_id"),
        correct_col=getattr(config, "correct_col", "correct"),
        time_col=getattr(config, "time_col", "timestamp"),
        min_seq_len=getattr(config, "min_seq_len", 3),
        drop_last=getattr(config, "drop_last", False),
        pad_val=getattr(config, "pad_val", 0),
        graph_window=getattr(config, "graph_window", None),
        self_loop=getattr(config, "self_loop", True),
        normalize_adj=getattr(config, "normalize_adj", True),
        sep=getattr(config, "sep", None),
    )
    return load_kt_data(cfg)


# =========================================================
# Quick test
# =========================================================

if __name__ == "__main__":
    # Example usage:
    # python data_loader.py
    #
    # Before running, modify the path below.
    demo_cfg = LoaderConfig(
        data_path="data/assistments2009.csv",
        batch_size=32,
        max_seq_len=100,
        train_ratio=0.8,
        user_col="user_id",
        question_col="question_id",
        correct_col="correct",
        time_col="timestamp",
        min_seq_len=3
    )

    if os.path.exists(demo_cfg.data_path):
        outputs = load_kt_data(demo_cfg)
        print("num_questions:", outputs["num_questions"])
        print("num_users:", outputs["num_users"])
        print("adj_matrix shape:", outputs["adj_matrix"].shape)

        batch = next(iter(outputs["train_loader"]))
        for k, v in batch.items():
            print(k, tuple(v.shape))
    else:
        print(f"Demo file not found: {demo_cfg.data_path}")