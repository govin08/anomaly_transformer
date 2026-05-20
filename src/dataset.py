"""
src/dataset.py
--------------
SMAP / MSL 데이터셋 로더

데이터 형식:
  data/train/P-1.npy  shape (T_train, n_features)
  data/test/P-1.npy   shape (T_test,  n_features)
  data/labeled_anomalies.csv  — 이상 구간 정보
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_channel(data_dir: str, channel: str):
    """
    단일 채널 파일 로드.
    Returns train_arr, test_arr, labels (0/1 per timestep)
    """
    train_path = os.path.join(data_dir, "train", f"{channel}.npy")
    test_path  = os.path.join(data_dir, "test",  f"{channel}.npy")
    label_path = os.path.join(data_dir, "labeled_anomalies.csv")

    train = np.load(train_path).astype(np.float32)
    test  = np.load(test_path).astype(np.float32)

    # 라벨 생성
    T = len(test)
    labels = np.zeros(T, dtype=np.int32)
    if os.path.exists(label_path):
        df = pd.read_csv(label_path)
        row = df[df["chan_id"] == channel]
        if not row.empty:
            anomaly_sequences = eval(row.iloc[0]["anomaly_sequences"])
            for s, e in anomaly_sequences:
                labels[s : e + 1] = 1

    return train, test, labels


def normalize(train: np.ndarray, test: np.ndarray):
    """Train 통계로 z-score 정규화."""
    mean = train.mean(axis=0, keepdims=True)
    std  = train.std(axis=0, keepdims=True) + 1e-8
    return (train - mean) / std, (test - mean) / std


class SlidingWindowDataset(Dataset):
    """
    슬라이딩 윈도우 방식으로 시계열을 잘라 반환.
    각 샘플: (win_len, n_features) 텐서
    """

    def __init__(self, data: np.ndarray, win_len: int, step: int = 1):
        self.data    = data
        self.win_len = win_len
        self.step    = step
        self.indices = list(range(0, len(data) - win_len + 1, step))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        window = self.data[start : start + self.win_len]
        return torch.tensor(window, dtype=torch.float32)


def list_channels(data_dir: str, dataset: str = "SMAP"):
    """
    dataset: "SMAP" | "MSL" | "all"
    Returns sorted list of channel IDs
    """
    train_dir = os.path.join(data_dir, "train")
    all_ch = sorted([f.replace(".npy", "") for f in os.listdir(train_dir)
                     if f.endswith(".npy")])
    if dataset == "SMAP":
        return [c for c in all_ch if c.startswith(("P-", "S-"))]
    elif dataset == "MSL":
        return [c for c in all_ch
                if not c.startswith(("P-", "S-"))]
    return all_ch
