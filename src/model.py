"""
src/model.py
------------
논문 "Anomaly Transformer" (ICLR 2022) 핵심 구조 구현

구성:
  AnomalyAttention  — Prior-Association + Series-Association (two-branch)
  EncoderLayer      — AnomalyAttention + FeedForward + LayerNorm
  AnomalyTransformer — L개 EncoderLayer 스택 + 재구성 헤드
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# Anomaly-Attention (Two-Branch)
# ══════════════════════════════════════════════════════════════════════════════
class AnomalyAttention(nn.Module):
    """
    논문 수식 (2):
      Prior-Association  P^l_ij  = Rescale( Gauss(|j-i|; sigma_i) )
      Series-Association S^l     = Softmax( Q K^T / sqrt(d) )
      Reconstruction     Z^l_hat = S^l V
    """

    def __init__(self, d_model: int, n_heads: int, seq_len: int,
                 attn_dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.seq_len  = seq_len

        self.W_q  = nn.Linear(d_model, d_model, bias=False)
        self.W_k  = nn.Linear(d_model, d_model, bias=False)
        self.W_v  = nn.Linear(d_model, d_model, bias=False)
        self.W_o  = nn.Linear(d_model, d_model, bias=False)

        # 학습 가능한 sigma (per time-step, shared across heads)
        self.log_sigma = nn.Parameter(torch.ones(seq_len) * math.log(25.0))

        self.dropout = nn.Dropout(attn_dropout)

        # 위치별 거리 행렬 사전 계산 (고정, 학습 안 함)
        idx = torch.arange(seq_len).float()
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()   # (L, L)
        self.register_buffer("dist", dist)

    def _prior_association(self) -> torch.Tensor:
        """
        Returns P of shape (seq_len, seq_len) — discrete distribution per row.
        sigma는 양수 강제를 위해 softplus 적용.
        """
        sigma = F.softplus(self.log_sigma).unsqueeze(1)   # (L, 1)
        # Gaussian kernel
        gauss = torch.exp(-self.dist ** 2 / (2 * sigma ** 2))  # (L, L)
        # Rescale → row-wise softmax (= 이산 분포)
        P = gauss / (gauss.sum(dim=-1, keepdim=True) + 1e-8)
        return P   # (L, L)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : (B, L, d_model)
        Returns:
            out          : (B, L, d_model)   재구성 벡터
            series_assoc : (B, L, L)          Series-Association (per head 평균)
            prior_assoc  : (L, L)             Prior-Association
        """
        B, L, _ = x.shape
        H, Dh = self.n_heads, self.d_head

        # Linear projection & reshape → (B, H, L, Dh)
        Q = self.W_q(x).view(B, L, H, Dh).transpose(1, 2)
        K = self.W_k(x).view(B, L, H, Dh).transpose(1, 2)
        V = self.W_v(x).view(B, L, H, Dh).transpose(1, 2)

        # Series-Association: standard scaled dot-product attention
        scale = math.sqrt(Dh)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale   # (B, H, L, L)
        S = torch.softmax(scores, dim=-1)                        # (B, H, L, L)
        S = self.dropout(S)

        # Reconstruction
        out = torch.matmul(S, V)                                 # (B, H, L, Dh)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        out = self.W_o(out)

        # Prior-Association (shared across heads)
        P = self._prior_association()                            # (L, L)

        # Series-Association averaged over heads
        S_avg = S.mean(dim=1)                                    # (B, L, L)

        return out, S_avg, P


# ══════════════════════════════════════════════════════════════════════════════
# Encoder Layer
# ══════════════════════════════════════════════════════════════════════════════
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, seq_len: int,
                 d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attn  = AnomalyAttention(d_model, n_heads, seq_len, dropout)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        # Anomaly-Attention
        attn_out, S, P = self.attn(x)
        x = self.norm1(x + self.drop(attn_out))
        # Feed-Forward
        x = self.norm2(x + self.drop(self.ff(x)))
        return x, S, P


# ══════════════════════════════════════════════════════════════════════════════
# Anomaly Transformer (Full Model)
# ══════════════════════════════════════════════════════════════════════════════
class AnomalyTransformer(nn.Module):
    """
    Args:
        win_len  : 입력 윈도우 길이 (= seq_len)
        enc_in   : 입력 변수 수 (채널 수)
        d_model  : 임베딩 차원
        n_heads  : 어텐션 헤드 수
        n_layers : 인코더 레이어 수
        d_ff     : Feed-Forward 은닉 차원 (기본 4*d_model)
        dropout  : 드롭아웃
    """

    def __init__(self, win_len: int, enc_in: int,
                 d_model: int = 512, n_heads: int = 8,
                 n_layers: int = 3, d_ff: int = None,
                 dropout: float = 0.1):
        super().__init__()
        self.win_len  = win_len
        self.n_layers = n_layers

        # Input Embedding
        self.embed = nn.Linear(enc_in, d_model)
        self.pos_enc = self._build_pos_enc(win_len, d_model)

        # Encoder Layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, win_len, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Reconstruction Head
        self.decode = nn.Linear(d_model, enc_in)

    @staticmethod
    def _build_pos_enc(seq_len: int, d_model: int) -> torch.Tensor:
        pe  = torch.zeros(seq_len, d_model)
        pos = torch.arange(seq_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        return pe.unsqueeze(0)   # (1, L, d_model)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : (B, L, enc_in)
        Returns:
            recon      : (B, L, enc_in)    재구성 출력
            series_list: list[(B, L, L)]   각 레이어의 Series-Association
            prior_list : list[(L, L)]      각 레이어의 Prior-Association
        """
        # Embedding + Positional Encoding
        h = self.embed(x)
        if self.pos_enc.device != h.device:
            self.pos_enc = self.pos_enc.to(h.device)
        h = h + self.pos_enc

        series_list, prior_list = [], []
        for layer in self.layers:
            h, S, P = layer(h)
            series_list.append(S)
            prior_list.append(P)

        recon = self.decode(h)
        return recon, series_list, prior_list
