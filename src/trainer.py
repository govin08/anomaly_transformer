"""
src/trainer.py
--------------
Anomaly Transformer 학습 / 평가 / 이상 점수 계산

핵심 구현:
  - Minimax 학습 전략 (논문 Section 3.2)
  - Association Discrepancy 기반 Anomaly Score (논문 수식 7)
  - 임계값 자동 탐색 (Percentile 방식)
"""

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ── KL Divergence ──────────────────────────────────────────────────────────
def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8):
    """KL(p || q), 마지막 차원 기준."""
    return (p * torch.log((p + eps) / (q + eps))).sum(dim=-1)


def association_discrepancy(
    series_list: list, prior_list: list
) -> torch.Tensor:
    """
    논문 수식 (5):
      AssDis(P, S; X) = (1/L) Σ_l [ KL(P^l || S^l) + KL(S^l || P^l) ] / 2

    Args:
        series_list : L개의 Series-Association, 각 (B, N, N)
        prior_list  : L개의 Prior-Association, 각 (N, N)
    Returns:
        AD : (B, N)  — 각 시점의 Association Discrepancy
    """
    n_layers = len(series_list)
    ad_sum = None
    for S, P in zip(series_list, prior_list):
        # P를 배치 차원으로 확장
        P_exp = P.unsqueeze(0).expand_as(S)   # (B, N, N)
        sym_kl = (kl_divergence(P_exp, S) + kl_divergence(S, P_exp)) / 2.0
        # sym_kl: (B, N)
        ad_sum = sym_kl if ad_sum is None else ad_sum + sym_kl
    return ad_sum / n_layers   # (B, N)


# ── Minimax Loss ───────────────────────────────────────────────────────────
def minimax_loss(
    series_list: list,
    prior_list: list,
    recon: torch.Tensor,
    x: torch.Tensor,
    lam: float = 3.0,
) -> tuple:
    """
    논문 수식 (6):
      L_total = ||X - X_hat||_F^2
               + λ * Σ [ KL(P_sg || S) + KL(S_sg || P) ]   ← minimize phase
               - λ * Σ [ KL(P || S_sg) + KL(S || P_sg) ]   ← maximize phase
      (sg = stop_gradient)

    실제로는 두 phase를 번갈아 수행하는 대신,
    단일 backward로 구현하기 위해:
      minimize term  : KL(P.detach() || S) → S를 P 방향으로
      maximize term  : -KL(P || S.detach()) → P를 S와 멀리

    Returns:
        loss_total, recon_loss, ad_loss_value
    """
    recon_loss = F.mse_loss(recon, x)

    ad_min = torch.tensor(0.0, device=x.device)
    ad_max = torch.tensor(0.0, device=x.device)

    for S, P in zip(series_list, prior_list):
        P_exp = P.unsqueeze(0).expand_as(S)

        # Minimize: S → P (P stop_grad)
        ad_min = ad_min + (
            kl_divergence(P_exp.detach(), S) +
            kl_divergence(S, P_exp.detach())
        ).mean()

        # Maximize: P → S (S stop_grad)
        ad_max = ad_max + (
            kl_divergence(P_exp, S.detach()) +
            kl_divergence(S.detach(), P_exp)
        ).mean()

    n = len(series_list)
    loss = recon_loss - lam * (ad_min / n) + lam * (ad_max / n)
    return loss, recon_loss, (ad_min / n)


# ── Trainer ────────────────────────────────────────────────────────────────
class Trainer:
    def __init__(self, model: nn.Module, device: str = "cpu",
                 lr: float = 1e-4, lam: float = 3.0):
        self.model  = model.to(device)
        self.device = device
        self.lam    = lam
        self.opt    = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt, step_size=5, gamma=0.7
        )
        self.history = {"loss": [], "recon": [], "ad": []}

    def train_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        total, r_total, ad_total, n = 0.0, 0.0, 0.0, 0
        for batch in loader:
            x = batch.to(self.device)
            self.opt.zero_grad()
            recon, series_list, prior_list = self.model(x)
            loss, rl, adl = minimax_loss(
                series_list, prior_list, recon, x, self.lam
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            bs = x.size(0)
            total   += loss.item() * bs
            r_total += rl.item()   * bs
            ad_total+= adl.item()  * bs
            n       += bs
        self.scheduler.step()
        return {"loss": total/n, "recon": r_total/n, "ad": ad_total/n}

    def fit(self, loader: DataLoader, epochs: int = 10,
            verbose: bool = True):
        print(f"{'Epoch':>6} {'Loss':>10} {'Recon':>10} {'AD':>10} {'Time':>8}")
        print("─" * 50)
        for ep in range(1, epochs + 1):
            t0 = time.time()
            metrics = self.train_epoch(loader)
            for k, v in metrics.items():
                self.history[k].append(v)
            if verbose:
                print(f"{ep:6d} {metrics['loss']:10.5f} "
                      f"{metrics['recon']:10.5f} "
                      f"{metrics['ad']:10.5f} "
                      f"{time.time()-t0:7.1f}s")

    # ── 추론: Anomaly Score 계산 ────────────────────────────────────────────
    @torch.no_grad()
    def score(self, loader: DataLoader, win_len: int, step: int,
              total_len: int) -> np.ndarray:
        """
        논문 수식 (7):
          AnomalyScore_t = Softmax(-AssDis_t) ⊙ ||x_t - x̂_t||²
        
        슬라이딩 윈도우 → 겹치는 구간은 평균 집계.
        """
        self.model.eval()
        score_buf = np.zeros(total_len, dtype=np.float64)
        count_buf = np.zeros(total_len, dtype=np.int32)

        idx = 0
        for batch in loader:
            x = batch.to(self.device)
            B, L, _ = x.shape
            recon, series_list, prior_list = self.model(x)

            # Association Discrepancy (B, L)
            ad = association_discrepancy(series_list, prior_list)

            # Recon error per timestep (B, L)
            re = ((x - recon) ** 2).mean(dim=-1)

            # Anomaly Score: Softmax(-AD) ⊙ recon_error
            ad_weight = torch.softmax(-ad, dim=-1)   # (B, L)
            sc = (ad_weight * re).cpu().numpy()       # (B, L)

            for b in range(B):
                start = idx * step
                end   = start + L
                if end > total_len:
                    end = total_len
                    sc[b] = sc[b][: end - start]
                score_buf[start:end] += sc[b][: end - start]
                count_buf[start:end] += 1
                idx += 1

        # 평균
        score_buf = np.where(count_buf > 0,
                             score_buf / count_buf,
                             score_buf)
        return score_buf


# ── Adjustment Strategy (논문 4장) ────────────────────────────────────────
def adjustment(pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    논문의 adjustment strategy:
    이상 구간 내에서 하나라도 탐지되면 해당 구간 전체를 탐지한 것으로 간주.
    실제 공정에서 하나의 알람이 울리면 전체 구간을 점검하는 것과 동일한 논리.
    """
    pred_adj = pred.copy()
    # 실제 이상 구간을 순회
    in_anom, start = False, 0
    for i, l in enumerate(labels):
        if l == 1 and not in_anom:
            start = i
            in_anom = True
        elif l == 0 and in_anom:
            # 해당 구간 내에 탐지된 시점이 하나라도 있으면 전체를 1로
            if pred[start:i].sum() > 0:
                pred_adj[start:i] = 1
            in_anom = False
    if in_anom and pred[start:].sum() > 0:
        pred_adj[start:] = 1
    return pred_adj


# ── 평가 지표 ──────────────────────────────────────────────────────────────
def evaluate(scores: np.ndarray, labels: np.ndarray,
             threshold_pct: float = 95.0,
             use_adjustment: bool = True) -> dict:
    """
    임계값 이상인 시점을 이상으로 판단 후 Precision / Recall / F1 계산.
    threshold_pct: 상위 몇 % 를 이상으로 판단할지
    use_adjustment: 논문의 adjustment strategy 적용 여부
    """
    thr  = np.percentile(scores, threshold_pct)
    pred = (scores >= thr).astype(int)

    if use_adjustment:
        pred = adjustment(pred, labels)

    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "threshold":  float(thr),
        "precision":  round(precision, 4),
        "recall":     round(recall, 4),
        "f1":         round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "anomaly_ratio": float(labels.mean()),
    }


def find_best_threshold(scores: np.ndarray, labels: np.ndarray,
                        pct_range=(80, 99), step=0.5,
                        use_adjustment: bool = True) -> dict:
    """F1을 최대화하는 임계값 백분위수 탐색."""
    best = {"f1": -1}
    pcts = np.arange(pct_range[0], pct_range[1] + step, step)
    for pct in pcts:
        m = evaluate(scores, labels, pct, use_adjustment)
        if m["f1"] > best["f1"]:
            best = {**m, "pct": pct}
    return best