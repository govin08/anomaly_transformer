"""
visualize.py
------------
저장된 Anomaly Score로 인터랙티브 시각화 생성

사용법:
  python visualize.py --channel P-1
  python visualize.py --channel P-1 --pct 95
"""

import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def plot(channel, save_dir, pct):
    score_path = os.path.join(save_dir, f"{channel}_scores.npy")
    label_path = os.path.join(save_dir, f"{channel}_labels.npy")
    hist_path  = os.path.join(save_dir, f"{channel}_train_history.npy")

    if not os.path.exists(score_path):
        print(f"결과 파일 없음: {score_path}")
        print("먼저 train.py를 실행하세요.")
        return

    scores = np.load(score_path)
    labels = np.load(label_path)
    history = np.load(hist_path) if os.path.exists(hist_path) else None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thr  = np.percentile(scores, pct)
    pred = (scores >= thr).astype(int)

    tp = int(((pred==1)&(labels==1)).sum())
    fp = int(((pred==1)&(labels==0)).sum())
    fn = int(((pred==0)&(labels==1)).sum())
    pr = tp / (tp+fp+1e-8)
    re = tp / (tp+fn+1e-8)
    f1 = 2*pr*re / (pr+re+1e-8)

    T   = len(scores)
    idx = np.arange(T)

    DARK, PANEL = "#0f0f0f", "#1a1a1a"
    GR, WH      = "#888888", "#e8e8e8"
    GRN, RED    = "#2dd4a4", "#f06060"
    AMB         = "#f5a623"

    fig, axes = plt.subplots(3, 1, figsize=(18, 11))
    fig.patch.set_facecolor(DARK)

    def sa(ax, t):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_color("#333")
        ax.tick_params(colors=GR, labelsize=9)
        ax.set_title(t, color=WH, fontsize=10, fontweight="bold")
        ax.xaxis.label.set_color(GR); ax.yaxis.label.set_color(GR)

    def shade(ax, lbl):
        in_a, s = False, 0
        for i, v in enumerate(lbl):
            if v==1 and not in_a: s=i; in_a=True
            elif v==0 and in_a: ax.axvspan(s,i,color=RED,alpha=0.2); in_a=False
        if in_a: ax.axvspan(s,len(lbl),color=RED,alpha=0.2)

    # 학습 곡선
    ax = axes[0]
    sa(ax, "학습 손실")
    if history is not None:
        ax.plot(history, color="#5b9cf6", lw=1.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Recon Loss")

    # Anomaly Score
    ax = axes[1]
    sa(ax, f"Anomaly Score  |  임계값: {pct}%ile")
    shade(ax, labels)
    ax.plot(idx, scores, color=GRN, lw=1.0, alpha=0.9)
    ax.axhline(thr, color=AMB, ls="--", lw=1.2, alpha=0.8, label=f"threshold={thr:.5f}")
    ax.fill_between(idx, 0, scores, where=(scores>=thr), color=RED, alpha=0.3)
    ax.legend(fontsize=8, facecolor=PANEL, labelcolor=WH, framealpha=0.5)
    ax.set_xlim(0, T)

    # 탐지 vs 실제
    ax = axes[2]
    sa(ax, f"탐지 결과  F1={f1:.3f}  Precision={pr:.3f}  Recall={re:.3f}")
    ax.plot(idx, labels,     color=RED, lw=1.0, alpha=0.6, label="실제 이상")
    ax.plot(idx, pred*0.95,  color=GRN, lw=1.0, alpha=0.8, label="탐지 결과", ls="--")
    ax.set_ylim(-0.1, 1.2); ax.set_xlim(0, T)
    ax.legend(fontsize=9, facecolor=PANEL, labelcolor=WH, framealpha=0.5)

    fig.suptitle(f"Anomaly Transformer — {channel}",
                 color=WH, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.96])

    out = os.path.join(save_dir, f"{channel}_viz_{pct}pct.png")
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"저장: {out}")
    print(f"F1={f1:.4f}  Precision={pr:.4f}  Recall={re:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--channel",  default="P-1")
    p.add_argument("--save_dir", default="results")
    p.add_argument("--pct",      type=float, default=95.0)
    args = p.parse_args()
    plot(args.channel, args.save_dir, args.pct)


if __name__ == "__main__":
    main()
