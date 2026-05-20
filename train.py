"""
train.py
--------
Anomaly Transformer 학습 + 평가 실행 스크립트

사용법:
  python train.py                          # 기본값으로 실행
  python train.py --channel P-1            # 특정 채널만
  python train.py --dataset MSL            # MSL 데이터셋
  python train.py --epochs 20 --d_model 256
  python train.py --channel P-1 --no_plot  # 시각화 없이 빠르게
"""

import os, sys, argparse, json
import numpy as np
import torch
from torch.utils.data import DataLoader

# 프로젝트 내부 모듈
sys.path.insert(0, os.path.dirname(__file__))
from src.model   import AnomalyTransformer
from src.dataset import load_channel, normalize, SlidingWindowDataset, list_channels
from src.trainer import Trainer, evaluate, find_best_threshold


# ── 인자 파싱 ──────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description="Anomaly Transformer 학습/평가")
    p.add_argument("--data_dir",  default="data",   help="데이터 루트 폴더")
    p.add_argument("--dataset",   default="SMAP",   choices=["SMAP","MSL","all"])
    p.add_argument("--channel",   default=None,     help="특정 채널 ID (예: P-1)")
    p.add_argument("--win_len",   type=int, default=100,  help="슬라이딩 윈도우 길이")
    p.add_argument("--step",      type=int, default=1,    help="슬라이딩 스텝")
    p.add_argument("--d_model",   type=int, default=128,  help="임베딩 차원")
    p.add_argument("--n_heads",   type=int, default=8,    help="어텐션 헤드 수")
    p.add_argument("--n_layers",  type=int, default=3,    help="인코더 레이어 수")
    p.add_argument("--d_ff",      type=int, default=None, help="FF 차원 (기본 4*d_model)")
    p.add_argument("--dropout",   type=float, default=0.1)
    p.add_argument("--epochs",    type=int, default=10,   help="학습 에폭")
    p.add_argument("--batch_size",type=int, default=64)
    p.add_argument("--lr",        type=float, default=1e-4)
    p.add_argument("--lam",       type=float, default=3.0, help="Minimax 가중치 λ")
    p.add_argument("--threshold_pct", type=float, default=None,
                   help="이상 판단 임계값 백분위수 (None이면 자동 탐색)")
    p.add_argument("--save_dir",  default="results", help="결과 저장 폴더")
    p.add_argument("--no_plot",   action="store_true", help="시각화 생략")
    p.add_argument("--device",    default=None,
                   help="'cuda' | 'mps' | 'cpu' (기본: 자동감지)")
    return p.parse_args()


# ── 디바이스 자동 감지 ─────────────────────────────────────────────────────
def get_device(preference=None):
    if preference:
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── 단일 채널 학습 + 평가 ─────────────────────────────────────────────────
def run_channel(channel: str, args, device: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  채널: {channel}  |  device: {device}")
    print(f"{'='*60}")

    # 데이터 로드 & 정규화
    train_raw, test_raw, labels = load_channel(args.data_dir, channel)
    train_norm, test_norm = normalize(train_raw, test_raw)

    n_feat = train_norm.shape[1]
    print(f"  Train: {train_norm.shape}  Test: {test_norm.shape}  "
          f"이상 비율: {labels.mean()*100:.1f}%")

    # 데이터로더
    train_ds = SlidingWindowDataset(train_norm, args.win_len, step=args.step)
    test_ds  = SlidingWindowDataset(test_norm,  args.win_len, step=args.step)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    # 모델
    model = AnomalyTransformer(
        win_len  = args.win_len,
        enc_in   = n_feat,
        d_model  = args.d_model,
        n_heads  = args.n_heads,
        n_layers = args.n_layers,
        d_ff     = args.d_ff,
        dropout  = args.dropout,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  모델 파라미터: {n_params:,}")

    # 학습
    trainer = Trainer(model, device=device, lr=args.lr, lam=args.lam)
    trainer.fit(train_loader, epochs=args.epochs, verbose=True)

    # 이상 점수 계산
    print("\n  Anomaly Score 계산 중...")
    scores = trainer.score(test_loader,
                           win_len=args.win_len,
                           step=args.step,
                           total_len=len(test_norm))

    # 평가
    if args.threshold_pct is not None:
        metrics = evaluate(scores, labels, args.threshold_pct)
        metrics["pct"] = args.threshold_pct
    else:
        print("  최적 임계값 탐색 중...")
        metrics = find_best_threshold(scores, labels)

    print(f"\n  ── 평가 결과 ──────────────────────────────")
    print(f"  임계값 백분위 : {metrics.get('pct', '?'):.1f}%")
    print(f"  Precision     : {metrics['precision']:.4f}")
    print(f"  Recall        : {metrics['recall']:.4f}")
    print(f"  F1 Score      : {metrics['f1']:.4f}")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}")

    # 결과 저장
    os.makedirs(args.save_dir, exist_ok=True)
    result = {
        "channel":  channel,
        "n_feat":   n_feat,
        "n_params": n_params,
        "metrics":  metrics,
        "args": vars(args),
    }
    result_path = os.path.join(args.save_dir, f"{channel}_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    # 모델 저장
    model_path = os.path.join(args.save_dir, f"{channel}_model.pt")
    torch.save(model.state_dict(), model_path)

    # Anomaly Score 저장 (시각화용)
    np.save(os.path.join(args.save_dir, f"{channel}_scores.npy"), scores)
    np.save(os.path.join(args.save_dir, f"{channel}_labels.npy"), labels)
    np.save(os.path.join(args.save_dir, f"{channel}_train_history.npy"),
            np.array(trainer.history["recon"]))

    # 시각화
    if not args.no_plot:
        plot_results(channel, test_norm, scores, labels, metrics,
                     trainer.history, args.save_dir)

    return result


# ── 시각화 ────────────────────────────────────────────────────────────────
def plot_results(channel, test_data, scores, labels, metrics,
                 history, save_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # Windows 기본 한글 폰트
        matplotlib.rcParams['axes.unicode_minus'] = False      # 마이너스 기호 깨짐 방지


    except ImportError:
        print("  matplotlib 없음 — 시각화 생략")
        return

    DARK, PANEL = "#0f0f0f", "#1a1a1a"
    GR, WH      = "#888888", "#e8e8e8"
    GRN, RED    = "#2dd4a4", "#f06060"
    AMB, BLU    = "#f5a623", "#5b9cf6"
    PUR         = "#b48ef5"

    fig = plt.figure(figsize=(18, 16))
    fig.patch.set_facecolor(DARK)
    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.45, wspace=0.28,
                           left=0.07, right=0.96,
                           top=0.93, bottom=0.05)

    def sa(ax, t):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_color("#333")
        ax.tick_params(colors=GR, labelsize=9)
        ax.set_title(t, color=WH, fontsize=10, fontweight="bold", pad=6)
        ax.xaxis.label.set_color(GR); ax.yaxis.label.set_color(GR)

    def shade(ax, lbl, a=0.2):
        in_a, s = False, 0
        for i, v in enumerate(lbl):
            if v==1 and not in_a: s=i; in_a=True
            elif v==0 and in_a:
                ax.axvspan(s, i, color=RED, alpha=a); in_a=False
        if in_a: ax.axvspan(s, len(lbl), color=RED, alpha=a)

    T = len(scores)
    idx = np.arange(T)

    # 1) 첫 번째 채널 시계열
    ax = fig.add_subplot(gs[0, :])
    sa(ax, f"[{channel}] 테스트 데이터 — 빨간 영역: 이상 구간")
    shade(ax, labels)
    ax.plot(idx, test_data[:T, 0], color=WH, lw=0.8, alpha=0.8, label="ch-0")
    if test_data.shape[1] > 1:
        ax.plot(idx, test_data[:T, 1], color=BLU, lw=0.8, alpha=0.6, label="ch-1")
    ax.legend(fontsize=8, facecolor=PANEL, labelcolor=WH, framealpha=0.5)
    ax.set_xlim(0, T)

    # 2) Anomaly Score
    ax = fig.add_subplot(gs[1, :])
    sa(ax, "Anomaly Score")
    shade(ax, labels)
    thr = metrics["threshold"]
    ax.plot(idx, scores, color=GRN, lw=1.0, alpha=0.9, label="Anomaly Score")
    ax.axhline(thr, color=AMB, ls="--", lw=1.2, alpha=0.8, label=f"임계값 ({metrics.get('pct','?'):.0f}%ile)")
    ax.fill_between(idx, 0, scores,
                    where=(scores >= thr),
                    color=RED, alpha=0.35, label="탐지됨")
    ax.legend(fontsize=9, facecolor=PANEL, labelcolor=WH, framealpha=0.5)
    ax.set_xlim(0, T)

    # 3) 학습 손실
    ax = fig.add_subplot(gs[2, 0])
    sa(ax, "학습 손실 (Reconstruction)")
    ep = np.arange(1, len(history["recon"]) + 1)
    ax.plot(ep, history["recon"], color=BLU, lw=1.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")

    # 4) Precision-Recall 바 차트
    ax = fig.add_subplot(gs[2, 1])
    sa(ax, "평가 지표")
    bars = ax.bar(["Precision", "Recall", "F1"],
                  [metrics["precision"], metrics["recall"], metrics["f1"]],
                  color=[BLU, PUR, GRN], width=0.5)
    for bar, v in zip(bars, [metrics["precision"], metrics["recall"], metrics["f1"]]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", color=WH, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score")

    # 5) 탐지 결과 확대 (이상 구간 1)
    thr_arr = (scores >= thr).astype(int)
    anom_starts = np.where(np.diff(np.pad(labels, 1)) == 1)[0]
    if len(anom_starts) > 0:
        s0 = max(0, anom_starts[0] - 50)
        e0 = min(T, anom_starts[0] + 150)
        ax = fig.add_subplot(gs[3, :])
        sa(ax, f"확대: 첫 번째 이상 구간 근처 (t={anom_starts[0]})")
        shade(ax, labels[s0:e0], a=0.25)
        ax_idx = np.arange(s0, e0)
        ax.plot(ax_idx, scores[s0:e0], color=GRN, lw=1.5, label="Anomaly Score")
        ax.axhline(thr, color=AMB, ls="--", lw=1.0, alpha=0.7)
        detect_idx = ax_idx[scores[s0:e0] >= thr]
        if len(detect_idx):
            ax.scatter(detect_idx,
                       scores[detect_idx],
                       color=RED, s=18, zorder=5, alpha=0.85, label="탐지")
        ax.legend(fontsize=9, facecolor=PANEL, labelcolor=WH, framealpha=0.5)
        ax.set_xlim(s0, e0)

    fig.suptitle(
        f"Anomaly Transformer  |  채널: {channel}  |  "
        f"F1: {metrics['f1']:.3f}  Precision: {metrics['precision']:.3f}  Recall: {metrics['recall']:.3f}",
        color=WH, fontsize=12, fontweight="bold"
    )
    out = os.path.join(save_dir, f"{channel}_result.png")
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"  시각화 저장: {out}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    args   = get_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    # 실행할 채널 목록
    if args.channel:
        channels = [args.channel]
    else:
        channels = list_channels(args.data_dir, args.dataset)
        print(f"채널 {len(channels)}개 발견: {channels[:5]} ...")

    all_results = []
    for ch in channels:
        try:
            res = run_channel(ch, args, device)
            all_results.append(res)
        except Exception as e:
            print(f"  ✗ {ch} 오류: {e}")

    # 전체 요약
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"  전체 요약 ({len(all_results)}채널)")
        print(f"{'='*60}")
        f1s = [r["metrics"]["f1"] for r in all_results]
        prs = [r["metrics"]["precision"] for r in all_results]
        res_vals = [r["metrics"]["recall"] for r in all_results]
        print(f"  평균 F1        : {np.mean(f1s):.4f}")
        print(f"  평균 Precision : {np.mean(prs):.4f}")
        print(f"  평균 Recall    : {np.mean(res_vals):.4f}")
        print(f"\n  결과 저장 위치: {os.path.abspath(args.save_dir)}/")


if __name__ == "__main__":
    main()
