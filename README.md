# Anomaly Transformer — 실행 가이드

논문: *Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy* (ICLR 2022)  
데이터: NASA SMAP / MSL (공개 데이터셋)

---

## 빠른 시작 (3단계)

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. 데이터 다운로드
python download_data.py

# 3. 학습 + 평가
python train.py --channel P-1 --epochs 10
```

결과는 `results/P-1_result.png` 에 저장됩니다.

---

## 데이터셋 설명

| 채널 접두사 | 데이터셋 | 변수 수 | 설명 |
|---|---|---|---|
| `P-`, `S-` | **SMAP** | 25 | NASA 토양수분 위성 원격측정 |
| `M-`, `C-`, `T-`, `D-`, `F-`, `G-` | **MSL** | 55 | NASA 화성 탐사 로버 |

- 학습: 정상 데이터만 포함 (`data/train/`)
- 테스트: 정상 + 이상 혼합 (`data/test/`)
- 라벨: `data/labeled_anomalies.csv` (이상 구간 시작/끝 인덱스)

---

## 파일 구조

```
anomaly_transformer_project/
├── download_data.py      ← 데이터 다운로드
├── train.py              ← 학습 + 평가 메인 스크립트
├── visualize.py          ← 저장된 결과 재시각화
├── requirements.txt
├── src/
│   ├── model.py          ← Anomaly Transformer 모델
│   ├── dataset.py        ← 데이터 로더
│   └── trainer.py        ← 학습 루프 + 손실 함수 + 평가
├── data/                 ← 다운로드 후 생성
│   ├── train/
│   ├── test/
│   └── labeled_anomalies.csv
└── results/              ← 실행 후 생성
    ├── P-1_result.png
    ├── P-1_result.json
    ├── P-1_model.pt
    ├── P-1_scores.npy
    └── P-1_labels.npy
```

---

## 주요 옵션

```bash
# GPU 사용
python train.py --channel P-1 --device cuda

# 모든 SMAP 채널
python train.py --dataset SMAP --epochs 10

# 더 강력한 모델
python train.py --channel P-1 --d_model 256 --n_layers 3 --epochs 20

# 빠른 테스트 (시각화 생략)
python train.py --channel P-1 --epochs 3 --no_plot

# 저장된 결과 다시 시각화
python visualize.py --channel P-1 --pct 95
```

---

## 핵심 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `--win_len` | 100 | 슬라이딩 윈도우 크기 |
| `--d_model` | 128 | Transformer 임베딩 차원 |
| `--n_heads` | 8 | Multi-head attention 헤드 수 |
| `--n_layers` | 3 | 인코더 레이어 수 |
| `--lam` | 3.0 | Minimax 손실 가중치 λ |
| `--epochs` | 10 | 학습 에폭 수 |
| `--lr` | 1e-4 | 학습률 |

---

## 논문 핵심 아이디어 (코드 대응)

| 논문 개념 | 구현 파일 | 함수/클래스 |
|---|---|---|
| Prior-Association (Gaussian Kernel) | `src/model.py` | `AnomalyAttention._prior_association()` |
| Series-Association (Self-Attention) | `src/model.py` | `AnomalyAttention.forward()` |
| Association Discrepancy (KL) | `src/trainer.py` | `association_discrepancy()` |
| Minimax 학습 전략 | `src/trainer.py` | `minimax_loss()` |
| Anomaly Score (수식 7) | `src/trainer.py` | `Trainer.score()` |

---

## 기대 성능 (참고)

논문 공식 결과 (F1 Score):

| 데이터셋 | 논문 F1 |
|---|---|
| SMAP | 88.31 |
| MSL | 90.24 |

이 구현은 단순화된 버전이므로 실제보다 낮을 수 있습니다.  
논문 수준의 성능을 원하면 공식 구현을 참고하세요:  
→ https://github.com/thuml/Anomaly-Transformer
