"""
SMAP / MSL 데이터셋 다운로드 스크립트
--------------------------------------
실행:  python download_data.py

다운로드 후 data/ 폴더 구조:
  data/
    train/   (*.npy  — 정상 학습 데이터)
    test/    (*.npy  — 테스트 데이터, 이상 포함)
    labeled_anomalies.csv  (이상 구간 라벨)
"""

import os, urllib.request, zipfile, shutil, sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_URL = "https://s3-us-west-2.amazonaws.com/telemanom/data.zip"
LABEL_URL = (
    "https://raw.githubusercontent.com/khundman/telemanom"
    "/master/labeled_anomalies.csv"
)

def download(url, dest):
    print(f"  Downloading {os.path.basename(dest)} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  ✓ {os.path.basename(dest)}")
        return True
    except Exception as e:
        print(f"  ✗ 실패: {e}")
        return False

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1) 데이터 zip 다운로드
    zip_path = os.path.join(DATA_DIR, "data.zip")
    if not (os.path.exists(os.path.join(DATA_DIR, "train")) and
            os.path.exists(os.path.join(DATA_DIR, "test"))):
        print("[1/2] SMAP/MSL npy 파일 다운로드 중...")
        if download(DATA_URL, zip_path):
            print("  압축 해제 중...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(DATA_DIR)
            # 압축 해제 후 내부 data/ 폴더가 중첩될 수 있으므로 정리
            inner = os.path.join(DATA_DIR, "data")
            if os.path.isdir(inner):
                for item in os.listdir(inner):
                    shutil.move(os.path.join(inner, item),
                                os.path.join(DATA_DIR, item))
                os.rmdir(inner)
            os.remove(zip_path)
            print("  ✓ 압축 해제 완료")
        else:
            print("\n  ⚠ 자동 다운로드 실패. 수동으로 받는 방법:")
            print("    1. 브라우저에서 아래 URL 접속:")
            print(f"       {DATA_URL}")
            print(f"    2. data.zip을 data/ 폴더에 저장 후 압축 해제")
            sys.exit(1)
    else:
        print("[1/2] 데이터 파일 이미 존재 — 건너뜀")

    # 2) 라벨 CSV 다운로드
    label_path = os.path.join(DATA_DIR, "labeled_anomalies.csv")
    if not os.path.exists(label_path):
        print("[2/2] 라벨 파일 다운로드 중...")
        download(LABEL_URL, label_path)
    else:
        print("[2/2] 라벨 파일 이미 존재 — 건너뜀")

    # 3) 확인
    train_files = os.listdir(os.path.join(DATA_DIR, "train"))
    test_files  = os.listdir(os.path.join(DATA_DIR, "test"))
    smap = [f for f in train_files if f.startswith("P-") or f.startswith("S-")]
    msl  = [f for f in train_files if f.startswith("M-") or f.startswith("C-") or f.startswith("T-") or f.startswith("D-") or f.startswith("F-") or f.startswith("G-")]

    print(f"\n=== 다운로드 완료 ===")
    print(f"  학습 파일: {len(train_files)}개")
    print(f"  테스트 파일: {len(test_files)}개")
    print(f"  SMAP 채널: {len(smap)}개  MSL 채널: {len(msl)}개")
    print(f"\n다음 단계:  python train.py")

if __name__ == "__main__":
    main()
