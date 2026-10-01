"""
sanity_check_real.py - 정답 라벨(CVAT 세그멘테이션 마스크)로 detect_baseline_mad를 채점한다.

사용법: baseline_v2.py와 같은 폴더에 두고
    python sanity_check_real.py            # 요약표
    python sanity_check_real.py detail     # 영상별 상세(전처리 없음 기준)

하는 일
  1. 라벨 로드: SegmentationClass(클래스 색) + SegmentationObject(인스턴스 색)로
     균열/포트홀 인스턴스 마스크를 만든다. 마스크는 원본 크기 = 원본 좌표계다.
  2. 세 가지 극단 전처리(없음 / 강한 가우시안 / Bilateral+CLAHE)를 원본 13장에 적용하고
     같은 추출기로 후보를 뽑는다.
  3. 인스턴스 재현율(정답 픽셀의 30% 이상이 후보 박스에 덮이면 hit), 픽셀 정밀도, f1px,
     화면 덮음 비율(cov)을 계산하고, 같은 크기의 무작위 박스(chance)와 비교한다.

판정 가이드
  - 세 전처리의 점수가 서로 벌어지는가(줄자가 전처리 차이를 구분하는가)
  - 추출기 점수가 chance(무작위 박스)보다 눈에 띄게 높은가
  라벨이 아직 원본 13장뿐이라 ROI/리사이즈 이전(A0) 기준이다.
"""
import os
import sys
import glob
import time
import cv2
import numpy as np
from PIL import Image

from src.baseline_v2 import detect_baseline_mad

IMG_DIR = r"C:\Users\kelly\Downloads\dataset\dataset"
GT_DIR = r"C:\Users\kelly\OneDrive\문서\카카오톡 받은 파일\road_ground_truth_all"
COL = {"crack": (255, 96, 55), "pothole": (42, 125, 209)}   # labelmap.txt (crak = crack 오타)
Z = 6.0
COVER_MIN = 0.3
N_RANDOM = 5
DETAIL = len(sys.argv) > 1 and sys.argv[1] == "detail"


# ---------------------------------------------------------------------------
# 라벨 / 영상 로드
# ---------------------------------------------------------------------------
def load_gt(name):
    """{'crack': [bool mask, ...], 'pothole': [bool mask, ...]}  인스턴스별 마스크."""
    cls = np.array(Image.open(os.path.join(GT_DIR, "SegmentationClass", name + ".png")).convert("RGB"))
    obj = np.array(Image.open(os.path.join(GT_DIR, "SegmentationObject", name + ".png")).convert("RGB"))
    inst = {"crack": [], "pothole": []}
    for typ, col in COL.items():
        cm = (cls == np.array(col)).all(-1)
        if not cm.any():
            continue
        for c in np.unique(obj[cm].reshape(-1, 3), axis=0):
            m = cm & (obj == c).all(-1)
            if m.any():
                inst[typ].append(m)
    return inst


def imread_gray(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def find_image(name):
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(IMG_DIR, name + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


# ---------------------------------------------------------------------------
# 전처리 (극단 세 가지)
# ---------------------------------------------------------------------------
def arm_none(g):
    return g


def arm_gauss(g):
    s = max(g.shape) / 640.0                        # 크기에 비례하는 강한 블러
    return cv2.GaussianBlur(g, (0, 0), 3 * s)


def arm_bilateral_clahe(g):
    b = cv2.bilateralFilter(g, 9, 75, 75)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(b)


ARMS = [("none", arm_none), ("gauss(strong)", arm_gauss), ("bilateral+CLAHE", arm_bilateral_clahe)]


# ---------------------------------------------------------------------------
# 채점
# ---------------------------------------------------------------------------
def box_mask(boxes, shape):
    m = np.zeros(shape, bool)
    for x, y, w, h in boxes:
        m[y:y + h, x:x + w] = True
    return m


def score_boxes(boxes, inst, shape):
    pred = box_mask(boxes, shape)
    hits = {"crack": [], "pothole": []}
    gt_all = np.zeros(shape, bool)
    for typ, ms in inst.items():
        for m in ms:
            hits[typ].append(float((pred & m).sum()) / m.sum() >= COVER_MIN)
            gt_all |= m
    return hits, int((pred & gt_all).sum()), int(pred.sum()), int(gt_all.sum())


def random_like(boxes, shape, rng):
    H, W = shape
    return [(int(rng.integers(0, max(W - w, 1))), int(rng.integers(0, max(H - h, 1))), w, h)
            for x, y, w, h in boxes]


def main():
    names = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(GT_DIR, "SegmentationClass", "*.png")))
    data = []
    n_cr = n_po = 0
    for nm in names:
        gray = imread_gray(find_image(nm))
        inst = load_gt(nm)
        assert inst["crack"] or inst["pothole"], nm
        assert all(m.shape == gray.shape for ms in inst.values() for m in ms), "라벨/영상 크기 불일치: " + nm
        data.append((nm, gray, inst))
        n_cr += len(inst["crack"])
        n_po += len(inst["pothole"])
    print("정답: 영상 %d장, 균열 인스턴스 %d개, 포트홀 인스턴스 %d개\n" % (len(data), n_cr, n_po))

    rows = []
    for arm_name, fn in ARMS:
        hit = {"crack": [], "pothole": []}
        chance = {"crack": [], "pothole": []}
        tp = pred_px = gt_px = 0
        covs, ndet = [], []
        t_det = 0.0
        rng = np.random.default_rng(0)
        for nm, gray, inst in data:
            img = fn(gray)
            t0 = time.perf_counter()
            boxes = detect_baseline_mad(img, z=Z)
            t_det += time.perf_counter() - t0
            h, t, p, g = score_boxes(boxes, inst, img.shape)
            for typ in hit:
                hit[typ] += h[typ]
            tp, pred_px, gt_px = tp + t, pred_px + p, gt_px + g
            covs.append(p / img.size)
            ndet.append(len(boxes))
            for _ in range(N_RANDOM):                      # 같은 크기 무작위 박스(우연 수준)
                hc, _, _, _ = score_boxes(random_like(boxes, img.shape, rng), inst, img.shape)
                for typ in chance:
                    chance[typ] += [v / N_RANDOM for v in hc[typ]]
            if DETAIL and arm_name == "none":
                print("  %-24s 균열 %d/%d  포트홀 %d/%d  박스 %d  cov %.2f" % (
                    nm, sum(h["crack"]), len(h["crack"]), sum(h["pothole"]), len(h["pothole"]),
                    len(boxes), p / img.size))
        m = lambda v: float(np.mean(v)) if len(v) else float("nan")
        cr, po = m(hit["crack"]), m(hit["pothole"])
        cc, cp = float(np.sum(chance["crack"]) / max(len(hit["crack"]), 1)), \
            float(np.sum(chance["pothole"]) / max(len(hit["pothole"]), 1))
        prec = tp / max(pred_px, 1)
        rec_px = tp / max(gt_px, 1)
        f1 = 2 * prec * rec_px / (prec + rec_px) if prec + rec_px > 0 else 0.0
        rows.append((arm_name, cr, po, (cr + po) / 2, (cc + cp) / 2, prec, f1,
                     float(np.mean(covs)), float(np.mean(ndet)), t_det))
    if DETAIL:
        print()

    print("%-18s %7s %8s %7s %8s %7s %6s %6s %6s %7s" %
          ("전처리", "crack", "pothole", "macro", "chance", "prec", "f1px", "cov", "n_det", "sec"))
    for r in rows:
        print("%-18s %7.2f %8.2f %7.2f %8.2f %7.2f %6.2f %6.2f %6.0f %7.1f" % r)
    print("\ncrack/pothole: 인스턴스 재현율(정답 픽셀의 %d%% 이상 덮임)  macro: 두 값의 평균" % int(COVER_MIN * 100))
    print("chance: 같은 크기 박스를 무작위로 뿌렸을 때의 macro 재현율  prec/f1px: 픽셀 단위")
    print("cov: 박스가 덮은 화면 비율  sec: 추출 시간 합계(13장)")


if __name__ == "__main__":
    main()
