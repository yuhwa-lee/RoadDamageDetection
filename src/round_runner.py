"""
round_runner.py - 전처리 후보를 같은 조건에서 채점하는 실험 러너 (1라운드 / 2라운드 공용)

사용법 (baseline_v2.py, candidates.py와 같은 폴더에서)
    python round_runner.py noise                 # 노이즈 축 후보를 채점
    python round_runner.py light                 # 조도 축
    python round_runner.py blur                  # 블러 축
    python round_runner.py all                   # 세 축을 차례로
    python round_runner.py noise --overlay Japan_001959,China_MotorBike_002159
    python round_runner.py noise --limit 3       # 앞의 3장만 (빠른 동작 확인용)
    python round_runner.py "combo:bilateral_c50|clahe_c2|unsharp_s1_a08"    # 2라운드: 조합 채점

구조
    - 전처리 후보 정의는 candidates.py (각자 자기 축만 수정). 이 파일은 공통이라 건드리지 않는다.
    - 추출기는 baseline_v2.detect_baseline_mad (동결된 기본 설정 그대로).
    - 정답은 CVAT 세그멘테이션 마스크(원본 좌표). 검출 박스를 원본 좌표로 되돌려서 채점한다.
    - 점수: 인스턴스 재현율(정답 픽셀의 30% 이상이 박스에 덮이면 hit), chance(같은 크기 무작위 박스),
            lift = macro - chance (순위 기준). 영상을 복원 추출(bootstrap)해서 baseline 대비 향상폭의 범위도 낸다.

나중에 바뀌는 곳
    - 데이터 분할: train_data/(개발: 제공 13 + 촬영 9), test_data/(최종 평가: 촬영 10). 라벨은 labeled_data/ 공용.
      기본은 train만 불러온다. test는 파라미터 선정에 쓰지 않고, 최종 성능 보고 때만 --split test 로 돌린다.
      제공/촬영 구분(source)은 파일 이름으로 정한다 (제공 데이터는 나라 이름으로 시작).
    - 박스 단위 정밀도, CNR 같은 열은 summarize() / run_arm() 에 추가하면 된다.
"""
import os
import sys
import csv
import glob
import time
import argparse
import datetime
from pathlib import Path
import numpy as np
import cv2 as cv
from PIL import Image

from image_ROI_Resize import define_roi, resize_image_with_info
from resize_long_side import resize_long_side_with_info

BASE_DIR = Path.cwd()   
GT_DIR = BASE_DIR / "labeled_data"      # train/test 라벨이 한 폴더에 있다
SPLIT_DIRS = {"train": BASE_DIR / "train_data", "test": BASE_DIR / "test_data"}
IMG_EXTS = (".jpg", ".jpeg", ".png")
PROVIDED_PREFIX = ("China_", "India_", "Japan_", "Norway_", "United_States_", "Czech_")   # 제공(RDD) 데이터 이름


def source_of(name):
    return "provided" if name.startswith(PROVIDED_PREFIX) else "captured"

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
OUT_DIR = "out_round"


def read_labelmap(gt_dir):
    """labelmap.txt의 'crack:255,96,55::' 줄에서 클래스별 색을 읽는다 (색을 코드에 박아두지 않기 위해)."""
    col = {}
    with open(os.path.join(gt_dir, "labelmap.txt"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or ":" not in line:
                continue
            name, rgb = line.strip().split(":")[:2]
            if name in ("crack", "pothole"):
                col[name] = tuple(int(v) for v in rgb.split(","))
    if set(col) != {"crack", "pothole"}:
        raise ValueError("labelmap.txt에서 crack/pothole 색을 찾지 못함: %s" % col)
    return col


COL = read_labelmap(str(GT_DIR))
COVER_MIN = 0.3        # 정답 픽셀의 30% 이상이 박스에 덮이면 hit
N_RANDOM = 5           # chance: 같은 크기 무작위 박스를 뿌리는 횟수
BOX_GT_MIN = 0.1       # 박스 정밀도: 박스 면적의 10% 이상이 정답 픽셀이면 맞는 박스
N_BOOT = 300           # 영상 복원 추출 횟수
TIE_LIFT = 0.02        # lift 차이가 이 안이면 동점으로 보고 더 빠른 쪽
FAMILY_LIFT = 0.10     # 같은 계열이면 이 안에서 다른 계열을 우선 고려


# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------
def imread_bgr(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv.imdecode(data, cv.IMREAD_COLOR)


def imread_gray(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv.imdecode(data, cv.IMREAD_GRAYSCALE)


def find_image(img_dir, name):
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG"):
        p = os.path.join(img_dir, name + ext)
        if os.path.exists(p):
            return p
    return None


def load_gt_instances(gt_dir, name):
    """정답 인스턴스 목록. 마스크는 메모리를 아끼려고 바운딩 박스 크기로 잘라 저장한다."""
    cls = np.array(Image.open(os.path.join(gt_dir, "SegmentationClass", name + ".png")).convert("RGB"))
    obj = np.array(Image.open(os.path.join(gt_dir, "SegmentationObject", name + ".png")).convert("RGB"))
    out = []
    for typ, col in COL.items():
        cm = (cls == np.array(col)).all(-1)
        if not cm.any():
            continue
        for c in np.unique(obj[cm].reshape(-1, 3), axis=0):
            m = cm & (obj == c).all(-1)
            if not m.any():
                continue
            ys, xs = np.where(m)
            y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
            out.append({"type": typ, "x0": x0, "y0": y0, "crop": m[y0:y1, x0:x1].copy()})
    return out, cls.shape[:2]


RESIZERS = {
    "fixed": resize_image_with_info,      # 640x480 강제 리사이즈 (비율 변경, 작은 ROI는 확대)
    "long": resize_long_side_with_info,   # 긴 변이 640을 넘을 때만 비율 유지 축소
}
RESIZE_MODE = "long"    # main()의 --resize 옵션으로 바뀐다


def prepare_input(bgr, name):
    """원본 BGR에서 ROI를 찾고, ROI crop을 고정 크기로 resize한 뒤 그레이로 변환한다.

    meta에는 원본 좌표 복원을 위한 ROI 좌표와 resize 정보가 함께 들어간다.
    """
    roi_bgr, road_mask, roi_coords_original = define_roi(bgr)
    resized_roi_bgr, resize_info = RESIZERS[RESIZE_MODE](roi_bgr)
    resized_roi_gray = cv.cvtColor(resized_roi_bgr, cv.COLOR_BGR2GRAY)
    # 도로 마스크(자유 모양)를 ROI 박스로 잘라 처리 영상과 같은 크기로 맞춘다 (정식 추출기가 사용)
    x1, y1, x2, y2 = roi_coords_original
    h, w = resized_roi_gray.shape
    road = cv.resize(road_mask[y1:y2, x1:x2], (w, h), interpolation=cv.INTER_NEAREST) > 0
    return resized_roi_gray, {"roi_coords_original": roi_coords_original, "resize_info": resize_info,
                              "road_mask": road}


def load_items(limit=0, split="train"):
    """split: 'train'(기본, 개발용) / 'test'(최종 평가용) / 'all'. 라벨(labeled_data)이 있는 영상만 불러온다."""
    dirs = list(SPLIT_DIRS.values()) if split == "all" else [SPLIT_DIRS[split]]
    items = []
    for d in dirs:
        paths = sorted(p for p in glob.glob(os.path.join(str(d), "*")) if p.lower().endswith(IMG_EXTS))
        for path in paths:
            nm = os.path.splitext(os.path.basename(path))[0]
            if not os.path.exists(os.path.join(str(GT_DIR), "SegmentationClass", nm + ".png")):
                continue                     # 라벨 없는 영상(capture_pothole_01 등)은 채점할 수 없어 건너뛴다
            bgr = imread_bgr(path)
            inst, shape = load_gt_instances(str(GT_DIR), nm)
            if bgr.shape[:2] != tuple(shape):
                raise ValueError("영상과 라벨 크기가 다름: %s %s vs %s" % (nm, bgr.shape[:2], shape))
            img, meta = prepare_input(bgr, nm)
            items.append({"name": nm, "source": source_of(nm), "path": path, "img": img, "meta": meta,
                          "inst": inst, "shape": tuple(shape)})
    if limit:
        items = items[:limit]
    if not items:
        raise SystemExit("불러온 영상이 없습니다. train_data/ test_data/ 와 labeled_data/ 경로를 확인하세요.")
    return items


# ---------------------------------------------------------------------------
# 좌표 변환 / 채점
# ---------------------------------------------------------------------------
def to_original(boxes, meta):
    roi_x1, roi_y1, roi_x2, roi_y2 = meta["roi_coords_original"]
    resize_info = meta["resize_info"]
    scale_x = resize_info["scale_x"]
    scale_y = resize_info["scale_y"]

    out = []
    for x, y, w, h in boxes:
        ox = int(round(x / scale_x)) + roi_x1
        oy = int(round(y / scale_y)) + roi_y1
        ow = max(1, int(round(w / scale_x)))
        oh = max(1, int(round(h / scale_y)))
        out.append((max(roi_x1, min(roi_x2 - 1, ox)), max(roi_y1, min(roi_y2 - 1, oy)), ow, oh))
    return out


def to_processed(box, meta):
    roi_x1, roi_y1, _, _ = meta["roi_coords_original"]
    resize_info = meta["resize_info"]
    scale_x = resize_info["scale_x"]
    scale_y = resize_info["scale_y"]
    x, y, w, h = box
    return (int(round((x - roi_x1) * scale_x)), int(round((y - roi_y1) * scale_y)),
            max(1, int(round(w * scale_x))), max(1, int(round(h * scale_y))))


def covered_fraction(boxes, inst):
    """정답 인스턴스의 픽셀 중 박스에 덮인 비율."""
    crop, x0, y0 = inst["crop"], inst["x0"], inst["y0"]
    h, w = crop.shape
    cov = np.zeros(crop.shape, bool)
    for bx, by, bw, bh in boxes:
        ix0, iy0 = max(bx, x0), max(by, y0)
        ix1, iy1 = min(bx + bw, x0 + w), min(by + bh, y0 + h)
        if ix0 < ix1 and iy0 < iy1:
            cov[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = True
    return float((cov & crop).sum()) / float(crop.sum())


def random_like(boxes, shape, rng):
    """같은 크기의 박스를 무작위 위치에 뿌린다 (chance 계산용)."""
    H, W = shape
    out = []
    for x, y, w, h in boxes:
        w, h = min(w, W), min(h, H)
        out.append((int(rng.integers(0, W - w + 1)), int(rng.integers(0, H - h + 1)), w, h))
    return out


def box_cover_ratio(boxes, shape):
    m = np.zeros(shape, bool)
    H, W = shape
    for x, y, w, h in boxes:
        m[max(y, 0):min(y + h, H), max(x, 0):min(x + w, W)] = True
    return float(m.mean())


def box_gt_fraction(box, insts):
    """박스(원본 좌표) 면적 중 정답 픽셀(균열/포트홀 전체)이 차지하는 비율."""
    bx, by, bw, bh = box
    if bw <= 0 or bh <= 0:
        return 0.0
    n = 0
    for inst in insts:
        crop, x0, y0 = inst["crop"], inst["x0"], inst["y0"]
        h, w = crop.shape
        ix0, iy0 = max(bx, x0), max(by, y0)
        ix1, iy1 = min(bx + bw, x0 + w), min(by + bh, y0 + h)
        if ix0 < ix1 and iy0 < iy1:
            n += int(crop[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0].sum())
    return n / float(bw * bh)


def score_image(item, img_shape, boxes_proc, rng):
    """한 영상의 채점 결과(원본 좌표 기준 재현율 + chance)."""
    boxes = to_original(boxes_proc, item["meta"])
    rec = {"n_crack": 0, "hit_crack": 0, "n_pot": 0, "hit_pot": 0, "ch_crack": 0.0, "ch_pot": 0.0}
    rand_sets = [random_like(boxes, item["shape"], rng) for _ in range(N_RANDOM)] if boxes else []
    for inst in item["inst"]:
        k = "crack" if inst["type"] == "crack" else "pot"
        rec["n_" + k] += 1
        if boxes and covered_fraction(boxes, inst) >= COVER_MIN:
            rec["hit_" + k] += 1
        if rand_sets:
            rec["ch_" + k] += sum(covered_fraction(rb, inst) >= COVER_MIN for rb in rand_sets) / N_RANDOM
    # 박스 정밀도: 박스 면적의 BOX_GT_MIN 이상이 정답 픽셀이면 '맞는 박스'
    rec["n_box"] = len(boxes)
    rec["tp_box"] = sum(box_gt_fraction(b, item["inst"]) >= BOX_GT_MIN for b in boxes)
    rec["ch_tp_box"] = (sum(box_gt_fraction(b, item["inst"]) >= BOX_GT_MIN for rb in rand_sets for b in rb)
                        / N_RANDOM) if rand_sets else 0.0
    rec["n_det"] = len(boxes_proc)
    rec["cov"] = box_cover_ratio(boxes_proc, img_shape)
    return rec


# ---------------------------------------------------------------------------
# 전처리 적용 / 한 arm 실행
# ---------------------------------------------------------------------------
def apply_stages(stages, img, meta=None):
    """전처리 단계를 차례로 적용. 함수에 uses_meta 속성이 있으면 meta(도로 마스크 등)도 넘긴다."""
    out = img
    for fn in stages:
        out = fn(out, meta) if getattr(fn, "uses_meta", False) else fn(out)
        if out.dtype != np.uint8 or out.shape != img.shape:
            raise ValueError("전처리 함수는 입력과 같은 크기의 uint8 영상을 돌려줘야 합니다 (받은 값: %s %s)"
                             % (out.dtype, out.shape))
    return out


def run_arm(arm, items, detector, overlay_names=(), overlay_dir=None):
    recs = []
    for i, it in enumerate(items):
        rng = np.random.default_rng(1000 + i)
        t0 = time.perf_counter()
        img = apply_stages(arm["stages"], it["img"], it["meta"])
        t1 = time.perf_counter()
        boxes = detector(img, it["meta"])
        t2 = time.perf_counter()
        rec = score_image(it, img.shape, boxes, rng)
        rec.update({"name": it["name"], "source": it["source"], "t_pre": t1 - t0, "t_det": t2 - t1})
        recs.append(rec)
        if overlay_dir and it["name"] in overlay_names:
            save_overlay(overlay_dir, arm["name"], it, img, boxes)
    return recs


def save_overlay(out_dir, arm_name, item, img, boxes):
    os.makedirs(out_dir, exist_ok=True)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    t = max(1, round(max(img.shape) / 400))
    for inst in item["inst"]:                                  # 정답 박스: 초록
        h, w = inst["crop"].shape
        x, y, bw, bh = to_processed((inst["x0"], inst["y0"], w, h), item["meta"])
        cv.rectangle(vis, (x, y), (x + bw, y + bh), (0, 200, 0), t)
    for x, y, w, h in boxes:                                   # 후보 박스: 빨강
        cv.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), t)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in arm_name)
    ok = cv.imwrite(os.path.join(out_dir, "%s__%s.png" % (item["name"], safe)), vis)
    if not ok:
        print("  오버레이 저장 실패:", item["name"], arm_name)


# ---------------------------------------------------------------------------
# 집계
# ---------------------------------------------------------------------------
KEYS = ["n_crack", "hit_crack", "n_pot", "hit_pot", "ch_crack", "ch_pot", "n_box", "tp_box", "ch_tp_box"]


def summarize(s):
    """합계(dict) -> crack, pothole, macro, chance, lift. 해당 유형의 정답이 없으면 그 유형은 제외."""
    nan = float("nan")
    crack = pothole = nan
    parts, chs = [], []
    if s["n_crack"] > 0:
        crack = s["hit_crack"] / s["n_crack"]
        parts.append(crack)
        chs.append(s["ch_crack"] / s["n_crack"])
    if s["n_pot"] > 0:
        pothole = s["hit_pot"] / s["n_pot"]
        parts.append(pothole)
        chs.append(s["ch_pot"] / s["n_pot"])
    macro = float(np.mean(parts)) if parts else nan
    chance = float(np.mean(chs)) if chs else nan
    prec = s["tp_box"] / s["n_box"] if s["n_box"] > 0 else nan
    ch_prec = s["ch_tp_box"] / s["n_box"] if s["n_box"] > 0 else nan
    return {"crack": crack, "pothole": pothole, "macro": macro, "chance": chance, "lift": macro - chance,
            "prec": prec, "ch_prec": ch_prec}


def sum_recs(recs):
    return {k: float(sum(r[k] for r in recs)) for k in KEYS}


def boot_indices(n_img, n_boot=N_BOOT, seed=0):
    return np.random.default_rng(seed).integers(0, n_img, (n_boot, n_img))


def boot_lifts(recs, idx):
    A = np.array([[r[k] for k in KEYS] for r in recs], float)
    out = []
    for b in range(idx.shape[0]):
        out.append(summarize(dict(zip(KEYS, A[idx[b]].sum(axis=0))))["lift"])
    return np.array(out)


def build_rows(arms, all_recs, do_boot=True):
    """arm별 결과 행. baseline 대비 lift 향상폭의 95% 범위(paired bootstrap)를 같이 낸다."""
    n_img = len(all_recs[0])
    idx = boot_indices(n_img) if do_boot else None
    lifts = [boot_lifts(r, idx) for r in all_recs] if do_boot else [None] * len(arms)
    base_i = next((i for i, a in enumerate(arms) if a["baseline"]), None)
    rows = []
    for i, (arm, recs) in enumerate(zip(arms, all_recs)):
        sm = summarize(sum_recs(recs))
        row = {"name": arm["name"], "family": arm["family"], "baseline": arm["baseline"], **sm,
               "n_det": float(np.mean([r["n_det"] for r in recs])),
               "cov": float(np.mean([r["cov"] for r in recs])),
               "t_total": float(sum(r["t_pre"] + r["t_det"] for r in recs)),
               "d_lo": None, "d_hi": None, "by_source": {}}
        if do_boot and base_i is not None and i != base_i:
            d = lifts[i] - lifts[base_i]
            row["d_lo"], row["d_hi"] = float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))
        for src in sorted({r["source"] for r in recs}):
            row["by_source"][src] = summarize(sum_recs([r for r in recs if r["source"] == src]))
        rows.append(row)
    base_lift = rows[base_i]["lift"] if base_i is not None else None
    for r in rows:
        r["d_lift"] = None if base_lift is None or r["baseline"] else r["lift"] - base_lift
    return rows


# ---------------------------------------------------------------------------
# 상위 2개 선정
# ---------------------------------------------------------------------------
def pick_top2(rows):
    cand = sorted([r for r in rows if not r["baseline"]], key=lambda r: -r["lift"])
    if len(cand) < 2:
        return cand, ["후보가 2개 미만입니다."]
    notes = []
    top1 = cand[0]
    notes.append("1위: %s (lift %.2f)" % (top1["name"], top1["lift"]))
    rest = cand[1:]
    best = rest[0]["lift"]
    group = [r for r in rest if r["lift"] >= best - TIE_LIFT]
    second = min(group, key=lambda r: r["t_total"])
    if second is not rest[0]:
        notes.append("2위 후보 %s와 lift 차이가 %.2f 이내여서 더 빠른 %s 선택"
                     % (rest[0]["name"], TIE_LIFT, second["name"]))
    if second["family"] == top1["family"]:
        others = [r for r in rest if r["family"] != top1["family"] and r["lift"] >= second["lift"] - FAMILY_LIFT]
        if others:
            alt = sorted(others, key=lambda r: (-round(r["lift"] / TIE_LIFT), r["t_total"]))[0]
            notes.append("%s는 1위와 같은 계열(%s)이어서, lift 차이 %.2f 이내인 다른 계열 %s 선택"
                         % (second["name"], second["family"], FAMILY_LIFT, alt["name"]))
            second = alt
    notes.append("2위: %s (lift %.2f)" % (second["name"], second["lift"]))
    return [top1, second], notes


# ---------------------------------------------------------------------------
# 출력
# ---------------------------------------------------------------------------
def f2(v):
    return "  nan" if v is None or v != v else "%5.2f" % v


def print_table(title, rows, items):
    srcs = sorted({it["source"] for it in items})
    n_cr = sum(1 for it in items for i in it["inst"] if i["type"] == "crack")
    n_po = sum(1 for it in items for i in it["inst"] if i["type"] == "pothole")
    print("\n[%s]  영상 %d장 (%s), 정답: 균열 %d개 / 포트홀 %d개" % (title, len(items), ", ".join(srcs), n_cr, n_po))
    print("%-20s %-16s %5s %7s %5s %6s %5s  %-24s %5s %5s %5s %7s" %
          ("arm", "family", "crack", "pothole", "macro", "chance", "lift",
           "baseline 대비 [95% 범위]", "prec", "n_det", "cov", "sec"))
    for r in rows:
        if r["d_lift"] is None:
            delta = "(기준)" if r["baseline"] else ""
        elif r["d_lo"] is None:
            delta = "%+.2f" % r["d_lift"]
        else:
            mark = "*" if r["d_lo"] > 0 else " "
            delta = "%+.2f [%+.2f,%+.2f]%s" % (r["d_lift"], r["d_lo"], r["d_hi"], mark)
        print("%-20s %-16s %s %7s %s %6s %s  %-24s %s %5.0f %5.2f %7.1f" %
              (r["name"], r["family"], f2(r["crack"]), f2(r["pothole"]).strip().rjust(7), f2(r["macro"]),
               f2(r["chance"]).strip().rjust(6), f2(r["lift"]), delta, f2(r["prec"]), r["n_det"], r["cov"],
               r["t_total"]))
    print("  lift = macro - chance (순위 기준).  * = baseline보다 확실히 나음(95% 범위의 하한이 0 초과)")
    print("  prec = 박스 정밀도(박스 면적의 %d%% 이상이 정답인 박스 비율). 무작위 박스 기준 약 %s"
          % (int(BOX_GT_MIN * 100), f2(float(np.nanmean([r["ch_prec"] for r in rows]))).strip()))
    if len(srcs) > 1:
        print("\n  출처별 lift")
        print("  %-20s %s" % ("arm", "  ".join("%-10s" % s for s in srcs)))
        for r in rows:
            print("  %-20s %s" % (r["name"], "  ".join("%-10s" % f2(r["by_source"][s]["lift"]).strip() for s in srcs)))


def write_csv(axis, rows, all_recs):
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in axis)
    p1 = os.path.join(OUT_DIR, "%s_%s_summary.csv" % (safe, stamp))
    p2 = os.path.join(OUT_DIR, "%s_%s_per_image.csv" % (safe, stamp))
    with open(p1, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["arm", "family", "baseline", "crack", "pothole", "macro", "chance", "lift",
                    "d_lift", "d_lo", "d_hi", "prec", "ch_prec", "n_det", "cov", "sec"])
        for r in rows:
            w.writerow([r["name"], r["family"], int(r["baseline"]), r["crack"], r["pothole"], r["macro"],
                        r["chance"], r["lift"], r["d_lift"], r["d_lo"], r["d_hi"], r["prec"], r["ch_prec"],
                        r["n_det"], r["cov"],
                        r["t_total"]])
    with open(p2, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["arm", "image", "source"] + KEYS + ["n_det", "cov", "t_pre", "t_det"])
        for r, recs in zip(rows, all_recs):
            for x in recs:
                w.writerow([r["name"], x["name"], x["source"]] + [x[k] for k in KEYS] +
                           [x["n_det"], x["cov"], x["t_pre"], x["t_det"]])
    print("\n저장: %s\n      %s" % (p1, p2))


def run_axis(axis, arms, items, detector, overlay_names=(), do_boot=True):
    all_recs = []
    for arm in arms:
        print("  - %s ..." % arm["name"], end="", flush=True)
        t0 = time.perf_counter()
        all_recs.append(run_arm(arm, items, detector, overlay_names, os.path.join(OUT_DIR, axis + "_overlay")))
        print(" %.1fs" % (time.perf_counter() - t0))
    rows = build_rows(arms, all_recs, do_boot)
    print_table(axis, rows, items)
    return rows, all_recs


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def get_detector(name="baseline"):
    """baseline = 테스트 추출기(줄자, 전처리 비교용, 동결) / crack = 정식 균열 추출기(src/crack_extractor.py)."""
    if name == "crack":
        from src.crack_extractor import detect
        return lambda img, meta: detect(img, meta)
    from src.baseline_v2 import detect_baseline_mad            # 동결된 기본 설정 그대로 사용
    return lambda img, meta: detect_baseline_mad(img)


def main(argv=None):
    ap = argparse.ArgumentParser(description="전처리 후보 채점 러너")
    ap.add_argument("axis", help="noise | light | blur | all | combo:이름1|이름2|이름3")
    ap.add_argument("--overlay", default="", help="오버레이를 저장할 영상 이름(콤마로 구분)")
    ap.add_argument("--limit", type=int, default=0, help="앞의 N장만 사용(빠른 확인용)")
    ap.add_argument("--split", choices=["train", "test", "all"], default="train",
                    help="데이터 세트 (기본 train. test는 최종 평가 때만)")
    ap.add_argument("--no-boot", action="store_true", help="bootstrap 범위 계산 생략")
    ap.add_argument("--resize", choices=sorted(RESIZERS), default="long",
                    help="long = 긴 변 640 이하로 비율 유지 축소(기본), fixed = 640x480 강제 리사이즈")
    ap.add_argument("--detector", choices=["baseline", "crack"], default="baseline",
                    help="baseline = 테스트 추출기(기본, 전처리 비교용), crack = 정식 균열 추출기")
    args = ap.parse_args(argv)

    global RESIZE_MODE
    RESIZE_MODE = args.resize

    import src.candidates as candidates
    items = load_items(args.limit, args.split)
    detector = get_detector(args.detector)
    overlay = tuple(s for s in args.overlay.split(",") if s)

    if args.axis.startswith("combo:"):
        names = args.axis[len("combo:"):].split("|")
        runs = {args.axis: [candidates.IDENTITY, candidates.compose(names)]}
    elif args.axis == "all":
        runs = dict(candidates.AXES)
    elif args.axis in candidates.AXES:
        runs = {args.axis: candidates.AXES[args.axis]}
    else:
        raise SystemExit("알 수 없는 축: %s (noise | light | blur | all | combo:...)" % args.axis)

    for axis, arms in runs.items():
        if RESIZE_MODE != "long":
            axis = "%s_%s" % (axis, RESIZE_MODE)   # 표 제목/결과 파일 이름에 리사이즈 방식을 남긴다
        if args.detector != "baseline":
            axis = "%s_%s" % (axis, args.detector)  # 정식 추출기 결과는 이름에 추출기 이름(_crack)
        print("\n== %s 실행 ==" % axis)
        rows, all_recs = run_axis(axis, arms, items, detector, overlay, not args.no_boot)
        if not axis.startswith("combo:"):
            top, notes = pick_top2(rows)
            print("\n상위 2개 선정:")
            for n in notes:
                print("  -", n)
        write_csv(axis, rows, all_recs)


if __name__ == "__main__":
    main()
