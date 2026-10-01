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
    - prepare_input(): ROI 크롭 + 리사이즈가 준비되면 이 함수만 바꾼다 (지금은 원본 그대로 = A0).
    - DATASETS: 직접 촬영분 라벨이 오면 항목을 하나 더 추가한다 (source 별 집계도 자동으로 나온다).
    - 박스 단위 정밀도, CNR 같은 열은 summarize() / run_arm() 에 추가하면 된다.
"""
import os
import sys
import csv
import glob
import time
import argparse
import datetime
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
DATASETS = [
    {"source": "provided",
     "img_dir": r"C:\Users\kelly\Downloads\dataset\dataset",
     "gt_dir": r"C:\Users\kelly\OneDrive\문서\카카오톡 받은 파일\road_ground_truth_all"},
    # 직접 촬영분 라벨이 오면 아래처럼 추가:
    # {"source": "captured", "img_dir": r"...", "gt_dir": r"..."},
]
OUT_DIR = "out_round"
COL = {"crack": (255, 96, 55), "pothole": (42, 125, 209)}   # labelmap.txt (crak = crack 오타)
COVER_MIN = 0.3        # 정답 픽셀의 30% 이상이 박스에 덮이면 hit
N_RANDOM = 5           # chance: 같은 크기 무작위 박스를 뿌리는 횟수
N_BOOT = 300           # 영상 복원 추출 횟수
TIE_LIFT = 0.02        # lift 차이가 이 안이면 동점으로 보고 더 빠른 쪽
FAMILY_LIFT = 0.10     # 같은 계열이면 이 안에서 다른 계열을 우선 고려


# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------
def imread_gray(path):
    import cv2                                           # 필요할 때만 불러온다
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def find_image(img_dir, name):
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(img_dir, name + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("영상을 찾을 수 없음: %s (%s)" % (name, img_dir))


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


def prepare_input(gray, name):
    """입력 준비(ROI 크롭 + 리사이즈)가 들어갈 자리. 지금은 원본 그대로(A0).
    ROI/리사이즈를 붙일 때는 (처리된 영상, meta)를 돌려주면 된다.
      meta = {"x1": ROI 좌상단 x, "y1": ROI 좌상단 y, "scale": 리사이즈 배율}
      원본 좌표 = 처리본 좌표 / scale + (x1, y1)"""
    return gray, {"x1": 0, "y1": 0, "scale": 1.0}


def load_items(limit=0):
    items = []
    for ds in DATASETS:
        names = sorted(os.path.splitext(os.path.basename(p))[0]
                       for p in glob.glob(os.path.join(ds["gt_dir"], "SegmentationClass", "*.png")))
        for nm in names:
            gray = imread_gray(find_image(ds["img_dir"], nm))
            inst, shape = load_gt_instances(ds["gt_dir"], nm)
            if gray.shape != tuple(shape):
                raise ValueError("영상과 라벨 크기가 다름: %s %s vs %s" % (nm, gray.shape, shape))
            img, meta = prepare_input(gray, nm)
            items.append({"name": nm, "source": ds["source"], "img": img, "meta": meta,
                          "inst": inst, "shape": tuple(shape)})
    if limit:
        items = items[:limit]
    if not items:
        raise SystemExit("불러온 영상이 없습니다. DATASETS 경로를 확인하세요.")
    return items


# ---------------------------------------------------------------------------
# 좌표 변환 / 채점
# ---------------------------------------------------------------------------
def to_original(boxes, meta):
    s = meta["scale"]
    return [(int(round(x / s + meta["x1"])), int(round(y / s + meta["y1"])),
             max(1, int(round(w / s))), max(1, int(round(h / s)))) for x, y, w, h in boxes]


def to_processed(box, meta):
    s = meta["scale"]
    x, y, w, h = box
    return (int(round((x - meta["x1"]) * s)), int(round((y - meta["y1"]) * s)),
            max(1, int(round(w * s))), max(1, int(round(h * s))))


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
    rec["n_det"] = len(boxes_proc)
    rec["cov"] = box_cover_ratio(boxes_proc, img_shape)
    return rec


# ---------------------------------------------------------------------------
# 전처리 적용 / 한 arm 실행
# ---------------------------------------------------------------------------
def apply_stages(stages, img):
    out = img
    for fn in stages:
        out = fn(out)
        if out.dtype != np.uint8 or out.shape != img.shape:
            raise ValueError("전처리 함수는 입력과 같은 크기의 uint8 영상을 돌려줘야 합니다 (받은 값: %s %s)"
                             % (out.dtype, out.shape))
    return out


def run_arm(arm, items, detector, overlay_names=(), overlay_dir=None):
    recs = []
    for i, it in enumerate(items):
        rng = np.random.default_rng(1000 + i)
        t0 = time.perf_counter()
        img = apply_stages(arm["stages"], it["img"])
        t1 = time.perf_counter()
        boxes = detector(img)
        t2 = time.perf_counter()
        rec = score_image(it, img.shape, boxes, rng)
        rec.update({"name": it["name"], "source": it["source"], "t_pre": t1 - t0, "t_det": t2 - t1})
        recs.append(rec)
        if overlay_dir and it["name"] in overlay_names:
            save_overlay(overlay_dir, arm["name"], it, img, boxes)
    return recs


def save_overlay(out_dir, arm_name, item, img, boxes):
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    t = max(1, round(max(img.shape) / 400))
    for inst in item["inst"]:                                  # 정답 박스: 초록
        h, w = inst["crop"].shape
        x, y, bw, bh = to_processed((inst["x0"], inst["y0"], w, h), item["meta"])
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 200, 0), t)
    for x, y, w, h in boxes:                                   # 후보 박스: 빨강
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), t)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in arm_name)
    ok = cv2.imwrite(os.path.join(out_dir, "%s__%s.png" % (item["name"], safe)), vis)
    if not ok:
        print("  오버레이 저장 실패:", item["name"], arm_name)


# ---------------------------------------------------------------------------
# 집계
# ---------------------------------------------------------------------------
KEYS = ["n_crack", "hit_crack", "n_pot", "hit_pot", "ch_crack", "ch_pot"]


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
    return {"crack": crack, "pothole": pothole, "macro": macro, "chance": chance, "lift": macro - chance}


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
    print("%-20s %-16s %5s %7s %5s %6s %5s  %-24s %5s %5s %7s" %
          ("arm", "family", "crack", "pothole", "macro", "chance", "lift",
           "baseline 대비 [95% 범위]", "n_det", "cov", "sec"))
    for r in rows:
        if r["d_lift"] is None:
            delta = "(기준)" if r["baseline"] else ""
        elif r["d_lo"] is None:
            delta = "%+.2f" % r["d_lift"]
        else:
            mark = "*" if r["d_lo"] > 0 else " "
            delta = "%+.2f [%+.2f,%+.2f]%s" % (r["d_lift"], r["d_lo"], r["d_hi"], mark)
        print("%-20s %-16s %s %7s %s %6s %s  %-24s %5.0f %5.2f %7.1f" %
              (r["name"], r["family"], f2(r["crack"]), f2(r["pothole"]).strip().rjust(7), f2(r["macro"]),
               f2(r["chance"]).strip().rjust(6), f2(r["lift"]), delta, r["n_det"], r["cov"], r["t_total"]))
    print("  lift = macro - chance (순위 기준).  * = baseline보다 확실히 나음(95% 범위의 하한이 0 초과)")
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
                    "d_lift", "d_lo", "d_hi", "n_det", "cov", "sec"])
        for r in rows:
            w.writerow([r["name"], r["family"], int(r["baseline"]), r["crack"], r["pothole"], r["macro"],
                        r["chance"], r["lift"], r["d_lift"], r["d_lo"], r["d_hi"], r["n_det"], r["cov"],
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
def default_detector():
    from src.baseline_v2 import detect_baseline_mad            # 동결된 기본 설정 그대로 사용
    return lambda img: detect_baseline_mad(img)


def main(argv=None):
    ap = argparse.ArgumentParser(description="전처리 후보 채점 러너")
    ap.add_argument("axis", help="noise | light | blur | all | combo:이름1|이름2|이름3")
    ap.add_argument("--overlay", default="", help="오버레이를 저장할 영상 이름(콤마로 구분)")
    ap.add_argument("--limit", type=int, default=0, help="앞의 N장만 사용(빠른 확인용)")
    ap.add_argument("--no-boot", action="store_true", help="bootstrap 범위 계산 생략")
    args = ap.parse_args(argv)

    import src.candidates as candidates
    items = load_items(args.limit)
    detector = default_detector()
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
