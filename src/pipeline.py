"""통합 실행 프로그램: 저품질 도로 영상 -> 전처리 -> 균열·포트홀 후보 추출 -> 시각화 + 전처리 전·후 정량 비교.

실행
    python src/pipeline.py train_data/China_MotorBike_002159.jpg       # 사진 한 장
    python src/pipeline.py train_data                                   # 폴더 (하위 폴더 포함)
    python src/pipeline.py train_data --no-vis                          # 수치만
    python src/pipeline.py train_data/IMG_1815.JPG --show crack         # 그림에 균열만 표시 (수치·CSV는 동일)
    python src/pipeline.py test_data --out out_pipeline/test_final      # 최종 평가 때 1회만 (시연 그림도 이때 저장)

처리 흐름 (사진마다 '전처리 전'과 '전처리 후'를 모두 실행)
    원본 -> ROI(도로 마스크) -> 긴 변 640 -> 회색조
      ├─ 균열:   [전: 없음 | 후: CRACK_PRE]   -> crack_extractor.extract_cracks
      └─ 포트홀: [전: 없음 | 후: POTHOLE_PRE] -> pothole_edge_extractor.detect_potholes (Edge 기본값)
    (전처리 단계는 회색조 영상 위에서 적용. 두 경로가 같은 입력에서 따로 전처리한다)
    ROI·resize·추출기는 채점기(crack_eval / pothole_eval)와 같은 함수를 그대로 쓴다.

전처리 설정 (근거: docs/pothole_experiments.md, crack_extractor/candidates docstring)
    CRACK_PRE   = adapt_gauss_s2  (train 22장, F2 0.399 -> 0.472, 노이즈·조도·흐림 축과 조합·순서 비교에서 최종)
    POTHOLE_PRE = laplacian_a05+adapt_gauss_s2  (샤프닝 후 적응 평활화. 포트홀 13장 + 음성 14장에서 규칙 ①~④ 유일 통과:
                  IoU 0.072 -> 0.125, 음성 오검출 1.63% -> 1.31%. docs/pothole_experiments.md 5절)

출력 (--out 폴더, 기본 out_pipeline/<시각>/ ; out_* 는 git 제외)
    vis/<이름>.jpg     | 입력(ROI 윤곽 + 정답) | 전처리 전 결과 | 전처리 후 결과 |
                       균열 = 빨강 마스크, 포트홀 = 파랑 영역 + 박스, 균열 정답 = 초록 윤곽, 포트홀 정답 = 자홍 윤곽,
                       도로 마스크 = 노랑 윤곽
    per_image.csv      사진 x 조건(before/after)마다 한 줄
    summary.csv        출처(제공/촬영/rdd) x 조건별 평균

정량 지표 (과제 안내서 '전처리 전·후 결과의 정량적 비교' 항목)
    edges        도로 안 Canny 에지 픽셀 수 (균열 경로 입력 영상 기준. pot_edges는 포트홀 경로)
                 임계값: '전처리 전' 회색조의 도로 안 기울기 크기(Sobel 3x3, L2) 90번째 백분위 = hi, lo = 0.5*hi.
                 같은 값을 전·후에 똑같이 쓴다 -> '전처리 전 가장 강한 변화 10% 수준의 에지가 전처리 후 얼마나 남나'.
                 (사진마다 대비가 달라 고정값은 부적절, 전·후에 다른 값을 쓰면 비교가 안 됨.
                  처음 쓴 '밝기 중앙값 x0.67/x1.33' 규칙은 밝기 단위라 밝은 사진에서 임계값이 기울기 범위를
                  벗어남: China_002159 중앙값 206 -> hi 274 > 기울기 95백분위 110, 에지 250개 -> 폐기)
    keypoints    도로 안 ORB 키포인트 수 (nfeatures=5000: 상한에 걸려 전·후 차이가 가려지지 않게 넉넉히)
    crack_n / crack_area%     균열 후보(연결 조각) 수 / 도로 면적 대비 %
    pot_n / pot_area%         포트홀 후보 수 / 도로 면적 대비 %
    roi_ms, *_pre_ms, *_ext_ms  단계별 처리 시간
    정확도(정답 라벨이 있을 때만, 채점기와 같은 방식):
        crack_F2/P/R  골격 허용오차 d=5(640 기준), 포트홀 주변 제외 (crack_eval.score_image)
                      summary.csv: crack_F2 = 사진별 F2 평균, crack_F2_micro = 사진 합산 후 F2 (crack_eval 보고값과 같은 방식)
                      포트홀만 라벨링한 영상(rdd_box)은 균열 정답이 없으므로 계산하지 않음
        pot_IoU       640 처리 좌표 픽셀 IoU (pothole_axis_eval과 같은 캔버스). 포트홀 정답이 있는 사진만
    주의: 평활화 후 에지·특징점이 줄어드는 것은 노이즈 감소일 수도, 균열 에지 손실일 수도 있다.
          그래서 개수는 정확도(F2, IoU)와 함께 읽는다.
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

import round_runner as rr                       # ROI·resize·정답 로더 (균열 채점기와 동일)
import crack_eval as CEV                        # 균열 영역 채점 (score_image)
from candidates import AXES, IDENTITY, compose
from crack_extractor import extract_cracks
from pothole_edge_extractor import EdgePotholeConfig, detect_potholes

CRACK_PRE = "adapt_gauss_s2"
POTHOLE_PRE = "laplacian_a05+adapt_gauss_s2"
ORB_N = 5000
CANNY_PCT, CANNY_RATIO = 90, 0.5

C_CRACK = (40, 40, 255)      # BGR 빨강
C_POT = (255, 120, 0)        # 파랑
C_GT = (0, 220, 0)           # 초록 (균열 정답)
C_GT_POT = (255, 0, 255)     # 자홍 (포트홀 정답)
C_ROAD = (0, 220, 255)       # 노랑


# ---------------------------------------------------------------------------
# 준비
# ---------------------------------------------------------------------------
def arm_of(name):
    if name in ("", "none"):
        return IDENTITY
    return compose(name.split("+"))


def imread(path):
    img = cv.imdecode(np.fromfile(str(path), np.uint8), cv.IMREAD_COLOR)
    if img is None:
        raise ValueError("읽을 수 없는 영상: %s" % path)
    return img


def prepare(bgr):
    """rr.prepare_input과 같은 처리. 시각화를 위해 컬러 ROI도 함께 돌려준다."""
    roi_bgr, road_mask, coords = rr.define_roi(bgr)
    color, info = rr.RESIZERS["long"](roi_bgr)
    gray = cv.cvtColor(color, cv.COLOR_BGR2GRAY)
    x1, y1, x2, y2 = coords
    h, w = gray.shape
    road = cv.resize(road_mask[y1:y2, x1:x2], (w, h), interpolation=cv.INTER_NEAREST) > 0
    return color, gray, {"roi_coords_original": coords, "resize_info": info, "road_mask": road}


def source_of(path):
    if path.parent.name == "rdd_box":
        return "rdd"
    return rr.source_of(path.stem)


def load_truth(path, meta, gray_shape, pothole_only):
    """정답이 있으면 (균열 채점용 item, 포트홀 640 마스크, 정답 윤곽용 마스크들)을 돌려준다. 없으면 None."""
    lp = rr.GT_DIR / "SegmentationClass" / (path.stem + ".png")
    if not lp.is_file():
        return None
    x1, y1, x2, y2 = meta["roi_coords_original"]
    h, w = gray_shape
    cls = np.array(Image.open(lp).convert("RGB"))

    def to_proc(col):
        m = np.all(cls == np.array(col), -1)[y1:y2, x1:x2].astype(np.uint8)
        return cv.resize(m, (w, h), interpolation=cv.INTER_NEAREST) > 0

    truth = {"pot": to_proc(rr.COL["pothole"]), "crack": None, "crack_item": None}
    if not pothole_only:
        truth["crack"] = to_proc(rr.COL["crack"])
        if (rr.GT_DIR / "SegmentationObject" / (path.stem + ".png")).is_file():
            inst, shape = rr.load_gt_instances(str(rr.GT_DIR), path.stem)
            truth["crack_item"] = {"name": path.stem, "source": source_of(path), "meta": meta,
                                   "inst": inst, "shape": tuple(shape)}
    return truth


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------
def canny_thresholds(gray, road):
    gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3); gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    hi = float(np.percentile(mag[road] if road.any() else mag, CANNY_PCT))
    return CANNY_RATIO * hi, hi


def count_edges(gray, road, th):
    e = cv.Canny(gray, th[0], th[1], L2gradient=True) > 0
    return int((e & road).sum())


_ORB = None


def count_keypoints(gray, road):
    global _ORB
    if _ORB is None:
        _ORB = cv.ORB_create(nfeatures=ORB_N)
    return len(_ORB.detect(gray, road.astype(np.uint8) * 255))


def crack_scores(item, mask):
    rec, _, _ = CEV.score_image(item, mask)
    d = CEV.D_REF
    cnt = (rec["tp_p_%d" % d], rec["np_%d" % d], rec["tp_r_%d" % d], rec["n_gt_skel"])
    p, r, _ = CEV.prf(*cnt)
    return p, r, CEV.fbeta(p, r, 2), cnt


def n_components(mask):
    return int(cv.connectedComponents(mask.astype(np.uint8))[0] - 1)


# ---------------------------------------------------------------------------
# 한 장 처리
# ---------------------------------------------------------------------------
def run_branch(gray, meta, arm, extractor):
    t0 = time.perf_counter()
    g = rr.apply_stages(arm["stages"], gray, meta)
    t1 = time.perf_counter()
    mask, boxes = extractor(g, meta["road_mask"])
    t2 = time.perf_counter()
    return g, mask, boxes, 1000 * (t1 - t0), 1000 * (t2 - t1)


def crack_extractor(g, road):
    mask, boxes, _ = extract_cracks(g, road)
    return mask, boxes


def make_pothole_extractor(cfg):
    def f(g, road):
        boxes, dbg = detect_potholes(g, road, cfg, return_debug=True)
        return dbg["candidate_mask"] > 0, boxes
    return f


def process(path, arms, pot_cfg, pothole_only):
    bgr = imread(path)
    t0 = time.perf_counter()
    color, gray, meta = prepare(bgr)
    roi_ms = 1000 * (time.perf_counter() - t0)
    road = meta["road_mask"]
    road_px = max(1, int(road.sum()))
    th = canny_thresholds(gray, road)
    truth = load_truth(path, meta, gray.shape, pothole_only)
    pot_ext = make_pothole_extractor(pot_cfg)

    rows, results = [], {}
    for cond in ("before", "after"):
        ca = IDENTITY if cond == "before" else arms["crack"]
        pa = IDENTITY if cond == "before" else arms["pothole"]
        cg, cm, cb, c_pre, c_ext = run_branch(gray, meta, ca, crack_extractor)
        pg, pm, pb, p_pre, p_ext = run_branch(gray, meta, pa, pot_ext)
        row = {"image": path.name, "source": source_of(path), "condition": cond,
               "crack_pre": ca["name"], "pothole_pre": pa["name"],
               "canny_lo": th[0], "canny_hi": th[1],
               "edges": count_edges(cg, road, th), "keypoints": count_keypoints(cg, road),
               "pot_edges": count_edges(pg, road, th),
               "crack_n": n_components(cm), "crack_area%": 100 * float((cm & road).sum()) / road_px,
               "pot_n": len(pb), "pot_area%": 100 * float((pm & road).sum()) / road_px,
               "roi_ms": roi_ms, "crack_pre_ms": c_pre, "crack_ext_ms": c_ext,
               "pot_pre_ms": p_pre, "pot_ext_ms": p_ext,
               "crack_P": "", "crack_R": "", "crack_F2": "", "pot_IoU": "", "crack_note": "",
               "_tp_p": "", "_np": "", "_tp_r": "", "_ng": ""}
        if truth is not None:
            if truth["crack_item"] is not None:
                p, r, f2, cnt = crack_scores(truth["crack_item"], cm)
                row.update(dict(zip(("_tp_p", "_np", "_tp_r", "_ng"), cnt)))
                if r == r:
                    row.update({"crack_P": p, "crack_R": r, "crack_F2": f2})
                else:
                    row["crack_note"] = "(no crack GT)"
            elif pothole_only:
                row["crack_note"] = "(pothole-only label)"
            if truth["pot"].any():
                row["pot_IoU"] = float((pm & truth["pot"]).sum()) / float((pm | truth["pot"]).sum())
        rows.append(row)
        results[cond] = {"crack": cm, "pot": pm, "pot_boxes": pb, "row": row}
    return color, meta, truth, results, rows


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------
def _fmt(v, nd=2):
    return "-" if v == "" or v is None or (isinstance(v, float) and v != v) else ("%.*f" % (nd, v))


def crack_txt(row):
    if row["crack_F2"] != "" and row["crack_F2"] == row["crack_F2"]:
        return "F2 %.2f (P %.2f R %.2f)" % (row["crack_F2"], row["crack_P"], row["crack_R"])
    return row.get("crack_note", "")


def pot_txt(row, truth):
    if row["pot_IoU"] != "":
        return "IoU %.2f" % row["pot_IoU"]
    return "" if truth is None else "(no pothole GT)"


def draw(color, meta, truth, results, title, show="all"):
    """show: all / crack / pothole. 그림에 표시할 대상만 고른다 (계산과 CSV는 항상 둘 다)."""
    road = meta["road_mask"]
    on = {"crack": show in ("all", "crack"), "pot": show in ("all", "pothole")}
    base = color.copy()

    def outline(img, mask, col, th=1):
        cnts, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cv.drawContours(img, cnts, -1, col, th)

    p0 = base.copy()
    outline(p0, road, C_ROAD, 2)
    if truth is not None:
        for key, col in (("crack", C_GT), ("pot", C_GT_POT)):
            if on[key] and truth[key] is not None and truth[key].any():
                outline(p0, truth[key], col, 2)
    gt_txt = "  ".join(t for k, t in (("crack", "crack=green"), ("pot", "pothole=magenta")) if on[k])
    pr_txt = "  ".join(t for k, t in (("crack", "crack=red"), ("pot", "pothole=blue")) if on[k])
    legend = ["INPUT  road=yellow" + ("" if show == "all" else "   [SHOW: %s only]" % show),
              ("GT: " + gt_txt) if truth is not None else "(no GT label)", "pred: " + pr_txt]
    panels = [(p0, legend)]
    for cond in ("before", "after"):
        r = results[cond]; row = r["row"]
        p = base.copy()
        for key, mask, col in (("pot", r["pot"], C_POT), ("crack", r["crack"], C_CRACK)):
            if on[key]:
                p[mask] = (0.45 * p[mask] + 0.55 * np.array(col)).astype(np.uint8)
        if on["pot"]:
            for x, y, w, h in r["pot_boxes"]:
                cv.rectangle(p, (x, y), (x + w, y + h), C_POT, 2)
        if truth is not None:
            for key, col in (("crack", C_GT), ("pot", C_GT_POT)):
                if on[key] and truth[key] is not None and truth[key].any():
                    outline(p, truth[key], col, 1)
        pre = "none" if cond == "before" else "crack:%s pothole:%s" % (row["crack_pre"], row["pothole_pre"])
        lines = ["%s  (%s)" % (cond.upper(), pre)]
        if on["crack"]:
            lines.append("crack %d (%.2f%%)  %s" % (row["crack_n"], row["crack_area%"], crack_txt(row)))
        if on["pot"]:
            lines.append("pothole %d (%.2f%%)  %s" % (row["pot_n"], row["pot_area%"], pot_txt(row, truth)))
        lines += [
                 "edges %d  ORB %d  time %.0fms" % (row["edges"], row["keypoints"],
                                                    row["roi_ms"] + row["crack_pre_ms"] + row["crack_ext_ms"]
                                                    + row["pot_pre_ms"] + row["pot_ext_ms"])]
        panels.append((p, lines))

    out = []
    for img, lines in panels:
        bar = np.full((22 * len(lines) + 8, img.shape[1], 3), 30, np.uint8)
        for i, t in enumerate(lines):
            cv.putText(bar, t, (6, 18 + 22 * i), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
        out.append((bar, img))
    hb = max(b.shape[0] for b, _ in out)
    cols = [np.vstack([cv.copyMakeBorder(b, 0, hb - b.shape[0], 0, 0, cv.BORDER_CONSTANT, value=(30, 30, 30)), i])
            for b, i in out]
    sep = np.full((cols[0].shape[0], 6, 3), 255, np.uint8)
    grid = np.hstack([cols[0], sep, cols[1], sep, cols[2]])
    head = np.full((28, grid.shape[1], 3), 15, np.uint8)
    cv.putText(head, title, (6, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv.LINE_AA)
    return np.vstack([head, grid])


def save_jpg(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise OSError(str(path))
    data.tofile(str(path))


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
NUM = ["edges", "keypoints", "pot_edges", "crack_n", "crack_area%", "pot_n", "pot_area%", "roi_ms",
       "crack_pre_ms", "crack_ext_ms", "pot_pre_ms", "pot_ext_ms", "crack_P", "crack_R", "crack_F2", "pot_IoU"]


def collect(inputs):
    paths = []
    for s in inputs:
        p = Path(s)
        if p.is_dir():
            paths += sorted(q for q in p.rglob("*") if q.suffix.lower() in rr.IMG_EXTS and q.is_file())
        elif p.is_file():
            paths.append(p)
        else:
            raise SystemExit("입력을 찾을 수 없습니다: %s" % s)
    return paths


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["source"], r["condition"])].append(r)
        groups[("all", r["condition"])].append(r)
    out = []
    for (src, cond), rs in sorted(groups.items()):
        s = {"source": src, "condition": cond, "images": len(rs)}
        # 균열 micro F2: 사진들의 맞힌 골격 픽셀을 합친 뒤 계산 (crack_eval의 '전체(micro)'와 같은 방식)
        cr = [r for r in rs if r["_ng"] != ""]
        if cr:
            p, rc, _ = CEV.prf(*(sum(r[k] for r in cr) for k in ("_tp_p", "_np", "_tp_r", "_ng")))
            s["crack_F2_micro"] = CEV.fbeta(p, rc, 2)
        else:
            s["crack_F2_micro"] = ""
        for k in NUM:
            v = [r[k] for r in rs if r[k] != "" and r[k] == r[k]]
            s[k] = float(np.mean(v)) if v else ""
            if k in ("crack_F2", "pot_IoU"):
                s[k + "_n"] = len(v)
        out.append(s)
    return out


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("%.4f" % v if isinstance(v, float) else v) for k, v in r.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description="통합 실행: 전처리 -> 균열·포트홀 후보 추출 -> 시각화 + 전·후 비교")
    ap.add_argument("inputs", nargs="+", help="사진 파일 또는 폴더(하위 폴더 포함)")
    ap.add_argument("--out", default="", help="결과 폴더 (기본 out_pipeline/<시각>)")
    ap.add_argument("--crack-pre", default=CRACK_PRE, help="균열 전처리 (+로 연결, none=없음)")
    ap.add_argument("--pothole-pre", default=POTHOLE_PRE, help="포트홀 전처리 (+로 연결, none=없음)")
    ap.add_argument("--no-vis", action="store_true", help="시각화 저장 생략")
    ap.add_argument("--show", choices=["all", "crack", "pothole"], default="all",
                    help="그림에 표시할 대상 (계산·CSV는 항상 균열·포트홀 모두). crack/pothole이면 그림 제목에 표시")
    a = ap.parse_args(argv)
    cv.setNumThreads(1)
    arms = {"crack": arm_of(a.crack_pre), "pothole": arm_of(a.pothole_pre)}
    out = Path(a.out) if a.out else ROOT / "out_pipeline" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    pothole_only = rr.pothole_only_names()
    pot_cfg = EdgePotholeConfig()
    paths = collect(a.inputs)
    print("영상 %d장 / 균열 전처리 %s / 포트홀 전처리 %s" % (len(paths), arms["crack"]["name"], arms["pothole"]["name"]))
    rows = []
    for i, p in enumerate(paths, 1):
        color, meta, truth, results, rr_rows = process(p, arms, pot_cfg, p.stem in pothole_only)
        rows += rr_rows
        b, f = rr_rows
        print("[%d/%d] %-28s 균열 %3d->%3d  포트홀 %2d->%2d  에지 %6d->%6d  F2 %s->%s  IoU %s->%s" % (
            i, len(paths), p.name, b["crack_n"], f["crack_n"], b["pot_n"], f["pot_n"], b["edges"], f["edges"],
            _fmt(b["crack_F2"]), _fmt(f["crack_F2"]), _fmt(b["pot_IoU"]), _fmt(f["pot_IoU"])), flush=True)
        if not a.no_vis:
            suffix = "" if a.show == "all" else "_" + a.show
            save_jpg(out / "vis" / (p.stem + suffix + ".jpg"),
                     draw(color, meta, truth, results, "%s  [%s]" % (p.name, b["source"]), a.show))
    write_csv(out / "per_image.csv", rows)
    summ = summarize(rows)
    write_csv(out / "summary.csv", summ)
    print("\n요약 (출처 x 조건 평균)")
    for s in summ:
        print("  %-9s %-6s %2d장 | 에지 %7.0f ORB %5.0f | 균열 %5.1f개 %5.2f%% F2 영상평균 %s micro %s | 포트홀 %4.1f개 %5.2f%% IoU %s | %4.0fms" % (
            s["source"], s["condition"], s["images"], s["edges"], s["keypoints"], s["crack_n"], s["crack_area%"],
            _fmt(s["crack_F2"], 3), _fmt(s["crack_F2_micro"], 3), s["pot_n"], s["pot_area%"], _fmt(s["pot_IoU"], 3),
            s["roi_ms"] + s["crack_pre_ms"] + s["crack_ext_ms"] + s["pot_pre_ms"] + s["pot_ext_ms"]))
    print("저장: %s" % out)


if __name__ == "__main__":
    main()
