"""
crack_eval.py - 정식 균열 추출기의 영역(픽셀) 채점과 선정 기준 판정

사용법 (cv2026 폴더에서)
    python src/crack_eval.py                    # 전처리 없음(none)만 채점
    python src/crack_eval.py --maps             # + 영상별 오류 지도 저장
    python src/crack_eval.py --axis noise       # 노이즈 축 후보 전체를 none과 비교해 판정
    python src/crack_eval.py --split test       # 최종 평가 세트 (파라미터 선정에는 쓰지 않는다)
    기본은 개발 세트(train_data/). 결과는 세트별로 따로 낸다 (train/test를 합산하지 않음)

채점 방식: 허용거리(tolerance) 정밀도 / 재현율
    1. 검출 마스크와 정답 균열 마스크를 각각 중심선(폭 1px 뼈대)으로 줄인다
       -> 라벨러가 균열을 굵게 칠한 정도(두께)가 점수에 섞이지 않게 하기 위함
    2. 정밀도 = 검출 중심선 픽셀 중, 정답 중심선에서 d 픽셀 이내에 있는 비율   ("헛것을 덜 그렸나")
       재현율 = 정답 중심선 픽셀 중, 검출 중심선에서 d 픽셀 이내에 있는 비율   ("진짜 균열을 찾았나")
    3. 포트홀 정답 주변(d 픽셀까지)에 떨어진 검출은 채점에서 제외한다 (포트홀 추출기의 몫)
    4. 집계는 모든 영상의 픽셀을 합쳐서 계산(micro). 영상별 값도 함께 출력한다

F2 (선정 점수) = 5PR / (4P + R)
    정밀도와 재현율을 모두 반영하되 재현율을 2배 중요하게 본다 (β=2, 재현율 우선일 때의 관례적 값).
    '손상 후보 추출'이라 놓침이 헛검출보다 나쁘지만, 정밀도 0.15 = 후보의 85%가 헛검출이라 정밀도도 무시할 수 없다

보고용 지표 (선정에는 쓰지 않음)
    F1      = 정밀도와 재현율을 같은 무게로 본 점수
    tIoU    = 허용거리 IoU = F1 / (2 - F1). 위의 허용거리 채점을 IoU 형태로 바꾼 값 (두께 영향 없음)
    pIoU    = 픽셀 IoU = (검출 ∩ 정답) / (검출 ∪ 정답), 픽셀 그대로. 정답 띠(약 12px)가 실제 균열보다 두꺼워
              위치가 완벽한 3px 검출도 상한이 약 0.30이고, 검출을 두껍게 할수록 오른다 -> 참고로만 보고

선정 기준
    ① 필수: 재현율이 COLLAPSE 넘게 떨어진 영상이 하나도 없음 (합산 점수는 한 영상의 붕괴를 가린다)
            단, 정답 균열 중심선이 MIN_GT_SKEL px(640 기준) 미만인 영상은 이 검사에서 제외한다
    ② ①을 통과한 것 중 F2 최대
    ③ F2 차이 TIE 이내면 동점으로 보고 덜 공격적인 쪽(목록 앞쪽, none이 가장 앞)
    ④ 확인: 고른 것이 d=3/8 에서도 기준보다 나쁘지 않은지 표에서 함께 본다

좌표 맞추기
    채점은 '원본 전체를 추출기와 같은 배율로 줄인 화면' 위에서 한다. 검출 마스크는 ROI 위치에 붙여 넣고,
    정답은 원본 전체를 같은 배율로 줄인다 -> ROI 밖 정답 균열도 '놓친 것'으로 계산된다.
    원본 해상도 채점은 큰 영상(Norway, 촬영본)이 픽셀 수만큼 가중되는 왜곡이 있어 쓰지 않는다

파라미터 근거
    D_REF = 5     정답 균열 띠 폭(640 기준) 중앙값 12.4px의 절반 정도. 3/8에서도 결론이 같은지 함께 출력
    COLLAPSE      0.2. 전체(micro) 재현율은 한 영상이 무너져도 다른 영상에 묻힌다
                  (예: India 1.00 -> 0.00 이 전체로는 -0.09로만 보임). 영상 단위 붕괴를 따로 막는다
    MIN_GT_SKEL   100 (640 기준 정답 중심선 길이). 정답이 아주 작으면 조각 하나로 재현율이 0과 1을 오가서
                  붕괴 판정이 우연에 좌우된다. 제공 13장에서 Norway_008315만 33px이고 나머지는 모두 315px 이상이라
                  둘 사이 값으로 잡았다 (100px = 640 화면 폭의 약 1/6). 해당 영상도 전체 점수에는 그대로 포함된다
    TIE           0.01 (F2). 이보다 작은 차이로 처리를 추가하지 않는다
    (이전 기준 '재현율 하락 <= 0.02 + 정밀도 최대'는 정밀도 개선을 지나치게 막아 F2로 바꿈)
"""
import os
import sys
import csv
import time
import argparse
import datetime

import numpy as np
import cv2 as cv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import round_runner as rr                     # 데이터 읽기 / ROI / resize / 전처리 적용은 러너와 공유
from candidates import AXES, IDENTITY
from crack_extractor import detect_mask, skeletonize   # 중심선은 추출기와 같은 함수 사용

D_REF = 5
D_CHECK = (3, 5, 8)
REF = 640
COLLAPSE = 0.2
MIN_GT_SKEL = 100
TIE = 0.01


def dist_to(mask):
    """모든 픽셀에서 mask의 가장 가까운 픽셀까지의 거리. mask가 비면 무한대."""
    if not mask.any():
        return np.full(mask.shape, np.inf, np.float32)
    return cv.distanceTransform((~mask).astype(np.uint8), cv.DIST_L2, 5)


# ---------------------------------------------------------------------------
# 좌표 맞추기: 채점 화면 = 원본 전체를 추출기와 같은 배율로 줄인 것
# ---------------------------------------------------------------------------
def eval_canvas(it, pred):
    meta = it["meta"]
    f = float(meta["resize_info"]["scale"])            # ROI -> 처리 영상 배율 (long 모드는 가로세로 동일)
    H, W = it["shape"]
    h, w = max(1, int(round(H * f))), max(1, int(round(W * f)))

    def gt_mask(typ):
        full = np.zeros((H, W), np.uint8)
        for g in it["inst"]:
            if g["type"] == typ:
                ch, cw = g["crop"].shape
                full[g["y0"]:g["y0"] + ch, g["x0"]:g["x0"] + cw] |= g["crop"].astype(np.uint8)
        if f == 1.0:
            return full > 0
        return cv.resize(full * 255, (w, h), interpolation=cv.INTER_AREA) >= 128

    x1, y1 = meta["roi_coords_original"][:2]
    ox, oy = int(round(x1 * f)), int(round(y1 * f))
    P = np.zeros((h, w), bool)
    ph, pw = pred.shape
    ph2, pw2 = min(ph, h - oy), min(pw, w - ox)
    P[oy:oy + ph2, ox:ox + pw2] = pred[:ph2, :pw2]
    return P, gt_mask("crack"), gt_mask("pothole"), f


def score_image(it, pred, keep_maps=False):
    P, G, Pot, f = eval_canvas(it, pred)
    s = max(P.shape) / float(REF)                       # 채점 화면 크기에 맞춘 d 배율
    sp, sg = skeletonize(P), skeletonize(G)
    d_to_g, d_to_p, d_to_pot = dist_to(sg), dist_to(sp), dist_to(Pot)
    rec = {"name": it["name"], "source": it["source"],
           "pred_area": float(P.mean()), "n_gt_skel": int(sg.sum()), "scale": s}
    pot_zone = d_to_pot <= D_REF * s
    Pi = P & ~pot_zone                                  # IoU도 포트홀 주변 검출은 제외
    rec["iou_i"] = int((Pi & G).sum())
    rec["iou_u"] = int((Pi | G).sum())
    gt_width = []
    if G.any():
        dt = cv.distanceTransform(G.astype(np.uint8), cv.DIST_L2, 5)
        gt_width = list(2.0 * dt[sg] / s)               # 640 기준으로 환산한 정답 띠 폭
    for d in D_CHECK:
        dd = d * s
        valid_p = sp & ~(d_to_pot <= dd)
        rec["np_%d" % d] = int(valid_p.sum())
        rec["tp_p_%d" % d] = int((valid_p & (d_to_g <= dd)).sum())
        rec["tp_r_%d" % d] = int((sg & (d_to_p <= dd)).sum())
    maps = None
    if keep_maps:
        dd = D_REF * s
        valid_p = sp & ~(d_to_pot <= dd)
        maps = {"tp": valid_p & (d_to_g <= dd), "fp": valid_p & ~(d_to_g <= dd),
                "fn": sg & ~(d_to_p <= dd), "pot": Pot, "f": f}
    return rec, gt_width, maps


# ---------------------------------------------------------------------------
# 집계
# ---------------------------------------------------------------------------
def fbeta(p, r, beta):
    if p != p or r != r or (p + r) == 0:
        return float("nan") if (p != p or r != r) else 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def prf(tp_p, np_, tp_r, ng):
    p = tp_p / np_ if np_ else float("nan")
    r = tp_r / ng if ng else float("nan")
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def summarize(recs, d):
    """(정밀도, 재현율, F1) - 13장 픽셀 합산(micro)."""
    tp_p = sum(r["tp_p_%d" % d] for r in recs)
    np_ = sum(r["np_%d" % d] for r in recs)
    tp_r = sum(r["tp_r_%d" % d] for r in recs)
    ng = sum(r["n_gt_skel"] for r in recs)
    return prf(tp_p, np_, tp_r, ng)


def report(recs):
    """d=D_REF 기준 보고용 묶음."""
    p, r, f1 = summarize(recs, D_REF)
    u = sum(x["iou_u"] for x in recs)
    iou = sum(x["iou_i"] for x in recs) / u if u else float("nan")
    tiou = f1 / (2 - f1) if f1 == f1 else float("nan")
    return {"P": p, "R": r, "F1": f1, "F2": fbeta(p, r, 2), "tIoU": tiou, "pIoU": iou}


def image_recall(rec, d=D_REF):
    return rec["tp_r_%d" % d] / rec["n_gt_skel"] if rec["n_gt_skel"] else None


def judge(base, recs):
    """선정 기준 ①: 기준(base) 대비 재현율 하락과 영상 단위 붕괴 검사."""
    r0, r1 = summarize(base, D_REF)[1], summarize(recs, D_REF)[1]
    collapsed, skipped = [], []
    for a, b in zip(base, recs):
        ra, rb = image_recall(a), image_recall(b)
        if ra is None:
            continue
        if a["n_gt_skel"] / a.get("scale", 1.0) < MIN_GT_SKEL:      # 정답이 너무 작은 영상은 붕괴 판정 제외
            if ra - rb > COLLAPSE:
                skipped.append("%s(%.2f->%.2f)" % (a["name"], ra, rb))
            continue
        if ra - rb > COLLAPSE:
            collapsed.append("%s(%.2f->%.2f)" % (a["name"], ra, rb))
    ok = not collapsed
    return {"dR": r1 - r0, "collapsed": collapsed, "skipped": skipped, "pass": ok}


def pick(rows, base_name="none"):
    """선정 기준 ②③: 통과한 것 중 F2 최대. 최고와 TIE 이내면 동점 -> 목록 앞쪽(덜 공격적, none 우선)."""
    cand = [(name, recs, j) for name, recs, j in rows if j["pass"]]
    if not cand:
        return base_name
    best = max(report(recs)["F2"] for _, recs, _ in cand)
    tied = [name for name, recs, _ in cand if best - report(recs)["F2"] <= TIE]
    return tied[0]


# ---------------------------------------------------------------------------
# 오류 지도: 초록 = 맞은 검출, 빨강 = 헛검출, 파랑 = 놓친 정답, 회색 = 포트홀(채점 제외)
# ---------------------------------------------------------------------------
def save_map(it, maps, out_dir, arm_name):
    bgr = rr.imread_bgr(it["path"])
    h, w = maps["tp"].shape
    base = cv.resize(bgr, (w, h), interpolation=cv.INTER_AREA)
    base = (base * 0.6).astype(np.uint8)
    base[maps["pot"]] = (base[maps["pot"]] * 0.5 + 64).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    for key, col in (("fn", (255, 80, 0)), ("fp", (0, 0, 255)), ("tp", (0, 220, 0))):
        m = cv.dilate(maps[key].astype(np.uint8), k) > 0
        base[m] = col
    os.makedirs(out_dir, exist_ok=True)
    cv.imwrite(os.path.join(out_dir, "%s__%s.png" % (it["name"], arm_name)), base)


# ---------------------------------------------------------------------------
def run(items, arm, out_dir=None):
    recs, widths = [], []
    for it in items:
        t0 = time.perf_counter()
        img = rr.apply_stages(arm["stages"], it["img"])
        mask, boxes = detect_mask(img, it["meta"])
        sec = time.perf_counter() - t0
        rec, w, maps = score_image(it, mask, keep_maps=out_dir is not None)
        rec.update({"n_box": len(boxes), "sec": sec})
        recs.append(rec)
        widths += w
        if maps is not None:
            save_map(it, maps, out_dir, arm["name"])
    return recs, widths


def f3(v):
    return "  -  " if v is None or v != v else "%.3f" % v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=sorted(AXES), default=None, help="이 축의 후보 전체를 none과 비교해 판정")
    ap.add_argument("--maps", action="store_true", help="영상별 오류 지도 저장")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", choices=["train", "test"], default="train", help="데이터 세트 (기본 train)")
    args = ap.parse_args()

    items = rr.load_items(args.limit, args.split)
    print("세트: %s (%d장: 제공 %d, 촬영 %d)" % (args.split, len(items), sum(i["source"] == "provided" for i in items),
                                          sum(i["source"] == "captured" for i in items)))
    arms = [IDENTITY] + ([a for a in AXES[args.axis] if a["name"] != "none"] if args.axis else [])
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "crackeval_%s_%s_%s" % (args.split, args.axis or "none", stamp)
    map_dir = os.path.join(rr.OUT_DIR, tag + "_maps") if args.maps else None

    rows, base = [], None
    for arm in arms:
        recs, widths = run(items, arm, map_dir)
        if base is None:
            base = recs
            print("\n[none] 영상별 (d=%d)" % D_REF)
            print("  %-24s %6s %6s %6s %8s %6s" % ("image", "P", "R", "F1", "검출면적", "n_box"))
            for r in recs:
                p, rc, f1 = prf(r["tp_p_%d" % D_REF], r["np_%d" % D_REF], r["tp_r_%d" % D_REF], r["n_gt_skel"])
                print("  %-24s %6s %6s %6s %7.1f%% %6d" % (r["name"], f3(p), f3(rc), f3(f1), 100 * r["pred_area"], r["n_box"]))
            if widths:
                print("  정답 균열 띠 폭(640 기준) 중앙값 %.1f px, 사분위 %.1f~%.1f  -> d 근거"
                      % tuple(np.percentile(widths, [50, 25, 75])))
        rows.append((arm["name"], recs, judge(base, recs)))

    print("\n전체 (micro, d=%d)   선정: ① 재현율 %.1f 넘게 무너진 영상 없음  ② F2 최대 (차이 %.2f 이내 동점 -> none 쪽)"
          % (D_REF, COLLAPSE, TIE))
    print("  %-22s %6s %6s %6s | %6s %6s %6s | F2@d3  F2@d8 | %7s %s" % ("arm", "F2", "P", "R", "F1", "tIoU", "pIoU", "재현율변화", "판정"))
    out_rows = []
    for name, recs, j in rows:
        m = report(recs)
        f2_3, f2_8 = fbeta(*summarize(recs, 3)[:2], 2), fbeta(*summarize(recs, 8)[:2], 2)
        verdict = "기준" if recs is base else ("통과" if j["pass"] else "탈락 (붕괴: " + " ".join(j["collapsed"]) + ")")
        if j.get("skipped"):
            verdict += "  [정답 작아 판정 제외: %s]" % " ".join(j["skipped"])
        print("  %-22s %6s %6s %6s | %6s %6s %6s | %5s  %5s | %+7.3f  %s"
              % (name, f3(m["F2"]), f3(m["P"]), f3(m["R"]), f3(m["F1"]), f3(m["tIoU"]), f3(m["pIoU"]), f3(f2_3), f3(f2_8), j["dR"], verdict))
        out_rows.append([name, m["F2"], m["P"], m["R"], m["F1"], m["tIoU"], m["pIoU"], f2_3, f2_8, j["dR"], j["pass"],
                         " ".join(j["collapsed"]), float(np.mean([r["sec"] for r in recs]))])
    if len(rows) > 1:
        print("\n  선정 결과: %s" % pick(rows))

    path = os.path.join(rr.OUT_DIR, tag + ".csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["arm", "F2", "P", "R", "F1", "tIoU", "pIoU", "F2_d3", "F2_d8", "dR", "pass", "collapsed", "sec"])
        w.writerows(out_rows)
    print("\n저장: %s" % path + ("  /  지도: %s" % map_dir if map_dir else ""))


if __name__ == "__main__":
    os.makedirs(rr.OUT_DIR, exist_ok=True)
    main()
