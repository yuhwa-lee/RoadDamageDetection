"""포트홀 박스(객체) 단위 평가. 검출기는 그대로(마스크/후보), 평가만 객체 단위.

실행: python src/pothole_box_eval.py [--config pothole_edge_baseline.json] [--merge 0,8]
정답:
  - train_data/*.jpg  : labeled_data 픽셀 마스크 -> 연결 영역마다 박스 1개
  - train_data/rdd_box: labeled_data에 같은 이름의 픽셀 라벨이 있으면 그것을 쓰고(위와 동일하게 박스화),
                        없으면 RDD2020 원본 박스(boxes.json). (균열 실행기는 하위 폴더를 읽지 않음)
채점 (모두 긴 변 640 처리 좌표, 균열 채점과 같은 캔버스):
  - 정답 박스와 검출 박스를 IoU가 큰 쌍부터 1:1로 짝지음.
  - IoU >= 0.5 를 '찾음'으로 셈 (PASCAL VOC / RDD 대회 기준). 작은 포트홀에 엄격하므로 0.3도 함께 보고.
  - 정밀도 = 맞은 검출 / 전체 검출, 재현율 = 찾은 정답 / 전체 정답.
  - 정답 없는 영상은 검출 수만 오검출로 셈.
정답 마스크 -> 박스: 같은 포트홀이 라벨 붓질로 끊긴 조각을 잇기 위해 3x3 closing 후 연결 영역,
  640 기준 면적 20px 미만 조각은 제외 (약 4x5px; 이보다 작으면 박스 IoU 자체가 불안정).
--merge m: 검출 박스를 m px(640 기준) 넓혀 겹치면 합침. 포트홀 하나가 여러 조각으로 잡히는 문제 확인용.
"""
import argparse, json, sys
from dataclasses import replace
from pathlib import Path
import cv2 as cv
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src'))
import pothole_eval as PE
from pothole_edge_extractor import EdgePotholeConfig, detect

GT_MIN_AREA = 20


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union else 0.0


def match(gt, det, thr):
    pairs = sorted(((box_iou(g, d), i, j) for i, g in enumerate(gt) for j, d in enumerate(det)), reverse=True)
    ug, ud, tp = set(), set(), 0
    for iou, i, j in pairs:
        if iou < thr: break
        if i in ug or j in ud: continue
        ug.add(i); ud.add(j); tp += 1
    return tp


def merge_boxes(boxes, m):
    boxes = [list(b) for b in boxes]
    changed = m > 0
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i+1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0]-m < b[2] and b[0]-m < a[2] and a[1]-m < b[3] and b[1]-m < a[3]:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j]; changed = True; break
            if changed: break
    return boxes


def to_proc(box, meta, gray):
    x1, y1, x2, y2 = meta['roi_coords_original']
    sx, sy = gray.shape[1] / (x2-x1), gray.shape[0] / (y2-y1)
    b = [(box[0]-x1)*sx, (box[1]-y1)*sy, (box[2]-x1)*sx, (box[3]-y1)*sy]
    c = [max(0, b[0]), max(0, b[1]), min(gray.shape[1], b[2]), min(gray.shape[0], b[3])]
    return c if c[2] > c[0] and c[3] > c[1] else None


def load(image_dir, gt_dir):
    color = PE.read_pothole_color(gt_dir)
    items = []
    for p in sorted(image_dir.iterdir()):
        lp = gt_dir / 'SegmentationClass' / (p.stem + '.png')
        if p.is_file() and lp.is_file():
            items.append((p, 'mask', lp))
    rdd = image_dir / 'rdd_box' / 'boxes.json'
    if rdd.is_file():
        for name, boxes in json.loads(rdd.read_text(encoding='utf-8'))['boxes'].items():
            lp = gt_dir / 'SegmentationClass' / (Path(name).stem + '.png')
            items.append((rdd.parent / name, 'mask', lp) if lp.is_file() else (rdd.parent / name, 'box', boxes))
    out = []
    for p, kind, gt in items:
        org = cv.imdecode(np.fromfile(str(p), np.uint8), cv.IMREAD_COLOR)
        gray, meta = PE.prepare_input(org, 'long')
        if kind == 'mask':
            full = np.all(np.array(Image.open(gt).convert('RGB')) == color, -1)
            x1, y1, x2, y2 = meta['roi_coords_original']
            m = cv.resize(full[y1:y2, x1:x2].astype(np.uint8), (gray.shape[1], gray.shape[0]), interpolation=cv.INTER_NEAREST)
            pix = m > 0   # 픽셀 채점용 (640 처리 좌표)
            m = cv.morphologyEx(m, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            n, _, st, _ = cv.connectedComponentsWithStats(m)
            boxes = [[x, y, x+w, y+h] for x, y, w, h, a in st[1:] if a >= GT_MIN_AREA]
            lost = 0
        else:
            proc = [to_proc(b, meta, gray) for b in gt]
            boxes = [b for b in proc if b]; lost = len(proc) - len(boxes); pix = None
        src = 'rdd_box' if p.parent.name == 'rdd_box' else ('captured' if p.name.startswith(('IMG', 'capture', 'cap_')) else 'provided')
        out.append(dict(name=p.name, src=src, kind=kind, gray=gray, meta=meta, gt=boxes, gt_outside_roi=lost, pix=pix,
                        label=str(gt) if kind == 'mask' else None))
    return out


def evaluate(items, cfg, merge=0, verbose=False):
    rows = []
    for it in items:
        dets = [[x, y, x+w, y+h] for x, y, w, h in detect(it['gray'], {'road_mask': it['meta']['road_mask'], 'pothole_edge_config': cfg})]
        dets = merge_boxes(dets, merge)
        rows.append(dict(it, det=len(dets), tp5=match(it['gt'], dets, .5), tp3=match(it['gt'], dets, .3)))
    def prf(rs, k):
        tp = sum(r[k] for r in rs); d = sum(r['det'] for r in rs); g = sum(len(r['gt']) for r in rs)
        p = tp/d if d else float('nan'); r = tp/g if g else float('nan')
        return tp, d, g, p, r, (2*p*r/(p+r) if p+r else 0.0)
    out = {}
    for name, rs in [('전체(포트홀 영상)', [r for r in rows if r['gt']]), ('  제공', [r for r in rows if r['gt'] and r['src'] == 'provided']),
                     ('  촬영', [r for r in rows if r['gt'] and r['src'] == 'captured']), ('  rdd_box', [r for r in rows if r['src'] == 'rdd_box'])]:
        a, b = prf(rs, 'tp5'), prf(rs, 'tp3')
        out[name.strip()] = (a, b)
        print(f'{name:<14} 영상{len(rs):>2} 정답{a[2]:>3} 검출{a[1]:>4} | IoU0.5 찾음 {a[0]:>2} P {a[3]:.3f} R {a[4]:.3f} F1 {a[5]:.3f} | IoU0.3 찾음 {b[0]:>2} R {b[4]:.3f} F1 {b[5]:.3f}')
    neg = [r for r in rows if not r['gt'] and r['src'] != 'rdd_box']
    print(f'정답 없는 영상 {len(neg)}장: 검출 발생 {sum(r["det"] > 0 for r in neg)}장, 오검출 박스 {sum(r["det"] for r in neg)}개')
    if verbose:
        for r in rows:
            if r['gt']: print(f'   {r["name"]:<28} 정답 {len(r["gt"])} 검출 {r["det"]:>3} 찾음(0.5/0.3) {r["tp5"]}/{r["tp3"]}' + (f'  ROI밖 정답 {r["gt_outside_roi"]}' if r['gt_outside_roi'] else ''))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image-dir', default=str(ROOT / 'train_data'))
    ap.add_argument('--gt-dir', default=str(ROOT / 'labeled_data'))
    ap.add_argument('--config', type=Path)
    ap.add_argument('--merge', default='0')
    a = ap.parse_args()
    cfg = EdgePotholeConfig(**json.loads(a.config.read_text(encoding='utf-8-sig'))) if a.config else EdgePotholeConfig()
    items = load(Path(a.image_dir), Path(a.gt_dir))
    for m in [float(x) for x in a.merge.split(',')]:
        print(f'\n=== merge {m:g}px ===')
        evaluate(items, cfg, m, verbose=True)


if __name__ == '__main__':
    main()
