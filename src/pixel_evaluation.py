"""포트홀 전처리 비교: 원본 해상도 전경 IoU. 박스를 채워서 평가하지 않는다."""
import csv
import datetime
import json
import time
from pathlib import Path
import cv2 as cv
import numpy as np


def restore_mask(mask, meta, original_shape):
    """현행 fixed/long resize를 역변환. 향후 letterbox 도입 시 padding 제거가 추가로 필요."""
    if mask.ndim != 2:
        raise ValueError('mask must be 2D')
    x1, y1, x2, y2 = meta['roi_coords_original']
    h, w = original_shape
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError('ROI is outside original image')
    out = np.zeros((h, w), np.uint8)
    out[y1:y2, x1:x2] = cv.resize((mask > 0).astype(np.uint8)*255,
                                  (x2-x1, y2-y1), interpolation=cv.INTER_NEAREST)
    return out


def ground_truth_mask(item, target='pothole'):
    gt = np.zeros(item['shape'], bool)
    for inst in item['inst']:
        if inst['type'] == target:
            y, x = inst['y0'], inst['x0']
            h, w = inst['crop'].shape
            gt[y:y+h, x:x+w] |= inst['crop']
    return gt


def pixel_counts(pred, gt):
    if pred.shape != gt.shape:
        raise ValueError('prediction and GT shapes differ')
    p, g = pred > 0, gt > 0
    return {'tp': int((p & g).sum()), 'fp': int((p & ~g).sum()), 'fn': int((~p & g).sum())}


def metrics(tp, fp, fn):
    union = tp+fp+fn
    # GT/예측 모두 없으면 포트홀 IoU는 정의되지 않음. 평균에서 제외하고 별도 계수.
    iou = tp/union if union else float('nan')
    return {'iou': iou, 'precision': tp/(tp+fp) if tp+fp else float('nan'),
            'recall': tp/(tp+fn) if tp+fn else float('nan'),
            'dice': 2*tp/(2*tp+fp+fn) if union else float('nan')}


def _write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv.imencode('.png', image)
    if not ok:
        raise OSError(str(path))
    data.tofile(str(path))  # Windows 한글 경로 지원


def _comparison(path, gt, pred, title):
    # 검정 배경 위 실제 픽셀. 흰색은 해당 영역. 아래 오류맵: TP초록 FP빨강 FN파랑.
    p = pred > 0
    left = cv.cvtColor(gt.astype(np.uint8)*255, cv.COLOR_GRAY2BGR)
    right = cv.cvtColor(p.astype(np.uint8)*255, cv.COLOR_GRAY2BGR)
    errors = np.zeros_like(left)
    errors[p & gt] = (0, 200, 0)
    errors[p & ~gt] = (0, 0, 255)
    errors[~p & gt] = (255, 0, 0)
    panels = []
    for panel, label in [(left, 'GT pothole'), (right, 'Prediction'), (errors, 'TP green / FP red / FN blue')]:
        scale = min(1.0, 640/max(panel.shape[:2]))
        panel = cv.resize(panel, (max(1, round(panel.shape[1]*scale)), max(1, round(panel.shape[0]*scale))), interpolation=cv.INTER_NEAREST)
        panel = cv.copyMakeBorder(panel, 35, 0, 0, 0, cv.BORDER_CONSTANT, value=(35,35,35))
        cv.putText(panel, label, (5,24), cv.FONT_HERSHEY_SIMPLEX, 0.43, (255,255,255), 1, cv.LINE_AA)
        panels.append(panel)
    _save_png(path, np.hstack(panels))