"""Ryu et al. (2015), DOI:10.1155/2015/968361 기반 독립 구현.

원저자 코드의 재현이 아니다. HST는 dark-tail triangle 근사, OHI는 정렬된
정규화 histogram intersection으로 구현한다. ROI/크기 정규화는 외부에서 수행.
GT, 학습 모델, 파일명은 탐지에 사용하지 않는다. 모든 box는 half-open xyxy.
"""
from dataclasses import dataclass
import math
import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class Config:
    segmentation: str = 'hst'
    min_area: int = 32  # 긴 변 640 영상 기준; 논문의 512px를 그대로 쓰지 않음
    max_area_fraction: float = .12
    min_compactness: float = .025  # area / perimeter**2
    max_aspect: float = 4.0
    min_solidity: float = .45
    median_size: int = 3
    closing_size: int = 5
    ring_radius: int = 12
    max_hull_growth: float = 1.8
    min_std: float = 8.0
    min_contrast: float = 6.0
    gray_ohi_threshold: float = .87
    sobel_ohi_threshold: float = .85
    nms_iou: float = .45


def box_iou(a, b):
    inter = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    aa = max(0., a[2]-a[0]) * max(0., a[3]-a[1])
    bb = max(0., b[2]-b[0]) * max(0., b[3]-b[1])
    return inter / max(aa + bb - inter, 1e-12)


def ordered_histogram_intersection(a, b):
    """밝기 위치 이동에 둔감한 정렬 histogram 비교. 높은 값은 유사함.

    정렬 과정에서 밝기 bin 위치 정보가 사라지므로 반드시 contrast와 함께 사용.
    Sobel은 두 영역 모두 동일한 0..255 스케일로 계산한다.
    """
    if not len(a) or not len(b):
        return 1.0
    ha = np.bincount(a.astype(np.uint8), minlength=256).astype(float)
    hb = np.bincount(b.astype(np.uint8), minlength=256).astype(float)
    return float(np.minimum(np.sort(ha / ha.sum()), np.sort(hb / hb.sum())).sum())


def dark_threshold(gray, valid):
    """HST 대용: histogram 주봉 왼쪽 꼬리에 triangle 임계값 적용."""
    values = gray[valid]
    if values.size == 0 or np.ptp(values) == 0:
        return -1
    hist = np.bincount(values, minlength=256).astype(np.float32)
    hist = cv.GaussianBlur(hist.reshape(1, -1), (9, 1), 0).ravel()
    left = int(np.percentile(values, 1))
    peak = int(np.argmax(hist))
    if peak <= left + 2:
        return int(cv.threshold(values.reshape(-1, 1), 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[0])
    xx = np.arange(left, peak+1)
    # 양 끝을 잇는 선에서 가장 떨어진 dark-tail 지점
    line = hist[left] + (hist[peak]-hist[left]) * (xx-left) / (peak-left)
    return int(xx[np.argmax(line-hist[xx])])


def nms(candidates, threshold):
    kept = []
    for c in sorted(candidates, key=lambda x: x['score'], reverse=True):
        if all(box_iou(c['box'], k['box']) < threshold for k in kept):
            kept.append(c)
    return kept


def detect(gray, road_mask=None, config=None):
    """return accepted, geometry_baseline, debug. 입력 크기 좌표계를 유지한다."""
    cfg = config or Config()
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError('gray must be a uint8 grayscale image')
    if road_mask is not None and road_mask.shape != gray.shape:
        raise ValueError('road_mask shape differs from image')
    valid = np.ones(gray.shape, bool) if road_mask is None else road_mask > 0
    threshold = dark_threshold(gray, valid)
    if cfg.segmentation == 'hst':
        binary = ((gray <= threshold) & valid).astype(np.uint8)*255
    elif cfg.segmentation == 'local':
        # RDD 조도 변화 대응 확장: 여러 크기의 주변 평균보다 어두운 영역.
        # HST의 재현이 아니며 나머지 형태/주변 비교 판정은 동일하다.
        source = cv.GaussianBlur(gray.astype(np.float32), (3,3), 0)
        dark = np.zeros(gray.shape, bool)
        for window in (31, 61, 121):
            background = cv.boxFilter(source, -1, (window,window))
            variance = np.maximum(cv.boxFilter(source*source,-1,(window,window))-background*background,0)
            dark |= background-source > np.maximum(8., .8*np.sqrt(variance))
        binary = (dark & valid).astype(np.uint8)*255
    else:
        raise ValueError('Unknown segmentation mode')
    binary = cv.medianBlur(binary, cfg.median_size)
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, np.ones((cfg.closing_size, cfg.closing_size), np.uint8))
    binary[~valid] = 0
    # 고정 스케일: 이미지별 min/max 정규화로 비교값이 달라지지 않게 함.
    gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
    sobel = np.clip(cv.magnitude(gx, gy)/4, 0, 255).astype(np.uint8)
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    candidates, decisions = [], []
    h, w = gray.shape
    for contour in contours:
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        if area < cfg.min_area or area > cfg.max_area_fraction * valid.sum():
            continue
        compactness = area / max(perimeter**2, 1)
        _, (rw, rh), _ = cv.minAreaRect(contour)
        aspect = max(rw, rh) / max(min(rw, rh), 1)
        hull = cv.convexHull(contour)
        hull_area = cv.contourArea(hull)
        solidity = area / max(hull_area, 1)
        if compactness < cfg.min_compactness or aspect > cfg.max_aspect or solidity < cfg.min_solidity:
            continue
        # 논문의 hull 기반 refinement를 성장량 제한과 함께 적용.
        refined = hull if hull_area <= cfg.max_hull_growth * area else contour
        x, y, bw, bh = cv.boundingRect(refined)
        r = cfg.ring_radius
        x0, y0, x1, y1 = max(0,x-r), max(0,y-r), min(w,x+bw+r), min(h,y+bh+r)
        local = np.zeros((y1-y0, x1-x0), np.uint8)
        cv.drawContours(local, [refined-np.array([[[x0,y0]]])], -1, 255, -1)
        inner = (local > 0) & valid[y0:y1,x0:x1]
        ring = (cv.dilate(local, cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*r+1,2*r+1))) > 0) & ~inner
        # 다른 어두운 후보는 정상 도로 비교 영역에서 제외.
        ring &= valid[y0:y1,x0:x1] & (binary[y0:y1,x0:x1] == 0)
        if inner.sum() < 8 or ring.sum() < 16:
            continue
        g, s = gray[y0:y1,x0:x1], sobel[y0:y1,x0:x1]
        std = float(g[inner].std())
        contrast = float(g[ring].mean()-g[inner].mean())
        gray_ohi = ordered_histogram_intersection(g[inner], g[ring])
        sobel_ohi = ordered_histogram_intersection(s[inner], s[ring])
        reasons = []
        if std < cfg.min_std:
            reasons.append('low_internal_variation')
        if contrast < cfg.min_contrast:
            reasons.append('low_contrast')
        if gray_ohi >= cfg.gray_ohi_threshold and sobel_ohi >= cfg.sobel_ohi_threshold:
            reasons.append('similar_to_background')
        # ranking 점수이며 확률이 아님. 평가시 모든 채택 후보를 사용한다.
        score = float(max(0,contrast)*(2-gray_ohi-sobel_ohi)*math.sqrt(max(compactness,0)))
        c = dict(box=[x,y,x+bw,y+bh], score=score, std=std, contrast=contrast,
                 gray_ohi=gray_ohi, sobel_ohi=sobel_ohi, compactness=compactness,
                 solidity=solidity, rejected_by=reasons)
        decisions.append(c)
        candidates.append(c)
    baseline = nms(candidates, cfg.nms_iou)
    accepted = nms([c for c in candidates if not c['rejected_by']], cfg.nms_iou)
    return accepted, baseline, dict(threshold=threshold if cfg.segmentation == 'hst' else None,
                                   candidate_mask=binary, decisions=decisions)
