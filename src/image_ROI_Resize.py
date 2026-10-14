
from __future__ import annotations

import cv2 as cv
import numpy as np

from resize_long_side import map_box_to_original, resize_long_side_with_info

# ---------------------------------------------------------------------------
# 고정값 (긴 변 640 작업 영상 기준)
# ---------------------------------------------------------------------------
GAP_CLOSE = 0.05     # 가로 틈 메우기: 영상 폭 비율
EDGE_MARGIN = 0.01   # 가장자리 여백: 영상 폭 비율

# 질감 필터
TEX_WIN = 9
TEX_SMOOTH = 3.0
TEX_K = 3.0
TEX_CRACK_CLOSE = 7
TEX_OPEN = 21

# 색 비율 판정
CH_REF = (0.80, 0.30, 0.70)
CH_GRN = 0.08
CH_YEL = 0.10
CH_RED = 0.12
CH_BLUE_SUN = 0.15
CH_BLUE_DARK = 0.80
CH_V_MIN = 15
CH_V_OVER = 45
CH_V_HI_MIN = 170

# 지평선 컷
HZ_CORE = 0.04
HZ_MARGIN = 0.03


# ---------------------------------------------------------------------------
# 1. 도로 후보
# ---------------------------------------------------------------------------
def _chroma_candidate(blurred):
    """밝기로 나눈 색 비율로 도로 후보를 만든다. 파란 쪽은 어두울수록 넓게, 나머지 방향은 좁게 허용."""
    f = blurred.astype(np.float32)
    s = f.sum(axis=2) + 1.0
    bn, gn, rn = (f[:, :, i] * 3.0 / s for i in range(3))
    yel, grn, red = (rn + gn) / 2 - bn, gn - (rn + bn) / 2, rn - (gn + bn) / 2
    v = s / 3.0
    h, w = v.shape
    ry, rx0, rx1 = CH_REF
    ref = (slice(int(h * ry), h), slice(int(w * rx0), int(w * rx1)))
    vr = v[ref].ravel()
    bright = vr >= np.median(vr)
    y0, g0, r0 = (float(np.median(c[ref].ravel()[bright])) for c in (yel, grn, red))
    v_sun = float(np.median(vr[bright]))
    dark = np.clip((v_sun - v) / max(v_sun * 0.6, 1.0), 0.0, 1.0)
    v_hi = float(np.clip(np.percentile(vr, 95) + CH_V_OVER, CH_V_HI_MIN, 255.0))
    ok = ((grn <= max(g0, 0.0) + CH_GRN) & (yel <= max(y0, 0.0) + CH_YEL) & (red <= max(r0, 0.0) + CH_RED)
          & (yel >= min(y0, 0.0) - (CH_BLUE_SUN + CH_BLUE_DARK * dark))
          & (v >= CH_V_MIN) & (v <= v_hi))
    return ok.astype(np.uint8) * 255


def _texture_mask(image):
    """넓게 거친 영역(자갈·풀·흙) = True."""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    se_c = cv.getStructuringElement(cv.MORPH_ELLIPSE, (TEX_CRACK_CLOSE, TEX_CRACK_CLOSE))
    gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, se_c).astype(np.float32)
    k = TEX_WIN
    mean = cv.blur(gray, (k, k))
    sq = cv.blur(gray * gray, (k, k))
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    std = cv.GaussianBlur(std, (0, 0), TEX_SMOOTH)
    h, w = gray.shape
    ref = float(np.median(std[int(h * 0.8):, int(w * 0.3):int(w * 0.7)])) + 1e-3
    rough = (std > TEX_K * ref).astype(np.uint8)
    se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (TEX_OPEN, TEX_OPEN))
    return cv.morphologyEx(rough, cv.MORPH_OPEN, se) > 0


def _low_saturation_mask(blurred):
    """대체용 후보. 색 비율 후보가 완전히 비었을 때만 쓰인다."""
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
    bottom_band = hsv[int(hsv.shape[0] * 0.65):, :]
    sat_threshold = int(np.clip(np.percentile(bottom_band[:, :, 1], 70) + 18, 55, 150))
    value_low = int(np.clip(np.percentile(bottom_band[:, :, 2], 8) - 20, 10, 120))
    value_high = int(np.clip(np.percentile(bottom_band[:, :, 2], 98) + 15, 180, 240))
    return cv.inRange(hsv, (0, 0, value_low), (180, sat_threshold, value_high))


def _horizon_cut(candidate_mask):
    """바닥에 닿는 가장 큰 도로 몸통의 꼭대기보다 위(나무·건물·하늘)를 지운다."""
    h, w = candidate_mask.shape
    k = max(5, int(HZ_CORE * w)) | 1
    core = cv.morphologyEx(candidate_mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k)))
    n, lab, st, _ = cv.connectedComponentsWithStats(core, connectivity=8)
    bottom = np.unique(lab[int(h * 0.85):])
    bottom = bottom[bottom != 0]
    if len(bottom) == 0:
        return candidate_mask
    best = bottom[np.argmax(st[bottom, cv.CC_STAT_AREA])]
    top = int(st[best, cv.CC_STAT_TOP]) - int(HZ_MARGIN * h)
    out = candidate_mask.copy()
    out[:max(0, top)] = 0
    return out


def _build_road_candidate_mask(image):
    height, width = image.shape[:2]
    blurred = cv.GaussianBlur(image, (5, 5), 0)

    candidate_mask = _chroma_candidate(blurred)
    chroma_only = candidate_mask.copy()           # 후보가 부족할 때의 대체값
    candidate_mask[_texture_mask(blurred)] = 0

    min_pixels = max(2000, int(0.01 * height * width))
    if np.count_nonzero(candidate_mask) < min_pixels:
        candidate_mask = chroma_only if np.count_nonzero(chroma_only) > 0 else _low_saturation_mask(blurred)

    se7 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    candidate_mask = cv.morphologyEx(candidate_mask, cv.MORPH_OPEN, se7)
    candidate_mask = cv.morphologyEx(candidate_mask, cv.MORPH_CLOSE, se7)

    g = max(3, int(GAP_CLOSE * width)) | 1
    candidate_mask = cv.morphologyEx(candidate_mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (g, 1)))

    return _horizon_cut(candidate_mask)


# ---------------------------------------------------------------------------
# 2. 도로 성분 선택 + 주변 조각 병합
# ---------------------------------------------------------------------------
def _select_road_component(candidate_mask):
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(candidate_mask, connectivity=8)
    height, width = candidate_mask.shape[:2]

    def present(row_slice):
        flags = np.zeros(num_labels, dtype=bool)
        flags[np.unique(labels[row_slice])] = True
        return flags

    touches_bottom = present(slice(int(height * 0.70), None))
    touches_lower_half = present(slice(int(height * 0.45), None))
    touches_top = present(slice(0, int(height * 0.20)))

    best_label, best_score = 0, -1.0
    for label in range(1, num_labels):
        area = stats[label, cv.CC_STAT_AREA]
        if area < 500 or not touches_lower_half[label]:
            continue
        centroid_x, centroid_y = centroids[label]
        center_bonus = 1.0 - abs(centroid_x - (width / 2.0)) / (width / 2.0)
        lower_bonus = centroid_y / max(height, 1)
        bottom_bonus = 1.0 if touches_bottom[label] else 0.0
        top_penalty = 0.35 if touches_top[label] else 0.0
        score = area * (1.0 + 0.35 * center_bonus + 0.35 * lower_bonus + 0.6 * bottom_bonus - top_penalty)
        if score > best_score:
            best_score, best_label = score, label

    if best_label == 0:
        best_label = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA])) if num_labels > 1 else 0

    road_mask = np.zeros_like(candidate_mask)
    if best_label == 0:
        return road_mask
    road_mask[labels == best_label] = 255

    grow = max(5, int(0.05 * width)) | 1
    near = cv.dilate(road_mask, cv.getStructuringElement(cv.MORPH_ELLIPSE, (grow, grow)))
    min_area = max(100, int(0.002 * height * width))
    for label in np.unique(labels[near > 0]):
        if label != 0 and stats[label, cv.CC_STAT_AREA] >= min_area:
            road_mask[labels == label] = 255
    return road_mask


# ---------------------------------------------------------------------------
# 3. 구멍 메우기
# ---------------------------------------------------------------------------
def _fill_holes(road_mask):
    contours, _ = cv.findContours(road_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(road_mask)
    cv.drawContours(filled, contours, -1, 255, thickness=cv.FILLED)
    return filled


# ---------------------------------------------------------------------------
# ROI 지정
# ---------------------------------------------------------------------------
def define_roi(image):
    """반환: roi(바운딩 박스 crop), roi_mask(원본 크기 도로 마스크), roi_coords(원본 좌표 x1, y1, x2, y2)."""
    original_height, original_width = image.shape[:2]
    work, info = resize_long_side_with_info(image)        # 긴 변 640으로 줄여 탐색

    road_mask = _select_road_component(_build_road_candidate_mask(work))
    road_mask = _fill_holes(road_mask)
    if road_mask.any():
        k = max(3, int(EDGE_MARGIN * road_mask.shape[1]) * 2 + 1)
        road_mask = cv.dilate(road_mask, cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k)))

    ys, xs = np.where(road_mask > 0)
    if len(xs) == 0:
        return image.copy(), np.zeros((original_height, original_width), dtype=np.uint8), (0, 0, original_width, original_height)

    margin = 8
    x1 = max(0, int(xs.min()) - margin)
    y1 = max(0, int(ys.min()) - margin)
    x2 = min(road_mask.shape[1], int(xs.max()) + margin + 1)
    y2 = min(road_mask.shape[0], int(ys.max()) + margin + 1)

    if info["scale"] != 1.0:
        road_mask = cv.resize(road_mask, (original_width, original_height), interpolation=cv.INTER_NEAREST)
        x1, y1, x2, y2 = map_box_to_original((x1, y1, x2, y2), info)

    return image[y1:y2, x1:x2].copy(), road_mask, (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# 고정 크기 resize (비교용 옵션 --resize fixed 전용. 기본 경로는 긴 변 640)
# ---------------------------------------------------------------------------
TARGET_WIDTH = 640
TARGET_HEIGHT = 480


def resize_image_with_info(image, target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT):
    """이미지를 target 크기로 강제 리사이즈한다 (비율 변경, 작은 영상은 확대).

    반환값:
    - resized: target 크기로 직접 리사이즈된 결과
    - resize_info: 원본 좌표로 되돌리기 위한 x/y 스케일 정보
    """
    height, width = image.shape[:2]

    scale_x = target_width / width
    scale_y = target_height / height
    interpolation = cv.INTER_AREA if scale_x < 1.0 or scale_y < 1.0 else cv.INTER_LINEAR
    resized = cv.resize(image, (target_width, target_height), interpolation=interpolation)

    resize_info = {
        "scale_x": scale_x,
        "scale_y": scale_y,
        "original_width": width,
        "original_height": height,
        "resized_width": target_width,
        "resized_height": target_height,
        "target_width": target_width,
        "target_height": target_height,
    }

    return resized, resize_info
