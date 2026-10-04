from __future__ import annotations

import cv2 as cv
import numpy as np

# ---------------------------------------------------------------------------
# 공통 전처리 설정
# ---------------------------------------------------------------------------
TARGET_WIDTH = 640
TARGET_HEIGHT = 480


# ---------------------------------------------------------------------------
# 이미지 크기 정규화
# ---------------------------------------------------------------------------
def resize_image_with_info(image, target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT):
    """이미지 비율을 유지하면서 target_size에 맞춘다. padding 방식으로 왜곡을 최소화.

    반환값:
    - canvas: padding이 적용된 리사이즈 결과
    - resize_info: 원본 좌표로 되돌리기 위한 scale/offset 정보
    """
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]

    # 원본 비율 유지: 가장 작은 scale을 선택해서 전체 이미지가 보존되도록 함
    scale = min(target_width / width, target_height / height)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    interpolation = cv.INTER_AREA if scale < 1.0 else cv.INTER_LINEAR
    resized = cv.resize(image, (new_width, new_height), interpolation=interpolation)

    # padding으로 최종 target 크기 맞춤
    canvas_shape = (target_height, target_width) if channels == 1 else (target_height, target_width, channels)
    canvas = np.zeros(canvas_shape, dtype=image.dtype)
    offset_y = (target_height - new_height) // 2
    offset_x = (target_width - new_width) // 2
    canvas[offset_y : offset_y + new_height, offset_x : offset_x + new_width] = resized

    resize_info = {
        "scale": scale,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "original_width": width,
        "original_height": height,
        "resized_width": new_width,
        "resized_height": new_height,
        "target_width": target_width,
        "target_height": target_height,
    }

    return canvas, resize_info


def resize_image(image, target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT):
    """호환용 wrapper. 기존처럼 리사이즈 결과만 반환한다."""
    resized, _ = resize_image_with_info(image, target_width=target_width, target_height=target_height)
    return resized


def map_box_to_original(box, resize_info):
    """리사이즈된 좌표계의 박스를 원본 이미지 좌표계로 되돌린다."""
    x1, y1, x2, y2 = box
    scale = resize_info["scale"]
    offset_x = resize_info["offset_x"]
    offset_y = resize_info["offset_y"]
    original_width = resize_info["original_width"]
    original_height = resize_info["original_height"]

    x1_original = int(np.floor((x1 - offset_x) / scale))
    y1_original = int(np.floor((y1 - offset_y) / scale))
    x2_original = int(np.ceil((x2 - offset_x) / scale))
    y2_original = int(np.ceil((y2 - offset_y) / scale))

    x1_original = max(0, min(original_width, x1_original))
    y1_original = max(0, min(original_height, y1_original))
    x2_original = max(0, min(original_width, x2_original))
    y2_original = max(0, min(original_height, y2_original))

    return (x1_original, y1_original, x2_original, y2_original)


def map_point_to_original(point, resize_info):
    """리사이즈된 좌표계의 점을 원본 이미지 좌표계로 되돌린다."""
    x, y = point
    scale = resize_info["scale"]
    offset_x = resize_info["offset_x"]
    offset_y = resize_info["offset_y"]
    original_width = resize_info["original_width"]
    original_height = resize_info["original_height"]

    x_original = int(np.round((x - offset_x) / scale))
    y_original = int(np.round((y - offset_y) / scale))

    x_original = max(0, min(original_width - 1, x_original))
    y_original = max(0, min(original_height - 1, y_original))

    return x_original, y_original


# ---------------------------------------------------------------------------
# ROI 지정
# ---------------------------------------------------------------------------
def _build_road_candidate_mask(image):
    """
    도로 후보 영역을 색상 기반으로 추출한다. (HSV 채널 활용)
        1) 이미지의 하단 35% 영역을 기준으로 채도와 명도의 임계값을 설정한다.
        2) 채도가 낮고 명도가 적절한 영역을 도로 후보로 간주한다.
        3) flood fill 알고리즘을 이용해 도로와 연결된 영역을 확장한다.
        4) GrabCut을 이용해 배경/하늘 영역을 제거하여 후보 마스크를 정제한다.

    반환값:
        - candidate_mask: 도로 후보 영역을 나타내는 이진 마스크
    """

    # 이미지 전처리: 블러링 + HSV 변환
    blurred = cv.GaussianBlur(image, (5, 5), 0)
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV) # 이 hsv안에는 3채널이 들어있음. hsv[:,:,0]은 hue(색상), hsv[:,:,1]은 saturation(채도), hsv[:,:,2]은 value(명도)

    # 원본 이미지의 height, width 값을 이용. 
    # bottom_band은 이미지 하단 35% 영역을 의미하며, 도로는 일반적으로 하단에 위치하고 채도가 낮다.
    height, width = image.shape[:2]
    bottom_band_start = int(height * 0.65)
    bottom_band = hsv[bottom_band_start:, :]

    # 도로는 일반적으로 하단에 위치하고 채도가 낮다.
    #
    # 1) percentile 기반으로 채도와 명도 임계값을 설정하여 도로 후보 영역을 추출한다.
    # np.percentile(a, q, axis)의 파라미터 의미는 다음과 같다 : 
    # - a : 입력 배열
    # - q : percentiles (0-100)
    # - axis : 집계를 수행할 축
    # 2) clip을 통해 임계값이 너무 낮거나 높지 않도록 제한한다.
    sat_threshold = int(np.clip(np.percentile(bottom_band[:, :, 1], 70) + 18, 55, 150))
    value_low = int(np.clip(np.percentile(bottom_band[:, :, 2], 8) - 20, 10, 120))
    value_high = int(np.clip(np.percentile(bottom_band[:, :, 2], 98) + 15, 180, 240))

    # cv.inRange() 함수는 HSV 이미지에서 특정 범위의 색상을 가진 픽셀을 선택하여 이진 마스크를 생성한다.
    low_sat_mask = cv.inRange(hsv, (0, 0, value_low), (180, sat_threshold, value_high))

    # 동작 원리
    # 1) 이미지를 여러 포인트로 나눈다. (이를 seed points라고 부른다)
    # 2) 각 seed point 영역 별로, 

    flood_union = np.zeros((height, width), dtype=np.uint8) # 빈 마스크 생성 (height, width) 크기의 0으로 채워진 배열 생성

    # y, x 영역 지정
    seed_y_positions = [
        max(0, height - max(5, height // 20)),
        max(0, int(height * 0.82)),
        max(0, int(height * 0.68)),
    ]
    seed_x_positions = np.linspace(width * 0.18, width * 0.82, num=5).astype(int)

    for seed_y in seed_y_positions:
        seed_y = int(np.clip(seed_y, 0, height - 1))

        for seed_x in seed_x_positions:
            seed_x = int(np.clip(seed_x, 0, width - 1))

            sample_x1 = max(0, seed_x - 6)
            sample_x2 = min(width, seed_x + 7)
            sample_y1 = max(0, seed_y - 6)
            sample_y2 = min(height, seed_y + 1)
            # blurred 이미지에서 seed point 주변의 작은 패치를 추출하여 local_patch에 저장한다.
            local_patch = blurred[sample_y1:sample_y2, sample_x1:sample_x2]

            # lo
            if local_patch.size == 0:
                continue

            local_std = np.std(local_patch.reshape(-1, 3), axis=0)
            diff = np.clip(local_std * 1.6 + 12, 12, 35).astype(int)

            flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
            cv.floodFill(
                blurred.copy(),
                flood_mask,
                (seed_x, seed_y),
                (0, 0, 0),
                loDiff=(int(diff[0]), int(diff[1]), int(diff[2])),
                upDiff=(int(diff[0]), int(diff[1]), int(diff[2])),
                flags=4 | cv.FLOODFILL_MASK_ONLY | cv.FLOODFILL_FIXED_RANGE,
            )

            flood_union = cv.bitwise_or(flood_union, flood_mask[1:-1, 1:-1])

    candidate_mask = cv.bitwise_and(low_sat_mask, flood_union)

    # GrabCut으로 road 후보를 한 번 더 분리해 sky/배경이 과하게 섞이는 현상을 줄인다.
    gc_mask = np.full((height, width), cv.GC_PR_BGD, dtype=np.uint8)
    gc_mask[: int(height * 0.18), :] = cv.GC_BGD
    gc_mask[flood_union > 0] = cv.GC_FGD
    gc_mask[low_sat_mask > 0] = cv.GC_PR_FGD

    if np.count_nonzero(flood_union) > 0:
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv.grabCut(blurred, gc_mask, None, bgd_model, fgd_model, 4, cv.GC_INIT_WITH_MASK)
            grabcut_mask = np.where((gc_mask == cv.GC_FGD) | (gc_mask == cv.GC_PR_FGD), 255, 0).astype(np.uint8)
            candidate_mask = cv.bitwise_and(candidate_mask, grabcut_mask)
        except cv.error:
            # GrabCut이 실패하는 경우에는 기존 후보 마스크를 그대로 사용한다.
            pass

    min_pixels = max(2000, int(0.01 * height * width))
    if np.count_nonzero(candidate_mask) < min_pixels:
        candidate_mask = flood_union if np.count_nonzero(flood_union) > 0 else low_sat_mask

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    candidate_mask = cv.morphologyEx(candidate_mask, cv.MORPH_OPEN, kernel)
    candidate_mask = cv.morphologyEx(candidate_mask, cv.MORPH_CLOSE, kernel)

    return candidate_mask


def _select_road_component(candidate_mask):
    """후보 마스크 중에서 바닥과 연결된 가장 큰 도로성 컴포넌트를 고른다."""
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(candidate_mask, connectivity=8)
    height, width = candidate_mask.shape[:2]

    bottom_band_start = int(height * 0.70)
    lower_half_start = int(height * 0.45)

    best_label = 0
    best_score = -1.0

    for label in range(1, num_labels):
        area = stats[label, cv.CC_STAT_AREA]
        if area < 500:
            continue

        component_mask = labels == label
        touches_bottom = np.any(component_mask[bottom_band_start:, :])
        touches_lower_half = np.any(component_mask[lower_half_start:, :])
        touches_top = np.any(component_mask[: int(height * 0.20), :])

        if not touches_lower_half:
            continue

        centroid_x, centroid_y = centroids[label]
        center_bonus = 1.0 - abs(centroid_x - (width / 2.0)) / (width / 2.0)
        lower_bonus = centroid_y / max(height, 1)
        bottom_bonus = 1.0 if touches_bottom else 0.0

        top_penalty = 0.35 if touches_top else 0.0
        score = area * (1.0 + 0.35 * center_bonus + 0.35 * lower_bonus + 0.6 * bottom_bonus - top_penalty)

        if score > best_score:
            best_score = score
            best_label = label

    if best_label == 0:
        # 후보가 부족하면 가장 큰 컴포넌트를 사용한다.
        best_label = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA])) if num_labels > 1 else 0

    road_mask = np.zeros_like(candidate_mask)
    if best_label != 0:
        road_mask[labels == best_label] = 255

    return road_mask


def define_roi(image):
    """
    고정 비율이 아니라, 이미지 내용에서 도로로 보이는 영역을 찾는다.
    1) 색상 기반 후보를 만들고
    2) 하단과 연결된 가장 큰 컴포넌트를 선택한 뒤
    3) 그 컴포넌트의 바운딩 박스를 ROI로 사용한다.

    반환값:
    - roi: 도로 영역을 감싸는 바운딩 박스 crop
    - roi_mask: 원본 크기의 도로 마스크
    - roi_coords: 원본 좌표계에서의 ROI 경계
    """

    # candidate_mask: 도로 후보 영역, road_mask: 최종 도로 영역
    candidate_mask = _build_road_candidate_mask(image)
    road_mask = _select_road_component(candidate_mask)

    ys, xs = np.where(road_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        height, width = image.shape[:2]
        return image.copy(), np.zeros((height, width), dtype=np.uint8), (0, 0, width, height)

    height, width = image.shape[:2]
    x1 = max(0, int(xs.min()) - 8)
    y1 = max(0, int(ys.min()) - 8)
    x2 = min(width, int(xs.max()) + 9)
    y2 = min(height, int(ys.max()) + 9)

    roi = image[y1:y2, x1:x2].copy()

    # roi_mask는 원본 크기를 유지해야 이후 원본 이미지 위에 결과를 다시 얹을 수 있다.
    roi_mask = road_mask

    return roi, roi_mask, (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# 조도 불균일 보정
# ---------------------------------------------------------------------------
def correct_illumination(gray_image):
    """
    전역 밝기 변화와 지역적 조도 차이를 줄이기 위한 보정 함수.
    CLAHE로 로컬 대비를 강화하고, 큰 Gaussian blur로 조명 분포를 추정한다.
    """
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray_image)

    # 큰 커널의 Gaussian blur로 배경/조명 성분을 근사적으로 추정
    # 작은 결함이나 균열은 이 배경에 거의 영향이 없도록 한다.
    illumination_map = cv.GaussianBlur(clahe_image, (0, 0), sigmaX=45, sigmaY=45)

    # 배경을 제거하여 실제 표면 회절/결함이 드러나도록 보정
    corrected = cv.divide(clahe_image, illumination_map, scale=255.0)
    corrected = cv.normalize(corrected, None, 0, 255, cv.NORM_MINMAX)

    return clahe_image, corrected


# ---------------------------------------------------------------------------
# 조도 보정 후 결함 강조
# ---------------------------------------------------------------------------
def emphasize_defects(corrected_image):
    """Top-Hat 변환으로 작은 결함과 균열을 강조한다."""
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
    tophat = cv.morphologyEx(corrected_image, cv.MORPH_TOPHAT, kernel)
    return tophat
