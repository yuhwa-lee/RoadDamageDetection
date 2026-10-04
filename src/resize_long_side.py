from __future__ import annotations

import cv2 as cv
import numpy as np

# ---------------------------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------------------------
MAX_LONG_SIDE = 640


# ---------------------------------------------------------------------------
# 긴 변 기준 resize
# ---------------------------------------------------------------------------
def resize_long_side_with_info(image, max_long_side=MAX_LONG_SIDE):
    """이미지의 긴 변이 max_long_side 이하가 되도록 비율을 유지하며 리사이즈한다.

    긴 변이 이미 max_long_side 이하이면 원본을 그대로 반환한다.

    반환값:
    - resized: 비율 유지로 리사이즈된 이미지
    - resize_info: 원본/리사이즈 크기와 좌표 복원에 필요한 스케일 정보
    """
    height, width = image.shape[:2]
    long_side = max(height, width)

    if long_side <= max_long_side:
        resize_info = {
            "scale_x": 1.0,
            "scale_y": 1.0,
            "original_width": width,
            "original_height": height,
            "resized_width": width,
            "resized_height": height,
            "target_long_side": max_long_side,
            "scale": 1.0,
        }
        return image.copy(), resize_info

    scale = max_long_side / float(long_side)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))

    interpolation = cv.INTER_AREA if scale < 1.0 else cv.INTER_LINEAR
    resized = cv.resize(image, (resized_width, resized_height), interpolation=interpolation)

    resize_info = {
        "scale_x": resized_width / float(width),
        "scale_y": resized_height / float(height),
        "original_width": width,
        "original_height": height,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "target_long_side": max_long_side,
        "scale": scale,
    }

    return resized, resize_info


def resize_long_side(image, max_long_side=MAX_LONG_SIDE):
    """호환용 wrapper. 리사이즈 결과만 반환한다."""
    resized, _ = resize_long_side_with_info(image, max_long_side=max_long_side)
    return resized


def map_box_to_original(box, resize_info):
    """리사이즈된 좌표계의 박스를 원본 이미지 좌표계로 되돌린다."""
    x1, y1, x2, y2 = box
    scale_x = resize_info["scale_x"]
    scale_y = resize_info["scale_y"]
    original_width = resize_info["original_width"]
    original_height = resize_info["original_height"]

    x1_original = int(np.floor(x1 / scale_x))
    y1_original = int(np.floor(y1 / scale_y))
    x2_original = int(np.ceil(x2 / scale_x))
    y2_original = int(np.ceil(y2 / scale_y))

    x1_original = max(0, min(original_width, x1_original))
    y1_original = max(0, min(original_height, y1_original))
    x2_original = max(0, min(original_width, x2_original))
    y2_original = max(0, min(original_height, y2_original))

    return (x1_original, y1_original, x2_original, y2_original)


def map_point_to_original(point, resize_info):
    """리사이즈된 좌표계의 점을 원본 이미지 좌표계로 되돌린다."""
    x, y = point
    scale_x = resize_info["scale_x"]
    scale_y = resize_info["scale_y"]
    original_width = resize_info["original_width"]
    original_height = resize_info["original_height"]

    x_original = int(np.round(x / scale_x))
    y_original = int(np.round(y / scale_y))

    x_original = max(0, min(original_width - 1, x_original))
    y_original = max(0, min(original_height - 1, y_original))

    return x_original, y_original