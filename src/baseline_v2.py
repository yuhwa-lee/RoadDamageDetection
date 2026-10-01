"""
baseline_v2.py - 노이즈 적응(MAD) 임계값을 쓰는 테스트 추출기 후보

기존 detect_baseline은 그래디언트 상위 5%를 항상 에지로 잡는다(백분위수 임계값).
그러면 깨끗한 영상에서는 질감을 잡아 화면을 뒤덮고, 노이즈가 조금만 있어도
에지가 이어져 거대한 덩어리가 되어 면적 상한에 버려진다.

여기서는 임계값을 "상위 몇 %"가 아니라 "그래디언트 분포의 중앙값에서 MAD의 z배 이상
튀는가"로 정한다. 노이즈가 크면 분포가 넓어져 임계값이 올라가고, 깨끗하면 내려간다.
대비를 일정 배율로 바꿔도 중앙값과 MAD가 같이 변하므로 결과가 유지된다.
"""
import cv2
import numpy as np


def detect_baseline_mad(gray, z=6.0, ref=640, max_pixels=0.02, max_bbox=0.25,
                        dilate=True, cap=300, return_dropped=False):
    # max_pixels: 에지 픽셀 수가 화면의 이 비율을 넘는 덩어리는 노이즈가 얽힌 것으로 보고 버린다.
    #             긴 균열 하나는 박스는 커도 픽셀 수는 적으므로 살아남는다.
    # max_bbox  : 박스 면적이 화면의 이 비율을 넘는 덩어리는 화면을 뒤덮는 박스라 버린다.
    H, W = gray.shape
    s = max(H, W) / ref                                    # 640 기준 크기 비율

    g = gray.astype(np.float32)
    mag = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1))
    med = float(np.median(mag))
    mad = float(np.median(np.abs(mag - med)))
    hi = med + z * 1.4826 * mad                            # 노이즈 수준에 적응한 임계값
    edges = cv2.Canny(gray, 0.5 * hi, hi)

    if dilate:
        k = max(3, int(round(3 * s)) | 1)                  # 홀수 커널
        edges = cv2.dilate(edges, np.ones((k, k), np.uint8))

    n, _, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    boxes, dropped = [], []
    for i in range(1, n):
        x, y, w, h, px = stats[i]
        if max(w, h) < 12 * s:                             # 너무 작은 조각
            continue
        box = (int(x), int(y), int(w), int(h))
        if px > max_pixels * H * W or w * h > max_bbox * H * W:   # 얽힌 덩어리 / 화면을 뒤덮는 박스
            dropped.append(box)
            continue
        boxes.append(box)
    boxes.sort(key=lambda b: max(b[2], b[3]), reverse=True)
    boxes = boxes[:cap]
    return (boxes, dropped) if return_dropped else boxes
