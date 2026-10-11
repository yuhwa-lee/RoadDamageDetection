from __future__ import annotations

import cv2 as cv
import numpy as np

from resize_long_side import resize_long_side_with_info

# ---------------------------------------------------------------------------
# ROI 탐색 옵션 (결함 누락 개선 항목별 스위치. 기본값 = 개선 적용 상태)
# ---------------------------------------------------------------------------
ROI_CONFIG = {
    "downscale": True,         # 긴 변 640으로 줄여 ROI를 찾고 결과를 원본 좌표로 복원 (속도)
    "relax_dark": False,       # 명도 하한 제거: 어두운 크랙/포트홀을 후보에서 배제하지 않음
    "wide_flood": False,       # floodFill 허용오차 확대
    "grabcut": False,          # GrabCut은 결함을 배경으로 밀어내므로 기본 off
    "light_morph": False,      # 모폴로지 커널 축소 (OPEN이 도로 가장자리를 깎는 것 완화)
    "merge_components": True,  # 선택된 컴포넌트 주변 조각도 합침 (결함으로 쪼개진 도로)
    "fill_holes": True,        # 도로 마스크 내부 구멍(결함) 메움
    "hull": False,             # 도로 영역의 convex hull을 ROI 마스크로 사용
    "texture_filter": True,    # 넓게 거친 영역(자갈·풀·흙)을 후보에서 제외 (아스팔트 표본보다 TEX_K배 이상 거칠면)
    "veg_filter": False,       # 채도가 있는 초록·노랑 색상(풀·나무)을 후보에서 제외
    "method": "chroma",        # 도로 후보 방식: flood = 기존(밝기 floodFill), chroma = 색 비율 기준(그늘 허용, 풀·흙 제외)
    "horizon_cut": True,       # 굵은 도로 몸통 꼭대기보다 위의 후보(나무·건물·하늘)를 지운다
    "gap_close": 0.05,         # 가로 방향으로 영상 폭의 이 비율 이하인 틈을 메운다 (차선이 도로를 가르는 것 방지)
    "edge_margin": 0.01,       # 최종 도로 마스크를 영상 폭의 이 비율만큼 바깥으로 넓힌다 (도로 경계에 걸친 균열 보호)
}

# texture_filter / veg_filter 설정 (긴 변 640 작업 영상 기준)
# 근거 (제공 13장, 균열 추출기 영역 채점 crack_eval.py, 선정 = 재현율 0.2 넘게 무너진 영상 없음 + F2 최대)
#   texture_filter 끔 : F2 0.360 (P 0.243 / R 0.410), 정답 균열 마스크 안 90.9%, 포트홀 100%
#   texture_filter 켬 : F2 0.457 (P 0.321 / R 0.512), 정답 균열 마스크 안 88.6%, 포트홀 100%
#     - Japan_000107 자갈 주차장 제외 (마스크 면적 50% -> 39%), 균열 재현율 0.05 -> 0.44
#     - TEX_K 2.5/3/3.5/4, TEX_SMOOTH 3/6/15, TEX_OPEN 15/21/31 비교. K 2.5, smooth 6 이상, open 15/31은
#       Japan_010993(균열 많은 젖은 도로)의 재현율이 0.71 -> 0.35~0.45로 무너져 탈락. K 3.5/4는 통과하지만 F2 낮음
#     - TEX_CRACK_CLOSE 없이(0)는 모든 K에서 Japan_010993 붕괴 -> 균열 선을 먼저 지우는 단계가 필요
#   주의: 이웃 설정에서 Japan_010993이 무너지는 경계에 있는 값이라 촬영본에서 다시 확인 필요
#   해결 못 한 것: Norway_008315 풀 비탈(640으로 줄이면 아스팔트만큼 매끈해 보임), US_005996 그림자 진 도로
#   veg_filter: F2 0.369 (+0.009, 차이 0.01 이내라 효과 없음으로 판단) -> 기본 꺼짐
TEX_WIN = 9          # 국소 표준편차를 재는 창 크기
TEX_SMOOTH = 3.0     # 표준편차 지도를 살짝 펴는 가우시안 크기 (크게 펴면 차선·그림자 경계 같은 가는 선이 넓게 번져 열림으로 안 지워진다)
TEX_K = 3.0          # 도로 표본(하단 중앙) 거칠기의 몇 배 이상이면 비도로로 볼지
TEX_CRACK_CLOSE = 7  # 질감을 재기 전에 이보다 가는 어두운 선(균열)을 닫힘으로 지운다 -> 균열 많은 도로를 거칠다고 보지 않게
TEX_OPEN = 21        # 이보다 가는 거친 영역(차선, 균열, 도로 경계선)은 제외 대상에서 뺀다
VEG_H = (25, 95)     # OpenCV 색상(0~180) 중 노랑~초록 범위
VEG_S_MIN = 40       # 채도가 이 값 이상일 때만 색상을 믿는다 (회색 아스팔트는 색상이 무의미)


def _texture_mask(image):
    """넓게 거친 영역 = True. 아스팔트는 매끈하고 자갈·풀·흙은 같은 색이어도 밝기가 잘게 들쭉날쭉하다."""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if TEX_CRACK_CLOSE > 1:
        se_c = cv.getStructuringElement(cv.MORPH_ELLIPSE, (TEX_CRACK_CLOSE, TEX_CRACK_CLOSE))
        gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, se_c)
    gray = gray.astype(np.float32)
    k = TEX_WIN
    mean = cv.blur(gray, (k, k))
    sq = cv.blur(gray * gray, (k, k))
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    std = cv.GaussianBlur(std, (0, 0), TEX_SMOOTH)
    h, w = gray.shape
    ref = float(np.median(std[int(h * 0.8):, int(w * 0.3):int(w * 0.7)])) + 1e-3   # 바로 앞 도로 = 아스팔트 표본
    rough = (std > TEX_K * ref).astype(np.uint8)
    se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (TEX_OPEN, TEX_OPEN))
    return cv.morphologyEx(rough, cv.MORPH_OPEN, se) > 0


def _vegetation_mask(hsv):
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    return (h >= VEG_H[0]) & (h <= VEG_H[1]) & (s >= VEG_S_MIN)


# chroma 방식 설정 (긴 변 640 작업 영상 기준)
# 근거 (32장 = 제공 13 + 촬영 19, flood 방식과 비교. 결함 포함률은 정답 결함 픽셀 중 ROI 박스/마스크 안 비율)
#   flood   : 결함 포함 박스 0.958 / 마스크 0.888. US_005996 그늘 도로가 통째로 빠져 0.00, Norway 풀 비탈·하늘 포함
#   chroma + horizon_cut + edge_margin 0.01 (기본):
#             결함 포함 박스 0.987 / 마스크 0.970. US_005996 0.87, IMG_1817·US_004830 그늘 포함, Norway 비탈·풀 제외
#   crack_eval (d=5) 전체 F2 0.225 -> 0.226, 픽셀 IoU(영상 평균) 0.0924 -> 0.0935 로 검출 점수는 거의 같다
#     - Norway_008315 재현율 1.00 -> 0.00: 균열은 마스크 안에 있고 검출도 되지만, 차선과 이어진 큰 조각이 되어
#       crack_extractor 5단계(골짜기 검사)가 조각째 버린다. flood에 여백만 줘도 같은 일이 생긴다 (추출기 쪽 한계)
#     - edge_margin 0: India 도로·흙 경계의 균열이 마스크 경계에 걸려 0.99 -> 0.56. 0.015 이상은 IMG_1825 붕괴
#   gap_close (가로 틈 메우기, 차선이 도로를 가르는 문제): 0 / 0.03 / 0.05 / 0.08 비교
#     - 0.05 (기본): 결함 포함 박스 0.987 -> 1.000 (32장 모두), 마스크 0.970 -> 0.988
#       China_002209·Japan_010993 반대편 차선, US_005996 먼 도로가 들어옴. crack_eval F2 0.226 -> 0.228
#     - 0.03: India 붕괴(0.99 -> 0.74). 0.08: F2 0.233이지만 정밀도가 더 내려가고 도로 밖이 더 붙는다
#     - 지평선 컷 몸통을 닫힘으로 먼저 메우는 방식은 나무·하늘과 이어져(Japan_000156, Norway_008315) 쓰지 않음
#   해결 못 한 것: 도로와 같은 회색 벽(cap_day_03)·집(Japan_000156), 차량(IMG_1824)은 색이 같아 남는다.
#     젖은 노면의 밝은 반사(Norway_009625)와 노란 도색 횡단보도(IMG_1811)는 빠진다 (CH_V_HI_MIN을 230까지
#     올려도 반사는 거의 그대로이고 India의 뿌연 하늘이 들어와서 170 유지)
# 색은 밝기로 나눈 정규화 색(r, g, b 비율 x 3, 회색 = 0)으로 잰다. Lab의 a/b는 어두울수록 0으로 줄어
# 그늘진 풀·나무가 회색처럼 보이지만, 비율은 어두워도 초록이면 초록으로 남는다.
#   yel = (r+g)/2 - b  (노랑·갈색 +, 파랑 -)   grn = g - (r+b)/2  (초록 +)   red = r - (g+b)/2  (빨강·흙 +)
CH_REF = (0.80, 0.30, 0.70)   # 아스팔트 표본: 아래 20% 행, 가로 30~70% 구간
CH_GRN = 0.08        # 표본보다 초록 쪽으로 이만큼 넘으면 제외 (풀·나무: 제공 영상에서 +0.15 이상)
CH_YEL = 0.10        # 표본보다 노랑 쪽으로 이만큼 넘으면 제외 (마른 풀 비탈·흙: +0.13 이상)
CH_RED = 0.12        # 표본보다 빨강 쪽으로 이만큼 넘으면 제외 (흙·벽돌)
CH_BLUE_SUN = 0.15   # 햇빛 받은 아스팔트만큼 밝은 픽셀은 파랑 쪽으로 이만큼만 허용 (밝고 파란 = 하늘, 맑은 하늘은 -0.3 이하)
CH_BLUE_DARK = 0.80  # 어두울수록 파랑 쪽 허용을 이만큼까지 늘린다 (그늘은 하늘빛을 받아 어둡고 파래진다)
CH_V_MIN = 15        # 밝기 평균(R+G+B)/3이 이보다 어두우면 제외 (색 비율을 믿을 수 없을 만큼 어두움)
CH_V_OVER = 45       # 표본 밝기 상위 95% + 이 값보다 밝으면 제외 (하늘·흰 벽)
CH_V_HI_MIN = 170    # 위 상한의 최솟값 (표본이 그늘이면 상한이 너무 낮아져 햇빛 받은 도로가 빠진다)
HZ_CORE = 0.04       # 지평선 컷: 도로 몸통을 찾을 때 쓰는 열림 크기 (영상 폭 비율). 이보다 가는 연결(먼 도로 끝)을 끊는다
HZ_MARGIN = 0.03     # 도로 몸통 꼭대기보다 이만큼(영상 높이 비율) 위까지는 후보로 남긴다


def _chroma_candidate(blurred):
    """밝기가 아니라 색 비율로 도로 후보를 만든다.

    그늘진 아스팔트는 밝기가 크게 떨어지지만 색은 회색에서 파란 쪽으로만 움직인다.
    풀·흙·마른 비탈은 밝기가 아스팔트와 비슷해도 초록/노랑/빨강 쪽으로 움직인다.
    그래서 파란 쪽은 어두울수록 넓게, 나머지 방향은 좁게 허용한다.
    """
    f = blurred.astype(np.float32)
    s = f.sum(axis=2) + 1.0
    bn, gn, rn = (f[:, :, i] * 3.0 / s for i in range(3))
    yel, grn, red = (rn + gn) / 2 - bn, gn - (rn + bn) / 2, rn - (gn + bn) / 2
    v = s / 3.0
    h, w = v.shape
    ry, rx0, rx1 = CH_REF
    ref = (slice(int(h * ry), h), slice(int(w * rx0), int(w * rx1)))
    vr = v[ref].ravel()
    bright = vr >= np.median(vr)          # 표본 중 밝은 절반 = 햇빛 받은 아스팔트 (그늘이 섞여도 기준이 파래지지 않게)
    y0, g0, r0 = (float(np.median(c[ref].ravel()[bright])) for c in (yel, grn, red))
    v_sun = float(np.median(vr[bright]))
    dark = np.clip((v_sun - v) / max(v_sun * 0.6, 1.0), 0.0, 1.0)     # 0 = 햇빛 아스팔트만큼 밝음, 1 = 아주 어두운 그늘
    v_hi = float(np.clip(np.percentile(vr, 95) + CH_V_OVER, CH_V_HI_MIN, 255.0))
    # 허용 범위는 표본 색과 중립 회색(0)을 모두 포함하게 잡는다. 표본이 노을빛(노랑)이거나 그늘(파랑)이어도
    # 햇빛 받은 회색 아스팔트가 빠지지 않게 하기 위함
    ok = ((grn <= max(g0, 0.0) + CH_GRN) & (yel <= max(y0, 0.0) + CH_YEL) & (red <= max(r0, 0.0) + CH_RED)
          & (yel >= min(y0, 0.0) - (CH_BLUE_SUN + CH_BLUE_DARK * dark))
          & (v >= CH_V_MIN) & (v <= v_hi))
    return ok.astype(np.uint8) * 255


def _horizon_cut(candidate_mask):
    """굵은 도로 몸통의 꼭대기보다 위에 있는 후보(나무·건물·하늘)를 지운다.

    먼 도로 끝은 가늘어서 그 너머의 색이 비슷한 나무·벽과 이어지기 쉽다. 큰 열림으로 가는 연결을 끊고
    바닥에 닿는 가장 큰 몸통을 찾은 뒤, 그 꼭대기 + 여유보다 위는 후보에서 뺀다.
    """
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

# ---------------------------------------------------------------------------
# 공통 전처리 설정
# ---------------------------------------------------------------------------
TARGET_WIDTH = 640
TARGET_HEIGHT = 480


# ---------------------------------------------------------------------------
# 이미지 크기 정규화
# ---------------------------------------------------------------------------
def resize_image_with_info(image, target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT):
    """이미지를 target_size로 강제 리사이즈한다.

    반환값:
    - resized: target_size로 직접 리사이즈된 결과
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


def resize_image(image, target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT):
    """호환용 wrapper. 기존처럼 리사이즈 결과만 반환한다."""
    resized, _ = resize_image_with_info(image, target_width=target_width, target_height=target_height)
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
    if ROI_CONFIG["relax_dark"]:
        # 크랙/포트홀은 도로보다 어둡다. 하한을 없애 후보에서 배제되지 않게 한다.
        value_low = 0
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
            if ROI_CONFIG["wide_flood"]:
                diff = np.clip(local_std * 2.5 + 20, 20, 60).astype(int)
            else:
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
    if ROI_CONFIG["method"] == "chroma":
        candidate_mask = _chroma_candidate(blurred)
        flood_union = candidate_mask.copy()   # 후보가 부족할 때의 대체값도 같은 기준으로
    if ROI_CONFIG["texture_filter"]:
        candidate_mask[_texture_mask(blurred)] = 0
    if ROI_CONFIG["veg_filter"]:
        candidate_mask[_vegetation_mask(hsv)] = 0

    # GrabCut으로 road 후보를 한 번 더 분리해 sky/배경이 과하게 섞이는 현상을 줄인다.
    gc_mask = np.full((height, width), cv.GC_PR_BGD, dtype=np.uint8)
    gc_mask[: int(height * 0.18), :] = cv.GC_BGD
    gc_mask[flood_union > 0] = cv.GC_FGD
    gc_mask[low_sat_mask > 0] = cv.GC_PR_FGD

    if ROI_CONFIG["grabcut"] and np.count_nonzero(flood_union) > 0:
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
    open_size = 3 if ROI_CONFIG["light_morph"] else 7
    open_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (open_size, open_size))
    candidate_mask = cv.morphologyEx(candidate_mask, cv.MORPH_OPEN, open_kernel)
    candidate_mask = cv.morphologyEx(candidate_mask, cv.MORPH_CLOSE, kernel)
    if ROI_CONFIG["gap_close"] > 0:
        # 가로 방향의 좁은 틈(차선, 균열 띠, 그늘 경계)만 메워 차선 양쪽 도로를 한 덩어리로 잇는다.
        # 가로로만 메우므로 도로 위쪽의 나무·건물과는 이어지지 않는다
        g = max(3, int(ROI_CONFIG["gap_close"] * width)) | 1
        candidate_mask = cv.morphologyEx(candidate_mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (g, 1)))
    if ROI_CONFIG["horizon_cut"]:
        candidate_mask = _horizon_cut(candidate_mask)

    return candidate_mask


def _select_road_component(candidate_mask):
    """후보 마스크 중에서 바닥과 연결된 가장 큰 도로성 컴포넌트를 고른다.

    merge_components가 켜져 있으면, 선택된 컴포넌트 근처에 있는 다른 조각도 함께 합친다.
    (크랙/포트홀이 도로를 가르면 조각이 나뉘는데, 하나만 고르면 반대편 결함이 누락된다.)
    """
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(candidate_mask, connectivity=8)
    height, width = candidate_mask.shape[:2]

    bottom_band_start = int(height * 0.70)
    lower_half_start = int(height * 0.45)
    top_end = int(height * 0.20)

    # 라벨별 "해당 행 구간에 픽셀이 있는가"를 한 번에 계산한다 (라벨마다 전체 이미지를 훑지 않는다).
    def present(row_slice):
        flags = np.zeros(num_labels, dtype=bool)
        flags[np.unique(labels[row_slice])] = True
        return flags

    touches_bottom = present(slice(bottom_band_start, None))
    touches_lower_half = present(slice(lower_half_start, None))
    touches_top = present(slice(0, top_end))

    best_label = 0
    best_score = -1.0

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
            best_score = score
            best_label = label

    if best_label == 0:
        # 후보가 부족하면 가장 큰 컴포넌트를 사용한다.
        best_label = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA])) if num_labels > 1 else 0

    road_mask = np.zeros_like(candidate_mask)
    if best_label == 0:
        return road_mask
    road_mask[labels == best_label] = 255

    if ROI_CONFIG["merge_components"]:
        # 선택된 영역을 넓게 팽창시킨 범위 안에 걸치는 조각(충분히 큰 것)을 합친다.
        grow = max(5, int(0.05 * width)) | 1
        near = cv.dilate(road_mask, cv.getStructuringElement(cv.MORPH_ELLIPSE, (grow, grow)))
        near_labels = np.unique(labels[near > 0])
        min_area = max(100, int(0.002 * height * width))
        for label in near_labels:
            if label != 0 and stats[label, cv.CC_STAT_AREA] >= min_area:
                road_mask[labels == label] = 255

    return road_mask


def _complete_road_mask(road_mask):
    """도로 마스크 안에서 결함 때문에 빠진 부분을 메운다."""
    if ROI_CONFIG["hull"]:
        contours, _ = cv.findContours(road_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            hull = cv.convexHull(np.vstack(contours))
            filled = np.zeros_like(road_mask)
            cv.fillConvexPoly(filled, hull, 255)
            return filled
    elif ROI_CONFIG["fill_holes"]:
        contours, _ = cv.findContours(road_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(road_mask)
        cv.drawContours(filled, contours, -1, 255, thickness=cv.FILLED)
        return filled
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

    original_height, original_width = image.shape[:2]

    # 긴 변 640 기준으로 줄여서 탐색한다 (고해상도 원본에서 GrabCut/floodFill은 매우 느리다).
    if ROI_CONFIG["downscale"]:
        work, info = resize_long_side_with_info(image)
    else:
        work, info = image, None

    # candidate_mask: 도로 후보 영역, road_mask: 최종 도로 영역
    candidate_mask = _build_road_candidate_mask(work)
    road_mask = _select_road_component(candidate_mask)
    road_mask = _complete_road_mask(road_mask)
    if ROI_CONFIG["edge_margin"] > 0 and road_mask.any():
        k = max(3, int(ROI_CONFIG["edge_margin"] * road_mask.shape[1]) * 2 + 1)
        road_mask = cv.dilate(road_mask, cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k)))

    ys, xs = np.where(road_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return image.copy(), np.zeros((original_height, original_width), dtype=np.uint8), (0, 0, original_width, original_height)

    margin = 8
    x1 = max(0, int(xs.min()) - margin)
    y1 = max(0, int(ys.min()) - margin)
    x2 = min(road_mask.shape[1], int(xs.max()) + margin + 1)
    y2 = min(road_mask.shape[0], int(ys.max()) + margin + 1)

    if info is not None and info["scale"] != 1.0:
        # 작은 영상에서 찾은 마스크/좌표를 원본 좌표계로 복원한다.
        road_mask = cv.resize(road_mask, (original_width, original_height), interpolation=cv.INTER_NEAREST)
        x1, y1, x2, y2 = map_box_to_original((x1, y1, x2, y2), info)

    roi = image[y1:y2, x1:x2].copy()

    # roi_mask는 원본 크기를 유지해야 이후 원본 이미지 위에 결과를 다시 얹을 수 있다.
    return roi, road_mask, (x1, y1, x2, y2)
