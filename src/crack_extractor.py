"""
crack_extractor.py - 정식 균열 추출기 (포트홀은 별도 파일로 만든다)

테스트 추출기(baseline_v2.detect_baseline_mad)와의 역할 차이
    - 테스트 추출기: 전처리 후보를 공정하게 비교하기 위한 '줄자'. Canny 기반이라 노이즈/블러/대비 변화에 민감하다.
    - 정식 추출기  : 실제로 균열 후보를 잘 찾기 위한 결과물. "균열 = 주변보다 어둡고, 가늘고, 길쭉한 선"이라는
                    특징을 직접 이용한다.

처리 순서 (모든 크기 기준은 긴 변 640px 기준이며 영상 크기에 비례해 조정한다)
    1. black-hat      : 붓(BH_KERNEL)보다 가늘고 어두운 구조만 밝게 남긴 점수 지도
    2. 임계값 처리      : 도로 마스크 안의 점수로 MAD 기준선 두 개(Z_LOW, Z_HIGH)를 계산하고 이력 임계값 적용
    3. 형태학 정리      : 면적 MIN_AREA 미만 조각 삭제 -> CLOSE_K 닫힘으로 작은 틈 메우기
    4. 모양 검사       : 조각마다 폭/길이/길쭉함을 재서 MIN_LENGTH, MIN_ELONG 미만이면 버림

파라미터 근거 (제공 데이터 13장, 균열 정답 56개 기준. 자세한 표는 보고서/실험 기록 참고)
    BH_KERNEL=15   7~31 비교. lift가 가장 높음(0.36). 균열 검출률은 0.64~0.71로 크기에 둔감하고,
                   21/31은 lift가 비슷(0.32/0.35)하면서 박스가 적고 정밀도가 높아 대안이 될 수 있다
    Z_LOW/HIGH     (1, 2) ~ (3, 6) 다섯 조합 비교. (1.5, 3)이 검출률 0.82 / 잡음 6.5%로 균형이 가장 좋았다
    MIN_AREA=20    5~30 비교. 검출률이 떨어지기 직전 값 (30에서 0.77 -> 0.75)
    CLOSE_K=3      5px 닫힘과 방향성 닫힘은 질감 점끼리 엉겨 큰 덩어리를 만들어 탈락
    MIN_LENGTH=15  0~30 비교. 검출률이 떨어지기 직전 값 (20에서 0.75 -> 0.71)
    MIN_ELONG=4    0~5 비교. 검출률이 떨어지기 직전 값 (5에서 0.75 -> 0.70)
    선정 규칙: "균열 검출률이 떨어지기 직전까지 기준을 최대한 엄격하게"

알려진 한계
    - 나뭇잎 그림자 가장자리는 가늘고 길어서 모양 검사를 통과한다 (조도 보정 전처리가 필요한 문제)
    - 도로 마스크가 진한 그늘을 도로가 아니라고 판단하면 그 안의 균열을 놓친다
    - 한 사진 안의 원근(먼 균열은 작게 찍힘)은 고려하지 않는다
"""
import cv2
import numpy as np

REF = 640            # 크기 기준이 되는 긴 변 길이
BH_KERNEL = 15       # 1단계 black-hat 붓 지름
Z_LOW, Z_HIGH = 1.5, 3.0   # 2단계 이력 임계값 (MAD 배수)
MIN_AREA = 20        # 3단계 작은 조각 기준 (픽셀 수)
CLOSE_K = 3          # 3단계 닫힘 커널
MIN_LENGTH = 15      # 4단계 최소 길이
MIN_ELONG = 4.0      # 4단계 최소 길쭉함 (길이 / 폭)
MIN_VALID_FRAC = 0.05   # 도로 마스크가 영상의 이 비율보다 작으면 마스크를 믿지 않고 전체를 쓴다


def _scale(gray):
    return max(gray.shape) / float(REF)


def _odd(v, lo=3):
    return max(lo, int(round(v)) | 1)


# ---------------------------------------------------------------------------
# 1단계: black-hat
# ---------------------------------------------------------------------------
def blackhat_map(gray, kernel=None):
    """붓(지름 kernel)으로 어두운 틈을 메운 영상(closing) - 원본. 붓보다 가늘고 어두운 것만 큰 값이 된다."""
    kernel = BH_KERNEL if kernel is None else kernel
    k = _odd(kernel * _scale(gray))
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se).astype(np.float32)


# ---------------------------------------------------------------------------
# 2단계: 임계값 처리 (MAD 기준선 + 이력 임계값)
# ---------------------------------------------------------------------------
def mad_threshold(values, z):
    """기준선 = 중앙값 + z * 1.4826 * MAD.  z는 '평소보다 표준편차 몇 개만큼 튀었나'로 읽는다."""
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med))) + 1e-6
    return med + z * 1.4826 * mad


def hysteresis(score, valid, z_low=None, z_high=None):
    """강한 기준선을 넘은 픽셀(씨앗)과 이어진, 약한 기준선 이상의 픽셀만 남긴다."""
    z_low = Z_LOW if z_low is None else z_low
    z_high = Z_HIGH if z_high is None else z_high
    v = score[valid]
    lo, hi = mad_threshold(v, z_low), mad_threshold(v, z_high)
    weak = ((score > lo) & valid).astype(np.uint8)
    strong = (score > hi) & valid
    n, lab = cv2.connectedComponents(weak, connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[strong])] = True
    keep[0] = False
    return keep[lab]


# ---------------------------------------------------------------------------
# 3단계: 형태학 정리
# ---------------------------------------------------------------------------
def remove_small(binary, min_area):
    """연결요소 중 픽셀 수가 min_area 미만인 조각을 지운다. 열림과 달리 가늘어도 길면 살아남는다."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    keep = st[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[lab]


def clean(binary, gray_shape):
    s = _scale(np.empty(gray_shape))
    b = remove_small(binary, max(1, int(round(MIN_AREA * s * s))))
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K))
    return cv2.morphologyEx(b.astype(np.uint8), cv2.MORPH_CLOSE, se) > 0


# ---------------------------------------------------------------------------
# 4단계: 연결요소 모양 검사
# ---------------------------------------------------------------------------
def component_shapes(binary):
    """조각마다 폭/길이/길쭉함을 잰다.
    폭   = 거리 변환(가장자리까지 거리)의 상위 10% 값 x 2  -> 조각의 두꺼운 정도
    길이 = 면적 / 폭                                       -> 구불구불해도 '쭉 폈을 때' 길이
    """
    n, lab, st, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    shapes = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in st[i])
        m = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        dt = cv2.distanceTransform(np.pad(m, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
        width = max(1.0, 2.0 * float(np.percentile(dt[m > 0], 90)))
        length = area / width
        shapes.append({"label": i, "box": (x, y, w, h), "area": area,
                       "width": width, "length": length, "elong": length / width})
    return lab, shapes


def is_crack_shape(sh, s=1.0):
    return sh["length"] >= MIN_LENGTH * s and sh["elong"] >= MIN_ELONG


# ---------------------------------------------------------------------------
# 전체
# ---------------------------------------------------------------------------
def detect_cracks(gray, road_mask=None, return_debug=False):
    """균열 후보 박스 목록 [(x, y, w, h), ...] (입력 영상 좌표)."""
    valid = np.ones(gray.shape, bool)
    if road_mask is not None and road_mask.shape == gray.shape and road_mask.mean() >= MIN_VALID_FRAC:
        valid = road_mask.astype(bool)

    score = blackhat_map(gray)                       # 1
    fg = hysteresis(score, valid)                    # 2
    fg = clean(fg, gray.shape)                       # 3
    lab, shapes = component_shapes(fg)               # 4
    s = _scale(gray)
    kept = [sh for sh in shapes if is_crack_shape(sh, s)]
    boxes = [sh["box"] for sh in kept]
    if return_debug:
        return boxes, {"score": score, "valid": valid, "fg": fg, "labels": lab, "shapes": shapes, "kept": kept}
    return boxes


def detect(gray, meta=None):
    """러너용 진입점. meta에 road_mask(입력 영상과 같은 크기)가 있으면 사용한다."""
    road = None if meta is None else meta.get("road_mask")
    return detect_cracks(gray, road)
