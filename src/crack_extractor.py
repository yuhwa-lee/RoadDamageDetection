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
    5. 골짜기 검사      : 조각 중심선의 양옆이 모두 더 밝은(골짜기) 비율이 MIN_VALLEY 미만이면 버림
                         (균열 = 양쪽보다 어두운 골짜기, 그림자 경계/질감 덩어리 = 한쪽만 밝은 계단)

파라미터 근거 (제공 데이터 13장, 균열 정답 56개 기준. 자세한 표는 보고서/실험 기록 참고)
    BH_KERNEL=15   7~31 비교. lift가 가장 높음(0.36). 균열 검출률은 0.64~0.71로 크기에 둔감하고,
                   21/31은 lift가 비슷(0.32/0.35)하면서 박스가 적고 정밀도가 높아 대안이 될 수 있다
    Z_LOW/HIGH     (1, 2) ~ (3, 6) 다섯 조합 비교. (1.5, 3)이 검출률 0.82 / 잡음 6.5%로 균형이 가장 좋았다
    MIN_AREA=20    5~30 비교. 검출률이 떨어지기 직전 값 (30에서 0.77 -> 0.75)
    CLOSE_K=3      5px 닫힘과 방향성 닫힘은 질감 점끼리 엉겨 큰 덩어리를 만들어 탈락
    MIN_LENGTH=15  0~30 비교. 검출률이 떨어지기 직전 값 (20에서 0.75 -> 0.71)
    MIN_ELONG=4    0~5 비교. 검출률이 떨어지기 직전 값 (5에서 0.75 -> 0.70)
    선정 규칙: "균열 검출률이 떨어지기 직전까지 기준을 최대한 엄격하게"  (위 1~4단계, 박스 채점 기준)

    5단계 골짜기 검사 (영역 채점 crack_eval.py, d=5, 선정 기준 = 영상 붕괴 없음 + F2 최대)
    VALLEY_DELTA=5, MIN_VALLEY=0.3
                   delta 2/3/5 x min 0~0.6 비교. 검사 없음: P 0.152 / R 0.445 / F2 0.322
                   delta 5, min 0.3: P 0.243 / R 0.410 / F2 0.360 (최고), 재현율 0.2 넘게 무너진 영상 없음
                   delta 3, min 0.5: F1은 최고(0.324)지만 F2 0.342, India(1.00->0.00)·Japan_010993 붕괴로 탈락
                   delta 3, min 0.4: F2 0.349, Japan_010993(0.70->0.21) 붕괴로 탈락
                   min을 더 올리면 '진짜 균열 + 주변 질감'이 이어진 거대 연결망이 조각째 지워져 재현율이 급락한다
                   -> 조각 안에서 비골짜기 부분만 잘라내는 방식이 다음 개선 대상
    2단계 지역 기준선 (train 22장 = 제공 13 + 촬영 9, 입력 = 노이즈 축 adapt_gauss_s2, 선정 = 붕괴 없음 + F2 최대)
    LOCAL_BLOCK=64, Z_LOW/Z_HIGH=1.0/2.5   (이전: 영상 전체 하나의 기준선, 1.5/3.0 -> F2 0.373, P 0.250, R 0.426)
                   원인: 가까이 찍은 사진은 아래쪽(카메라 가까운 곳)일수록 알갱이가 커서, 기준선 하나로는
                   아래쪽은 알갱이가 헛검출되고 위쪽은 균열이 기준을 못 넘었다 -> 블록마다 기준선을 따로 잡음
                   - 지역 기준선 + 1.0/2.5 : F2 0.460 (P 0.245 / R 0.590), 붕괴 없음 -> 선택
                   - 지역 기준선 + 1.0/2.0 : F2 0.466 (P 0.234 / R 0.619) 동점(차이 0.01 이내) -> 덜 공격적인 2.5
                   - 전체 기준선 + 1.0/2.5 : F2 0.422, India 1.00->0.00 붕괴 (기준만 낮추면 질감이 균열에 이어 붙음)
                   - 지역 기준선 + 1.5/3.0 (기준 그대로) : F2 0.368~0.381, 붕괴
                   - 블록 48 / 128 : F2 0.438 / 0.447 통과, 96 : India 붕괴 -> 블록 크기에 민감한 편. 촬영본 추가 시 재확인
                   - 전체 기준선과 지역 기준선 중 큰 값 사용 : 정밀도만 오르고 재현율이 떨어져 F2 0.285~0.332
    4.5단계 PRUNE(조각 안 비골짜기 구간 잘라내기)은 F2 0.343~0.375 로 개선이 없어 꺼둠
    전체 재확인 (train 22장, 입력 adapt_gauss_s2, 한 번에 하나씩 바꿈. 기준 F2 0.460 / P 0.245 / R 0.590)
      규칙: 기존 값보다 F2가 0.01 넘게 오르고 붕괴가 없을 때만 바꾼다 (차이 0.01 이내는 같은 성능으로 보고 유지)
      -> 바뀐 값 없음. 기존 값 그대로 유지
      Z_LOW/Z_HIGH : 1.0/2.0 0.466, 1.0/3.0 0.456 (동점). 1.25/2.5 0.432 붕괴, 1.5/2.5 0.379 붕괴, 0.75/* IMG_1817 붕괴
                     -> Z_HIGH는 2~3에서 평평, Z_LOW는 1.0 근처에서 민감
      BH_KERNEL    : 9 붕괴, 11 0.461 붕괴(China_002209), 21 0.449, 27 붕괴 -> 15 유지
      MIN_AREA     : 10 0.462, 30 0.457, 40 0.456 (평평)
      MIN_LENGTH   : 10 0.463, 20 0.459, 30 0.456 (평평)
      MIN_ELONG    : 2 0.459, 3 0.458, 5 0.463, 6 0.454 (평평)
      MIN_VALLEY   : 0 0.411, 0.2 0.441, 0.4 붕괴 -> 골짜기 검사는 유효 (0 -> 0.3 에서 F2 +0.05)
      VALLEY_DELTA : 3 0.441, 3/0.5 붕괴, 8/0.2 붕괴 -> 5 유지
      LOCAL_BLOCK  : 32 붕괴, 48 0.438, 80 0.450, 96 붕괴(0.469), 128 0.447, 192 붕괴(0.461) -> 32~192 에서 F2 0.44~0.47
    3.5단계 문맥 걸러내기 (헛검출 출처 분석: 정답 없는 사진 23%, 페인트 근처 14%, 도로 경계 근처 11%)
      PAINT_K=3, PAINT_MARGIN=4, EDGE_MARGIN=10 : F2 0.460 -> 0.472, P 0.245 -> 0.273, R 0.590 -> 0.578, 붕괴 없음
      - 페인트만: K 2 0.464, 3 0.468, 4 0.464, 6 0.460 (거의 변화 없음). 여유 7은 US_005996 붕괴
      - 도로 경계만: 3 0.464, 6 0.463, 10 0.467, 15/20은 Japan_010993 붕괴 (경계에 걸친 균열 손실)
      - 개선 폭이 작다 (+0.012). 페인트·경계 헛검출의 일부만 지워지고, 나머지 헛검출(그림자 얼룩, 보수 자국,
        정답 균열이 없는 사진)은 이 방법으로 줄지 않는다. 정밀도의 근본 한계는 '어둡고 가는 선'이라는 단서만 쓰는 것

알려진 한계
    - 나뭇잎 그림자 가장자리는 가늘고 길어서 모양 검사를 통과한다 (조도 보정 전처리가 필요한 문제)
    - 도로 마스크가 진한 그늘을 도로가 아니라고 판단하면 그 안의 균열을 놓친다
    - 한 사진 안의 원근(먼 균열은 작게 찍힘)은 고려하지 않는다
"""
import cv2
import numpy as np

REF = 640            # 크기 기준이 되는 긴 변 길이
BH_KERNEL = 15       # 1단계 black-hat 붓 지름
Z_LOW, Z_HIGH = 1.0, 2.5   # 2단계 이력 임계값 (MAD 배수)
MIN_AREA = 20        # 3단계 작은 조각 기준 (픽셀 수)
CLOSE_K = 3          # 3단계 닫힘 커널
MIN_LENGTH = 15      # 4단계 최소 길이
MIN_ELONG = 4.0      # 4단계 최소 길쭉함 (길이 / 폭)
VALLEY_DELTA = 5.0   # 5단계: 양옆이 중심보다 이만큼(밝기 단계) 이상 밝아야 골짜기로 인정
VALLEY_SIGMA = 2.0   # 5단계: 선 방향을 잴 때 쓰는 구조 텐서 평활 크기
MIN_VALLEY = 0.3     # 5단계: 중심선 중 골짜기 픽셀 비율이 이 값 미만인 조각은 버림
LOCAL_BLOCK = 64     # 2단계: 기준선을 이 크기(640 기준) 블록마다 따로 계산한다. 0이면 영상 전체에서 하나
PAINT_K = 3          # 3.5단계: >0이면 도로 밝기 중앙값 + PAINT_K*MAD보다 밝은 넓은 영역(차선·횡단보도 페인트)을 찾는다
                     #          (도로 마스크 밖도 포함해서 찾는다: 새 도로 마스크가 밝은 페인트를 빼는 경우가 있어서)
PAINT_MARGIN = 4     # 3.5단계: 페인트에서 이 거리(640 기준 px) 안의 후보 픽셀을 지운다 (페인트 가장자리의 어두운 띠)
EDGE_MARGIN = 10     # 3.5단계: >0이면 도로 마스크 경계에서 이 거리 안의 후보 픽셀을 지운다 (연석·벽·인도 경계)
PRUNE = False        # 4.5단계: 조각 안에서 골짜기가 아닌 구간만 잘라낸다 (거대 연결망 쪼개기)
PRUNE_WIN = 9        # 4.5단계: 중심선 픽셀 주변 이 크기 창 안에서 골짜기 비율을 본다 (점 하나 단위로 자르면 조각이 잘게 끊긴다)
PRUNE_FRAC = 0.5     # 4.5단계: 창 안 중심선 중 골짜기 비율이 이 값 미만인 구간을 잘라낸다
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


def local_thresholds(score, valid, z_low, z_high):
    """블록마다 (중앙값 + z·MAD) 기준선을 계산해 영상 크기로 부드럽게 펼친다.
    알갱이가 큰 구역(가까운 노면)은 기준선이 높아지고, 매끈한 구역은 낮아진다."""
    h, w = score.shape
    b = max(8, int(round(LOCAL_BLOCK * max(h, w) / float(REF))))
    gh, gw = max(1, int(np.ceil(h / b))), max(1, int(np.ceil(w / b)))
    med = np.zeros((gh, gw), np.float32)
    mad = np.zeros((gh, gw), np.float32)
    ok = np.zeros((gh, gw), bool)
    for i in range(gh):
        for j in range(gw):
            sl = (slice(i * b, (i + 1) * b), slice(j * b, (j + 1) * b))
            v = score[sl][valid[sl]]
            if v.size >= 0.25 * b * b:
                m = float(np.median(v))
                med[i, j], mad[i, j], ok[i, j] = m, float(np.median(np.abs(v - m))) + 1e-6, True
    if not ok.any():
        return np.zeros_like(score), np.zeros_like(score)
    # 표본이 부족한 블록은 전체 값으로 채운다
    vv = score[valid]
    m0 = float(np.median(vv)); d0 = float(np.median(np.abs(vv - m0))) + 1e-6
    med[~ok], mad[~ok] = m0, d0
    lo = cv2.resize(med + z_low * 1.4826 * mad, (w, h), interpolation=cv2.INTER_LINEAR)
    hi = cv2.resize(med + z_high * 1.4826 * mad, (w, h), interpolation=cv2.INTER_LINEAR)
    return lo, hi


def hysteresis(score, valid, z_low=None, z_high=None):
    """강한 기준선을 넘은 픽셀(씨앗)과 이어진, 약한 기준선 이상의 픽셀만 남긴다."""
    z_low = Z_LOW if z_low is None else z_low
    z_high = Z_HIGH if z_high is None else z_high
    v = score[valid]
    lo, hi = mad_threshold(v, z_low), mad_threshold(v, z_high)
    if LOCAL_BLOCK > 0:
        lo, hi = local_thresholds(score, valid, z_low, z_high)
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
# 5단계: 골짜기 검사
# ---------------------------------------------------------------------------
def skeletonize(mask):
    """폭 1px 중심선. opencv-contrib가 있으면 내장 함수, 없으면 같은 알고리즘(Zhang-Suen)을 직접 돈다."""
    m = mask.astype(np.uint8)
    if not m.any():
        return m.astype(bool)
    if hasattr(cv2, "ximgproc"):
        return cv2.ximgproc.thinning(m * 255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN) > 0
    return _zhang_suen(m.astype(bool))


def _zhang_suen(img):
    img = np.pad(img, 1).astype(np.uint8)
    while True:
        changed = False
        for step in (0, 1):
            p2, p3, p4 = img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:]
            p5, p6, p7 = img[2:, 2:], img[2:, 1:-1], img[2:, :-2]
            p8, p9 = img[1:-1, :-2], img[:-2, :-2]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]
            b = sum(n.astype(np.int32) for n in nb)
            a = sum(((nb[i] == 0) & (nb[(i + 1) % 8] == 1)).astype(np.int32) for i in range(8))
            c = img[1:-1, 1:-1] == 1
            if step == 0:
                cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            rm = c & (b >= 2) & (b <= 6) & (a == 1) & cond
            if rm.any():
                img[1:-1, 1:-1][rm] = 0
                changed = True
        if not changed:
            return img[1:-1, 1:-1] > 0


def valley_map(gray, skel, labels, shapes, s=1.0, delta=None):
    """중심선 픽셀마다 '양옆이 모두 더 밝은가'를 판정한다.

    선의 가로 방향(법선)은 구조 텐서로 구한다: 밝기가 가장 크게 변하는 방향 = 선을 가로지르는 방향.
    법선 양쪽으로 (조각 폭/2 + 2px) 떨어진 곳의 밝기를 재서, 둘 다 중심보다 delta 이상 밝으면 골짜기.
    """
    delta = VALLEY_DELTA if delta is None else delta
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0 * s)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    sig = VALLEY_SIGMA * s
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sig)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sig)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sig)
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy)          # 가장 큰 고유벡터 방향 = 법선

    ys, xs = np.where(skel)
    width = np.ones(labels.max() + 1, np.float32)
    for sh in shapes:
        width[sh["label"]] = sh["width"]
    r = width[labels[ys, xs]] / 2.0 + 2.0 * s
    nx, ny = np.cos(theta[ys, xs]), np.sin(theta[ys, xs])
    h, w = gray.shape

    def sample(px, py):
        px = np.clip(np.round(px).astype(int), 0, w - 1)
        py = np.clip(np.round(py).astype(int), 0, h - 1)
        return g[py, px]

    c = g[ys, xs]
    a = sample(xs + nx * r, ys + ny * r)
    b = sample(xs - nx * r, ys - ny * r)
    out = np.zeros(gray.shape, bool)
    out[ys, xs] = (a - c > delta) & (b - c > delta)
    return out


def context_guard(gray, fg, valid, s=1.0):
    """3.5단계: 균열처럼 어둡고 가늘지만 균열이 아닌 것(페인트 가장자리, 도로 경계)을 지운다."""
    out = fg.copy()
    if PAINT_K > 0 and valid.any():
        v = gray[valid].astype(np.float32)
        med = float(np.median(v)); mad = float(np.median(np.abs(v - med))) * 1.4826 + 1e-6
        paint = (gray > med + PAINT_K * mad).astype(np.uint8)
        k = _odd(3 * s)
        paint = cv2.morphologyEx(paint, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        m = _odd(2 * PAINT_MARGIN * s + 1)
        zone = cv2.dilate(paint, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (m, m))) > 0
        out &= ~zone
    if EDGE_MARGIN > 0 and not valid.all():
        m = _odd(2 * EDGE_MARGIN * s + 1)
        inner = cv2.erode(valid.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (m, m))) > 0
        out &= inner
    return out


def prune_non_valley(gray, labels, shapes, s=1.0):
    """4.5단계: 조각 안에서 '양옆이 모두 더 밝은 골짜기'가 아닌 구간만 잘라낸 마스크를 돌려준다.

    조각째 버리면 진짜 균열이 주변 질감·차선과 이어진 거대 연결망에서 균열까지 사라진다.
    1) 조각들의 중심선을 구하고 픽셀마다 골짜기 여부를 판정
    2) 중심선 픽셀마다 주변 창(PRUNE_WIN) 안의 골짜기 비율을 보고 PRUNE_FRAC 이상인 구간만 남긴다
    3) 조각의 각 픽셀은 가장 가까운 중심선 픽셀을 따라 남기거나 지운다
    """
    fg = np.isin(labels, [sh["label"] for sh in shapes]) if shapes else np.zeros(labels.shape, bool)
    if not fg.any():
        return fg
    skel = skeletonize(fg)
    vmap = valley_map(gray, skel, labels, shapes, s)
    k = _odd(PRUNE_WIN * s)
    n_sk = cv2.boxFilter(skel.astype(np.float32), -1, (k, k), normalize=False)
    n_v = cv2.boxFilter(vmap.astype(np.float32), -1, (k, k), normalize=False)
    keep_sk = skel & (n_v >= PRUNE_FRAC * np.maximum(n_sk, 1))
    # 각 픽셀에서 가장 가까운 중심선 픽셀의 번호
    _, near = cv2.distanceTransformWithLabels((~skel).astype(np.uint8), cv2.DIST_L2, 5,
                                              labelType=cv2.DIST_LABEL_PIXEL)
    keep_lab = np.zeros(near.max() + 1, bool)
    keep_lab[near[keep_sk]] = True
    return fg & keep_lab[near]


def valley_fraction(gray, labels, shapes, s=1.0, delta=None):
    """조각 번호 -> 중심선 중 골짜기 픽셀 비율."""
    fg = np.isin(labels, [sh["label"] for sh in shapes]) if shapes else np.zeros(labels.shape, bool)
    skel = skeletonize(fg)
    vmap = valley_map(gray, skel, labels, shapes, s, delta)
    n = labels.max() + 1
    tot = np.bincount(labels[skel], minlength=n)
    val = np.bincount(labels[vmap], minlength=n)
    return {sh["label"]: (val[sh["label"]] / tot[sh["label"]] if tot[sh["label"]] else 0.0) for sh in shapes}


# ---------------------------------------------------------------------------
# 전체
# ---------------------------------------------------------------------------
def extract_cracks(gray, road_mask=None):
    """균열 검출 본체. 마스크와 박스를 함께 만든다 (둘 다 입력 영상 좌표).

    반환
        mask  : 입력 영상과 같은 크기의 bool 배열. 모양 검사를 통과한 조각의 픽셀만 True
        boxes : 그 조각마다 감싸는 박스 [(x, y, w, h), ...]  (mask에서 계산되므로 둘은 항상 일치)
        debug : 단계별 중간 결과 (시각화/분석용)
    """
    valid = np.ones(gray.shape, bool)
    if road_mask is not None and road_mask.shape == gray.shape and road_mask.mean() >= MIN_VALID_FRAC:
        valid = road_mask.astype(bool)

    score = blackhat_map(gray)                       # 1
    fg = hysteresis(score, valid)                    # 2
    fg = clean(fg, gray.shape)                       # 3
    fg = context_guard(gray, fg, valid, _scale(gray))                            # 3.5
    lab, shapes = component_shapes(fg)               # 4
    s = _scale(gray)
    kept = [sh for sh in shapes if is_crack_shape(sh, s)]
    if PRUNE:                                                                    # 4.5
        pruned = prune_non_valley(gray, lab, kept, s)
        lab, shapes = component_shapes(pruned)                                   # 잘라낸 뒤 모양 검사를 다시
        kept = [sh for sh in shapes if is_crack_shape(sh, s)]
    vfrac = valley_fraction(gray, lab, kept, s)                                  # 5
    for sh in kept:
        sh["valley"] = vfrac[sh["label"]]
    kept = [sh for sh in kept if sh["valley"] >= MIN_VALLEY]

    # 통과한 조각 번호의 픽셀만 칠한다 (lab: 픽셀마다 조각 번호가 적힌 지도)
    keep_id = np.zeros(lab.max() + 1, bool)
    for sh in kept:
        keep_id[sh["label"]] = True
    mask = keep_id[lab]
    boxes = [sh["box"] for sh in kept]
    debug = {"score": score, "valid": valid, "fg": fg, "labels": lab, "shapes": shapes, "kept": kept}
    return mask, boxes, debug


def detect_cracks(gray, road_mask=None, return_debug=False):
    """균열 후보 박스 목록 [(x, y, w, h), ...] (입력 영상 좌표). 기존 박스 채점용으로 유지."""
    mask, boxes, debug = extract_cracks(gray, road_mask)
    if return_debug:
        return boxes, debug
    return boxes


def detect(gray, meta=None):
    """러너용 진입점 (박스). meta에 road_mask(입력 영상과 같은 크기)가 있으면 사용한다."""
    road = None if meta is None else meta.get("road_mask")
    return detect_cracks(gray, road)


def detect_mask(gray, meta=None):
    """러너용 진입점 (마스크). (mask, boxes)를 반환한다. 영역 채점은 mask로 한다."""
    road = None if meta is None else meta.get("road_mask")
    mask, boxes, _ = extract_cracks(gray, road)
    return mask, boxes
