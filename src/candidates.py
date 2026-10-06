"""
candidates.py - 전처리 후보 정의 (각자 자기 축만 수정)

규칙
  - 전처리 함수는 uint8 그레이 영상을 받아 '같은 크기'의 uint8 그레이 영상을 돌려준다.
  - 커널/반경은 영상 크기에 비례하게 정의한다 (_scale: 긴 변 640 기준). 크기가 정규화되면 scale은 거의 1이 된다.
  - 방법마다 강도(파라미터) 변형은 2~3개만 둔다. 많이 두면 13장에 과적합되기 쉽다.
  - arm(이름, 계열, 함수...) 으로 후보를 만든다.
      이름   : 공백 없이 (combo 실행에서 이름으로 찾는다)
      계열   : 같은 계열이 상위 2개를 독식하지 않도록 선정 규칙에서 쓴다
      baseline=True : 기준선("그대로") 행. 선정 대상에서 빠진다
  - 1라운드는 축 안에서 한 단계만 바꾼다. 2라운드 조합은 compose()로 만든다.
"""
import cv2
import numpy as np


def _scale(g):
    return max(g.shape) / 640.0


def _odd(v, lo=3, hi=None):
    k = max(lo, int(round(v)) | 1)
    return min(k, hi) if hi else k


def arm(name, family, *stages, baseline=False):
    return {"name": name, "family": family, "stages": list(stages), "baseline": baseline}


def _tag_value(v):
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return ("%.1f" % v).replace(".", "p")


def sweep(prefix, family, builder, values):
    return [arm("%s_%s" % (prefix, _tag_value(v)), family, builder(v)) for v in values]


# ---------------------------------------------------------------------------
# 노이즈 제거
# ---------------------------------------------------------------------------
def gaussian(sigma640):
    return lambda g: cv2.GaussianBlur(g, (0, 0), sigma640 * _scale(g))


def median(k640):
    return lambda g: cv2.medianBlur(g, _odd(k640 * _scale(g)))


def bilateral(sigma_color, d640=9, sigma_space=75):
    return lambda g: cv2.bilateralFilter(g, _odd(d640 * _scale(g), hi=15), sigma_color, sigma_space)


def nlm(h):
    return lambda g: cv2.fastNlMeansDenoising(g, None, h, 7, 21)


# ---------------------------------------------------------------------------
# 조도 보정
# ---------------------------------------------------------------------------
def auto_gamma():
    """전역 감마 보정.

    원리: 영상의 평균 밝기를 기준으로 감마를 자동 추정한다.
    기능: 전체가 너무 어둡거나 밝은 경우를 빠르게 중간 톤으로 끌어온다.
    """
    def f(g):
        m = float(np.clip(g.mean() / 255.0, 0.02, 0.98))
        gamma = np.log(0.5) / np.log(m)
        lut = (np.power(np.arange(256) / 255.0, gamma) * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.LUT(g, lut)
    return f


def hist_eq():
    """히스토그램 평활화.

    원리: 밝기 분포를 전체적으로 펼쳐 대비를 강하게 만든다.
    기능: 균일한 조명에서는 효과적이지만, 조도 기울기가 크면 과도하게 튈 수 있다.
    """
    return lambda g: cv2.equalizeHist(g)


def clahe(clip, tiles=8):
    """CLAHE 기반 로컬 대비 강화.

    원리: 영상 전체가 아니라 타일 단위로 대비를 늘리고, clipLimit으로 과증폭을 막는다.
    기능: 조명 변화가 있어도 국소 구조를 더 잘 보이게 만든다.
    """
    return lambda g: cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(g)


def illumination_divide(clip=2.0, tiles=8, sigma640=45.0):
    """CLAHE 후 큰 블러로 조명 맵을 추정해 나누는 보정.

    원리: 저주파 조명 성분은 큰 Gaussian blur로 근사하고, divide로 제거한다.
    기능: 밝기 기울기와 그림자를 줄여 같은 표면을 비슷한 밝기로 맞춘다.
    """
    def f(g):
        clahe_img = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(g)
        sigma = max(1.0, sigma640 * _scale(g))
        illumination_map = cv2.GaussianBlur(clahe_img, (0, 0), sigmaX=sigma, sigmaY=sigma)
        corrected = cv2.divide(clahe_img, illumination_map, scale=255.0)
        return cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return f


def retinex_single(sigma640=45.0):
    """Single-Scale Retinex.

    원리: 로그 공간에서 원본과 큰 블러 배경의 차이를 취해 조명 성분을 분리한다.
    기능: 그림자와 넓은 조도 편차를 줄이면서 물체의 반사 성분을 상대적으로 드러낸다.
    """
    def f(g):
        gf = g.astype(np.float32) + 1.0
        sigma = max(1.0, sigma640 * _scale(g))
        bg = cv2.GaussianBlur(gf, (0, 0), sigmaX=sigma, sigmaY=sigma) + 1.0
        retinex = np.log(gf) - np.log(bg)
        return cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return f


def homomorphic_filter(sigma640=45.0):
    """Homomorphic filtering.

    원리: 조명이 곱셈 성분이라고 보고 로그 변환 후 저주파 성분을 제거한다.
    기능: 넓게 퍼진 밝기 편차를 누르고 국소 대비를 올려, 조도 불균일에 강하게 만든다.
    """
    def f(g):
        gf = g.astype(np.float32) + 1.0
        sigma = max(1.0, sigma640 * _scale(g))
        log_image = np.log(gf)
        low_freq = cv2.GaussianBlur(log_image, (0, 0), sigmaX=sigma, sigmaY=sigma)
        high_freq = log_image - low_freq
        return cv2.normalize(high_freq, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return f


def bg_flatten(k640):
    """Black-hat 기반 배경 조도 제거.

    원리: 큰 구조 요소로 배경을 추정한 뒤, 원본과의 차이(black-hat)를 이용해 어두운 구조를 강조한다.
    기능: 도로처럼 밝은 배경 위의 어두운 결함이나 굴곡을 상대적으로 도드라지게 만든다.
    """
    def f(g):
        k = _odd(k640 * _scale(g), hi=101)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.bitwise_not(cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, kernel))
    return f


# ---------------------------------------------------------------------------
# 블러 보정 (샤프닝)
# ---------------------------------------------------------------------------
def unsharp(sigma640, amount):
    def f(g):
        blur = cv2.GaussianBlur(g, (0, 0), sigma640 * _scale(g))
        return cv2.addWeighted(g, 1.0 + amount, blur, -amount, 0)
    return f


def laplacian_sharpen(alpha):
    def f(g):
        lap = cv2.Laplacian(g, cv2.CV_16S, ksize=3).astype(np.float32)
        return np.clip(g.astype(np.float32) - alpha * lap, 0, 255).astype(np.uint8)
    return f


# ---------------------------------------------------------------------------
# 후보 목록 (초안: 담당자가 자기 축의 변형/파라미터를 조정한다)
# ---------------------------------------------------------------------------
IDENTITY = arm("none", "-", baseline=True)

NOISE = [
    IDENTITY,
    arm("gauss_s1", "isotropic", gaussian(1.0)),
    arm("gauss_s2", "isotropic", gaussian(2.0)),
    arm("median_k3", "rank", median(3)),
    arm("median_k5", "rank", median(5)),
    arm("bilateral_c25", "edge-preserving", bilateral(25)),
    arm("bilateral_c50", "edge-preserving", bilateral(50)),
    arm("bilateral_c75", "edge-preserving", bilateral(75)),
    arm("nlm_h5", "edge-preserving", nlm(5)),
    arm("nlm_h10", "edge-preserving", nlm(10)),
    arm("nlm_h15", "edge-preserving", nlm(15)),
]

LIGHT = [
    IDENTITY,
    arm("autogamma", "global", auto_gamma()),
    arm("histeq", "global", hist_eq()),
    *sweep("clahe_c", "local", lambda clip: clahe(clip), [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]),
    *sweep("divide_c", "local", lambda clip: illumination_divide(clip), [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]),
    *sweep("retinex_s", "retinex", lambda sigma: retinex_single(sigma), [25.0, 35.0, 45.0, 60.0, 75.0]),
    *sweep("homo_s", "homomorphic", lambda sigma: homomorphic_filter(sigma), [25.0, 35.0, 45.0, 60.0, 75.0]),
    *sweep("bgflat_k", "morph", lambda k: bg_flatten(k), [21, 31, 41, 61, 81]),
]

# 고주파 강조(high-boost)는 언샤프와 같은 계열이라 생략했다. 필요하면 아래에 추가한다.
BLUR = [
    IDENTITY,
    arm("unsharp_s1_a08", "unsharp", unsharp(1.0, 0.8)),
    arm("unsharp_s2_a12", "unsharp", unsharp(2.0, 1.2)),
    arm("unsharp_s3_a15", "unsharp", unsharp(3.0, 1.5)),
    arm("laplacian_a05", "laplacian", laplacian_sharpen(0.5)),
    arm("laplacian_a10", "laplacian", laplacian_sharpen(1.0)),
]

AXES = {"noise": NOISE, "light": LIGHT, "blur": BLUR}


def compose(names):
    """이름 목록으로 조합 arm을 만든다. 앞에서부터 차례로 적용한다 (노이즈 -> 조도 -> 블러 순서 권장)."""
    by_name = {a["name"]: a for axis in AXES.values() for a in axis}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise SystemExit("알 수 없는 후보 이름: %s\n사용 가능: %s" % (missing, sorted(by_name)))
    stages = [s for n in names for s in by_name[n]["stages"]]
    return arm("+".join(names), "combo", *stages)
