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
    """평균 밝기가 중간(128)이 되도록 감마를 영상마다 자동으로 정한다."""
    def f(g):
        m = float(np.clip(g.mean() / 255.0, 0.02, 0.98))
        gamma = np.log(0.5) / np.log(m)
        lut = (np.power(np.arange(256) / 255.0, gamma) * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.LUT(g, lut)
    return f


def hist_eq():
    return lambda g: cv2.equalizeHist(g)


def clahe(clip, tiles=8):
    return lambda g: cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(g)


def bg_flatten(k640):
    """Black-hat 기반 배경 조도 제거: 255 - blackhat = 국소 배경을 일정하게 맞추고 어두운 구조만 남긴다."""
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
    arm("clahe_c2", "local", clahe(2.0)),
    arm("clahe_c4", "local", clahe(4.0)),
    arm("bgflat_k31", "morph", bg_flatten(31)),
    arm("bgflat_k61", "morph", bg_flatten(61)),
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
