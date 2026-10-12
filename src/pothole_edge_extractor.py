"""경계 기반 포트홀 후보 검출 실험 모듈 (딥러닝/GT 사용 없음).

회색조 ROI와 같은 크기의 도로 마스크를 입력받는다. ROI와 resize는 호출자가 수행.
기본: 여러 공간 크기의 주변 대비 어두운 영역 -> 형태 -> 외곽 안팎 밝기 변화 검증.
proposal_mode='closed_edges'이면 기존 Canny 닫힌 영역 방식을 비교용으로 사용.
닫힌 균열, 맨홀, 보수 흔적도 통과할 수 있으며 실데이터 성능은 아직 미검증.
도로/영상 경계로 닫힌 영역은 제외하므로 경계에 걸친 실제 포트홀도 놓칠 수 있다.

인터페이스:
    detect_potholes(gray, road_mask=None, config=None, return_debug=False)
    detect(gray, meta=None)       -> (x, y, w, h) 박스 목록
    detect_mask(gray, meta=None)  -> uint8 0/255 마스크
meta 설정 키: road_mask, pothole_edge_config

기본값 조정 기록 (train 22장 = 포트홀 있음 8 / 없음 14, 640 처리 해상도, 전처리 없음):
  주 점수 = 포트홀 있는 영상별 IoU 평균. 보조 = 정답 없는 영상 중 오검출 장수, 평균 오검출 면적 비율.
  (전체 픽셀 합산 IoU는 FP/FN의 99%가 촬영본이라 참고로만 사용)
                                     평균IoU  오검출 장수  오검출 면적
  regional 기본(seed 8, growth on)    0.052    13/14     5.68%
  regional growth off                  0.051    13/14     3.25%
  seed 12 + growth off                 0.085    13/14     2.20%
  seed 16 + growth off  <- 채택        0.106     8/14     1.63%
  seed 20 + growth off                 0.053     7/14     0.82%
  closed_edges(json)                   0.108     9/14     1.35%
  - 오검출 정체: 미세 질감이 아니라 골재 틈이 모인 '약하게 어두운 덩어리'(12~48px).
    근거: 가우시안 σ2~6 적응 평활화로도 오검출 면적이 줄지 않음(5.7% -> 5.1~6.5%).
  - seed 16: 약하게 어두운 출발점을 버림. 20은 오검출이 더 줄지만 정답 출발점까지 잃어 평균 IoU 하락.
  - growth off: 밝기 확장이 덩어리를 레이스 모양으로 키워 오검출 면적만 늘림(IoU 동일, 면적 약 절반).
  - 한계: IoU 상승은 India_002374 한 장 영향이 큼(0.18 -> 0.65). 확실한 개선은 오검출 감소.
    촬영본 포트홀 3장(IMG_1810/1811/1816)은 어떤 설정에서도 IoU 0~0.07:
    주변보다 밝고 매끈한 포트홀이라 '어두운 영역' 가정으로는 원리상 검출 불가.
"""
from dataclasses import dataclass

import cv2 as cv
import numpy as np


AUDIT_STAGES = ('road_mask', 'proposal_regions', 'after_merge', 'after_min_area', 'after_max_area',
                'after_shape', 'after_boundary', 'after_surrounding_road',
                'after_region_difference', 'final')


@dataclass(frozen=True)
class EdgePotholeConfig:
    # 공간 크기는 입력 긴 변 640 기준이며 입력 크기에 비례한다.
    blur_sigma: float = 1.0
    canny_low: float = 35.0
    canny_high: float = 90.0
    close_sizes: tuple = (3, )
    proposal_mode: str = 'regional'
    region_scales: tuple = (12.0, 24.0, 48.0)  # 주변 밝기 추정 Gaussian sigma
    region_seed_contrast: float = 16.0  # 8 -> 16: 약하게 어두운 덩어리 오검출 감소 (docstring 기록)
    region_grow_contrast: float = 3.0
    boundary_step: float = 4.0
    boundary_min_contrast: float = 3.0
    boundary_min_valid: float = 0.80
    merge_regions: bool = False  # 이전 병합 실험은 기본 비활성화
    merge_gap: float = 3.0  # 640 기준, 사이 경계의 최대 두께
    merge_gradient: float = 60.0  # Gaussian 영상의 Sobel 크기 (Canny와 같은 단위)
    merge_gray_difference: float = 12.0
    merge_texture_difference: float = 0.30
    merge_gray_range: float = 20.0  # 연쇄 병합 시 원래 영역 평균의 전체 범위 제한
    min_area: float = 120.0
    max_area_ratio: float = 0.18
    min_thickness: float = 5.0
    max_axis_ratio: float = 4.0
    min_fill: float = 0.30
    min_solidity: float = 0.55
    min_edge_support: float = 0.55
    ring_radius: float = 8.0
    min_abs_contrast: float = 5.0
    min_texture_difference: float = 0.15
    # 후보 내부가 거의 흰색이거나 주변보다 현저히 밝으면 제외.
    max_inside_median: float = 215.0  # 0~255 grayscale
    bright_pixel_threshold: float = 225.0
    max_bright_fraction: float = 0.50
    max_brighter_than_road: float = 6.0  # 내부 중앙값 - 주변 중앙값 허용치
    nms_iou: float = 0.5
    nms_containment: float = 0.85
    max_candidates: int = 50
    # Edge가 채택한 후보를 seed로, 연결된 유사 밝기 영역만 제한적으로 복원.
    enable_brightness_growth: bool = False  # 오검출 면적만 늘려 기본 비활성 (docstring 기록)
    growth_radius: float = 24.0        # 긴 변 640 기준 최대 이동 거리
    growth_tolerance: float = 20.0    # 시드 중앙 밝기와의 최대 차이
    growth_road_margin: float = 6.0   # 주변 도로보다 허용하는 최대 밝기
    growth_max_area_ratio: float = 4.0  # 시드 면적 대비 확장 상한
    growth_max_gradient: float = 65.0 # 호환용 기존 설정
    growth_background_sigma: float = 24.0  # 국소 도로 배경 추정
    growth_seed_response_min: float = 3.0   # 시드의 최소 상대 어두움
    growth_response_min: float = 1.0        # 확장 픽셀의 최소 상대 어두움
    growth_step_contrast: float = 16.0      # 단계별 밝기 변화 허용
    growth_max_bright_fraction: float = 0.35


def _kernel(size):
    return cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size))


def _odd(value):
    return max(3, int(round(value)) | 1)


def _validate(cfg):
    numeric = [v for v in vars(cfg).values() if isinstance(v, (int, float))]
    if not all(np.isfinite(v) for v in numeric):
        raise ValueError('configuration values must be finite')
    if cfg.proposal_mode not in ('regional', 'closed_edges'):
        raise ValueError('proposal_mode must be regional or closed_edges')
    if cfg.proposal_mode == 'regional' and cfg.merge_regions:
        raise ValueError('merge_regions is only supported with closed_edges')
    if not cfg.region_scales or any(not np.isfinite(k) or k <= 0 for k in cfg.region_scales):
        raise ValueError('region_scales must contain positive finite values')
    if not 0 < cfg.region_grow_contrast < cfg.region_seed_contrast:
        raise ValueError('region contrasts must satisfy 0 < grow < seed')
    if min(cfg.boundary_step, cfg.boundary_min_contrast) <= 0 or not 0 < cfg.boundary_min_valid <= 1:
        raise ValueError('invalid boundary parameters')
    if not 0 <= cfg.canny_low < cfg.canny_high or cfg.blur_sigma <= 0:
        raise ValueError('invalid Canny thresholds or blur sigma')
    if not cfg.close_sizes or any(not np.isfinite(k) or k < 3 for k in cfg.close_sizes):
        raise ValueError('close_sizes must contain sizes >= 3')
    if min(cfg.min_area, cfg.min_thickness, cfg.ring_radius) <= 0 or cfg.max_axis_ratio < 1:
        raise ValueError('invalid size/shape parameters')
    if cfg.min_abs_contrast < 0 or cfg.max_candidates < 1 or int(cfg.max_candidates) != cfg.max_candidates:
        raise ValueError('invalid contrast or candidate limit')
    if not isinstance(cfg.merge_regions, bool) or min(
            cfg.merge_gap, cfg.merge_gradient, cfg.merge_gray_difference,
            cfg.merge_texture_difference, cfg.merge_gray_range) <= 0:
        raise ValueError('invalid region merge parameters')
    if not (0 <= cfg.max_inside_median <= 255 and 0 <= cfg.bright_pixel_threshold <= 255
            and 0 <= cfg.max_bright_fraction <= 1 and 0 <= cfg.max_brighter_than_road <= 255):
        raise ValueError('invalid brightness rejection parameters')
    if (cfg.growth_background_sigma <= 0 or cfg.growth_seed_response_min < 0
            or cfg.growth_response_min < 0 or cfg.growth_step_contrast <= 0
            or not 0 <= cfg.growth_max_bright_fraction <= 1):
        raise ValueError('invalid relative brightness growth parameters')
    if (not isinstance(cfg.enable_brightness_growth, bool)
            or cfg.growth_radius <= 0 or cfg.growth_tolerance <= 0
            or cfg.growth_road_margin < 0 or cfg.growth_max_area_ratio < 1
            or cfg.growth_max_gradient <= 0):
        raise ValueError('invalid brightness growth parameters')
    if cfg.merge_texture_difference > 1:
        raise ValueError('merge_texture_difference must be <= 1')
    for name in ('max_area_ratio', 'min_fill', 'min_solidity', 'min_edge_support',
                 'min_texture_difference', 'nms_iou', 'nms_containment'):
        if not 0 < getattr(cfg, name) <= 1:
            raise ValueError(f'{name} must be in (0, 1]')


def _mask_overlap(a, b):
    """박스가 아니라 실제 후보 픽셀로 중복 판정."""
    x, y, w, h = a['box']
    xx, yy, ww, hh = b['box']
    xa, ya, xb, yb = max(x, xx), max(y, yy), min(x+w, xx+ww), min(y+h, yy+hh)
    if xa >= xb or ya >= yb:
        return 0.0, 0.0
    inter = int(np.count_nonzero(
        a['_mask'][ya-y:yb-y, xa-x:xb-x] & b['_mask'][ya-yy:yb-yy, xa-xx:xb-xx]
    ))
    return inter / (a['area']+b['area']-inter), inter / min(a['area'], b['area'])


def _merge_regions(labels, stats, forbidden, smooth, texture, valid, cfg, scale):
    """닫힌 영역끼리만 약한 경계 너머로 병합. 외부 영역/GT는 사용하지 않는다.

    짧은 수평/수직 경계 구간을 모아 인접 관계를 만든다. 공유 구간의
    70% 이상이 약할 때만 병합하고, 병합 후 그룹 특징을 매번 다시 비교한다.
    """
    count = len(stats)
    members = (~forbidden).astype(np.int32)
    if not cfg.merge_regions or members.sum() < 2:
        return labels, stats, forbidden, members
    gradient = cv.magnitude(cv.Sobel(smooth, cv.CV_32F, 1, 0),
                            cv.Sobel(smooth, cv.CV_32F, 0, 1))
    links = []
    height, width = labels.shape
    gap = max(1, int(round(cfg.merge_gap*scale)))
    for axis in (0, 1):
        for distance in range(2, min(gap+2, labels.shape[axis])):
            first = (slice(None, -distance), slice(None)) if axis == 0 else (slice(None), slice(None, -distance))
            last = (slice(distance, None), slice(None)) if axis == 0 else (slice(None), slice(distance, None))
            a, b = labels[first], labels[last]
            adjacent = (~forbidden[a]) & (~forbidden[b]) & (a != b)
            strength = np.zeros(a.shape, np.float32)
            for offset in range(1, distance):
                middle = (slice(offset, height-distance+offset), slice(None)) if axis == 0 else (slice(None), slice(offset, width-distance+offset))
                adjacent &= (labels[middle] == 0) & valid[middle]
                strength = np.maximum(strength, gradient[middle])
            yy, xx = np.nonzero(adjacent)
            if len(yy):
                aa, bb = a[yy, xx], b[yy, xx]
                keys = np.minimum(aa, bb).astype(np.int64)*count + np.maximum(aa, bb)
                links.append((keys, strength[yy, xx], yy, xx, axis, distance))
    if not links:
        return labels, stats, forbidden, members
    keys = np.concatenate([link[0] for link in links])
    strengths = np.concatenate([link[1] for link in links])
    pairs, inverse = np.unique(keys, return_inverse=True)
    contacts = np.bincount(inverse)
    weak = np.bincount(inverse, weights=strengths <= cfg.merge_gradient)
    average = np.bincount(inverse, weights=strengths)/contacts
    flat = labels.ravel()
    area = stats[:, cv.CC_STAT_AREA].astype(np.float64).copy()
    gray_sum = np.bincount(flat, weights=smooth.ravel(), minlength=count)
    tex_sum = np.bincount(flat, weights=texture.ravel(), minlength=count)
    tex_sq = np.bincount(flat, weights=np.square(texture.astype(np.float64)).ravel(), minlength=count)
    low = gray_sum/np.maximum(area, 1)
    high = low.copy()
    parent = np.arange(count)
    bounds = stats[:, :4].copy()
    bounds[:, 2:] += bounds[:, :2]  # exclusive x2/y2
    limit = cfg.max_area_ratio*int(valid.sum())

    def root(label):
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    accepted_pairs = set()
    for index in np.argsort(average, kind='stable'):
        if contacts[index] < max(2, int(round(3*scale))) or weak[index]/contacts[index] < .70:
            continue
        a, b = divmod(int(pairs[index]), count)
        a, b = root(a), root(b)
        if a == b:
            accepted_pairs.add(int(pairs[index]))
            continue
        mean_a, mean_b = gray_sum[a]/area[a], gray_sum[b]/area[b]
        ta = np.sqrt(max(0, tex_sq[a]/area[a]-(tex_sum[a]/area[a])**2))
        tb = np.sqrt(max(0, tex_sq[b]/area[b]-(tex_sum[b]/area[b])**2))
        box = np.r_[np.minimum(bounds[a, :2], bounds[b, :2]),
                    np.maximum(bounds[a, 2:], bounds[b, 2:])]
        if (abs(mean_a-mean_b) > cfg.merge_gray_difference
                or abs(ta-tb)/(ta+tb+1) > cfg.merge_texture_difference
                or max(high[a], high[b])-min(low[a], low[b]) > cfg.merge_gray_range
                or area[a]+area[b] > limit
                or (box[2]-box[0])*(box[3]-box[1]) > 2*limit):
            continue
        parent[b] = a
        area[a] += area[b]
        gray_sum[a] += gray_sum[b]
        tex_sum[a] += tex_sum[b]
        tex_sq[a] += tex_sq[b]
        low[a], high[a] = min(low[a], low[b]), max(high[a], high[b])
        bounds[a] = box
        members[a] += members[b]
        accepted_pairs.add(int(pairs[index]))
    mapping = np.array([root(i) for i in range(count)], dtype=np.int32)
    merged = mapping[labels]
    # 합쳐진 두 영역 사이의 약한 경계만 채운다. 외곽 팽창/convex hull 채우기는 하지 않는다.
    for keys, strength, yy, xx, axis, distance in links:
        selected = np.isin(keys, list(accepted_pairs)) & (strength <= cfg.merge_gradient)
        ys, xs = yy[selected], xx[selected]
        owners = mapping[labels[ys, xs]]
        for offset in range(1, distance):
            merged[ys + (offset if axis == 0 else 0), xs + (offset if axis == 1 else 0)] = owners
    new_stats = stats.copy()
    new_stats[:, cv.CC_STAT_AREA] = np.bincount(merged.ravel(), minlength=count)
    for label in np.flatnonzero((mapping == np.arange(count)) & ~forbidden):
        x, y, x2, y2 = bounds[label]
        new_stats[label, :4] = (x, y, x2-x, y2-y)
    excluded = forbidden | (mapping != np.arange(count))
    # 내부 경계 픽셀 추가로 면적 제한을 넘으면 기존 max-area 필터가 처리한다.
    return merged, new_stats, excluded, members


def _candidate_regions(smooth, edges, texture, valid, safe, cfg, scale, debug, audit):
    """후보 생성과 검증을 분리한다. regional은 Canny 폐곡선을 요구하지 않는다."""
    if cfg.proposal_mode == 'closed_edges':
        for size in sorted({_odd(k*scale) for k in cfg.close_sizes}):
            barrier = cv.morphologyEx(edges, cv.MORPH_CLOSE, _kernel(size))
            barrier[~safe] = 0
            if audit:
                debug['barrier_masks'][f'close_{size}'] = barrier.copy()
            count, labels, stats, _ = cv.connectedComponentsWithStats(
                (barrier == 0).astype(np.uint8), connectivity=4)
            outside = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0],
                                                labels[:, -1], labels[~valid])))
            forbidden = np.zeros(count, bool)
            forbidden[outside] = True
            forbidden[0] = True
            if audit:
                debug['stage_masks']['proposal_regions'][~forbidden[labels]] = 255
                debug['stage_counts']['proposal_regions'] += int((~forbidden).sum())
            labels, stats, forbidden, members = _merge_regions(
                labels, stats, forbidden, smooth, texture, valid, cfg, scale)
            yield f'edge_close_{size}', labels, stats, forbidden, members
        return

    # ROI 밖을 검게 채워 생기는 가짜 명암 차이를 막기 위한 정규화 필터.
    weights = valid.astype(np.float32)
    weighted = smooth.astype(np.float32)*weights

    def road_blur(sigma):
        denominator = cv.GaussianBlur(weights, (0, 0), sigma)
        numerator = cv.GaussianBlur(weighted, (0, 0), sigma)
        return numerator/np.maximum(denominator, 1e-6)

    for base_sigma in sorted(set(cfg.region_scales)):
        sigma = max(1.0, base_sigma*scale)
        # 미세 골재보다 넓은 변화로 영역을 만들고, 더 넓은 주변 밝기와 비교한다.
        interior = road_blur(max(.7, sigma/6.0))
        response = road_blur(sigma)-interior
        seed = (response >= cfg.region_seed_contrast) & safe
        weak = ((response >= cfg.region_grow_contrast) & valid).astype(np.uint8)
        # 작은 내부 구멍만 메운다. 외곽 팽창이나 볼록 껍질 채우기는 하지 않는다.
        _, holes, hole_stats, _ = cv.connectedComponentsWithStats(1-weak, connectivity=4)
        exterior = np.unique(np.concatenate((holes[0], holes[-1], holes[:, 0],
                                             holes[:, -1], holes[~valid])))
        fill = hole_stats[:, cv.CC_STAT_AREA] <= max(1, round(16*scale*scale))
        fill[exterior] = False
        fill[0] = False
        weak[fill[holes]] = 1
        count, labels, stats, _ = cv.connectedComponentsWithStats(weak, connectivity=8)
        seeded = np.zeros(count, bool)
        seeded[np.unique(labels[seed])] = True
        forbidden = ~seeded
        # 도로/영상 끝에서 잘려 인위적으로 닫힌 후보는 제외.
        forbidden[np.unique(labels[~safe])] = True
        forbidden[0] = True
        branch = f'regional_sigma_{base_sigma:g}'
        if audit:
            debug['stage_masks']['proposal_regions'][~forbidden[labels]] = 255
            debug['stage_counts']['proposal_regions'] += int((~forbidden).sum())
            debug['region_maps'][f'{branch}_response'] = np.clip(response, 0, 255).astype(np.uint8)
            debug['region_maps'][f'{branch}_seeds'] = seed.astype(np.uint8)*255
        yield branch, labels, stats, forbidden, (~forbidden).astype(np.int32)


def _outer_boundary_evidence(region, intensity, valid, cfg, scale):
    """후보 외곽 법선의 안/밖을 직접 샘플링. 후보를 만든 Canny 지지율과 독립적."""
    mask = region.astype(np.uint8)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    points = np.concatenate(contours)[:, 0, :]
    # 둘레의 균일한 간격으로 최대 512점만 검사한다.
    points = points[::max(1, int(np.ceil(len(points)/512)))]
    signed = cv.distanceTransform(1-mask, cv.DIST_L2, 5)-cv.distanceTransform(mask, cv.DIST_L2, 5)
    signed = cv.GaussianBlur(signed, (0, 0), max(.5, scale))
    gx, gy = cv.Sobel(signed, cv.CV_32F, 1, 0), cv.Sobel(signed, cv.CV_32F, 0, 1)
    x, y = points[:, 0], points[:, 1]
    length = np.hypot(gx[y, x], gy[y, x])
    nx, ny = gx[y, x]/np.maximum(length, 1e-6), gy[y, x]/np.maximum(length, 1e-6)
    step = max(1.0, cfg.boundary_step*scale)
    xi, yi = np.rint(x-step*nx).astype(int), np.rint(y-step*ny).astype(int)
    xo, yo = np.rint(x+step*nx).astype(int), np.rint(y+step*ny).astype(int)
    h, w = region.shape
    usable = ((length > 1e-6) & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
              & (xo >= 0) & (xo < w) & (yo >= 0) & (yo < h))
    ids = np.flatnonzero(usable)
    usable[ids] &= (region[yi[ids], xi[ids]] & ~region[yo[ids], xo[ids]]
                    & valid[yi[ids], xi[ids]] & valid[yo[ids], xo[ids]])
    ids = np.flatnonzero(usable)
    fraction = len(ids)/max(1, len(points))
    if not len(ids):
        return 0.0, 0.0, fraction
    delta = intensity[yo[ids], xo[ids]].astype(float)-intensity[yi[ids], xi[ids]].astype(float)
    # 샘플 실패도 지지하지 않는 점으로 센다. 한쪽 경계만 뚜렷한 후보를 억제.
    support = float(np.count_nonzero(delta >= cfg.boundary_min_contrast)/len(points))
    return support, float(np.median(delta)), fraction


def _grow_brightness_seed(seed, gray, valid, cfg, scale):
    """Edge 시드에서 국소 배경 대비 어두운 인접 영역 복원. GT 미사용."""
    seed = seed.astype(bool)
    seed_area = int(seed.sum())
    if not seed_area:
        return seed, 'empty_seed'

    radius = max(1, int(round(cfg.growth_radius * scale)))
    limit = cv.dilate(seed.astype(np.uint8), _kernel(2*radius+1)) > 0
    limit &= valid.astype(bool)
    if np.count_nonzero(limit & ~seed) < 8:
        return seed, 'insufficient_surrounding'

    f = gray.astype(np.float32)
    # 도로 마스크 외부의 0 값이 배경 밝기에 섞이지 않게 가중 평균 적용.
    sigma = max(2.0, cfg.growth_background_sigma * scale)
    weights = valid.astype(np.float32)
    background = cv.GaussianBlur(f * weights, (0, 0), sigma)
    denominator = cv.GaussianBlur(weights, (0, 0), sigma)
    background /= np.maximum(denominator, 1e-5)

    # 골재 텍스처의 픽셀별 변동을 완화해 포트홀 넓은 영역에 반응.
    local = cv.GaussianBlur(f, (0, 0), max(.5, 1.2*scale))
    response = background - local
    median_response = float(np.median(response[seed]))
    if median_response < cfg.growth_seed_response_min:
        return seed, 'weak_seed_response'

    # 전체 밝기 유사도가 아닌 주변 도로 대비 상대 밝기를 사용.
    eligible = (response >= cfg.growth_response_min) & limit
    eligible &= (local <= cfg.max_inside_median)
    eligible &= (local <= background + cfg.growth_road_margin)
    eligible &= (f < cfg.bright_pixel_threshold)

    # 한 번에 멀리 떨어진 픽셀로 건너뛰지 않고 연결 성분만 취함.
    eligible |= seed
    _, labels = cv.connectedComponents(eligible.astype(np.uint8), connectivity=8)
    seed_ids = np.unique(labels[seed])
    grown = np.isin(labels, seed_ids[seed_ids != 0]) & limit
    grown |= seed

    # 얇은 연결고리로 전혀 다른 노면까지 확장되는 것을 방지.
    # 추가 영역의 대부분이 시드의 주변 밝기와 크게 다르면 원래 시드 보존.
    added = grown & ~seed
    if not added.any():
        return seed, 'unchanged'
    if np.median(np.abs(local[added] - np.median(local[seed]))) > cfg.growth_tolerance + cfg.growth_step_contrast:
        return seed, 'brightness_jump'
    if np.mean(f[grown] >= cfg.bright_pixel_threshold) > cfg.growth_max_bright_fraction:
        return seed, 'too_bright_after_growth'

    area = int(grown.sum())
    if area > cfg.growth_max_area_ratio * seed_area:
        return seed, 'area_limit'
    return grown, 'grown'

def detect_potholes(gray, road_mask=None, config=None, return_debug=False, audit=False):
    cfg = config or EdgePotholeConfig()
    if not isinstance(cfg, EdgePotholeConfig):
        raise TypeError('config must be EdgePotholeConfig')
    _validate(cfg)
    if not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.dtype != np.uint8 or not gray.size:
        raise ValueError('gray must be a nonempty 2D uint8 array')
    if road_mask is not None and np.shape(road_mask) != gray.shape:
        raise ValueError('road_mask shape must match gray')
    valid = np.ones(gray.shape, bool) if road_mask is None else np.asarray(road_mask) > 0
    debug = {'valid_mask': valid.astype(np.uint8)*255, 'edges': np.zeros_like(gray),
             'proposal_masks': {}, 'seed_mask': np.zeros_like(gray),
             'candidate_mask': np.zeros_like(gray), 'candidates': [], 'kept': [],
             'region_maps': {}}
    if audit:
        debug['stage_masks'] = {name: np.zeros_like(gray) for name in AUDIT_STAGES}
        debug['stage_masks']['road_mask'] = debug['valid_mask'].copy()
        debug['stage_counts'] = {name: 0 for name in AUDIT_STAGES}
        debug['barrier_masks'] = {}

    def mark(name, x, y, local):
        if audit:
            hh, ww = local.shape
            debug['stage_masks'][name][y:y+hh,x:x+ww][local] = 255
            debug['stage_counts'][name] += 1
    if not valid.any():
        return ([], debug) if return_debug else []

    scale = max(gray.shape) / 640.0
    smooth = cv.GaussianBlur(gray, (0, 0), max(0.5, cfg.blur_sigma*scale))
    # 도로 밖을 0으로 칠하지 않는다. 그 경우 마스크 경계가 가짜 에지가 된다.
    edges = cv.Canny(smooth, cfg.canny_low, cfg.canny_high, L2gradient=True)
    safe = cv.erode(valid.astype(np.uint8), _kernel(3),
                    borderType=cv.BORDER_CONSTANT, borderValue=0) > 0
    edges[~safe] = 0
    debug['edges'] = edges.copy()
    debug['smoothed'] = smooth
    near_edge = cv.dilate(edges, _kernel(3)) > 0
    f = gray.astype(np.float32)
    texture = f - cv.GaussianBlur(f, (0, 0), max(0.5, 1.2*scale))
    min_area = max(8.0, cfg.min_area*scale*scale)
    radius = max(3, int(round(cfg.ring_radius*scale)))
    if cfg.proposal_mode == 'regional':
        radius = max(radius, int(np.ceil(cfg.boundary_step*scale))+3)
    boundary_image = cv.GaussianBlur(smooth, (0, 0), max(.5, scale))
    candidates = []

    for branch, labels, stats, forbidden, members in _candidate_regions(
            smooth, edges, texture, valid, safe, cfg, scale, debug, audit):
        proposal = np.zeros_like(gray)
        for label in range(1, len(stats)):
            if forbidden[label]:
                continue
            x, y, w, h, area = map(int, stats[label])
            if audit:
                local = labels[y:y+h,x:x+w] == label
                mark('after_merge', x, y, local)
            if area < min_area:
                if audit:
                    debug['candidates'].append({
                        'branch': branch, 'box': (x,y,w,h),
                        'area': area, 'merged_members': int(members[label]),
                        'reason': 'too_small', 'score': 0.0})
                continue
            local = labels[y:y+h, x:x+w] == label
            mark('after_min_area', x, y, local)
            proposal[y:y+h, x:x+w][local] = 255
            rec = {'branch': branch, 'box': (x,y,w,h), 'area': area,
                   'reason': '', 'score': 0.0, 'merged_members': int(members[label])}
            debug['candidates'].append(rec)
            if area > cfg.max_area_ratio*valid.sum():
                rec['reason'] = 'too_large'
                continue
            mark('after_max_area', x, y, local)
            contours, _ = cv.findContours(local.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            pts = np.concatenate(contours)
            sides = cv.minAreaRect(pts)[1]
            axis = max(sides)/max(1.0, min(sides))
            solidity = min(1.0, area/max(1.0, cv.contourArea(cv.convexHull(pts))))
            dist = cv.distanceTransform(np.pad(local.astype(np.uint8), 1), cv.DIST_L2, 5)[1:-1,1:-1]
            thickness = 2*float(np.percentile(dist[local], 90))
            rec.update(axis_ratio=axis, solidity=solidity, thickness=thickness, fill=area/(w*h))
            if (axis > cfg.max_axis_ratio or solidity < cfg.min_solidity
                    or area/(w*h) < cfg.min_fill or thickness < max(2.5, cfg.min_thickness*scale)):
                rec['reason'] = 'shape'
                if audit:
                    rec['shape_failures'] = ','.join(name for name, failed in (
                        ('axis_ratio', axis > cfg.max_axis_ratio),
                        ('solidity', solidity < cfg.min_solidity),
                        ('fill', area/(w*h) < cfg.min_fill),
                        ('thickness', thickness < max(2.5, cfg.min_thickness*scale))) if failed)
                continue
            mark('after_shape', x, y, local)
            xa, ya, xb, yb = max(0,x-radius), max(0,y-radius), min(gray.shape[1],x+w+radius), min(gray.shape[0],y+h+radius)
            region = labels[ya:yb,xa:xb] == label
            boundary = (cv.dilate(region.astype(np.uint8), _kernel(3)) > 0) & ~region
            support = float(near_edge[ya:yb,xa:xb][boundary].mean()) if boundary.any() else 0.0
            if cfg.proposal_mode == 'regional':
                support, boundary_contrast, usable = _outer_boundary_evidence(
                    region, boundary_image[ya:yb,xa:xb], valid[ya:yb,xa:xb], cfg, scale)
                rec.update(boundary_contrast=boundary_contrast, boundary_valid_fraction=usable)
                if usable < cfg.boundary_min_valid:
                    rec['reason'] = 'insufficient_boundary_samples'
                    rec['edge_support'] = support
                    continue
            rec['edge_support'] = support
            if support < cfg.min_edge_support:
                rec['reason'] = 'weak_boundary'
                continue
            mark('after_boundary', x, y, local)
            # 경계선 자체의 명도 차이만 비교하지 않도록 바로 옆 1픽셀은 제외.
            ring = (cv.dilate(region.astype(np.uint8), _kernel(2*radius+1)) > 0)
            ring &= ~region & ~boundary & valid[ya:yb,xa:xb]
            if ring.sum() < max(8, .1*area):
                rec['reason'] = 'insufficient_surrounding_road'
                continue
            mark('after_surrounding_road', x, y, local)
            crop, tex = f[ya:yb,xa:xb], texture[ya:yb,xa:xb]
            inside_median = float(np.median(crop[region]))
            outside_median = float(np.median(crop[ring]))
            contrast = outside_median - inside_median
            bright_fraction = float(np.mean(crop[region] >= cfg.bright_pixel_threshold))
            rec.update(inside_median=inside_median, outside_median=outside_median,
                       bright_fraction=bright_fraction)
            # 매우 밝은 후보 / 거의 흰색 후보 / 주변보다 현저히 밝은 후보 제거.
            if (inside_median >= cfg.max_inside_median or
                    bright_fraction >= cfg.max_bright_fraction or
                    contrast < -cfg.max_brighter_than_road):
                rec.update(contrast=contrast, reason='too_bright')
                continue
            tin, tout = float(tex[region].std()), float(tex[ring].std())
            difference = abs(tin-tout)/(tin+tout+1.0)
            rec.update(contrast=contrast, texture_difference=difference)
            if abs(contrast) < cfg.min_abs_contrast and difference < cfg.min_texture_difference:
                rec['reason'] = 'weak_region_difference'
                continue
            mark('after_region_difference', x, y, local)
            rec['score'] = float(support + difference + min(abs(contrast)/32.0, 2.0))
            candidates.append(dict(rec, _mask=local, _record=rec))
        if return_debug:
            debug['proposal_masks'][branch] = proposal

    kept = []
    # NMS는 기존 seed를 기준으로 수행해 과거와 같은 후보 선택 유지.
    for candidate in sorted(candidates, key=lambda c: -c['score']):
        rec = candidate['_record']
        if any(iou >= cfg.nms_iou or containment >= cfg.nms_containment
               for iou, containment in (_mask_overlap(candidate, old) for old in kept)):
            rec['reason'] = 'duplicate'
            continue
        if len(kept) >= cfg.max_candidates:
            rec['reason'] = 'candidate_limit'
            continue
        kept.append(candidate)
        x, y, w, h = candidate['box']
        # 기존 마스크는 비교용으로 보존.
        debug['seed_mask'][y:y+h, x:x+w][candidate['_mask']] = 255
        if cfg.enable_brightness_growth:
            pad = max(1, int(round(cfg.growth_radius*scale)))
            xa, ya = max(0, x-pad), max(0, y-pad)
            xb, yb = min(gray.shape[1], x+w+pad), min(gray.shape[0], y+h+pad)
            local_seed = np.zeros((yb-ya, xb-xa), dtype=bool)
            local_seed[y-ya:y-ya+h, x-xa:x-xa+w] = candidate['_mask']
            grown, status = _grow_brightness_seed(
                local_seed, gray[ya:yb, xa:xb], valid[ya:yb, xa:xb], cfg, scale)
            rec['growth_status'] = status
            rec['growth_pixels'] = int(grown.sum()) - int(local_seed.sum())
            debug['candidate_mask'][ya:yb, xa:xb][grown] = 255
            yy, xx = np.nonzero(grown)
            rec['box'] = (xa+int(xx.min()), ya+int(yy.min()),
                          int(xx.max()-xx.min()+1), int(yy.max()-yy.min()+1))
        else:
            rec['growth_status'] = 'disabled'
            rec['growth_pixels'] = 0
            debug['candidate_mask'][y:y+h, x:x+w][candidate['_mask']] = 255
        debug['kept'].append(rec)
    if audit:
        debug['stage_masks']['final'] = debug['candidate_mask'].copy()
        debug['stage_counts']['final'] = len(kept)
    # seed_mask는 확장 전, candidate_mask는 확장 후의 최종 예측.
    debug['region_maps']['growth_seed_mask'] = debug['seed_mask'].copy()
    debug['region_maps']['growth_final_mask'] = debug['candidate_mask'].copy()
    debug['region_maps']['growth_added_pixels'] = cv.subtract(debug['candidate_mask'], debug['seed_mask'])
    boxes = [rec['box'] for rec in debug['kept']]
    return (boxes, debug) if return_debug else boxes


def detect(gray, meta=None):
    meta = {} if meta is None else meta
    return detect_potholes(gray, meta.get('road_mask'), meta.get('pothole_edge_config'))


def detect_mask(gray, meta=None):
    meta = {} if meta is None else meta
    _, debug = detect_potholes(gray, meta.get('road_mask'),
                               meta.get('pothole_edge_config'), return_debug=True)
    return debug['candidate_mask']
