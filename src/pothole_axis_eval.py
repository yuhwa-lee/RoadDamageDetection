"""포트홀 전처리 축 비교: 같은 검출기(Edge 기본값) 앞에 전처리 후보를 하나씩 붙여 전·후를 비교한다.

실행: python src/pothole_axis_eval.py --axis noise [--arms none,gauss_s2] [--out out_round/pothole_axis.csv]
      python src/pothole_axis_eval.py --combo nlm_h5+homo_s_25,... [--orders]   (조합/순서 비교)
입력: train_data (픽셀 정답 22장 + rdd_box 박스 정답 6장). test는 사용하지 않는다.
처리: ROI -> 긴 변 640 -> 회색조 -> [전처리 후보] -> 검출. 채점 캔버스는 640 처리 좌표 (균열과 동일).

지표
  과제 요구(전처리 전·후 비교): 영상당 후보 수, 후보 면적(도로 대비 %), 영상당 처리 시간(전처리+검출, ms)
  정확도(정답 라벨 사용):
    주 점수  pos_iou   = 픽셀 정답 포트홀 영상(8장)의 영상별 IoU 평균 (전체 픽셀 합산은 촬영본이 지배해 미사용)
    박스     found05/03 = 정답 포트홀 20개 중 박스 IoU>=0.5/0.3 로 찾은 수 (pothole_box_eval과 같은 1:1 짝짓기)
    오검출   neg_imgs   = 정답 없는 영상 14장 중 검출이 생긴 장수, neg_fp% = 그 영상들의 평균 오검출 면적 비율
선택 규칙 (균열과 같은 틀)
  ① 붕괴 없음: 포트홀 영상 중 한 장이라도 IoU가 0.2 넘게 떨어지면 탈락
     (0.2: 균열의 재현율 붕괴 기준과 동일. 8장 평균에서 한 장 0.2 하락 = 평균 0.025 하락)
  ② 오검출 악화 없음: neg_fp%가 none보다 커지면 탈락 (오검출 감소가 지금 검출기의 유일한 확실한 성과)
  ③ 남은 것 중 pos_iou 최대. none과 0.01 이내면 동률 -> none 유지 (덜 처리하는 쪽).
     후보끼리 0.01 이내 동률이면 단계 수가 적은 쪽.
  ④ 한 장 빼기 검사: 포트홀 영상을 한 장씩 뺀 평균이 어느 경우에도 none보다 0.01 넘게 높아야 통과.
     (0.01 = ③의 동률 기준. 처음엔 '낮지만 않으면 통과'로 썼으나 '한 장 빼니 none과 같음'까지 통과시켜
      의도(한 장 효과 걸러내기)와 달라 같은 날, 새 데이터 보기 전에 수정)
     (2026-10-12, 라벨 6장 추가 전 확정: 8장 결과에서 nlm_h5/adapt_gauss_s2의 상승이 각각 영상 한 장
      (China_002159 / Japan_001959) 효과였기 때문. 새 데이터를 보기 전에 추가한 규칙)
판정: python src/pothole_axis_eval.py --judge out_round/pothole_axis.csv,out_round/pothole_combo.csv,...
"""
import argparse, csv, os, pickle, sys, time
from pathlib import Path
import cv2 as cv
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src'))
import pothole_box_eval as B
import pothole_eval as PE
from itertools import permutations
from candidates import AXES, IDENTITY, compose
from round_runner import apply_stages
from pothole_edge_extractor import EdgePotholeConfig, detect_potholes

CACHE = ROOT / 'out_round' / 'pothole_items.pkl'


def _signature():
    """캐시가 만들어진 뒤 라벨/박스/영상이 바뀌었는지 확인하는 지문 (파일 이름 + 수정 시각)."""
    files = sorted((ROOT / 'labeled_data' / 'SegmentationClass').glob('*.png'))
    files += sorted((ROOT / 'train_data').rglob('*.*'))
    return [(str(f.relative_to(ROOT)), f.stat().st_mtime_ns) for f in files if f.is_file()]


def load_items():
    sig = _signature()
    if CACHE.is_file():
        cached = pickle.load(open(CACHE, 'rb'))
        if isinstance(cached, dict) and cached.get('sig') == sig:
            return cached['items']
        print('라벨/영상이 바뀌어 입력 캐시를 다시 만듭니다.', flush=True)
    items = B.load(ROOT / 'train_data', ROOT / 'labeled_data')
    for it in items:
        it['road'] = it['meta']['road_mask']
        del it['meta']
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump({'sig': sig, 'items': items}, open(CACHE, 'wb'))
    n_pix = sum(it['pix'] is not None and it['pix'].any() for it in items)
    print(f'입력: 영상 {len(items)}장 (픽셀 정답 포트홀 {n_pix}장, 박스 정답 {sum(it["kind"] == "box" for it in items)}장)', flush=True)
    return items


def run_arm(arm, items, cfg):
    per, cand, area, secs = {}, [], [], []
    f5 = f3 = ngt = 0; neg = []
    for it in items:
        t0 = time.perf_counter()
        g = apply_stages(arm['stages'], it['gray'], {'road_mask': it['road']})
        boxes, dbg = detect_potholes(g, it['road'], cfg, return_debug=True)
        secs.append(time.perf_counter() - t0)
        m = dbg['candidate_mask'] > 0
        cand.append(len(boxes)); area.append(m.sum() / max(1, it['road'].sum()))
        det = [[x, y, x + w, y + h] for x, y, w, h in boxes]
        if it['gt']:
            f5 += B.match(it['gt'], det, .5); f3 += B.match(it['gt'], det, .3); ngt += len(it['gt'])
        if it['pix'] is not None:
            gt = it['pix']
            if gt.any():
                tp = (m & gt).sum(); per[it['name']] = tp / ((m | gt).sum())
            else:
                neg.append(m.sum() / m.size)
    return dict(arm=arm['name'], family=arm['family'], pos_iou=float(np.mean(list(per.values()))),
                found05=f5, found03=f3, n_gt=ngt, neg_imgs=sum(v > 0 for v in neg), n_neg=len(neg),
                neg_fp=100 * float(np.mean(neg)), cand=float(np.mean(cand)), area=100 * float(np.mean(area)),
                ms=1000 * float(np.mean(secs)), per=per)


def judge(paths):
    """①~④ 규칙으로 CSV 결과를 판정한다. 같은 이름의 후보는 처음 나온 것만 쓴다."""
    rows, seen = [], set()
    for path in paths:
        for r in csv.DictReader(open(path, encoding='utf-8-sig')):
            if r['arm'] not in seen:
                seen.add(r['arm']); rows.append(r)
    keys = [k for k in rows[0] if k.lower().endswith(('.jpg', '.jpeg', '.png'))]
    base = next(r for r in rows if r['arm'] == 'none')
    iou = lambda r: np.array([float(r[k]) for k in keys])
    b = iou(base); n = len(keys)
    loo = lambda v: (v.sum() - v) / (n - 1)          # 한 장씩 뺀 평균
    for r in rows:
        v = iou(r); r['n_stage'] = r['arm'].count('+') + 1 if r['arm'] != 'none' else 0
        if (b - v).max() > 0.2: r['v'] = '① 붕괴'
        elif float(r['neg_fp%']) > float(base['neg_fp%']) + 1e-9: r['v'] = '② 오검출↑'
        elif r['arm'] != 'none' and (loo(v) - loo(b)).min() <= 0.01:
            i = int(np.argmin(loo(v) - loo(b)))
            r['v'] = f'④ 한장효과({keys[i][:14]} 빼면 이득 {(loo(v) - loo(b))[i]:+.3f})'
        else: r['v'] = 'ok'
        r['iou'] = float(v.mean())
    ok = [r for r in rows if r['v'] == 'ok']
    top = max(r['iou'] for r in ok)
    tied = [r for r in ok if r['iou'] >= top - 0.01]
    best = min(tied, key=lambda r: (r['n_stage'], -r['iou']))
    if best['arm'] != 'none' and best['iou'] <= float(base['pos_iou']) + 0.01:
        best = base
    print(f'포트홀 영상 {n}장, 후보 {len(rows)}개: ' + ', '.join(f'{k} {sum(r["v"].startswith(k) for r in rows)}'
          for k in ('ok', '①', '②', '④')))
    for r in sorted(rows, key=lambda r: -r['iou'])[:15]:
        print(f"  {r['arm']:<40} IoU {r['iou']:.3f}  음성 {float(r['neg_fp%']):.2f}%  박스 {r['found05']}/{r['n_gt']}  {r['v']}")
    print(f"선택: {best['arm']} (IoU {float(best['pos_iou']):.3f})")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--axis', choices=list(AXES))
    ap.add_argument('--combo', default='', help='+로 이은 후보 이름들을 쉼표로 구분 (예: nlm_h5+homo_s_25)')
    ap.add_argument('--orders', action='store_true', help='--combo 각각의 모든 적용 순서를 비교')
    ap.add_argument('--judge', default='', help='쉼표로 구분한 결과 CSV들을 ①~④ 규칙으로 판정')
    ap.add_argument('--arms', default='', help='쉼표로 구분한 후보 이름(생략 시 축 전체)')
    ap.add_argument('--out', default=str(ROOT / 'out_round' / 'pothole_axis.csv'))
    a = ap.parse_args()
    if a.judge:
        judge(a.judge.split(',')); return
    if not (a.axis or a.combo):
        ap.error('--axis, --combo, --judge 중 하나가 필요합니다.')
    cv.setNumThreads(1)
    items = load_items(); cfg = EdgePotholeConfig()
    if a.combo:
        arms, seen = [], set()
        for c in a.combo.split(','):
            names = c.split('+')
            for order in (permutations(names) if a.orders else [tuple(names)]):
                if order not in seen:
                    seen.add(order); arms.append(IDENTITY if list(order) == ['none'] else compose(list(order)))
        a.axis = 'combo'
    else:
        arms = AXES[a.axis] if not a.arms else [x for x in AXES[a.axis] if x['name'] in a.arms.split(',')]
    out = Path(a.out); new = not out.is_file()
    pos_names = sorted(it['name'] for it in items if it['pix'] is not None and it['pix'].any())
    if not new:
        header = next(csv.reader(open(out, encoding='utf-8-sig')))
        if header[13:] != pos_names:
            raise SystemExit(f'{out}의 포트홀 영상 목록이 현재 데이터와 다릅니다(라벨 추가?). 새 --out 파일을 쓰세요.')
    with out.open('a', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['axis', 'arm', 'family', 'pos_iou', 'found05', 'found03', 'n_gt', 'neg_imgs', 'n_neg',
                        'neg_fp%', 'cand/img', 'area%', 'ms/img'] + pos_names)
        for arm in arms:
            r = run_arm(arm, items, cfg)
            print(f"{r['arm']:<22} IoU {r['pos_iou']:.3f} | 박스 {r['found05']}/{r['n_gt']} (0.3: {r['found03']}) | "
                  f"음성 {r['neg_imgs']}/{r['n_neg']}장 {r['neg_fp']:.2f}% | 후보 {r['cand']:.1f}개 면적 {r['area']:.2f}% | {r['ms']:.0f}ms", flush=True)
            w.writerow([a.axis, r['arm'], r['family'], f"{r['pos_iou']:.4f}", r['found05'], r['found03'], r['n_gt'],
                        r['neg_imgs'], r['n_neg'], f"{r['neg_fp']:.3f}", f"{r['cand']:.2f}", f"{r['area']:.3f}", f"{r['ms']:.1f}"]
                       + [f"{r['per'][k]:.4f}" for k in sorted(r['per'])])
            f.flush()


if __name__ == '__main__':
    main()
