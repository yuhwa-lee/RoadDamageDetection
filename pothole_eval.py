"""포트홀 검출기 단일 설정 평가. 전처리 후보 반복 없이 원본 좌표 픽셀 IoU 계산.

실행: python pothole_eval.py --image-dir dataset --gt-dir labeled_data
필요: src/pothole_extractor.py, src/pixel_evaluation.py 및 기존 ROI/resize 모듈.
"""
import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from src.image_ROI_Resize import define_roi, resize_image_with_info
from src.resize_long_side import resize_long_side_with_info
from src.pothole_edge_extractor import EdgePotholeConfig, detect_mask as edge_detect_mask
from src.pothole_edge_extractor import detect_potholes as edge_detect_debug, AUDIT_STAGES
from src.pixel_evaluation import restore_mask, pixel_counts, metrics


def read_pothole_color(gt_dir):
    """CVAT 클래스 RGB 색상."""
    path = gt_dir / 'labelmap.txt'
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        parts = line.split(':')
        if parts[0].strip().lower() == 'pothole':
            color = tuple(int(x) for x in parts[1].split(','))
            if len(color) != 3 or any(x < 0 or x > 255 for x in color):
                raise ValueError(f'잘못된 pothole 색상: {line}')
            return color
    raise ValueError(f'pothole 클래스를 찾지 못했습니다: {path}')


def prepare_input(original, resize_mode):
    """새 ROI → 동일 범위의 영상/마스크 crop → resize → grayscale."""
    h, w = original.shape[:2]
    _, road_mask, coords = define_roi(original)
    x1, y1, x2, y2 = map(int, coords)

    if road_mask.shape != (h, w):
        raise ValueError(
            f"도로 마스크 크기 오류: {road_mask.shape}, 원본 크기: {(h, w)}"
        )
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError(f"잘못된 ROI 좌표: {coords}")

    roi_image = original[y1:y2, x1:x2].copy()
    roi_mask = (road_mask[y1:y2, x1:x2] > 0).astype(np.uint8) * 255

    if resize_mode == "long":
        resized, info = resize_long_side_with_info(roi_image)
    elif resize_mode == "fixed":
        resized, info = resize_image_with_info(roi_image)
    else:
        raise ValueError(f"지원하지 않는 resize 방식: {resize_mode}")

    gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
    road = cv.resize(
        roi_mask, (gray.shape[1], gray.shape[0]),
        interpolation=cv.INTER_NEAREST,
    ) > 0
    return gray, {
        "roi_coords_original": (x1, y1, x2, y2),
        "resize_info": info,
        "road_mask": road,
    }


def save_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv.imencode('.png', image)
    if not ok:
        raise OSError(f'저장 실패: {path}')
    data.tofile(str(path))


def comparison(original, gt, pred):
    """원본 / 정답 / 예측 / 오류 지도. 표시용 크기만 축소."""
    p = pred > 0
    gt_view, pred_view = original.copy(), original.copy()
    gt_view[gt] = (0.55 * original[gt] + 0.45 * np.array([0,255,0])).astype(np.uint8)
    pred_view[p] = (0.55 * original[p] + 0.45 * np.array([0,0,255])).astype(np.uint8)
    errors = np.zeros_like(original)
    errors[p & gt] = (0,200,0)
    errors[p & ~gt] = (0,0,255)
    errors[~p & gt] = (255,0,0)
    panels = []
    for image, title in [(original, 'Original'), (gt_view, 'GT pothole'),
                         (pred_view, 'Prediction'), (errors, 'TP green / FP red / FN blue')]:
        h, w = image.shape[:2]
        scale = min(1.0, 600 / max(h,w))
        view = cv.resize(image, (max(1,round(w*scale)), max(1,round(h*scale))), interpolation=cv.INTER_NEAREST)
        view = cv.copyMakeBorder(view, 32, 0, 0, 0, cv.BORDER_CONSTANT, value=(35,35,35))
        cv.putText(view, title, (8,22), cv.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1, cv.LINE_AA)
        panels.append(view)
    return np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_edge_audit(output, path, debug, meta, gt):
    """검출 후에만 GT와 비교한다. 검출기에 GT를 넘기지 않는다."""
    folder = output/'edge_audit'/path.stem
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    previous_tp = None
    for order, stage in enumerate(AUDIT_STAGES):
        mask = debug['stage_masks'][stage]
        restored = restore_mask(mask, meta, gt.shape)
        counts = pixel_counts(restored, gt)
        rows.append({
            'image': path.name, 'stage_order': order, 'stage': stage,
            'candidate_count': debug['stage_counts'][stage] if stage != 'road_mask' else '',
            'gt_pixels': int(gt.sum()), **counts, **metrics(**counts),
            'gt_coverage': counts['tp']/int(gt.sum()) if gt.any() else float('nan'),
            'gt_pixels_lost_from_previous': previous_tp-counts['tp'] if previous_tp is not None else 0,
        })
        previous_tp = counts['tp']
        # 중간 마스크는 처리 해상도로 저장, CSV 지표는 원본 해상도로 계산.
        save_png(folder/f'{order:02d}_{stage}.png', mask)
    save_png(folder/'canny_edges.png', debug['edges'])
    if 'smoothed' in debug:
        save_png(folder/'smoothed.png', debug['smoothed'])
    for name, mask in debug['barrier_masks'].items():
        save_png(folder/f'{name}_barrier.png', mask)
    for name, mask in debug.get('region_maps', {}).items():
        save_png(folder/f'{name}.png', mask)
    fields = ['image', 'candidate_id', 'branch', 'x', 'y', 'w', 'h', 'area', 'merged_members',
              'reason', 'shape_failures', 'score', 'axis_ratio', 'solidity',
              'thickness', 'fill', 'edge_support', 'boundary_contrast',
              'boundary_valid_fraction', 'contrast', 'texture_difference']
    with (folder/'candidates.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, rec in enumerate(debug['candidates']):
            row = {key: rec.get(key, '') for key in fields}
            row.update(image=path.name, candidate_id=index,
                       reason=rec['reason'] or 'accepted',
                       **dict(zip(('x','y','w','h'), rec['box'])))
            writer.writerow(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description='포트홀 검출기 1개 설정의 픽셀 IoU 평가')
    parser.add_argument('--detector', choices=['classic', 'edge'], default='classic',
                        help='classic: 기존 어두운 영역 기반, edge: 영역 후보와 외곽 검증 (config로 기존 Canny 선택 가능)')
    parser.add_argument('--image-dir', default='dataset')
    parser.add_argument('--gt-dir', default='labeled_data')
    parser.add_argument('--output', default='out_pothole_eval')
    parser.add_argument('--resize', choices=['long','fixed'], default='long')
    parser.add_argument('--limit', type=int, default=0, help='평가 이미지 수 제한. 0이면 전체')
    parser.add_argument('--config', type=Path, help='선택한 검출기 설정값을 지정한 JSON 파일(선택)')
    parser.add_argument('--seed', type=int, default=0, help='각 영상의 GrabCut 난수 시드(재현 가능한 비교)')
    parser.add_argument('--preprocess', default='none')
    parser.add_argument('--refinement', choices=['grow','none'],
                        help='grow: 후보 영역 확장, none: 이전 후보 마스크. 생략 시 config/default 사용')
    parser.add_argument('--show', action='store_true', help='비교 창 표시. 아무 키: 다음, ESC/q: 창 표시 종료')
    parser.add_argument('--audit-edge', action='store_true',
                        help='edge 단계별 정답 포함률, 후보 탈락 사유 및 중간 마스크 저장')
    args = parser.parse_args(argv)
    if args.audit_edge and args.detector != 'edge':
        parser.error('--audit-edge는 --detector edge와 함께 사용하세요.')
    # 잘못된 이름은 ROI 계산/결과 폴더 생성 전에 알린다.
    cv.setNumThreads(1)
    image_dir, gt_dir = Path(args.image_dir), Path(args.gt_dir)
    if not image_dir.is_dir() or not (gt_dir/'SegmentationClass').is_dir():
        parser.error('--image-dir 및 --gt-dir/SegmentationClass 경로를 확인하세요.')
    if args.limit < 0:
        parser.error('--limit는 0 이상이어야 합니다.')
    color = read_pothole_color(gt_dir)
    config_values = json.loads(args.config.read_text(encoding='utf-8-sig')) if args.config else {}
    config = EdgePotholeConfig(**config_values)
    detector_fn = edge_detect_mask
    config_key = 'pothole_edge_config'
    extensions = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
    paths = sorted(p for p in image_dir.rglob('*') if p.is_file() and p.suffix.lower() in extensions)
    paired, missing = [], []
    for path in paths:
        label_path = gt_dir/'SegmentationClass'/(path.stem+'.png')
        if label_path.is_file():
            paired.append((path,label_path))
        else:
            missing.append(str(path))
    if not paired:
        parser.error('같은 이름의 원본 이미지와 정답 마스크가 없습니다.')
    # 평면 CVAT 라벨 폴더와 중복 파일명을 잘못 연결하지 않도록 검사
    stems = [p.stem for p,_ in paired]
    if len(stems) != len(set(stems)):
        parser.error('하위 폴더에 같은 이름의 이미지가 중복됩니다. 입력 폴더를 구분하세요.')
    if args.limit:
        paired = paired[:args.limit]
    output = Path(args.output)/args.detector/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    (output/'run_config.json').write_text(json.dumps({
        'image_dir': str(image_dir.resolve()), 'gt_dir': str(gt_dir.resolve()),
        'resize': args.resize, 'pothole_rgb': color, 'detector_name': args.detector, 'detector': asdict(config),
        'seed_per_image': args.seed, 'opencv_version': cv.__version__,
        'preprocessing': args.preprocess.split('|'),
        'audit_edge': args.audit_edge,
        'evaluation': 'original image foreground pixel IoU; empty union = NaN',
        'images': [str(p) for p,_ in paired], 'unlabeled_images_skipped': missing,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'평가 {len(paired)}장 / 정답 없는 원본 {len(missing)}장 제외 / 전처리 후보 반복 없음')
    rows, show = [], args.show
    audit_rows = []
    for index, (path,label_path) in enumerate(paired,1):
        original = cv.imdecode(np.fromfile(str(path), dtype=np.uint8), cv.IMREAD_COLOR)
        if original is None:
            raise ValueError(f'읽을 수 없는 이미지: {path}')
        with Image.open(label_path) as label_image:
            label = np.array(label_image.convert('RGB'))
        if label.shape[:2] != original.shape[:2]:
            raise ValueError(f'원본과 정답 크기가 다릅니다: {path.name}')
        gt = np.all(label == color, axis=-1)
        start = time.perf_counter()
        cv.setRNGSeed(args.seed)
        gray, meta = prepare_input(original, args.resize)
        if args.preprocess != 'none':
            parser.error('현재 Edge 전용 실행기는 --preprocess none만 지원합니다.')
        save_png(output/'preprocessed'/(path.stem+'.png'), gray)
        meta[config_key] = config
        if args.audit_edge:
            _, edge_debug = edge_detect_debug(gray, meta['road_mask'], config,
                                              return_debug=True, audit=True)
            pred_processed = edge_debug['candidate_mask']
        else:
            pred_processed = detector_fn(gray, meta)
        pred = restore_mask(pred_processed, meta, original.shape[:2])
        seconds = time.perf_counter()-start
        if args.audit_edge:
            image_audit = save_edge_audit(output, path, edge_debug, meta, gt)
            audit_rows.extend(image_audit)
            # 중간에 실행을 멈추어도 완료된 사진의 진단 수치를 남긴다.
            write_csv(output/'edge_stage_per_image.csv', audit_rows)
            print('  GT 포함률: ' + ' -> '.join(
                f"{r['stage']}={100*r['gt_coverage']:.1f}%" for r in image_audit), flush=True)
        counts = pixel_counts(pred,gt)
        scores = metrics(**counts)
        x1,y1,x2,y2 = meta['roi_coords_original']
        rows.append({'image':path.name, **counts, **scores,
                     'gt_pixels':int(gt.sum()), 'pred_pixels':int((pred>0).sum()),
                     'gt_outside_roi_pixels':int(gt.sum()-gt[y1:y2,x1:x2].sum()),
                     'seconds':seconds})
        save_png(output/'masks'/(path.stem+'.png'),pred)
        view = comparison(original,gt,pred)
        save_png(output/'comparisons'/(path.stem+'.png'),view)
        print(f'[{index}/{len(paired)}] {path.name}: IoU={scores["iou"]:.4f} '
              f'TP={counts["tp"]} FP={counts["fp"]} FN={counts["fn"]}')
        if show:
            try:
                cv.imshow('Pothole evaluation',view)
                if (cv.waitKey(0) & 0xFF) in (27,ord('q')):
                    cv.destroyAllWindows()
                    show = False
            except cv.error:
                print('이미지 창을 지원하지 않는 환경입니다. PNG 저장을 계속합니다.')
                show = False
    if show:
        cv.destroyAllWindows()
    total = {key:sum(row[key] for row in rows) for key in ('tp','fp','fn')}
    valid = [r['iou'] for r in rows if np.isfinite(r['iou'])]
    summary = {'images':len(rows), **total, 'global_iou':metrics(**total)['iou'],
               'global_precision':metrics(**total)['precision'],
               'global_recall':metrics(**total)['recall'],
               'mean_positive_image_iou':float(np.mean([r['iou'] for r in rows if r['gt_pixels']>0]))
                   if any(r['gt_pixels']>0 for r in rows) else float('nan'),
               'mean_image_iou':float(np.mean(valid)) if valid else float('nan'),
               'empty_union_images':sum(r['tp']+r['fp']+r['fn']==0 for r in rows),
               'negative_images':sum(r['gt_pixels']==0 for r in rows),
               'negative_images_with_fp':sum(r['gt_pixels']==0 and r['fp']>0 for r in rows),
               'seconds':sum(r['seconds'] for r in rows)}
    write_csv(output/'per_image.csv',rows)
    write_csv(output/'summary.csv',[summary])
    if args.audit_edge:
        stage_summary = []
        for order, stage in enumerate(AUDIT_STAGES):
            selected = [r for r in audit_rows if r['stage'] == stage]
            counts = {key: sum(r[key] for r in selected) for key in ('tp','fp','fn')}
            gt_total = sum(r['gt_pixels'] for r in selected)
            stage_summary.append({
                'stage_order': order, 'stage': stage, 'images': len(selected),
                'candidate_count': sum(r['candidate_count'] for r in selected) if stage != 'road_mask' else '',
                'gt_pixels': gt_total, **counts, **metrics(**counts),
                'gt_coverage': counts['tp']/gt_total if gt_total else float('nan'),
                'gt_pixels_lost_from_previous': sum(r['gt_pixels_lost_from_previous'] for r in selected),
            })
        write_csv(output/'edge_stage_summary.csv', stage_summary)
        (output/'edge_audit_README.txt').write_text(
            '검출 조건은 동일하며 GT는 검출 후 평가에만 사용합니다.\n'
            'road_mask: crop/resize/원본 복원을 거친 실제 도로 마스크.\n'
            'proposal_regions: 크기 필터 전 후보. closed_edges 모드에서는 이전 closed_regions와 같습니다.\n'
            'regional: 여러 크기의 주변 대비 어두운 영역에서 강한 시작점과 연결된 약한 영역을 후보로 만듭니다.\n'
            'after_merge: 병합 후 후보. 기본 병합 비활성화에서는 proposal_regions와 같습니다.\n'
            '그 다음은 최소면적, 최대면적, 형태, 경계지지, 주변노면, 내부/주변차이, NMS/개수제한 순서입니다.\n'
            'gt_coverage는 0~1 비율이며 GT가 없는 사진은 NaN입니다.\n'
            'gt_pixels_lost_from_previous가 음수이면 병합으로 내부 경계가 채워져 GT 포함 픽셀이 증가한 것입니다.\n'
            'proposal_regions부터 포함률이 낮으면 후보 생성에서 이미 누락된 것입니다.\n'
            '후속 단계에서 포함률이 떨어지면 해당 필터가 GT 포함 영역을 제거한 것입니다.\n'
            '포함률이 높아도 FP가 클 수 있습니다. 필터를 풀면 성능이 좋아진다는 뜻은 아닙니다.\n'
            '여러 공간 크기/closing 분기의 영역 합집합을 평가하므로 같은 GT 픽셀은 중복 계산하지 않습니다.\n'
            'candidate_count는 분기별 후보 개수 합계이며 road_mask에는 적용되지 않습니다.\n'
            'CSV의 TP/FP/FN은 원본 해상도 픽셀 수입니다. candidates.csv 좌표/면적과 중간 PNG는 처리 해상도입니다.\n'
            'candidates.csv는 생성된 후보의 최초 탈락 사유입니다. accepted는 최종 채택, 빈 특징값은 계산 전에 탈락한 경우입니다.\n'
            'merged_members는 후보를 구성하는 원래 닫힌 영역 수입니다. 1이면 병합되지 않은 후보입니다.\n'
            'regional의 edge_support는 외곽 안보다 밖이 밝은 점의 비율입니다. closed_edges는 기존 Canny 지지율입니다.\n'
            'boundary_contrast는 유효 외곽 샘플의 밖-안 밝기 차 중앙값, boundary_valid_fraction은 유효 샘플 비율입니다.\n'
            'regional *_response.png는 양수 밝기 차(0~255), *_seeds.png는 강한 시작점입니다.\n'
            'regional은 Canny가 닫히지 않아도 후보를 만들지만, 마스크/영상 끝에 닿는 후보는 제외합니다.\n',
            encoding='utf-8')
    print(f'전체 합산 IoU={summary["global_iou"]:.4f} / 영상 평균 IoU={summary["mean_image_iou"]:.4f}')
    if not any(r['gt_pixels'] for r in rows):
        print('주의: 포트홀 정답 픽셀이 없습니다. 라벨 색상/데이터 구성을 확인하세요.')
    print(f'저장 위치: {output.resolve()}')


if __name__ == '__main__':
    main()
