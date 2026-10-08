"""독립적인 RDD Supervisely 박스 평가/단일 이미지 실행 CLI."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from src.image_ROI_Resize import define_roi
from ryu_detector import Config, detect, box_iou


def read_image(path):
    image = cv.imdecode(np.fromfile(str(path), np.uint8), cv.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot read image: {path}')
    return image


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv.imencode('.jpg', image)
    if not ok:
        raise ValueError(f'Cannot encode: {path}')
    data.tofile(str(path))


def infer(image, roi_mode='existing', segmentation='hst'):
    cv.setNumThreads(1)
    cv.setRNGSeed(2026)
    h, w = image.shape[:2]
    scale = min(1., 640 / max(h,w))
    resized = cv.resize(image, (max(1,round(w*scale)), max(1,round(h*scale))), interpolation=cv.INTER_AREA)
    rh, rw = resized.shape[:2]
    if roi_mode == 'existing':
        _, mask, _ = define_roi(resized)
    else:
        mask = np.full((rh,rw),255,np.uint8)
    gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
    accepted, baseline, debug = detect(gray, mask, Config(segmentation=segmentation))
    def original(c):
        result = dict(c)
        x0,y0,x1,y1 = c['box']
        result['box'] = [x0*w/rw,y0*h/rh,x1*w/rw,y1*h/rh]
        return result
    return [original(c) for c in accepted], [original(c) for c in baseline], {
        'threshold':debug['threshold'], 'roi_fraction':float(np.mean(mask>0)),
        'decisions':[original(c) for c in debug['decisions']]}, resized, mask, debug['candidate_mask']


def load_gt(path, shape):
    data = json.loads(path.read_text(encoding='utf-8'))
    h,w = shape[:2]
    if data['size'] != {'height':h,'width':w}:
        raise ValueError(f'Annotation/image size mismatch: {path}')
    boxes = []
    for obj in data['objects']:
        if obj['classTitle'] != 'pothole':
            continue
        if obj['geometryType'] != 'rectangle':
            raise ValueError(f'Non-rectangle pothole: {path}')
        (x0,y0),(x1,y1) = obj['points']['exterior']
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError(f'Invalid box: {path}')
        # 좌표 연속값으로 해석: 모든 GT/prediction에 동일하게 x2-x1 적용.
        boxes.append([x0,y0,x1,y1])
    return boxes, not data['objects']


def match(predictions, gt, threshold=.5):
    """score 내림차순, 동일 GT 중복 매칭 금지. unmatched prediction은 FP."""
    used, tp = set(), 0
    for p in sorted(predictions, key=lambda c:c['score'], reverse=True):
        options = [(box_iou(p['box'], b),j) for j,b in enumerate(gt) if j not in used]
        best,j = max(options, default=(0.,-1))
        if j >= 0 and best >= threshold:
            tp += 1
            used.add(j)
    return tp, len(predictions)-tp, len(gt)-tp


def draw(image, predictions, gt):
    canvas = image.copy()
    for boxes,color in [(gt,(0,220,0)),([c['box'] for c in predictions],(0,0,255))]:
        for b in boxes:
            x0,y0,x1,y1 = map(lambda x:int(round(x)), b)
            cv.rectangle(canvas,(x0,y0),(x1,y1),color,2)
    canvas = cv.copyMakeBorder(canvas,32,0,0,0,cv.BORDER_CONSTANT,value=(30,30,30))
    cv.putText(canvas,'GT green | prediction red',(8,22),cv.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
    return canvas


def process(job):
    image_path, ann_path, roi_mode, preview_dir, segmentation = job
    start = time.perf_counter()
    image = read_image(image_path)
    predictions, baseline, info, resized, mask, candidates = infer(image,roi_mode,segmentation)
    gt, empty = load_gt(Path(ann_path),image.shape) if ann_path else ([],False)
    if preview_dir:
        dest = Path(preview_dir)
        stem = Path(image_path).stem
        save_image(dest/f'{stem}_result.jpg',draw(image,predictions,gt))
        save_image(dest/f'{stem}_roi.jpg',mask)
        save_image(dest/f'{stem}_candidates.jpg',candidates)
    return dict(image=Path(image_path).name, gt=gt, empty_annotation=empty,
                predictions=predictions, geometry_baseline=baseline, **info,
                elapsed_seconds=time.perf_counter()-start)


def summarize(rows,key):
    counts = np.array([match(r[key],r['gt']) for r in rows],dtype=int).reshape(-1,3).sum(axis=0)
    tp,fp,fn = map(int,counts)
    best_ious = [max((box_iou(g,p['box']) for p in r[key]),default=0.) for r in rows for g in r['gt']]
    negatives = [r for r in rows if not r['gt']]
    return dict(images=len(rows),gt_boxes=tp+fn,tp=tp,fp=fp,fn=fn,
                precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn) if tp+fn else 0.,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                mean_gt_best_box_iou=float(np.mean(best_ious)) if best_ious else None,
                negative_images=len(negatives),negative_images_with_fp=sum(bool(r[key]) for r in negatives))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'RDD2020_train'/'train')
    parser.add_argument('--image',type=Path,help='단일 이미지 탐지; 정답 없이 실행')
    parser.add_argument('--roi',choices=['existing','none'],default='existing')
    parser.add_argument('--segmentation',choices=['hst','local'],default='hst')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--limit',type=int,default=0,help='0은 전체 평가')
    parser.add_argument('--previews',type=int,default=12)
    args = parser.parse_args()
    out = args.out or ROOT/'outputs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    metadata = dict(opencv=cv.__version__,numpy=np.__version__,python=sys.version,
                    roi=args.roi,config=asdict(Config(segmentation=args.segmentation)),
                    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                   [Path(__file__),ROOT/'ryu_detector.py',ROOT.parent/'src'/'image_ROI_Resize.py']})
    (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    if args.image:
        row = process((args.image,None,args.roi,out/'previews',args.segmentation))
        (out/'predictions.json').write_text(json.dumps(row,indent=2,ensure_ascii=False),encoding='utf-8')
        print(f"Detected {len(row['predictions'])} boxes. Output: {out}")
        return
    images = sorted(p for p in (args.data/'img').iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'})
    if args.limit:
        images = images[:args.limit]
    if not images:
        raise ValueError('No images found')
    jobs = []
    manifest = []
    for p in images:
        ann = args.data/'ann'/(p.name+'.json')
        if not ann.is_file():
            raise ValueError(f'Missing annotation: {ann}')
        manifest.append(dict(image=p.name,image_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                             annotation_sha256=hashlib.sha256(ann.read_bytes()).hexdigest()))
        jobs.append((str(p),str(ann),args.roi,None,args.segmentation))
    # 정답 유무별 예시만 균등 선택. 선택은 알고리즘/평가에는 영향 없음.
    positives = [j for j in range(len(jobs)) if any(o['classTitle']=='pothole' for o in json.loads(Path(jobs[j][1]).read_text(encoding='utf-8'))['objects'])]
    negatives = [j for j in range(len(jobs)) if j not in positives]
    for j in positives[:args.previews//2]+negatives[:args.previews-args.previews//2]:
        jobs[j] = (*jobs[j][:3],str(out/'previews'),args.segmentation)
    rows = []
    with ProcessPoolExecutor(max_workers=max(1,args.workers)) as pool, (out/'predictions.jsonl').open('w',encoding='utf-8') as f:
        for index,row in enumerate(pool.map(process,jobs),1):
            rows.append(row)
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
            f.flush()
            if index%25==0 or index==len(jobs):
                print(f'{index}/{len(jobs)} completed',flush=True)
    summary = dict(config=asdict(Config(segmentation=args.segmentation)),roi=args.roi,box_convention='continuous xyxy; area=(x2-x1)*(y2-y1)',
                   evaluation='Development dataset evaluation; local variant added after inspecting HST results. Not an independent test.',
                   matching='score-ordered one-to-one at box IoU >= 0.5',
                   empty_annotation_policy='Included as negatives; additionally report excluding empty annotations.',
                   empty_annotation_images=sum(r['empty_annotation'] for r in rows),
                   geometry_baseline=summarize(rows,'geometry_baseline'),ryu=summarize(rows,'predictions'),
                   excluding_empty_annotations=summarize([r for r in rows if not r['empty_annotation']],'predictions'),
                   mean_seconds_per_image=float(np.mean([r['elapsed_seconds'] for r in rows])))
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    lines = ['# Ryu-inspired box evaluation','',f"Images: {len(rows)} | ROI: {args.roi}",'',
             '| Metric | Geometry baseline | Ryu-inspired |','|---|---:|---:|']
    for key in ['tp','fp','fn','precision','recall','f1','mean_gt_best_box_iou','negative_images_with_fp']:
        a,b = summary['geometry_baseline'][key],summary['ryu'][key]
        lines.append(f'| {key} | {a} | {b} |')
    lines += ['', 'Baseline disables only the final brightness/texture decision; it is not the old pixel detector.',
              'Mean GT-best IoU includes missed GT as zero, allows shared predictions, and does not penalize FP. Interpret alongside F1.',
              'Boxes are evaluated, not pixel masks. Development evaluation, not an independent test.',
              f"Empty annotations counted as negatives: {summary['empty_annotation_images']}. See summary.json for exclusion sensitivity."]
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    print(f'Output: {out}')


if __name__ == '__main__':
    main()
