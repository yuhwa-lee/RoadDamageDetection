# Ryu-inspired box evaluation

Images: 2 | ROI: none

| Metric | Geometry baseline | Ryu-inspired |
|---|---:|---:|
| tp | 0 | 0 |
| fp | 16 | 6 |
| fn | 0 | 0 |
| precision | 0.0 | 0.0 |
| recall | 0.0 | 0.0 |
| f1 | 0.0 | 0.0 |
| mean_gt_best_box_iou | None | None |
| negative_images_with_fp | 2 | 2 |

Baseline disables only the final brightness/texture decision; it is not the old pixel detector.
Mean GT-best IoU includes missed GT as zero, allows shared predictions, and does not penalize FP. Interpret alongside F1.
Boxes are evaluated, not pixel masks. Development evaluation, not an independent test.
Empty annotations counted as negatives: 1. See summary.json for exclusion sensitivity.