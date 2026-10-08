# Ryu-inspired box evaluation

Images: 804 | ROI: none

| Metric | Geometry baseline | Ryu-inspired |
|---|---:|---:|
| tp | 0 | 0 |
| fp | 22213 | 12721 |
| fn | 133 | 133 |
| precision | 0.0 | 0.0 |
| recall | 0.0 | 0.0 |
| f1 | 0.0 | 0.0 |
| mean_gt_best_box_iou | 0.01630013571127449 | 0.009366028228979063 |
| negative_images_with_fp | 719 | 710 |

Baseline disables only the final brightness/texture decision; it is not the old pixel detector.
Mean GT-best IoU includes missed GT as zero, allows shared predictions, and does not penalize FP. Interpret alongside F1.
Boxes are evaluated, not pixel masks. Development evaluation, not an independent test.
Empty annotations counted as negatives: 266. See summary.json for exclusion sensitivity.