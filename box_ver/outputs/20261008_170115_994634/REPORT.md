# Ryu-inspired box evaluation

Images: 804 | ROI: existing

| Metric | Geometry baseline | Ryu-inspired |
|---|---:|---:|
| tp | 0 | 0 |
| fp | 1677 | 799 |
| fn | 133 | 133 |
| precision | 0.0 | 0.0 |
| recall | 0.0 | 0.0 |
| f1 | 0.0 | 0.0 |
| mean_gt_best_box_iou | 0.00572579543957679 | 0.004335918625247298 |
| negative_images_with_fp | 519 | 343 |

Baseline disables only the final brightness/texture decision; it is not the old pixel detector.
Mean GT-best IoU includes missed GT as zero, allows shared predictions, and does not penalize FP. Interpret alongside F1.
Boxes are evaluated, not pixel masks. No parameter selection was performed on this evaluation.
Empty annotations counted as negatives: 266. See summary.json for exclusion sensitivity.