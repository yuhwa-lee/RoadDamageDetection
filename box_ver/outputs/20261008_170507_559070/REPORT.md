# Ryu-inspired box evaluation

Images: 804 | ROI: none

| Metric | Geometry baseline | Ryu-inspired |
|---|---:|---:|
| tp | 0 | 0 |
| fp | 1749 | 1181 |
| fn | 133 | 133 |
| precision | 0.0 | 0.0 |
| recall | 0.0 | 0.0 |
| f1 | 0.0 | 0.0 |
| mean_gt_best_box_iou | 0.0011970156378774668 | 0.0011813419813197488 |
| negative_images_with_fp | 404 | 330 |

Baseline disables only the final brightness/texture decision; it is not the old pixel detector.
Mean GT-best IoU includes missed GT as zero, allows shared predictions, and does not penalize FP. Interpret alongside F1.
Boxes are evaluated, not pixel masks. No parameter selection was performed on this evaluation.
Empty annotations counted as negatives: 266. See summary.json for exclusion sensitivity.