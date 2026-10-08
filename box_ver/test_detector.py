import unittest
import cv2 as cv
import numpy as np
from ryu_detector import Config, box_iou, ordered_histogram_intersection, detect
from run import match


class DetectorTests(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(box_iou([0,0,10,10],[5,0,15,10]),1/3)
        self.assertEqual(box_iou([0,0,10,10],[10,0,20,10]),0)

    def test_duplicates_are_false_positives(self):
        p = [dict(box=[0,0,10,10],score=s) for s in [2,1]]
        self.assertEqual(match(p,[[0,0,10,10]]),(1,1,0))
        self.assertEqual(match(p,[]),(0,2,0))
        self.assertEqual(match([],[[0,0,10,10]]),(0,0,1))

    def test_ohi_shift_invariance_and_range(self):
        a = np.array([10,10,20,30],np.uint8)
        self.assertAlmostEqual(ordered_histogram_intersection(a,a+40),1)
        self.assertLess(ordered_histogram_intersection(a,np.zeros(100,np.uint8)),1)

    def test_constant_and_empty_roi(self):
        a = np.full((100,100),150,np.uint8)
        self.assertEqual(detect(a)[0],[])
        a[40:60,40:60] = 30
        self.assertEqual(detect(a,np.zeros_like(a))[0],[])

    def test_textured_dark_region_and_mask_containment(self):
        rng = np.random.default_rng(7)
        gray = np.clip(rng.normal(155,3,(200,200)),0,255).astype(np.uint8)
        region = np.zeros_like(gray)
        cv.ellipse(region,(100,120),(25,16),0,0,360,255,-1)
        gray[region>0] = rng.integers(30,100,size=np.count_nonzero(region),dtype=np.uint8)
        pred,_,_ = detect(gray)
        self.assertTrue(pred)
        self.assertGreater(max(box_iou(p['box'],[75,104,126,137]) for p in pred),.7)
        valid = np.full_like(gray,255)
        valid[90:150,65:135] = 0
        self.assertEqual(detect(gray,valid)[0],[])

    def test_local_mode_with_bright_background(self):
        rng = np.random.default_rng(9)
        gray = np.clip(rng.normal(160,2,(200,200)),0,255).astype(np.uint8)
        gray[:50] = 250
        gray[110:140,80:120] = rng.integers(60,115,(30,40),dtype=np.uint8)
        pred,_,_ = detect(gray,config=Config(segmentation='local'))
        self.assertTrue(any(box_iou(p['box'],[80,110,120,140])>.5 for p in pred))


if __name__ == '__main__':
    unittest.main()
