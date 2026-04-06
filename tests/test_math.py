import unittest
import numpy as np
import math
from spine_analysis import _segment_inclination, cobb_angle, multi_segment_cobb, lateral_deviation_angle, trunk_forward_lean

class TestSpineMath(unittest.TestCase):
    def test_segment_inclination_vertical(self):
        """Verify vertical segment inclination is exactly 0.000000."""
        p1, p2 = (100.0, 100.0), (100.0, 200.0)
        self.assertAlmostEqual(_segment_inclination(p1, p2), 0.0, places=6)

    def test_segment_inclination_45deg(self):
        """Verify 45-degree tilt is exactly 45.000000."""
        p1, p2 = (100.0, 100.0), (200.0, 200.0)
        self.assertAlmostEqual(_segment_inclination(p1, p2), 45.0, places=6)

    def test_multi_segment_cobb_monotonic(self):
        """Verify monotonic Cobb angle is net difference (max-min)."""
        from spine_analysis import SPINE_IDX
        kpts = np.zeros((40, 2))
        # Seg 1: (100,100) to (100,150) -> 0 deg
        # Seg 2: (100,150) to (150,200) -> 45 deg
        # Seg 3: (150,200) to (250,250) -> 63.434948 deg (atan2(100, 50))
        kpts[SPINE_IDX["C7"]] = (100, 100)
        kpts[SPINE_IDX["T3"]] = (100, 150)
        kpts[SPINE_IDX["T8"]] = (150, 200)
        kpts[SPINE_IDX["L1"]] = (250, 250)
        
        expected = math.degrees(math.atan2(100, 50)) - 0.0
        self.assertAlmostEqual(multi_segment_cobb(kpts, ["C7", "T3", "T8", "L1"]), expected, places=6)

    def test_multi_segment_cobb_non_monotonic(self):
        """Verify non-monotonic Cobb uses range (max-min)."""
        from spine_analysis import SPINE_IDX
        kpts = np.zeros((40, 2))
        # Seg 1: 0 deg
        # Seg 2: 45 deg
        # Seg 3: 10 deg
        kpts[SPINE_IDX["C7"]] = (100, 100)
        kpts[SPINE_IDX["T3"]] = (100, 150)
        kpts[SPINE_IDX["T8"]] = (150, 200)
        kpts[SPINE_IDX["L1"]] = (150 + 50*math.tan(math.radians(10)), 250)
        
        # Max is 45, Min is 0. Cobb = 45.
        self.assertAlmostEqual(multi_segment_cobb(kpts, ["C7", "T3", "T8", "L1"]), 45.0, places=6)

    def test_lateral_deviation(self):
        # Straight line
        p1 = (100, 100)
        p2 = (100, 200)
        p3 = (100, 300)
        self.assertAlmostEqual(lateral_deviation_angle(p1, p2, p3), 0.0, places=6)
        
        # 90 deg bend at p2
        p1 = (0, 200)
        p2 = (100, 200)
        p3 = (100, 300)
        # Angle p1-p2-p3 is 90 deg. Supplement is 180 - 90 = 90 deg.
        self.assertAlmostEqual(lateral_deviation_angle(p1, p2, p3), 90.0, places=6)

    def test_trunk_forward_lean(self):
        # Vertical
        c7 = (100, 100)
        sacrum = (100, 200)
        self.assertAlmostEqual(trunk_forward_lean(c7, sacrum), 0.0, places=6)
        
        # 10 deg lean
        # v[1] = 100. v[0] = 100 * tan(10 deg)
        lean_rad = math.radians(10)
        sacrum = (100 + 100 * math.tan(lean_rad), 200)
        self.assertAlmostEqual(trunk_forward_lean(c7, sacrum), 10.0, places=6)

if __name__ == '__main__':
    unittest.main()
