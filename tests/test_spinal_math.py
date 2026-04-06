import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to sys.path to allow importing from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spine_analysis
import surface_curvature_analysis

class TestSpinalMath(unittest.TestCase):
    
    def test_vertical_line_inclination(self):
        """Case 1: Vertical line should result in 0 degrees inclination."""
        pt_top = [100, 0]
        pt_bot = [100, 100]
        angle = spine_analysis._segment_inclination(pt_top, pt_bot)
        self.assertAlmostEqual(angle, 0.0, places=5)

    def test_known_slope_inclination(self):
        """Case 2: 45 degree slope (dx = dy) should result in 45 degrees."""
        # In image coordinates, y increases downward.
        # dx=1, dy=1 -> 45 degrees
        pt_top = [100, 100]
        pt_bot = [200, 200]
        angle = spine_analysis._segment_inclination(pt_top, pt_bot)
        self.assertAlmostEqual(angle, 45.0, places=5)

    def test_symmetric_case_positivity(self):
        """Case 3: Symmetric segments should have same magnitude (abs check)."""
        pt_top = [100, 100]
        pt_bot_pos = [150, 200] # tilted right
        pt_bot_neg = [50, 200]  # tilted left
        
        angle_pos = spine_analysis._segment_inclination(pt_top, pt_bot_pos)
        angle_neg = spine_analysis._segment_inclination(pt_top, pt_bot_neg)
        
        # Both should be positive due to abs(atan2(...))
        self.assertGreaterEqual(angle_pos, 0)
        self.assertGreaterEqual(angle_neg, 0)
        self.assertAlmostEqual(angle_pos, angle_neg, places=5)

    def test_frame_id_synchronization(self):
        """Case 4: Synchronized merge using Frame-ID (handling dropped frames)."""
        # Anatomical dropped frame 1
        df_a = pd.DataFrame({"frame_id": [0, 2, 3], "anatomical": [10.0, 12.0, 13.0]})
        # Geometric dropped frame 2
        df_g = pd.DataFrame({"frame_id": [0, 1, 3], "geometric": [8.0, 9.0, 11.0]})
        
        df_sync = pd.merge(df_a, df_g, on="frame_id")
        
        # Only frame 0 and 3 should survive
        self.assertEqual(len(df_sync), 2)
        self.assertListEqual(list(df_sync["frame_id"]), [0, 3])
        
        # Differences should be correct for matched frames
        diff = df_sync["anatomical"] - df_sync["geometric"]
        self.assertAlmostEqual(diff.iloc[0], 2.0)
        self.assertAlmostEqual(diff.iloc[1], 2.0)

if __name__ == "__main__":
    unittest.main()
