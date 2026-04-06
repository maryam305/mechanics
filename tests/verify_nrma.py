import os
import numpy as np
import cv2
import config
from spine_analysis import analyze_video, SPINE_IDX

class MockEstimator:
    def detect(self, frame):
        return [ [0, 0, 100, 100, 0.9] ] # Mock bbox
    def estimate(self, frame, bboxes):
        # Mock 37 keypoints with known angles
        pts = np.zeros((37, 3))
        # N: Neck inclination
        pts[36] = [100, 10, 0.9] # C1
        pts[35] = [110, 20, 0.9] # C4 (atan2(10, 10) = 45 deg)
        pts[18] = [110, 30, 0.9] # C7 (atan2(0, 10) = 0 deg)
        # R: Thoracic kyphosis
        pts[30] = [120, 40, 0.9] # T3 (atan2(10, 10) = 45 deg)
        pts[29] = [120, 50, 0.9] # T8 (atan2(0, 10) = 0 deg)
        pts[28] = [110, 60, 0.9] # L1 (atan2(-10, 10) = -45 deg)
        # M: Lumbar lordosis
        pts[27] = [110, 70, 0.9] # L3 (atan2(0, 10) = 0 deg)
        pts[26] = [120, 80, 0.9] # L5 (atan2(10, 10) = 45 deg)
        pts[19] = [130, 90, 0.9] # Sacrum (atan2(10, 10) = 45 deg)
        return [ pts ]

def test_nrma_branching():
    # Create a fake video file for testing
    vid_path = "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(vid_path, fourcc, 30.0, (640, 480))
    for _ in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    vid_info = {
        "filename": vid_path,
        "display_name": "Test_Subject",
        "condition": "unloaded"
    }
    
    estimator = MockEstimator()
    res = analyze_video(vid_info, estimator)
    
    print(f"Results keys: {res.keys()}")
    print(f"Neck (N) mean: {np.mean(res['neck_n'])}")
    print(f"Thoracic (R) mean: {np.mean(res['thoracic_r'])}")
    
    # Check directory structure
    branch_dir = os.path.join(config.OUTPUT_DIR, "Test_Subject")
    print(f"Checking branch dir: {branch_dir}")
    assert os.path.exists(branch_dir)
    assert os.path.exists(os.path.join(branch_dir, "videos"))
    assert os.path.exists(os.path.join(branch_dir, "plots"))
    assert os.path.exists(os.path.join(branch_dir, "reports"))
    
    # Cleanup
    import shutil
    # os.remove(vid_path)
    # shutil.rmtree(os.path.join(config.OUTPUT_DIR, "Test_Subject"))
    print("Test passed!")

if __name__ == "__main__":
    test_nrma_branching()
