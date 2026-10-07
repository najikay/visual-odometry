# RGB-D Visual Odometry & Sparse Mapping

A visual odometry pipeline with sparse 3D mapping, built from scratch in Python with OpenCV and NumPy and evaluated on the TUM RGB-D benchmark. Each frame is tracked against recent frames using ORB features, depth-based 3D-2D correspondences, EPnP + RANSAC and a custom Gauss-Newton refinement. A separate keyframe stage triangulates a sparse point cloud from pairs of RGB views. No IMU is used.

![SLAM Demo](assets/demo_1.png)
![Trajectory vs ground truth](assets/demo_2.png)
*(Pangolin 3D viewer: estimated trajectory in green, ground truth in red, triangulated point cloud in white.)*

## How it works

* **Features:** up to 3000 ORB keypoints per frame, refined to sub-pixel accuracy with `cv2.cornerSubPix`.
* **Tracking (RGB-D):** the current frame is matched (Hamming distance, Lowe ratio test 0.75) against up to 3 of the last 5 frames. Matched keypoints in the reference frame are lifted to 3D using the **depth image**, so metric scale comes from the depth sensor.
* **Pose estimation:** `solvePnPRansac` with EPnP gives an initial pose and inlier set; a custom **Gauss-Newton** loop (10 iterations) then minimizes reprojection error over the inliers.
* **Keyframe mapping (two-view):** when the camera has moved more than 20 cm since the last mapping keyframe, matched points are **triangulated from the two RGB views** (not read from depth). Mapping only across a wide baseline avoids the smeared "ray" points that appear when triangulating during near-pure rotation.
* **Map filtering:** a triangulated point is kept only if it reprojects within 2.0 px in both views and lies inside depth/height bounds suited to this indoor scene.
* **Loop-closure detection:** every 10 frames, older frames within 1 m of the current position estimate are checked by descriptor matching (ratio test); more than 40 good matches marks a loop. **The loop is detected and reported, but no pose-graph correction is applied.**
* **Evaluation:** the estimated trajectory is aligned to ground truth with SVD (Kabsch/Horn) and the Absolute Trajectory Error (ATE, RMSE) is reported.

## Results

TUM RGB-D `freiburg2_pioneer_slam3`, single run:

* **Ground-truth path length:** 18.80 m (estimated: 20.09 m)
* **ATE (SVD-aligned RMSE):** 0.3399 m, about 1.8% of the path length

Please read these numbers with the caveats below:

* The translation scale factor (`SCALE_FIX = 1.06`) and the RANSAC threshold were selected by `tune_slam.py` to minimize ATE **on this same sequence**; no held-out sequence was used.
* A short moving-average filter smooths recent positions before evaluation.
* "1.8%" is ATE divided by path length, not a standard relative-drift (RPE) metric; RPE is not computed.
* No bundle adjustment and no loop correction are performed.

## Limitations

* Tracking is frame-to-frame using depth; the triangulated map is built for output and is not used for tracking.
* Lens distortion is ignored.
* Evaluated on a single sequence.

## Project structure

* `src/main.py` - main loop, keyframe/mapping logic, smoothing, ATE evaluation.
* `src/tracker.py` - ORB + sub-pixel refinement, EPnP + RANSAC, Gauss-Newton, triangulation and filtering, loop detection.
* `src/viewer.py` - real-time 3D rendering with Pangolin.
* `src/map.py`, `src/frame.py`, `src/point.py` - data structures.
* `src/dataset.py` - TUM file parsing, RGB-depth association, ground truth.
* `src/config.py` - camera intrinsics and thresholds.
* `src/tune_slam.py` - headless grid search over the scale factor and RANSAC threshold.

## Installation & usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/najikay/rgbd-visual-odometry.git
   cd rgbd-visual-odometry
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset:** download the TUM RGB-D `freiburg2_pioneer_slam3` sequence and set its path in `src/config.py`.

4. **Run:**
   ```bash
   cd src
   python main.py
   ```

## Academic context
Final project for the "Navigation, Mapping, and Pose Estimation" course at the University of Haifa.

**Author:** Naji Kayal
