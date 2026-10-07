"""
Main entry point for the Visual Odometry SLAM system.
Handles synchronization, tracking loop, mapping, and metric evaluation.
"""
import os
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"

import cv2
import numpy as np

import config
import dataset
from frame import Frame
from map import Map
from point import Point
from tracker import Tracker
from viewer import Viewer

def smooth_recent_trajectory(trajectory, window=10):
    """Applies a moving average to smooth local high-frequency jitter."""
    if len(trajectory) < window: return
    subset = np.array(trajectory[-window:])
    smoothed_subset = []
    for i in range(len(subset)):
        start, end = max(0, i - 2), min(len(subset), i + 3)
        smoothed_subset.append(np.mean(subset[start:end], axis=0))
    for i in range(window):
        trajectory[-(window - i)] = smoothed_subset[i]

def calculate_svd_trajectory_alignment(est_traj, gt_traj):
    """Calculates Absolute Trajectory Error (ATE) using SVD alignment."""
    est, gt = np.array(est_traj), np.array(gt_traj)
    mu_est, mu_gt = np.mean(est, axis=0), np.mean(gt, axis=0)
    W = np.dot((est - mu_est).T, (gt - mu_gt))
    U, S, Vt = np.linalg.svd(W)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    aligned_est = np.dot(est, R.T) + (mu_gt - np.dot(R, mu_est))
    return np.sqrt(np.mean(np.sum((aligned_est - gt)**2, axis=1)))

def main():
    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking", 640, 480)

    rgb_list = dataset.read_file_list(os.path.join(config.DATASET_PATH, "rgb.txt"))
    depth_list = dataset.read_file_list(os.path.join(config.DATASET_PATH, "depth.txt"))
    gt_data = dataset.load_ground_truth(os.path.join(config.DATASET_PATH, "groundtruth.txt"))
    matches = dataset.associate_data(rgb_list, depth_list)

    # Initialize ground truth synchronization if data is present
    gt_synchronized = []
    if gt_data:
        gt_timestamps = np.array(sorted(gt_data.keys()))
        gt_start_pos = gt_data[gt_timestamps[np.abs(gt_timestamps - matches[0][0]).argmin()]]
        for m in matches:
            idx = min(np.searchsorted(gt_timestamps, m[0]), len(gt_timestamps) - 1)
            gt_synchronized.append(gt_data[gt_timestamps[idx]] - gt_start_pos)

    vo_map = Map()
    tracker = Tracker()
    viewer = Viewer()

    first_rgb = cv2.imread(os.path.join(config.DATASET_PATH, matches[0][1]), cv2.IMREAD_GRAYSCALE)
    prev_depth = cv2.imread(os.path.join(config.DATASET_PATH, matches[0][3]), cv2.IMREAD_UNCHANGED)

    last_keyframe = Frame(0, first_rgb, matches[0][0], config.K)
    tracker.compute_features(last_keyframe)
    vo_map.add_frame(last_keyframe)

    mapping_keyframe = last_keyframe
    trajectory = [np.zeros(3)]
    gt_traj_draw = [gt_synchronized[0]] if gt_synchronized else []

    frame_history = [last_keyframe]
    depth_history = [prev_depth]
    skipped_frames, max_skip = 0, 3

    for i in range(1, len(matches)):
        if viewer.should_quit(): break

        curr_rgb = cv2.imread(os.path.join(config.DATASET_PATH, matches[i][1]), cv2.IMREAD_GRAYSCALE)
        curr_depth = cv2.imread(os.path.join(config.DATASET_PATH, matches[i][3]), cv2.IMREAD_UNCHANGED)
        if curr_rgb is None: continue

        curr_frame = Frame(i, curr_rgb, matches[i][0], config.K)
        tracker.compute_features(curr_frame)

        # Execute tracking pipeline
        T_motion, _, _, valid_matches, shake_reason = tracker.match_and_track(frame_history[-5:], curr_frame, depth_history[-5:])
        should_update_keyframe = False
        display_text, color = "Tracking", (0, 255, 0)

        if T_motion is not None and shake_reason is None:
            T_motion[:3, 3] *= config.SCALE_FIX
            curr_frame.set_pose(last_keyframe.pose @ T_motion)
            vo_map.add_frame(curr_frame)

            # Detect loop closures periodically
            if i % 10 == 0:
                T_correction, loop_id = tracker.detect_loop(curr_frame, vo_map.frames)
                if T_correction is not None:
                    display_text, color = f"Loop detected: {loop_id}", (255, 0, 255)

            aligned_pos = viewer.get_aligned_pos(curr_frame.pose)
            trajectory.append(aligned_pos)

            # Generate map points only when a sufficient baseline is achieved
            dist_moved = np.linalg.norm(curr_frame.pose[:3, 3] - mapping_keyframe.pose[:3, 3])

            if dist_moved > 0.20:
                new_map_points = tracker.triangulate_keyframes(mapping_keyframe, curr_frame)
                for global_pt in new_map_points:
                    aligned_pt = viewer.get_aligned_pos(np.eye(4)) + (viewer.vis_align[:3, :3] @ global_pt)
                    vo_map.add_point(Point(0, aligned_pt))

                mapping_keyframe = curr_frame

            should_update_keyframe = True
            skipped_frames = 0
        else:
            # Coast and repeat last pose if tracking fails
            trajectory.append(trajectory[-1] if len(trajectory) > 0 else np.zeros(3))
            skipped_frames += 1
            display_text, color = f"Lost Track: {shake_reason}", (0, 0, 255)

            if skipped_frames > max_skip:
                should_update_keyframe = True
                skipped_frames = 0
                display_text, color = "Force resync", (0, 255, 255)
                mapping_keyframe = curr_frame
            else:
                curr_frame.pose = last_keyframe.pose

        smooth_recent_trajectory(trajectory, window=15)
        if gt_synchronized:
            gt_traj_draw.append(gt_synchronized[i])

        viewer.update(vo_map, trajectory, gt_traj_draw, last_keyframe.pose)

        # Update 2D OpenCV display
        try:
            vis = cv2.cvtColor(curr_frame.image, cv2.COLOR_GRAY2BGR)
            if curr_frame.keypoints:
                cv2.drawKeypoints(curr_frame.image, curr_frame.keypoints, vis, color=(0, 255, 0), flags=0)
            vis = cv2.resize(vis, (640, 480))
            cv2.putText(vis, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("Tracking", vis)
            cv2.waitKey(1)
        except Exception: pass

        if should_update_keyframe:
            last_keyframe = curr_frame
            frame_history.append(curr_frame)
            depth_history.append(curr_depth)
            if len(frame_history) > 20:
                frame_history.pop(0)
                depth_history.pop(0)

    # Perform evaluation if ground truth data was loaded
    if gt_traj_draw:
        est_dist = sum(np.linalg.norm(trajectory[i] - trajectory[i-1]) for i in range(1, len(trajectory)))
        gt_dist = sum(np.linalg.norm(gt_traj_draw[i] - gt_traj_draw[i-1]) for i in range(1, len(gt_traj_draw)))
        min_len = min(len(trajectory), len(gt_traj_draw))
        print(f"\nTotal distance (Estimated):    {est_dist:.2f} meters")
        print(f"Total distance (Ground truth): {gt_dist:.2f} meters")
        print(f"Final ATE (SVD aligned):       {calculate_svd_trajectory_alignment(trajectory[:min_len], gt_traj_draw[:min_len]):.4f} meters")

if __name__ == "__main__":
    main()