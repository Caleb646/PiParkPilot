import cv2 as cv
import numpy as np
import pandas as pd
import os
import time

import matplotlib.pyplot as plt

class KTTISequence:
    def __init__(self, path_to_sequence: str, read_first: int = -1) -> None:
        self.left_image_fnames = [
            os.path.join(os.path.join(path_to_sequence, "image_0", fname)) 
            for fname in sorted(os.listdir(os.path.join(path_to_sequence, "image_0")))
            ]
        self.right_image_fnames = [
            os.path.join(os.path.join(path_to_sequence, "image_1", fname)) 
            for fname in sorted(os.listdir(os.path.join(path_to_sequence, "image_1")))
            ]
        self.image_paths = list(zip(self.left_image_fnames, self.right_image_fnames))
        if read_first == -1: # if -1 read all of the images
            read_first = len(self.image_paths)
        self.image_paths = self.image_paths[:min(read_first, len(self.image_paths))]
        self.num_frames = len(self.image_paths)
        self.calib = pd.read_csv(os.path.join(path_to_sequence, "calib.txt"), delimiter=' ', header=None, index_col=0)

        # These matrices contain intrinsic information about the camera's focal length and optical center.
        # They also contain tranformation information which relates each camera's coordinate frame to 
        # the global coordinate frame (in this case that of the left grayscale camera).
        #
        # The camera projection matrices in the calibration file are
        # for each camera AFTER RECTIFICATION in terms of the stereo rig.
        #
        # These matrices are taking 3D points from the coordinate frame of the camera they are associated with, 
        # and projecting them onto the image plane of the left camera.
        self.left_proj_mat = np.array(self.calib.loc['P0:']).reshape((3,4))
        self.right_proj_mat = np.array(self.calib.loc['P1:']).reshape((3,4))
        
        self.poses = pd.read_csv(os.path.join(path_to_sequence, "poses.txt"), delimiter=' ', header=None)
        self.times = np.array(pd.read_csv(os.path.join(path_to_sequence, "times.txt"), delimiter=' ', header=None))
        self.ground_truths = np.zeros((self.num_frames, 3, 4))
        for i in range(self.num_frames):
            self.ground_truths[i] = np.array(self.poses.iloc[i]).reshape((3, 4))

    def read_images(self):
        for i in range(self.num_frames - 1):
            current_left = cv.imread(self.image_paths[i][0], cv.IMREAD_GRAYSCALE)
            current_right = cv.imread(self.image_paths[i][1], cv.IMREAD_GRAYSCALE)
            next_left = cv.imread(self.image_paths[i+1][0], cv.IMREAD_GRAYSCALE)
            next_right = cv.imread(self.image_paths[i+1][1], cv.IMREAD_GRAYSCALE)
            yield current_left, current_right, next_left, next_right

    def calculate_error(self, estimated):
        n_examples = estimated.shape[0]
        def get_mse(ground_truth, estimated):
            se = np.sqrt((ground_truth[:, 0, 3] - estimated[:, 0, 3])**2 
                        + (ground_truth[:, 1, 3] - estimated[:, 1, 3])**2 
                        + (ground_truth[:, 2, 3] - estimated[:, 2, 3])**2)**2
            mse = se.mean()
            return mse
        
        def get_mae(ground_truth, estimated):
            ae = np.sqrt((ground_truth[:, 0, 3] - estimated[:, 0, 3])**2 
                        + (ground_truth[:, 1, 3] - estimated[:, 1, 3])**2 
                        + (ground_truth[:, 2, 3] - estimated[:, 2, 3])**2)
            mae = ae.mean()
            return mae
        
        def get_rmse(ground_truth, estimated):
            sr = (ground_truth[:, 0, 3] - estimated[:, 0, 3])**2\
                  + (ground_truth[:, 1, 3] - estimated[:, 1, 3])**2\
                  + (ground_truth[:, 2, 3] - estimated[:, 2, 3])**2
            return np.sqrt(np.sum(sr) / ground_truth.shape[0])
        mae = get_mae(self.ground_truths, estimated)
        mse = get_mse(self.ground_truths, estimated)
        rmse = get_rmse(self.ground_truths, estimated)
        final_rmse = get_rmse(self.ground_truths[n_examples-2 : n_examples], estimated[n_examples-2 : n_examples])
        return {'mae': round(mae, 1),
                'rmse': round(rmse, 1),
                "final_rmse": round(final_rmse, 1),
                'mse': round(mse, 1)}


if __name__ == "__main__":
    from visual_odometry import VisualOdometry
    # Best Scores:
    # Sequence 00 & 4540 Frames -> mae: 35.9, rmse: 43.4, mse: 1887.3
    sequence = KTTISequence("./data/test_data/KITTI/00") #, 100)
    vo = VisualOdometry(sequence.left_proj_mat, sequence.right_proj_mat)
    T_tot = np.eye(4)
    trajectory = np.zeros((sequence.num_frames, 3, 4))
    trajectory[0] = T_tot[:3, :]
    print(f"Total Number of Frames to Process: {sequence.num_frames}")
    total_start_time = time.time()
    for i, (current_left, current_right, next_left, next_right) in enumerate(sequence.read_images()):
        start_time = time.time()
        rmat, tvec, img1_points, img2_points = vo.estimate_motion(
            current_left, current_right, next_left, next_right
            )
        Tmat = np.eye(4)
        Tmat[:3, :3] = rmat
        Tmat[:3, 3] = tvec.T
        T_tot = T_tot.dot(np.linalg.inv(Tmat))
        trajectory[i + 1, :, :] = T_tot[:3, :]
        print(f"Finished frame: {i + 1} in {time.time() - start_time} seconds")
    total_duration = time.time() - total_start_time
    print(f"Total Time: {total_duration} seconds - Avg Frame Time: {total_duration / sequence.num_frames} - Scores: {sequence.calculate_error(trajectory)}")

    should_plot = True
    if should_plot:
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(trajectory[:, :, 3][:, 0], 
                trajectory[:, :, 3][:, 1], 
                trajectory[:, :, 3][:, 2], label='estimated', color='orange')

        ax.plot(sequence.ground_truths[:, :, 3][:, 0], 
                sequence.ground_truths[:, :, 3][:, 1], 
                sequence.ground_truths[:, :, 3][:, 2], label='ground truth')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(elev=-20, azim=270)