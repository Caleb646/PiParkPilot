import cv2 as cv
import numpy as np
import logging
from typing import Tuple, List, Union




import pathlib
import os
import sys
parent = pathlib.Path(os.path.abspath(os.path.curdir))
path = os.path.join(str(parent))
sys.path.append(path)


import pi_park.utils as utils


logger = logging.getLogger(__name__)

class KalmanFilter:
    def __init__(self, fps: int):
        self.logger = logger
        self.num_states = 18
        """
        The states are 
        position: (x, y, z, x', y', z', x'', y'', z'')
        angle: (xr, yr, zr, xr', yr', zr', xr'', yr'', zr'')
        """
        self.num_measurements = 6
        """
        The measurements are
        position: (x, y, z)
        angle: (xr, yr, zr)
        """
        self.num_inputs = 0
        self.dt = 1 / float(fps)
        """
        The differential time between measurements which in this case is 1/T, 
        where T is the frame rate of the video.
        """
        self.kalman = cv.KalmanFilter(
            self.num_states, self.num_measurements, self.num_inputs, cv.CV_64F
            )
        
        self.measurements = np.zeros((self.num_measurements, 1))
        
        # SOURCE: https://docs.opencv.org/3.4/dc/d2c/tutorial_real_time_pose.html

                 # DYNAMIC MODEL
        #  [1 0 0 dt  0  0 dt2   0   0 0 0 0  0  0  0   0   0   0]
        #  [0 1 0  0 dt  0   0 dt2   0 0 0 0  0  0  0   0   0   0]
        #  [0 0 1  0  0 dt   0   0 dt2 0 0 0  0  0  0   0   0   0]
        #  [0 0 0  1  0  0  dt   0   0 0 0 0  0  0  0   0   0   0]
        #  [0 0 0  0  1  0   0  dt   0 0 0 0  0  0  0   0   0   0]
        #  [0 0 0  0  0  1   0   0  dt 0 0 0  0  0  0   0   0   0]
        #  [0 0 0  0  0  0   1   0   0 0 0 0  0  0  0   0   0   0]
        #  [0 0 0  0  0  0   0   1   0 0 0 0  0  0  0   0   0   0]
        #  [0 0 0  0  0  0   0   0   1 0 0 0  0  0  0   0   0   0]
        #  [0 0 0  0  0  0   0   0   0 1 0 0 dt  0  0 dt2   0   0]
        #  [0 0 0  0  0  0   0   0   0 0 1 0  0 dt  0   0 dt2   0]
        #  [0 0 0  0  0  0   0   0   0 0 0 1  0  0 dt   0   0 dt2]
        #  [0 0 0  0  0  0   0   0   0 0 0 0  1  0  0  dt   0   0]
        #  [0 0 0  0  0  0   0   0   0 0 0 0  0  1  0   0  dt   0]
        #  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  1   0   0  dt]
        #  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   1   0   0]
        #  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   1   0]
        #  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   0   1]

        self.kalman.transitionMatrix[0, 3] = self.dt
        self.kalman.transitionMatrix[1, 4] = self.dt
        self.kalman.transitionMatrix[2, 5] = self.dt
        self.kalman.transitionMatrix[3, 6] = self.dt
        self.kalman.transitionMatrix[4, 7] = self.dt
        self.kalman.transitionMatrix[5, 8] = self.dt
        self.kalman.transitionMatrix[0, 6] = 0.5 * self.dt**2
        self.kalman.transitionMatrix[1, 7] = 0.5 * self.dt**2
        self.kalman.transitionMatrix[2, 8] = 0.5 * self.dt**2

        self.kalman.transitionMatrix[9, 12] = self.dt
        self.kalman.transitionMatrix[10, 13] = self.dt
        self.kalman.transitionMatrix[11, 14] = self.dt
        self.kalman.transitionMatrix[12, 15] = self.dt
        self.kalman.transitionMatrix[13, 16] = self.dt
        self.kalman.transitionMatrix[14, 17] = self.dt
        self.kalman.transitionMatrix[9, 15] = 0.5 * self.dt**2
        self.kalman.transitionMatrix[10, 16] = 0.5 * self.dt**2
        self.kalman.transitionMatrix[11, 17] = 0.5 * self.dt**2

        
            # MEASUREMENT MODEL
        #  [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        #  [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        #  [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        #  [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
        #  [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
        #  [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]
        
        self.kalman.measurementMatrix = cv.setIdentity(
            self.kalman.measurementMatrix, 1
            )
        self.kalman.measurementMatrix[0, 0] = 1  # x
        self.kalman.measurementMatrix[1, 1] = 1  # y
        self.kalman.measurementMatrix[2, 2] = 1  # z
        self.kalman.measurementMatrix[3, 9] = 1  # roll
        self.kalman.measurementMatrix[4, 10] = 1 # pitch
        self.kalman.measurementMatrix[5, 11] = 1 # yaw


        self.kalman.processNoiseCov = cv.setIdentity(
            self.kalman.processNoiseCov, 1e-5
            )
        # self.kalman.processNoiseCov = np.eye(
        #     self.num_states, self.num_states
        #     ) * 1e-5
        """
        The process noise covariance matrix self.kalman.processNoiseCov represents 
        the uncertainty in our motion model and affects how the Kalman 
        filter predicts the next state.
        """
        self.kalman.measurementNoiseCov = cv.setIdentity(
            self.kalman.measurementNoiseCov, 0.1 #1e-4
            )
        # self.kalman.measurementNoiseCov = np.eye(
        #     self.num_measurements, self.num_measurements
        #     ) * 1e-4
        self.kalman.errorCovPost = cv.setIdentity(
            self.kalman.errorCovPost, 0.1
            )
        # self.kalman.errorCovPost = np.eye(
        #     self.num_states, self.num_states
        #     ) * 1

        # self.kalman.statePre = np.zeros((self.num_states, 1))
        # self.kalman.statePost = np.zeros((self.num_states, 1))


        self.min_inliers_kalman = 30 # Kalman threshold updating
        #self.min_inliers_kalman = 15 # Kalman threshold updating

    def predict_update(self, trans_measured, rot_measured, num_inliers):
        if num_inliers >= self.min_inliers_kalman:
            self.logger.debug(f"Measured translation vector: {np.squeeze(trans_measured)}")
            xr, yr, zr = utils.decompose_rotation_matrix(rot_measured)
            self.measurements[0, 0] = trans_measured[0] # x
            self.measurements[1, 0] = trans_measured[1] # y
            self.measurements[2, 0] = trans_measured[2] # z
            self.measurements[3, 0] = xr # xr
            self.measurements[4, 0] = yr # yr
            self.measurements[5, 0] = zr # zr

        np.squeeze(self.kalman.correct(self.measurements))
        estimate = self.kalman.predict()
        #estimate = np.squeeze(self.kalman.correct(self.measurements))
        
        new_trans = np.zeros((3,))
        new_trans[0] = estimate[0]
        new_trans[1] = estimate[1]
        new_trans[2] = estimate[2]

        new_rot = utils.euler2rotation(
            estimate[9], estimate[10], estimate[11]
        )
        return True, new_trans, new_rot
        #return False, None, None

if __name__ == "__main__":

    kf = KalmanFilter(1)

    print(kf.predict_update(np.array([0, 0, 0]), np.eye(3, 3), 35)[1])
    print(kf.predict_update(np.array([1, 0, 1]), np.eye(3, 3), 35)[1])
    print(kf.predict_update(np.array([2, 0, 1]), np.eye(3, 3), 35)[1])
    print(kf.predict_update(np.array([3, 0, 1]), np.eye(3, 3), 35)[1])

            