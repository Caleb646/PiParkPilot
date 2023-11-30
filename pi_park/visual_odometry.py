import numpy as np
import cv2 as cv
from scipy.optimize import minimize
import math
import logging
import time
from typing import Tuple, List

from pi_park.kp_matcher import KPMatcher
import pi_park.utils as utils

def undistort_rectify(
        left: cv.Mat, 
        right: cv.Mat,
        left_x_map: cv.Mat,
        left_y_map: cv.Mat,
        right_x_map: cv.Mat,
        right_y_map: cv.Mat
        ):
    left_undistorted = cv.remap(
        left, left_x_map, left_y_map, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0
        )
    right_undistorted = cv.remap(
        right, right_x_map, right_y_map, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0
        )
    return left_undistorted, right_undistorted

def calculate_disparity_map(img_left: cv.Mat, img_right: cv.Mat):
        sad_window = 6
        num_disparities = sad_window*16
        block_size = 11
        matcher = cv.StereoSGBM_create(
            numDisparities=num_disparities,
            minDisparity=0,
            blockSize=block_size,
            P1 = 8 * 3 * sad_window ** 2,
            P2 = 32 * 3 * sad_window ** 2,
            mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
            )
        if len(img_left.shape) > 2:
            img_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
            img_right = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)
        disp_left = matcher.compute(img_left, img_right).astype(np.float32) / 16
        return disp_left

def decompose_euler_angles_from_proj_mat(proj_mat: np.ndarray):
    _, _, _, _, _, _, euler_angles = cv.decomposeProjectionMatrix(proj_mat)
    return euler_angles

def decompose_projection_matrix(
        proj_mat: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    ---------
    camera intrinsic matrix: shape -> (3, 3)
    rotation matrix: shape -> (3, 3)
    translation matrix: shape -> (3, 1). Has been de-homogenized.
    """
    # decomposeProjectionMatrix has 4 additional returns values 
    # rotMatrX 3x3 rotation matrix around x-axis.
    # rotMatrY 3x3 rotation matrix around y-axis.
    # rotMatrZ 3x3 rotation matrix around z-axis.
    # eulerAngles 3-element vector containing three Euler angles of rotation in degrees.
    cam_intrinsic, cam_rotation, cam_translation, _, _, _, euler_angles = cv.decomposeProjectionMatrix(proj_mat)
    cam_translation = (cam_translation / cam_translation[3])[:3]
    return cam_intrinsic, cam_rotation, cam_translation

def decompose_rotation_matrix(rot: np.ndarray):
    x_theta = math.atan2(rot[2, 1], rot[2, 2])
    y_theta = math.atan2(rot[2, 0], math.sqrt(rot[2, 1]**2 + rot[0, 0]**2))
    z_theta = math.atan2(rot[1, 0], rot[0, 0])
    return np.array([math.degrees(x_theta), math.degrees(y_theta), math.degrees(z_theta)])

def calculate_depth_map(
        disp_left_pixels: cv.Mat, 
        focal_length_pixels: float, 
        cams_horizontal_dist_meters: float
        ) -> cv.Mat:
    disp_left_pixels[disp_left_pixels == 0.0] = 0.1
    disp_left_pixels[disp_left_pixels == -1.0] = 0.1
    depth_map = np.ones(disp_left_pixels.shape)
    # focal length and disparity are measured in pixels, then the pixel units will cancel, 
    # and if we have our baseline measured in meters, then our Z measurement will be in meters
    depth_map = focal_length_pixels * cams_horizontal_dist_meters / disp_left_pixels
    return depth_map # depth is in meters

def stereo_to_depth(
        img_left: cv.Mat, 
        img_right: cv.Mat, 
        left_proj_mat: np.ndarray, 
        right_proj_mat: np.ndarray
        ) -> cv.Mat:
    disp = calculate_disparity_map(img_left, img_right)
    left_intrinsic, r_left, trans_left = decompose_projection_matrix(left_proj_mat)
    right_intrinsic, r_right, trans_right = decompose_projection_matrix(right_proj_mat)
    # focal length and disparity are measured in pixels, then the pixel units will cancel, 
    # and if we have our baseline measured in meters, then our Z measurement will be in meters
    # Distance between two cameras in meters i.e. the baseline: abs(trans_right[0] - trans_left[0])
    return calculate_depth_map(disp, left_intrinsic[0, 0], abs(trans_right[0] - trans_left[0]))

class VisualOdometry:
    def __init__(
            self, 
            left_cam_3dto2d_proj_mat: np.ndarray, 
            right_cam3dto2d_proj_mat: np.ndarray,
            keypt_matcher: KPMatcher = None,
            max_depth_meters: int = 15 # meters
            ) -> None:
        self.logger = logging.getLogger(__name__)
        self.left_proj_mat = left_cam_3dto2d_proj_mat
        self.right_proj_mat = right_cam3dto2d_proj_mat
        self.keypt_matcher = KPMatcher() if keypt_matcher is None else keypt_matcher
        self.max_depth_m = max_depth_meters
        self.left_cam_intrinsic, self.left_cam_rotation, self.left_cam_translation = decompose_projection_matrix(self.left_proj_mat)
        self.right_cam_intrinsic, self.right_cam_rotation, self.right_cam_translation = decompose_projection_matrix(self.right_proj_mat)
        self.window_size = 3
        """
        Determines how many previous frames information to save
        """
        self.pts_2d: List[List[np.ndarray]] = []
        """
        shape -> (window size, number of matched pts, 2)
        """
        self.pts_3d: List[np.ndarray] = []
        """
        shape -> (number of matched pts, 3)
        """
        self.poses: List[np.ndarray] = []
        """
        shape -> (window size, 3, 4).
        Saved camera projection matrices. The last index is the current camera pose
        """


    def windowed_bundle_adjustment(self):
        """
        Optimize the current camera pose using the previous self.window_size number of frames of saved
        3D and 2D points. The 3D points should be present in all of the previous frames. Meaning that each 3D point 
        at index i should project onto the 2D point at index k, i given the camera pose at index k. 
        Parameters
        ----------
        Returns
        ----------
        Notes
        ----------
        """
        def loss(pose: np.ndarray):
            error = 0
            projection_error = lambda twod, threed, pose: np.sum((twod - (pose.dot(threed) / threed[2])[:2])**2)  
            for k in range(self.window_size - 1):
                for i in range(len(self.pts_3d)):
                    error += projection_error(self.pts_2d[k][i], self.pts_3d[i], self.poses[k])
            reshaped = pose.reshape((3, 4))
            for i in range(len(self.pts_3d)):
                    error += projection_error(self.pts_2d[-1][i], self.pts_3d[i], reshaped)
            return error
        return np.reshape(minimize(loss, self.poses[-1].flatten()).x, (3, 4))
        
    def estimate_motion(
            self, 
            cur_left_img: cv.Mat, 
            cur_right_img: cv.Mat, 
            next_left_img: cv.Mat, 
            next_right_img: cv.Mat
            ):
        start_time = time.time()
        matches, cur_img_matched_pts, next_img_matched_pts = None, None, None
        with utils.Timer(self.logger.info, "KP Matching Time: {} seconds"):
            matches, cur_img_matched_pts, next_img_matched_pts = self.keypt_matcher.find_keypoints(cur_left_img, next_left_img)

        depth_map = None
        with utils.Timer(self.logger.info, "Depth Map Creation Time: {} seconds"):
            depth_map = stereo_to_depth(cur_left_img, cur_right_img, self.left_proj_mat, self.right_proj_mat)

        rmat = np.eye(3)
        tvec = np.zeros((3, 1))
        # focal length and disparity are measured in pixels, then 
        # the pixel units will cancel, and if we have our baseline measured in meters,
        cx = self.left_cam_intrinsic[0, 2]
        cy = self.left_cam_intrinsic[1, 2]
        fx = self.left_cam_intrinsic[0, 0]
        fy = self.left_cam_intrinsic[1, 1]
        cur_img_3d_object_points = np.zeros((0, 3))
        delete = []

        with utils.Timer(self.logger.info, "Filtering 3D pts beyond max depth Time: {} seconds"):
            # Extract depth information of query image at match points and build 3D positions
            for i, (u, v) in enumerate(cur_img_matched_pts):
                z = depth_map[int(v), int(u)]
                # remove matched points with a depth greater than max_depth.
                # Because these points are most likely noise
                if z > self.max_depth_m:
                    delete.append(i)
                    continue
                    
                x = z * (u - cx) / fx
                y = z * (v - cy) / fy
                cur_img_3d_object_points = np.vstack([cur_img_3d_object_points, np.array([x, y, z])])

        cur_points = np.delete(cur_img_matched_pts, delete, axis=0)
        next_points = np.delete(next_img_matched_pts, delete, axis=0)
        
        if cur_img_3d_object_points.shape[0] < 5 or next_points.shape[0] < 5:
            self.logger.error(
                f"Not enough 3D object pts or 2D image points "
                f"to estimate new camera projection matrix."
                )
            return None, None, None, None, False

        succeeded, rvec, tvec, inliers = False, None, None, None
        with utils.Timer(self.logger.info, "Computing Camera Proj Mat Time: {} seconds"):
            succeeded, rvec, tvec, inliers = cv.solvePnPRansac(
                cur_img_3d_object_points, 
                next_points, 
                self.left_cam_intrinsic, 
                None
                )
        if succeeded is False:
            self.logger.error(f"Failed to solve for camera projection matrix")
            return None, None, None, None, False
        
        rmat = cv.Rodrigues(rvec)[0]
        self.logger.info(f"Took {time.time() - start_time} seconds to successfully estimate motion.")
        return rmat, tvec, cur_points, next_points, True

if __name__ == "__main__":
    pass
    #test_undistort()