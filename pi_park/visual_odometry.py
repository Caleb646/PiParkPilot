import numpy as np
import cv2 as cv
from scipy.optimize import minimize
import math
import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union





import pathlib
import os
import sys

parent = pathlib.Path(os.path.abspath(os.path.curdir))
path = os.path.join(str(parent))
sys.path.append(path)




from pi_park.kp_matcher import KPMatcher, KPResult
from pi_park.filter import KalmanFilter
import pi_park.utils as utils

logger = logging.getLogger(__name__)

def filter_disparity_map(
        left_img: cv.Mat, right_img: cv.Mat, left_disp: np.ndarray, left_matcher
        ):
    right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
    wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(80000)
    wls_filter.setSigmaColor(1.5)

    right_disp = right_matcher.compute(right_img, left_img)#.astype(np.float32) / 16
    left_filter_disp = wls_filter.filter(left_disp, left_img, None, right_disp)
    return left_filter_disp

def calculate_disparity_map(img_left: cv.Mat, img_right: cv.Mat, filter_disp=True):
        sad_window = 5
        num_disparities = sad_window*16
        block_size = 16 #19 #11
        # matcher = cv.StereoSGBM_create(
        #     numDisparities=num_disparities,
        #     minDisparity=0,
        #     blockSize=block_size,
        #     P1 = 8 * 3 * sad_window ** 2,
        #     P2 = 32 * 3 * sad_window ** 2,
        #     mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
        #     )
        
        matcher = cv.StereoSGBM_create(
            numDisparities=16,
            minDisparity=0,
            blockSize=9,
            P1 = 8*3*sad_window**2,
            P2 = 32*3*sad_window**2,
            mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
            )
        if len(img_left.shape) > 2:
            img_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
            img_right = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

        disp_left = matcher.compute(img_left, img_right)
        if filter_disp:
            disp_left = filter_disparity_map(img_left, img_right, disp_left, matcher)
        return disp_left.astype(np.float32) / 16

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
        right_proj_mat: np.ndarray,
        resize: Union[float, None] = 0.5,
        filter_disp = True
        ) -> cv.Mat:
    left, right = img_left, img_right
    if resize is not None:
        assert resize in [0.5, 0.25], f"Resize must be in [{[0.5, 0.25]}] NOT [{resize}]"
        left = cv.resize(left, None, fx=resize, fy=resize, interpolation=cv.INTER_AREA)
        right = cv.resize(right, None, fx=resize, fy=resize, interpolation=cv.INTER_AREA)
    disp = calculate_disparity_map(left, right, filter_disp=filter_disp)
    left_intrinsic, r_left, trans_left = utils.decompose_projection_matrix(left_proj_mat)
    right_intrinsic, r_right, trans_right = utils.decompose_projection_matrix(right_proj_mat)
    # focal length and disparity are measured in pixels, then the pixel units will cancel, 
    # and if we have our baseline measured in meters, then our Z measurement will be in meters
    # Distance between two cameras in meters i.e. the baseline: abs(trans_right[0] - trans_left[0])
    depth_map = calculate_depth_map(disp, left_intrinsic[0, 0], abs(trans_right[0] - trans_left[0]))
    if resize is not None:
        scale_back = {0.5 : 2, 0.25 : 4}[resize]
        depth_map = cv.resize(depth_map, None, fx=scale_back, fy=scale_back, interpolation=cv.INTER_AREA)
    return depth_map

@dataclass
class TriangulateResult:
    pts_3d: np.ndarray
    cur_2d_pts: np.ndarray
    success: bool = False
    depth_map: Optional[cv.Mat] = None
    reproj_error: Optional[float] = None
    prev_leftnright_kp_result: Optional[KPResult] = None
    left_prevncur_kp_result: Optional[KPResult] = None


def depth_map_triangulate(
        cur_img_left: cv.Mat, 
        cur_img_right: cv.Mat, 
        prev_left_img: cv.Mat,
        left_proj_mat: np.ndarray, 
        right_proj_mat: np.ndarray,
        kp_matcher: KPMatcher,
        max_depth_m=300, # max depth in meters
        resize: Union[float, None] = 0.5,
        filter_disp = True
        ):
    matches, prev_left_img_matched_keypts, _, cur_left_img_matched_keypts, _ = kp_matcher.find_keypoints(prev_left_img, cur_img_left)
    prev_left_img_matched_pts = utils.keypoints2points(prev_left_img_matched_keypts)
    cur_left_img_matched_pts = utils.keypoints2points(cur_left_img_matched_keypts)

    left_cam_intrinsic, left_cam_rotation, left_cam_translation = utils.decompose_projection_matrix(left_proj_mat)
    depth_map = None
    with utils.Timer(logger.debug, "Depth Map Creation Time: {} seconds"):
        depth_map = stereo_to_depth(
            cur_img_left, 
            cur_img_right, 
            left_proj_mat, 
            right_proj_mat,
            resize=resize,
            filter_disp=filter_disp
            )
    cx = left_cam_intrinsic[0, 2]
    cy = left_cam_intrinsic[1, 2]
    fx = left_cam_intrinsic[0, 0]
    fy = left_cam_intrinsic[1, 1]
    left_3d_pts = np.zeros((0, 3))
    delete = []

    with utils.Timer(logger.debug, "Filtering 3D pts beyond max depth Time: {} seconds"):
        for i, (u, v) in enumerate(prev_left_img_matched_pts):
            z = depth_map[int(v), int(u)]
            # remove matched points with a depth greater than max_depth.
            # Because these points are most likely noise
            if z > max_depth_m:
                delete.append(i)
                continue
                
            x = z * (u - cx) / fx
            y = z * (v - cy) / fy
            left_3d_pts = np.vstack([left_3d_pts, np.array([x, y, z])])
        logger.debug(
            f"Removed [{len(delete)}] out of [{prev_left_img_matched_pts.shape[0]}] 3D pts because "
            f"because they exceed the max depth of [{max_depth_m}]"
            )
    prev_2d_pts = np.delete(prev_left_img_matched_pts, delete, axis=0)
    cur_2d_pts = np.delete(cur_left_img_matched_pts, delete, axis=0)
    return depth_map, left_3d_pts, prev_2d_pts, cur_2d_pts

def keypoint_triangulate(
        prev_left_img,
        prev_right_img,
        cur_left_img,
        left_proj: np.ndarray, 
        right_proj: np.ndarray, 
        kp_matcher: KPMatcher,
        use_ssc=False
        ):
    with utils.Timer(logger.debug, "Key Point triangulating took: {} seconds"):
        lr_kpres = kp_matcher.find_keypoints(
            prev_left_img, prev_right_img, use_ssc=False
            )
        left_matched_pts = utils.keypoints2points(lr_kpres.matched_A_kpts)
        right_matched_pts = utils.keypoints2points(lr_kpres.matched_B_kpts)

        cp_kpres = kp_matcher.find_keypoints_with_known(
            lr_kpres.matched_A_kpts, lr_kpres.matched_A_desc, cur_left_img, use_ssc=use_ssc
            )
        
        left_final_pts = left_matched_pts
        right_final_pts = right_matched_pts
        # If ssc was used to prune clustered key points
        # remove the points that were not selected by ssc
        if cp_kpres.ssc_A_selected_pts_idxs is not None:
            left_final_pts = left_matched_pts[cp_kpres.ssc_A_selected_pts_idxs]
            right_final_pts = right_matched_pts[cp_kpres.ssc_A_selected_pts_idxs]
        else:
            left_final_pts = np.asarray([left_final_pts[m.queryIdx] for m in cp_kpres.matches])
            right_final_pts = np.asarray([right_final_pts[m.queryIdx] for m in cp_kpres.matches])

        draw_matches = False
        if draw_matches:
            img = cv.drawMatches(
                cur_left_img, 
                cp_kpres.all_A_kpts, 
                prev_left_img, 
                cp_kpres.all_B_kpts, 
                cp_kpres.matches[:15], 
                None, 
                flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            cv.imshow("Selected keypoints", img)
            cv.waitKey(0)
        
        # left_final_pts = utils.normalize_img_coords(left_proj, left_final_pts)
        # right_final_pts = utils.normalize_img_coords(right_proj, right_final_pts)

        # shape -> (4, number of points)
        # Only triangulate points that appear in both the left and right current images 
        # and also appear in the left previous image
        
        # Polynomial triangulation is slightly more accurate
        contains_pts_at_inf, cur_3d_pts = utils.polynomial_triangulation(
            left_final_pts, left_proj, right_final_pts, right_proj
        )
        # print(f"Number of 3D pts: {cur_3d_pts.shape} --- Number of 2D pts: {left_final_pts.shape}")
        # print(f"B Shape: {cp_kpres.matched_B_kpts.shape}")
        reproj_error = utils.reprojection_error(left_proj, cur_3d_pts, left_final_pts)
        print(f"Reprojection Error: {reproj_error}")
        #print(cur_3d_pts[:3], cur_3d_pts.max(), np.abs(cur_3d_pts).min())
        #print(cur_3d_pts.max(), np.abs(cur_3d_pts).min())
        #assert False
        return TriangulateResult(
            pts_3d=cur_3d_pts[:, :3], 
            cur_2d_pts=utils.keypoints2points(cp_kpres.matched_B_kpts),
            reproj_error=reproj_error,
            success=True,
            prev_leftnright_kp_result=lr_kpres,
            left_prevncur_kp_result=cp_kpres
        )

def project_position(position: np.ndarray, rot_mat: np.ndarray, trans_vec: np.ndarray):
    Tmat = np.eye(4)
    Tmat[:3, :3] = rot_mat
    Tmat[:3, 3] = trans_vec.T
    return np.dot(position, np.linalg.inv(Tmat))[:3, :]

@dataclass
class MotionEstimateResult: 
    total_time: float
    rmat: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    estimate_successful: bool = False
    triangulate_result: Optional[TriangulateResult] = None
    depth_map: Optional[cv.Mat] = None
    reproj_error: Optional[float] = None
    ransac_pnp_inlier_idxs: Optional[np.ndarray] = None
    
class VisualOdometry:
    def __init__(
            self, 
            left_cam_3dto2d_proj_mat: np.ndarray, 
            right_cam3dto2d_proj_mat: np.ndarray,
            keypt_matcher: KPMatcher = None,
            max_depth_meters: int = 5 # meters
            ) -> None:
        #self.logger = logging.getLogger(__name__)
        self.logger = logger
        self.left_proj_mat = left_cam_3dto2d_proj_mat
        self.right_proj_mat = right_cam3dto2d_proj_mat
        self.keypt_matcher = KPMatcher() if keypt_matcher is None else keypt_matcher
        self.max_depth_m = max_depth_meters
        self.left_cam_intrinsic, self.left_cam_rotation, self.left_cam_translation = utils.decompose_projection_matrix(self.left_proj_mat)
        self.right_cam_intrinsic, self.right_cam_rotation, self.right_cam_translation = utils.decompose_projection_matrix(self.right_proj_mat)
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
        # TODO: pass fps to kalman
        self.kalman_filter = KalmanFilter(5)


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
            prev_left_img: cv.Mat, 
            prev_right_img: cv.Mat, 
            cur_left_img: cv.Mat, 
            cur_right_img: cv.Mat,
            use_kalman_filter: bool
            ):
        start_time = time.time()

        # depth_map, left_3d_pts, prev_2d_pts, cur_2d_pts = depth_map_triangulate(
        #     cur_left_img,
        #     cur_right_img,
        #     prev_left_img,
        #     self.left_proj_mat,
        #     self.right_proj_mat,
        #     self.keypt_matcher,
        #     max_depth_m=self.max_depth_m,
        #     resize=0.5,
        #     filter_disp=False
        # )
        depth_map = None
        tr_res = keypoint_triangulate(
            prev_left_img,
            prev_right_img,
            cur_left_img,
            self.left_proj_mat,
            self.right_proj_mat,
            self.keypt_matcher,
            use_ssc=False
        )
        if tr_res.pts_3d.shape[0] < 6 or tr_res.cur_2d_pts.shape[0] < 6:
            self.logger.error(
                f"Not enough 3D object pts or 2D image points "
                f"to estimate new camera projection matrix."
                )
            return MotionEstimateResult(
                total_time= time.time() - start_time,
                depth_map=depth_map,
                estimate_successful=False
            )

        succeeded, rvec, tvec, inliers = False, None, None, None
        with utils.Timer(self.logger.debug, "Computing Camera Proj Mat Time: {} seconds"):
            succeeded, rvec, tvec, inliers = cv.solvePnPRansac(
                tr_res.pts_3d, 
                tr_res.cur_2d_pts, 
                self.left_cam_intrinsic, 
                None,
                #iterationsCount=200,
                iterationsCount=400,
                #reprojectionError=5.0,
                reprojectionError=2.0,
                confidence=0.99
                )
        if succeeded is False:
            self.logger.error(f"Failed to solve for camera projection matrix")
            return MotionEstimateResult(
                total_time= time.time() - start_time,
                depth_map=depth_map,
                estimate_successful=False
            )
        
        rmat = cv.Rodrigues(rvec)[0]
        # if use_kalman_filter is True:
        #     kal_success, trans_est, rot_est = self.kalman_filter.predict_update(tvec, rmat, inliers.shape[0])
        #     self.logger.debug(f"Kalman Filter predicted translation: {np.squeeze(trans_est)}")
        #     if not kal_success:
        #         self.logger.error(
        #             f"Kalman Filter failed to estimate new position"
        #             f" while estimating motion."
        #             )
        #         return None, None, depth_map, None, None, False
        #     self.logger.info(f"Took {time.time() - start_time} seconds to successfully estimate motion.")
        #     return rot_est, trans_est, depth_map, prev_2d_pts, cur_2d_pts, True
        
        return MotionEstimateResult(
            total_time=time.time() - start_time,
            depth_map=depth_map,
            estimate_successful=True,
            rmat=rmat,
            tvec=tvec,
            triangulate_result=tr_res,
            ransac_pnp_inlier_idxs=inliers
        )
    

def test_with_kitti_sequence():
    from pi_park.ktti import KTTISequence
    from cProfile import Profile
    from pstats import SortKey, Stats

    logger.setLevel(logging.DEBUG)
    with Profile() as profile:
        sequence = KTTISequence(f"{path}/data/test_data/KITTI/00", read_first=20)

        rproj = sequence.right_proj_mat

        vo = VisualOdometry(sequence.left_proj_mat, sequence.right_proj_mat, max_depth_meters=300)
        position = np.eye(4)[:3, :]
        for i, (current_left, current_right, next_left, next_right) in enumerate(sequence.read_images()):
            rmat, tvec, depth_map, cur_2d_pts, success = vo.estimate_motion(
                current_left, current_right, next_left, next_right, False
                )
            #utils.log_imgs([utils.depth2img(depth_map)], f"{path}/logs/img_log/kitti/depth/")
            if success is True:
                ground = sequence.ground_truths[i]     
                position = project_position(position, rmat, tvec)
                print("New Position: ", np.squeeze(position[:, 3]), "Target Position: ", np.squeeze(ground[:, 3]))
                diff_whole = np.sum(abs(ground - position))
                diff_trans = abs(ground[:, 3] - position[:, 3])
                print("Total & Trans Error: ", diff_whole, diff_trans)
                #print(ground, "\n\n", position)
            else:
                print("Failed to estimate motion")
        (
         Stats(profile)
         .strip_dirs()
         .sort_stats(SortKey.CUMULATIVE)
         .print_stats()
        )

def test_with_my_sequence(config_path: str, sequence_dir: str):
    from pi_park.stereo_camera import StereoCamera

    config = utils.load_yaml_file(config_path)
    camera = StereoCamera(rectify_map_path=config["stereo_map_path"])
    vo = VisualOdometry(camera.left_proj, camera.right_proj)
    position = np.eye(4)[:3, :]
    for prev_left_img, \
        prev_right_img, \
        cur_left_img, \
        cur_right_img in utils.read_stereo_directory(sequence_dir, "left", "right"):
        
        estimate_res: MotionEstimateResult = vo.estimate_motion(
            prev_left_img, prev_right_img, cur_left_img, cur_right_img, False
            )
        position = project_position(position, estimate_res.rmat, estimate_res.tvec)
        rotation = utils.decompose_rotation_matrix(position[:3, :3])
        print(f"New Position: {np.squeeze(position[:3, 3])}")
        print(f"New Rotation: {rotation}")
        

if __name__ == "__main__":
    #test_with_kitti_sequence()
    test_with_my_sequence(f"{path}/pi_park/configs/basic_config.yml", f"{path}/data/test_data/my_cam/00/")
    #test_undistort()