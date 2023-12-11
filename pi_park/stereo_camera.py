import numpy as np
import cv2 as cv
import logging
import time
from typing import Tuple, List, Union
import sys
import os

# import pathlib
# import os
# import sys
# parent = pathlib.Path(os.path.abspath(os.path.curdir))
# path = os.path.join(str(parent))
# sys.path.append(path)

import pi_park.utils as utils


logger = logging.getLogger(__name__)
cv.Mat = np.ndarray

class StereoCamera:
    def __init__(
            self, 
            left_id: int, 
            right_id: int,
            rectify_map_path: Union[str, None] = None,
            img_log_dir: Union[str, None] = None
            ) -> None:
        #self.logger = logging.getLogger(__name__)
        self.logger = logger      
        self.rectify_map_path = rectify_map_path
        self.img_log_dir = img_log_dir
        if self.rectify_map_path is not None:
            assert os.path.exists(self.rectify_map_path), f"Path to Stereo Map [{self.rectify_map_path}] does NOT exist."
            cv_file = cv.FileStorage()
            cv_file.open(self.rectify_map_path, cv.FileStorage_READ)

            self.left_x_stereo_map = cv_file.getNode('left_x_stereo_map').mat()
            self.left_y_stereo_map = cv_file.getNode('left_y_stereo_map').mat()
            self.right_x_stereo_map = cv_file.getNode('right_x_stereo_map').mat()
            self.right_y_stereo_map = cv_file.getNode('right_y_stereo_map').mat()
            self.left_proj = cv_file.getNode("left_cam_projection").mat()
            """
            shape -> (3, 4).
            The left camera's initial extrinsic matrix will be the origin for the world.
            Every point's position will be relative to it.
            """
            self.right_proj = cv_file.getNode("right_cam_projection").mat()

            self.left_intrinsic, self.rotation_left, self.trans_left = utils.decompose_projection_matrix(self.left_proj)
            self.right_intrinsic, self.rotation_right, self.trans_right = utils.decompose_projection_matrix(self.right_proj)

        if "win" in sys.platform:
            self.logger.info(f"Using Windows VideoCapture setup: [{sys.platform}]")
            self.left_cap = cv.VideoCapture(left_id, cv.CAP_DSHOW)
            self.right_cap = cv.VideoCapture(right_id, cv.CAP_DSHOW)
        else:
            self.logger.info(f"Using Default VideoCapture setup: [{sys.platform}]")
            self.left_cap = cv.VideoCapture(left_id)
            self.right_cap = cv.VideoCapture(right_id)
        self.left_cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc("M", "J", "P", "G"))
        self.right_cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc("M", "J", "P", "G"))
        assert self.left_cap.isOpened() and self.right_cap.isOpened()

    @property
    def focal_length_pixel(self):
        return self.left_intrinsic[0, 0]
    
    @property
    def baseline_meters(self):
        return abs(self.trans_right[0] - self.trans_left[0])

    def shutdown(self):
        self.left_cap.release()
        self.right_cap.release()

    def is_opened(self):
        return self.left_cap.isOpened() and self.right_cap.isOpened()

    def undistort_rectify(
            self, left: cv.Mat, right: cv.Mat
        ):
        left_undistorted = cv.remap(
            left, 
            self.left_x_stereo_map, 
            self.left_y_stereo_map, 
            cv.INTER_LANCZOS4, 
            cv.BORDER_CONSTANT, 
            0
            )
        right_undistorted = cv.remap(
            right, 
            self.right_x_stereo_map, 
            self.right_y_stereo_map, 
            cv.INTER_LANCZOS4, 
            cv.BORDER_CONSTANT, 
            0
            )
        return left_undistorted, right_undistorted

    def read(self, to_gray_scale=True):
        assert self.is_opened()
        #if self.left_cap.isOpened() and self.right_cap.isOpened():
        # TODO: reading the left and right images may need to be synchronized 
        # better than this.
        s1 = time.time()
        left_success, left = self.left_cap.read()
        e1 = time.time()
        s2 = time.time()
        right_success, right = self.right_cap.read()
        e2 = time.time()

        if not left_success or not right_success:
            self.logger.error(f"Left Camera Failed: {not left_success} ----- Right Camera Failed: {not right_success}")
            return None, None
        self.logger.debug(
            f"Left Cam -> Duration: {round(e1 - s1, 4)} ---- Right Cam -> Duration: {round(e2 - s2, 4)} "
            f"--- Out of Sync by {round(abs(e2-e1), 4)} seconds"
            )
        if to_gray_scale:
            left, right = cv.cvtColor(left, cv.COLOR_BGR2GRAY), cv.cvtColor(right, cv.COLOR_BGR2GRAY)

        if self.rectify_map_path is not None:
            left, right = self.undistort_rectify(left, right)
            
        return left, right

if __name__ == "__main__":
    cam = StereoCamera(1, 0, f"{path}/data/calib/stereo_map.xml")

    left, right = cv.imread(f"{path}/data/calib/left_imgs/img_1.png"), cv.imread(f"{path}/data/calib/right_imgs/img_1.png")
    l, r = cam.undistort_rectify(left, right)
    cv.imwrite("ltest.jpeg", l)
    cv.imwrite("rtest.jpeg", r)