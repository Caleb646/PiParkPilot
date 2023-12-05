import yaml
import os
import logging
import time
import numpy as np
import cv2 as cv
import math
from typing import Tuple, Union, List

logger = logging.getLogger(__name__)


def euler2rotation(xr, yr, zr):
    xrads = math.radians(xr)
    yrads = math.radians(yr)
    zrads = math.radians(zr)
    sa = math.sin(xrads)
    ca = math.cos(xrads)
    sb = math.sin(yrads)
    cb = math.cos(yrads)
    sh = math.sin(zrads)
    ch = math.cos(zrads)
    r = np.zeros((3, 3))
    r[0] = [ch * ca, -ch*sa*cb+sh*sb, ch*sa*sb+sh*cb]
    r[1] = [sa, ca*cb, -ca*sb]
    r[2] = [-sh*ca, sh*ca*cb+ch*sb, -sh*sa*sb+ch*cb]
    return r

def decompose_euler_angles_from_proj_mat(proj_mat: np.ndarray):
    _, _, _, _, _, _, euler_angles = cv.decomposeProjectionMatrix(proj_mat)
    # eulerAngles 3-element vector containing three Euler angles of rotation in degrees.
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

class Timer:
    def __init__(self, out, m: str):
        self.out = out
        self.m = m

    def __enter__(self):#, out, m: str):
        self.start_time = time.time()
        return self

    def __exit__(self, *args, **kwargs):
        self.out(self.m.format(time.time() - self.start_time))

def clear_dir(dirpath: str):
    assert os.path.isdir(dirpath), f"Path is not a dir or doesnt exist: {dirpath}"
    files = os.listdir(dirpath)
    for file in files:
        file_path = os.path.join(dirpath, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

def load_yaml_file(path):
    if not os.path.exists(path):
        logger.error(f"YAML file at {path} does NOT exist")
        raise FileNotFoundError(f"YAML file at {path} does NOT exist")
    with open(path, mode="r") as f:
        return yaml.safe_load(f)
    
def get_current_time(format="%dd_%Hh_%Mm_%Ss"):
    return time.strftime(format)