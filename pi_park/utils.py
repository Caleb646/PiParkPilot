import yaml
import os
import logging
import time
import numpy as np
import cv2 as cv
import math
import shutil
from scipy import linalg
from typing import Tuple, Union, List, Optional

logger = logging.getLogger(__name__)
cv.Mat = np.ndarray

def get_time():
    return time.perf_counter()

def depth2img(depth_map: cv.Mat):
    return np.array(cv.normalize(depth_map, None, 0, 255, cv.NORM_MINMAX), dtype=np.uint8)

def euler2rotation(xr, yr, zr):
    """
    Parameters
    --------------------
    xr: (float) `should be in degrees`
    yr: (float) `should be in degrees`
    zr: (float) `should be in degrees`
    """
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
    return np.asarray([math.degrees(x_theta), math.degrees(y_theta), math.degrees(z_theta)])

def decompose_proj2intrisic_extrinsic(proj: np.ndarray):
    intrin, rot, trans = decompose_projection_matrix(proj)
    extrin = np.eye(4)
    extrin[:3, :3] = rot
    extrin[:3, 3] = trans.T
    return intrin, extrin[:3]

def keypoints2points(kpts):
    return np.asarray([kp.pt for kp in kpts])

class Timer:
    def __init__(self, out, m: str):
        self.out = out
        self.m = m

    def __enter__(self):#, out, m: str):
        self.start_time = time.time()
        return self

    def __exit__(self, *args, **kwargs):
        self.out(self.m.format(time.time() - self.start_time))

def clear_dir(dirpath: str, remove_dirs=True):
    if not os.path.isdir(dirpath):
        logger.error(f"Path is not a dir or doesnt exist: {dirpath}")
        return
    files = os.listdir(dirpath)
    for file in files:
        file_path = os.path.join(dirpath, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        if remove_dirs and os.path.isdir(file_path):
            shutil.rmtree(file_path)

def load_yaml_file(path):
    if not os.path.exists(path):
        logger.error(f"YAML file at {path} does NOT exist")
        raise FileNotFoundError(f"YAML file at {path} does NOT exist")
    with open(path, mode="r") as f:
        return yaml.safe_load(f)
    
def get_current_time(format="%Hh%Mm%S.%f"):
    from datetime import datetime
    curr_time = datetime.now()
    return curr_time.strftime(format)
    #return time.strftime(format)

def create_directory(directory: Optional[str]):
    if directory is not None:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

def log_imgs(imgs: List[cv.Mat], directory: Union[str, None]):
    create_directory(directory)
    for i, img in enumerate(imgs):
        cur_time = get_current_time()
        path = os.path.join(directory, f"{cur_time}__{i}.jpg")
        saved = cv.imwrite(path, img)

def read_stereo_directory(
        main_directory: str, left_dir: str, right_dir: str, read_first=-1, to_gray_scale=True
        ):
    left_path = os.path.join(main_directory, left_dir)
    right_path = os.path.join(main_directory, right_dir)
    assert os.path.exists(left_path), f"Left Image path [{left_path}] does NOT exist"
    assert os.path.exists(right_path), f"Right Image path [{right_path}] does NOT exist"
    left_img_fnames = [
        os.path.join(os.path.join(left_path, fname)) 
        for fname in sorted(os.listdir(os.path.join(left_path)))
    ]
    right_img_fnames = [
            os.path.join(os.path.join(right_path, fname)) 
            for fname in sorted(os.listdir(os.path.join(right_path)))
        ]
    image_paths = list(zip(left_img_fnames, right_img_fnames))
    if read_first == -1: # if -1 read all of the images
            read_first = len(image_paths)
    image_paths = image_paths[:min(read_first, len(image_paths))]
    for i in range(len(image_paths) - 1):
        cur_left = cv.imread(image_paths[i][0])
        cur_right = cv.imread(image_paths[i][1])
        next_left = cv.imread(image_paths[i+1][0])
        next_right = cv.imread(image_paths[i+1][1])
        if to_gray_scale:
            cur_left, cur_right = cv.cvtColor(cur_left, cv.COLOR_BGR2GRAY), cv.cvtColor(cur_right, cv.COLOR_BGR2GRAY)
            next_left, next_right = cv.cvtColor(next_left, cv.COLOR_BGR2GRAY), cv.cvtColor(next_right, cv.COLOR_BGR2GRAY)
        yield cur_left, cur_right, next_left, next_right


def normalize_img_coords(proj: np.ndarray, img_2d_pts: np.ndarray, distortionCoefs=None):
    """
    img_2d_pts: shape -> (number of points, 2)

    x = (u — cx) / fx
    y = (v — cy) / fy
    Here, (c_x, c_y) are the principal point (optical center) coordinates, 
    and f_x and f_y are the focal lengths along the x and y axes, respectively.
    """
    intrinsic, _, _ = decompose_projection_matrix(proj)
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    img_2d_pts[:, 0] = (img_2d_pts[:, 0] - cx) / fx
    img_2d_pts[:, 1] = (img_2d_pts[:, 1] - cy) / fy
    #return img_2d_pts
    return np.squeeze(cv.undistortPoints(img_2d_pts, intrinsic, distortionCoefs))

def linear_eigen_triangulation(
        u1, P1, u2, P2, max_coordinate_value=1.e16, output_dtype=np.float64
        ):
    """
    SOURCE: https://github.com/Eliasvan/Multiple-Quadrotor-SLAM/blob/master/Work/python_libs/triangulation.py

    Linear Eigenvalue based (using SVD) triangulation.
    Wrapper to OpenCV's "triangulatePoints()" function.
    Relative speed: 1.0
    
    (u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
    (u2, P2) is the second pair.
    "max_coordinate_value" is a threshold to decide whether points are at infinity
    
    The status-vector is based on the assumption that all 3D points have finite coordinates.
    """
    x = cv.triangulatePoints(P1[0:3, 0:4], P2[0:3, 0:4], u1.T, u2.T)
    
    x[0:3, :] /= x[3:4, :]    # normalize coordinates
    x_status = (np.max(abs(x[0:3, :]), axis=0) <= max_coordinate_value)    # NaN or Inf will receive status False
    
    return x_status, x[0:3, :].T.astype(output_dtype)

def linear_LS_triangulation(u1, P1, u2, P2, output_dtype=np.float64):
    """
    SOURCE: https://github.com/Eliasvan/Multiple-Quadrotor-SLAM/blob/master/Work/python_libs/triangulation.py

    Linear Least Squares based triangulation.
    Relative speed: 0.1
    
    (u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
    (u2, P2) is the second pair.
    
    u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.
    
    The status-vector will be True for all points.
    """
    # Initialize consts to be used in linear_LS_triangulation()
    linear_LS_triangulation_C = -np.eye(2, 3)
    A = np.zeros((4, 3))
    b = np.zeros((4, 1))
    
    # Create array of triangulated points
    x = np.zeros((3, len(u1)))
    
    # Initialize C matrices
    C1 = np.array(linear_LS_triangulation_C)
    C2 = np.array(linear_LS_triangulation_C)
    
    for i in range(len(u1)):
        # Derivation of matrices A and b:
        # for each camera following equations hold in case of perfect point matches:
        #     u.x * (P[2,:] * x)     =     P[0,:] * x
        #     u.y * (P[2,:] * x)     =     P[1,:] * x
        # and imposing the constraint:
        #     x = [x.x, x.y, x.z, 1]^T
        # yields:
        #     (u.x * P[2, 0:3] - P[0, 0:3]) * [x.x, x.y, x.z]^T     +     (u.x * P[2, 3] - P[0, 3]) * 1     =     0
        #     (u.y * P[2, 0:3] - P[1, 0:3]) * [x.x, x.y, x.z]^T     +     (u.y * P[2, 3] - P[1, 3]) * 1     =     0
        # and since we have to do this for 2 cameras, and since we imposed the constraint,
        # we have to solve 4 equations in 3 unknowns (in LS sense).

        # Build C matrices, to construct A and b in a concise way
        C1[:, 2] = u1[i, :]
        C2[:, 2] = u2[i, :]
        
        # Build A matrix:
        # [
        #     [ u1.x * P1[2,0] - P1[0,0],    u1.x * P1[2,1] - P1[0,1],    u1.x * P1[2,2] - P1[0,2] ],
        #     [ u1.y * P1[2,0] - P1[1,0],    u1.y * P1[2,1] - P1[1,1],    u1.y * P1[2,2] - P1[1,2] ],
        #     [ u2.x * P2[2,0] - P2[0,0],    u2.x * P2[2,1] - P2[0,1],    u2.x * P2[2,2] - P2[0,2] ],
        #     [ u2.y * P2[2,0] - P2[1,0],    u2.y * P2[2,1] - P2[1,1],    u2.y * P2[2,2] - P2[1,2] ]
        # ]
        A[0:2, :] = C1.dot(P1[0:3, 0:3])    # C1 * R1
        A[2:4, :] = C2.dot(P2[0:3, 0:3])    # C2 * R2
        
        # Build b vector:
        # [
        #     [ -(u1.x * P1[2,3] - P1[0,3]) ],
        #     [ -(u1.y * P1[2,3] - P1[1,3]) ],
        #     [ -(u2.x * P2[2,3] - P2[0,3]) ],
        #     [ -(u2.y * P2[2,3] - P2[1,3]) ]
        # ]
        b[0:2, :] = C1.dot(P1[0:3, 3:4])    # C1 * t1
        b[2:4, :] = C2.dot(P2[0:3, 3:4])    # C2 * t2
        b *= -1
        
        cv.solve(A, b, x[:, i:i+1], cv.DECOMP_SVD)
    
    return np.ones(len(u1), dtype=bool), x.T.astype(output_dtype)


def direct_linear_transform(apts: np.ndarray, aproj, bpts, bproj):
    num_pts = apts.shape[0]
    pts = np.zeros((num_pts, 3))
    for i in range(num_pts):
        a = apts[i]
        b = bpts[i]
        A = [
            a[1] * aproj[2,:] - aproj[1,:],
            aproj[0,:] - a[0] * aproj[2,:],
            b[1] * bproj[2,:] - bproj[1,:],
            bproj[0,:] - b[0] * bproj[2,:]
            ]
        A = np.array(A).reshape((4,4))
        B = A.transpose() @ A
        U, s, Vh = linalg.svd(B, full_matrices = False)
        pts[i] = Vh[3,0:3] / Vh[3,3]
    return np.ones(num_pts, dtype=bool), pts


def iterative_LS_triangulation(u1, P1, u2, P2, tolerance=3.e-5):
    """
    SOURCE: https://github.com/Eliasvan/Multiple-Quadrotor-SLAM/blob/master/Work/python_libs/triangulation.py

    Iterative (Linear) Least Squares based triangulation.
    From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997.
    Relative speed: 0.025
    
    (u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
    (u2, P2) is the second pair.
    "tolerance" is the depth convergence tolerance.
    
    Additionally returns a status-vector to indicate outliers:
        1: inlier, and in front of both cameras
        0: outlier, but in front of both cameras
        -1: only in front of second camera
        -2: only in front of first camera
        -3: not in front of any camera
    Outliers are selected based on non-convergence of depth, and on negativity of depths (=> behind camera(s)).
    
    u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.
    """
    iterative_LS_triangulation_C = -np.eye(2, 3)

    A = np.zeros((4, 3))
    b = np.zeros((4, 1))
    
    # Create array of triangulated points
    x = np.empty((4, len(u1))); x[3, :].fill(1)    # create empty array of homogenous 3D coordinates
    x_status = np.empty(len(u1), dtype=int)
    
    # Initialize C matrices
    C1 = np.array(iterative_LS_triangulation_C)
    C2 = np.array(iterative_LS_triangulation_C)
    
    for xi in range(len(u1)):
        # Build C matrices, to construct A and b in a concise way
        C1[:, 2] = u1[xi, :]
        C2[:, 2] = u2[xi, :]
        
        # Build A matrix
        A[0:2, :] = C1.dot(P1[0:3, 0:3])    # C1 * R1
        A[2:4, :] = C2.dot(P2[0:3, 0:3])    # C2 * R2
        
        # Build b vector
        b[0:2, :] = C1.dot(P1[0:3, 3:4])    # C1 * t1
        b[2:4, :] = C2.dot(P2[0:3, 3:4])    # C2 * t2
        b *= -1
        
        # Init depths
        d1 = d2 = 1.
        
        for i in range(10):
            cv.solve(A, b, x[0:3, xi:xi+1], cv.DECOMP_SVD)
            
            # Calculate new depths
            d1_new = P1[2, :].dot(x[:, xi])
            d2_new = P2[2, :].dot(x[:, xi])
            
            # Convergence criterium
            if abs(d1_new - d1) <= tolerance and abs(d2_new - d2) <= tolerance:
                break
            
            # Re-weight A matrix and b vector with the new depths
            A[0:2, :] *= 1 / d1_new
            A[2:4, :] *= 1 / d2_new
            b[0:2, :] *= 1 / d1_new
            b[2:4, :] *= 1 / d2_new
            
            # Update depths
            d1 = d1_new
            d2 = d2_new
        
        # Set status
        x_status[xi] = ( i < 10 and                       # points should have converged by now
                         (d1_new > 0 and d2_new > 0) )    # points should be in front of both cameras
        if d1_new <= 0: 
            x_status[xi] -= 1
        if d2_new <= 0: 
            x_status[xi] -= 2
    
    return x_status, x[0:3, :].T.astype(np.float32)


def polynomial_triangulation(u1, P1, u2, P2):
    """
    SOURCE: https://github.com/Eliasvan/Multiple-Quadrotor-SLAM/blob/master/Work/python_libs/triangulation.py

    Polynomial (Optimal) triangulation.
    Uses Linear-Eigen for final triangulation.
    Relative speed: 0.1
    
    (u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
    (u2, P2) is the second pair.
    
    u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.
    
    The status-vector is based on the assumption that all 3D points have finite coordinates.
    """
    P1_full = np.eye(4); P1_full[0:3, :] = P1[0:3, :]    # convert to 4x4
    P2_full = np.eye(4); P2_full[0:3, :] = P2[0:3, :]    # convert to 4x4
    P_canon = P2_full.dot(cv.invert(P1_full)[1])    # find canonical P which satisfies P2 = P_canon * P1
    
    # "F = [t]_cross * R" [HZ 9.2.4]; transpose is needed for numpy
    F = np.cross(P_canon[0:3, 3], P_canon[0:3, 0:3], axisb=0).T
    
    # Project 2D matches to closest pair of epipolar lines
    u1_new, u2_new = cv.correctMatches(F, u1.reshape(1, len(u1), 2), u2.reshape(1, len(u1), 2))
    
    # For a purely sideways trajectory of 2nd cam, correctMatches() returns NaN for all possible points!
    if np.isnan(u1_new).all() or np.isnan(u2_new).all():
        F = cv.findFundamentalMat(u1, u2, cv.FM_8POINT)[0]    # so use a noisy version of the fund mat
        u1_new, u2_new = cv.correctMatches(F, u1.reshape(1, len(u1), 2), u2.reshape(1, len(u1), 2))
    
    # Triangulate using the refined image points
    return linear_eigen_triangulation(u1_new[0], P1, u2_new[0], P2)


def reprojection_error(camera_proj: np.ndarray, pts_3d, pts_2d):
    num_points = pts_3d.shape[0]
    assert num_points == pts_2d.shape[0], f"Number of 3D and 2D points must match."
    intrin, rot_mat, trans = decompose_projection_matrix(camera_proj)
    rvec = cv.Rodrigues(rot_mat)[0]
    repts, jacob = cv.projectPoints(pts_3d, rvec, trans, intrin, None)
    repts = np.squeeze(repts)
    error = np.sqrt(((repts - pts_2d)**2).sum() / float(num_points))
    # Should be somewhere between 0.5 and 5
    return error
        