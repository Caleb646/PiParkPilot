import numpy as np
import cv2 as cv
import glob
import argparse
import os
import pi_park.utils as utils

parser = argparse.ArgumentParser()
parser.add_argument('--path', help='Config Path', type=str)
args = parser.parse_args()
assert os.path.exists(args.path), f"Config path doesnt exist: {args.path}"
# Example: python pi_park/calibrate.py --path="./pi_park/configs/pi_config.yml"
config = utils.load_yaml_file(path=args.path)

chessboard_size = tuple(config["chessboard_size"])
# Height, Width
frame_size = tuple(config["frame_size"])
height_img, width_img = frame_size
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
# objp[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)

# chessboard_squares_size = config["square_size_meters"] # in meters == ~0.56 inches
# objp = objp * chessboard_squares_size

# # Arrays to store object points and image points from all the images.
# objpoints = [] # 3d point in real world space
# imgpointsL = [] # 2d points in image plane.
# imgpointsR = [] # 2d points in image plane.


# imagesLeft = sorted(glob.glob(f'{config["left_imgs_path"]}/*.png'))
# imagesRight = sorted(glob.glob(f'{config["right_imgs_path"]}/*.png'))

# show_images = False
# for imgLeft, imgRight in zip(imagesLeft, imagesRight):

#     left_img = cv.imread(imgLeft)
#     right_img = cv.imread(imgRight)
#     grayL = cv.cvtColor(left_img, cv.COLOR_BGR2GRAY)
#     grayR = cv.cvtColor(right_img, cv.COLOR_BGR2GRAY)


#     assert frame_size == left_img.shape[:2]
#     assert frame_size == right_img.shape[:2]
#     # Find the chess board corners
#     retL, cornersL = cv.findChessboardCorners(grayL, chessboard_size, None)
#     retR, cornersR = cv.findChessboardCorners(grayR, chessboard_size, None)

#     # If found, add object points, image points (after refining them)
#     if retL and retR == True:
#         objpoints.append(objp)
#         cornersL = cv.cornerSubPix(grayL, cornersL, (11, 11), (-1,-1), criteria)
#         imgpointsL.append(cornersL)
#         cornersR = cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
#         imgpointsR.append(cornersR)

#         if show_images:
#             # Draw and display the corners
#             cv.drawChessboardCorners(left_img, chessboard_size, cornersL, retL)
#             cv.imshow('img left', left_img)
#             cv.drawChessboardCorners(right_img, chessboard_size, cornersR, retR)
#             cv.imshow('img right', right_img)
#             cv.waitKey(5000)


# cv.destroyAllWindows()

# ############## CALIBRATION #######################################################

# retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frame_size, None, None)
# heightL, widthL, channelsL = left_img.shape

# retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frame_size, None, None)
# heightR, widthR, channelsR = right_img.shape


# print(f"Left Cam RMSE: {retL} ----- Right Cam RMSE: {retR}") # Should be between 0.15 and 0.25
# assert retL <= config["max_allowable_rmse"]
# assert retR <= config["max_allowable_rmse"]


# newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))
# newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))

# ########## Stereo Vision Calibration #############################################
# flags = 0
# flags |= cv.CALIB_FIX_INTRINSIC
# # Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# # Hence intrinsic parameters are the same 
# criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# # This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
# retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)

########## Stereo Rectification #################################################
# rectifyScale = 1
# rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale,(0,0))

left_intrinic = np.array([
    [959.604574978534, 0, 322.604597729790],
    [0, 959.604574978534, 263.008067216411],
    [0, 0, 1]
])

right_intrinsic = np.array([
    [958.869412957424, 0, 332.089057541143],
    [0, 959.714909436994, 266.538462538104],
    [0, 0, 1]
])

# K1, K2 = Radial distortion parameters. P1, P2 = Tangential distortion parameters
# distCoeffs Input vector of distortion coefficients [k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,taux,tauy] of 4, 5, 8, 12 or 14 elements.
left_dist_coeffs = np.array([-0.00318201510884808, 0.499363395273345, 0, 0])
right_dist_coeffs = np.array([-0.0180408634858719, 0.241940448224419, 0, 0])

""" 
Matlab 2nd Camera's Pose Extrinisc matrix. Translation is in centimeters
0.999602474277881	-0.0270621023530861	0.00790797279773163	-11.6027173219198
0.0271048132928062	0.999618308999263	-0.00534466180564957	-0.489765789046087
-0.00776031661085352	0.00555688129131292	0.999954448240725	0.453553935626270
0	0	0	1
"""
# The inverse of Matlab's CameraPose 2 Extrinsic matrix
# right_extrinsic = np.array([
#     [0.999602474277881,	0.0271048132928062,	-0.00776031661085354, 11.6148996757477 / 100],
#     [-0.0270621023530861, 0.999618308999263, 0.00555688129131290, 0.173064580732736 / 100],
#     [0.00790797279773161, -0.00534466180564959,	0.999954448240725, -0.364396934991493 / 100],
#     [0, 0, 0, 1]
# ])

right_extrinsic = np.array([
    [0.999602474277881,	-0.0270621023530861, 0.00790797279773163, -11.6027173219198 / 100],
    [0.0271048132928062, 0.999618308999263,	-0.00534466180564957, -0.489765789046087 / 100],
    [-0.00776031661085352, 0.00555688129131292, 0.999954448240725, 0.453553935626270 / 100],
    [0, 0, 0, 1]
])
rotation_right = right_extrinsic[:3, :3]
trans_right = right_extrinsic[:3, 3]

rectifyScale = 1
left_rectify, right_rectify, left_proj, right_proj, Q, left_region_of_interest, right_region_of_interest = cv.stereoRectify(
    left_intrinic, left_dist_coeffs, right_intrinsic, right_dist_coeffs, (width_img, height_img), rotation_right, trans_right, rectifyScale, (0, 0)
    )

left_stereo_map = cv.initUndistortRectifyMap(
    left_intrinic, left_dist_coeffs, left_rectify, left_proj, (width_img, height_img), cv.CV_16SC2
    )

right_stereo_map = cv.initUndistortRectifyMap(
    right_intrinsic, right_dist_coeffs, right_rectify, right_proj, (width_img, height_img), cv.CV_16SC2
    )

cv_file = cv.FileStorage(config["stereo_map_path"], cv.FILE_STORAGE_WRITE)

cv_file.write('left_x_stereo_map', left_stereo_map[0])
cv_file.write('left_y_stereo_map', left_stereo_map[1])
cv_file.write('right_x_stereo_map', right_stereo_map[0])
cv_file.write('right_y_stereo_map', right_stereo_map[1])
cv_file.write("left_cam_projection", left_proj)
cv_file.write("right_cam_projection", right_proj)

cv_file.release()