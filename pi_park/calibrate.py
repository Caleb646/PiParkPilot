import numpy as np
import cv2 as cv
import glob

import utils


config = utils.load_yaml_file()

chessboardSize = config["chessboard_size"]
frameSize = config["frame_size"]
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

chessboard_squares_size = config["square_size_meters"] # in meters == ~0.56 inches
objp = objp * chessboard_squares_size

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.


imagesLeft = sorted(glob.glob(f'{config["left_imgs_path"]}/*.png'))
imagesRight = sorted(glob.glob(f'{config["right_imgs_path"]}/*.png'))

show_images = False
for imgLeft, imgRight in zip(imagesLeft, imagesRight):

    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if retL and retR == True:
        objpoints.append(objp)
        cornersL = cv.cornerSubPix(grayL, cornersL, (11, 11), (-1,-1), criteria)
        imgpointsL.append(cornersL)
        cornersR = cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
        imgpointsR.append(cornersR)

        if show_images:
            # Draw and display the corners
            cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
            cv.imshow('img left', imgL)
            cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
            cv.imshow('img right', imgR)
            cv.waitKey(5000)


cv.destroyAllWindows()

############## CALIBRATION #######################################################

retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
heightL, widthL, channelsL = imgL.shape

retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
heightR, widthR, channelsR = imgR.shape


print(f"Left Cam RMSE: {retL} ----- Right Cam RMSE: {retR}") # Should be between 0.15 and 0.25
assert retL <= config["max_allowable_rmse"]
assert retR <= config["max_allowable_rmse"]


newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))
newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))

########## Stereo Vision Calibration #############################################
flags = 0
flags |= cv.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# Hence intrinsic parameters are the same 
criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)

width_camera_mm = 32 # in mm
height_camera_mm = 32 # in mm
pixel_focal_x = newCameraMatrixL[0, 0]
pixel_focal_y = newCameraMatrixL[1, 1]
focal_x_mm = pixel_focal_x * width_camera_mm / widthR
focal_y_mm = pixel_focal_y * height_camera_mm / heightR
print(f"Focal Length in mm - {focal_x_mm} - {focal_y_mm} - diff: {abs(focal_x_mm - focal_y_mm)}")

########## Stereo Rectification #################################################
rectifyScale = 1
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale,(0,0))

stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

# The cameras baseline distance in meters
cams_horizontal_distance_apart = 0.115 # in meters == 4.5 inches

left_cam_proj = np.hstack([np.array(cameraMatrixL), np.array([[0], [0], [0]])])
right_cam_proj = np.hstack([np.array(cameraMatrixR), np.array([[cams_horizontal_distance_apart], [0], [0]])])

left_cam_proj = projMatrixL
right_cam_proj = projMatrixR

print(f"Left Camera Projection:\n{left_cam_proj}\n")
print(f"Right Camera Projection:\n{right_cam_proj}\n")

cam_intrinsic, cam_rotation, cam_translation, _, _, _, euler_angles = cv.decomposeProjectionMatrix(right_cam_proj)
print(f"Right Camera Translation: {cam_translation / cam_translation[3]}")

print("Saving parameters!")
cv_file = cv.FileStorage(config["stereo_map_path"], cv.FILE_STORAGE_WRITE)

cv_file.write('left_x_stereo_map', stereoMapL[0])
cv_file.write('left_y_stereo_map', stereoMapL[1])
cv_file.write('right_x_stereo_map', stereoMapR[0])
cv_file.write('right_y_stereo_map', stereoMapR[1])
cv_file.write("left_cam_projection", left_cam_proj)
cv_file.write("right_cam_projection", right_cam_proj)

cv_file.release()