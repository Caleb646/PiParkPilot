#Tune your depth map with real-time video from the cameras.
import cv2
import os
import threading
import numpy as np
import time
from datetime import datetime
import json

from pi_park.stereo_camera import StereoCamera
import pi_park.visual_odometry as vo

loading = False
variables = ["SWS", "SpeckleSize", "SpeckleRange", "UniqRatio", "TxtrThrshld", "NumofDisp",
    "MinDisp", "PreFiltCap", "PreFiltSize", "P1", "P2"]

variable_mapping = {"SWS" : 15, "SpeckleSize" : 100, "SpeckleRange" : 15, "UniqRatio" : 10, "TxtrThrshld" : 100, "NumofDisp" : 1,
    "MinDisp": -25, "PreFiltCap" : 30, "PreFiltSize" : 105, "P1": 8, "P2": 32}

def calculate_disparity_map(img_left, img_right, camera: StereoCamera):
        sad_window = 7
        num_disparities = sad_window*16
        block_size = 11
        matcher = cv2.StereoSGBM_create(
            minDisparity=variable_mapping["MinDisp"],
            numDisparities=variable_mapping["NumofDisp"],
            blockSize=variable_mapping["SWS"],
            P1 = variable_mapping["P1"],
            P2 = variable_mapping["P2"],
            #mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            #mode=cv2.STEREO_SGBM_MODE_HH
            )
        disp_left = matcher.compute(img_left, img_right).astype(np.float32) / 16
        disparity_normalized = cv2.normalize(disp_left, None, 0, 255, cv2.NORM_MINMAX)
        image = np.array(disparity_normalized, dtype = np.uint8)
        disparity_color = cv2.applyColorMap(image, cv2.COLORMAP_JET)
        return disparity_color, vo.calculate_depth_map(disp_left, camera.focal_length_pixel, camera.baseline_meters)

def stereo_depth_map(rectified_pair, variable_mapping):

    '''print ('SWS='+str(SWS)+' PFS='+str(PFS)+' PFC='+str(PFC)+' MDS='+\
           str(MDS)+' NOD='+str(NOD)+' TTH='+str(TTH))
    print (' UR='+str(UR)+' SR='+str(SR)+' SPWS='+str(SPWS))'''

    #blockSize is the SAD Window Size
    #Filter settings
    sbm = cv2.StereoBM_create(numDisparities=16, blockSize=variable_mapping["SWS"]) 
    sbm.setPreFilterType(1)    
    sbm.setPreFilterSize(variable_mapping['PreFiltSize'])
    sbm.setPreFilterCap(variable_mapping['PreFiltCap'])
    sbm.setSpeckleRange(variable_mapping['SpeckleRange'])
    sbm.setSpeckleWindowSize(variable_mapping['SpeckleSize'])
    sbm.setMinDisparity(variable_mapping['MinDisp'])
    sbm.setNumDisparities(variable_mapping['NumofDisp'])
    sbm.setTextureThreshold(variable_mapping['TxtrThrshld'])
    sbm.setUniquenessRatio(variable_mapping['UniqRatio'])
    

    c, r = rectified_pair[0].shape
    dmLeft = rectified_pair[0]
    dmRight = rectified_pair[1]
    disparity = sbm.compute(dmLeft, dmRight)
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    #Convering Numpy Array to CV_8UC1
    image = np.array(disparity_normalized, dtype = np.uint8)
    disparity_color = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return disparity_color, disparity_normalized

def save_load_map_settings(current_save, current_load, variable_mapping):
    global loading

    if current_load != 0:
        if os.path.isfile('../3dmap_set.txt') == True:
            loading = True
            fName = '../3dmap_set.txt'
            print('Loading parameters from file...')
            f=open(fName, 'r')
            data = json.load(f)

            cv2.setTrackbarPos("SWS", "Stereo", data['SADWindowSize'])
            cv2.setTrackbarPos("PreFiltSize", "Stereo", data['preFilterSize'])
            cv2.setTrackbarPos("PreFiltCap", "Stereo", data['preFilterCap'])
            cv2.setTrackbarPos("MinDisp", "Stereo", data['minDisparity']+100)
            cv2.setTrackbarPos("NumofDisp", "Stereo", int(data['numberOfDisparities']/16))
            cv2.setTrackbarPos("TxtrThrshld", "Stereo", data['textureThreshold'])
            cv2.setTrackbarPos("UniqRatio", "Stereo", data['uniquenessRatio'])
            cv2.setTrackbarPos("SpeckleRange", "Stereo", data['speckleRange'])
            cv2.setTrackbarPos("SpeckleSize", "Stereo", data['speckleWindowSize'])

            f.close()
            print ('Parameters loaded from file '+fName)
            print ('Redrawing depth map with loaded parameters...')
            print ('Done!') 

        else: 
            print ("File to load from doesn't exist.")

def activateTrackbars(x):
    global loading
    loading = False
    

def create_trackbars() :
    global loading

    #SWS cannot be larger than the image width and image heights.
    #In this case, width = 320 and height = 240
    cv2.createTrackbar("SWS", "Stereo", 115, 230, activateTrackbars)
    cv2.createTrackbar("SpeckleSize", "Stereo", 0, 300, activateTrackbars)
    cv2.createTrackbar("SpeckleRange", "Stereo", 0, 40, activateTrackbars)
    cv2.createTrackbar("UniqRatio", "Stereo", 1, 20, activateTrackbars)
    cv2.createTrackbar("TxtrThrshld", "Stereo", 0, 1000, activateTrackbars)
    cv2.createTrackbar("NumofDisp", "Stereo", 1, 256, activateTrackbars)
    cv2.createTrackbar("MinDisp", "Stereo", -100, 200, activateTrackbars)
    cv2.createTrackbar("PreFiltCap", "Stereo", 1, 63, activateTrackbars)
    cv2.createTrackbar("PreFiltSize", "Stereo", 5, 255, activateTrackbars)


    cv2.createTrackbar("P1", "Stereo", 0, 1024, activateTrackbars)
    cv2.createTrackbar("P2", "Stereo", 0, 2056, activateTrackbars)


    cv2.createTrackbar("Save Settings", "Stereo", 0, 1, activateTrackbars)
    cv2.createTrackbar("Load Settings","Stereo", 0, 1, activateTrackbars)

def onMouse(event, x, y, flag, disparity_normalized):
    if event == cv2.EVENT_LBUTTONDOWN:
        distance = disparity_normalized[y][x]
        print("Distance in centimeters {}".format(distance))



if __name__ == '__main__':
    camera = StereoCamera(1, 0, "data/calib/stereo_map.xml")
    # Initialise trackbars and windows
    cv2.namedWindow("Stereo")
    create_trackbars()

    print ("Cameras Started")

    while True:
        left_frame, right_frame = camera.read()

        if left_frame is not None and right_frame is not None:
            rectified_pair = (left_frame, right_frame)

            #getting trackbar position and assigning to the variables
            if loading == False:
                for v in variables:
                    current_value = cv2.getTrackbarPos(v, "Stereo")
                    if v == "SWS" or v == "PreFiltSize":
                        if current_value < 5:
                            current_value = 5
                        if current_value % 2 == 0:
                            current_value += 1
                    
                    if v == "NumofDisp":
                        if current_value == 0:
                            current_value = 1
                        current_value = current_value * 16
                    if v == "MinDisp":
                        current_value = current_value - 100
                    if v == "UniqRatio" or v == "PreFiltCap":
                        if current_value == 0:
                            current_value = 1
                    
                    variable_mapping[v] = current_value
            
           #getting save and load trackbar positions

            current_save = cv2.getTrackbarPos("Save Settings", "Stereo")
            current_load = cv2.getTrackbarPos("Load Settings", "Stereo")
 
            save_load_map_settings(current_save, current_load, variable_mapping)
            cv2.setTrackbarPos("Save Settings", "Stereo", 0)
            cv2.setTrackbarPos("Load Settings", "Stereo", 0)
            #disparity_color, disparity_normalized = stereo_depth_map(rectified_pair, variable_mapping)
            disparity_color, disparity_normalized = calculate_disparity_map(*rectified_pair, camera)

            #What happens when the mouse is clicked
            cv2.setMouseCallback("Disparity", onMouse, disparity_normalized)
            #cv2.resizeWindow("Stereo", 640, 640)  
            cv2.imshow("Disparity", disparity_color)        
            #cv2.imshow("Stereo", disparity_color)
            cv2.imshow("Frame", np.hstack((rectified_pair[0], rectified_pair[1])))
            
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break

            else:
                continue
    
    camera.shutdown()
    cv2.destroyAllWindows()
                
