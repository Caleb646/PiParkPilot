import argparse
import cv2
import os
import time

if __name__ == "__main__":
    max_value = 255
    max_value_H = 360//2
    low_H = 0
    low_S = 0
    low_V = 0
    high_H = max_value_H
    high_S = max_value
    high_V = max_value
    hsv_range_window_name = 'HSV Range'
    low_H_name = 'Low H'
    low_S_name = 'Low S'
    low_V_name = 'Low V'
    high_H_name = 'High H'
    high_S_name = 'High S'
    high_V_name = 'High V'

    def on_low_H_thresh_trackbar(val):
        global low_H
        global high_H
        low_H = val
        low_H = min(high_H-1, low_H)
        cv2.setTrackbarPos(low_H_name, hsv_range_window_name, low_H)

    def on_high_H_thresh_trackbar(val):
        global low_H
        global high_H
        high_H = val
        high_H = max(high_H, low_H+1)
        cv2.setTrackbarPos(high_H_name, hsv_range_window_name, high_H)

    def on_low_S_thresh_trackbar(val):
        global low_S
        global high_S
        low_S = val
        low_S = min(high_S-1, low_S)
        cv2.setTrackbarPos(low_S_name, hsv_range_window_name, low_S)

    def on_high_S_thresh_trackbar(val):
        global low_S
        global high_S
        high_S = val
        high_S = max(high_S, low_S+1)
        cv2.setTrackbarPos(high_S_name, hsv_range_window_name, high_S)

    def on_low_V_thresh_trackbar(val):
        global low_V
        global high_V
        low_V = val
        low_V = min(high_V-1, low_V)
        cv2.setTrackbarPos(low_V_name, hsv_range_window_name, low_V)

    def on_high_V_thresh_trackbar(val):
        global low_V
        global high_V
        high_V = val
        high_V = max(high_V, low_V+1)
        cv2.setTrackbarPos(high_V_name, hsv_range_window_name, high_V)

    cv2.namedWindow(hsv_range_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(hsv_range_window_name, 400, 300)
    cv2.createTrackbar(low_H_name, hsv_range_window_name, low_H, max_value_H, on_low_H_thresh_trackbar)
    cv2.createTrackbar(high_H_name, hsv_range_window_name, high_H, max_value_H, on_high_H_thresh_trackbar)
    cv2.createTrackbar(low_S_name, hsv_range_window_name, low_S, max_value, on_low_S_thresh_trackbar)
    cv2.createTrackbar(high_S_name, hsv_range_window_name, high_S, max_value, on_high_S_thresh_trackbar)
    cv2.createTrackbar(low_V_name, hsv_range_window_name, low_V, max_value, on_low_V_thresh_trackbar)
    cv2.createTrackbar(high_V_name, hsv_range_window_name, high_V, max_value, on_high_V_thresh_trackbar)

    parser = argparse.ArgumentParser(description='Input image path')
    parser.add_argument('--path', help = 'Input image path', type = str)
    parser.add_argument('--camera', help = 'Camera to use', default = 0, type = int)
    args = parser.parse_args()


    if args.path is not None:
        path = os.path.abspath(args.path)
        assert os.path.exists(path), f"Path [{path}] doesn't exist"
        assert os.path.isfile(path), f"Path [{path}] isn't a file."

        image = cv2.imread(path)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        print("press q to close the application")
        while True:
            threshold = cv2.inRange(hsv_image, (low_H, low_S, low_V), (high_H, high_S, high_V))
            threshold = cv2.resize(threshold, (1024, 720)) 
            cv2.imshow("Thresholded", threshold)
            key = cv2.waitKey(30)
            if key == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
            time.sleep(0.05)
    if args.camera is not None and args.path is None:
        print(f"Camera being used: {args.camera}")
        cap = cv2.VideoCapture(args.camera)
        window_capture_name = "Window Capture"
        cv2.namedWindow(window_capture_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_capture_name, 720, 480)
        while True:
            ret, frame = cap.read()
            if frame is None:
                break
            hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            threshold = cv2.inRange(hsv_image, (low_H, low_S, low_V), (high_H, high_S, high_V))
            threshold = cv2.resize(threshold, (1024, 720)) 
            cv2.imshow(window_capture_name, frame)
            cv2.imshow("Thresholded", threshold)
            key = cv2.waitKey(30)
            if key == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
            time.sleep(0.05)
    
    with open("hsv_values.txt", "w") as f:
        f.write("H Low High\n")
        f.write(f"H {low_H} {high_H}\n")

        f.write("S Low High\n")
        f.write(f"S {low_S} {high_S}\n")

        f.write("V Low High\n")
        f.write(f"V {low_V} {high_V}\n")