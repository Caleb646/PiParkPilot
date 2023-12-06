import cv2 as cv
import argparse
import os


import pi_park.utils as utils
from pi_park.stereo_camera import StereoCamera



parser = argparse.ArgumentParser()
parser.add_argument('--path', help='Config Path', type=str)
args = parser.parse_args()
assert os.path.exists(args.path), f"Config path doesnt exist: {args.path}"

config = utils.load_yaml_file(args.path)
camera = StereoCamera(config["left_cam_id"], config["right_cam_id"])
num = 0

while camera.is_opened():
    left, right = camera.read(to_gray_scale=False)
    k = cv.waitKey(5)
    if k == ord("q") or k == 27: # press q or esc to quit
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv.imwrite(f'{config["left_imgs_path"]}/img_' + str(num) + '.png', left)
        cv.imwrite(f'{config["right_imgs_path"]}/img_' + str(num) + '.png', right)
        print("images saved!")
        num += 1

    cv.imshow("Left Image", left)
    cv.imshow("Right Image", right)

camera.shutdown()
cv.destroyAllWindows()