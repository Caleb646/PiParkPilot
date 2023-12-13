import cv2 as cv
import argparse
import os

import pi_park.utils as utils
from pi_park.stereo_camera import StereoCamera

parser = argparse.ArgumentParser()
parser.add_argument('--path', help='Config Path', type=str)
parser.add_argument('--tseq', help='The name of the directory to store the test images', type=str, default="")
parser.add_argument('--inc', help='The straight line distance from one camera measurement to the next in feet', type=str, default=1)
args = parser.parse_args()
assert os.path.exists(args.path), f"Config path doesnt exist: {args.path}"

config = utils.load_yaml_file(args.path)
camera = StereoCamera(config["left_cam_id"], config["right_cam_id"])


generating_test_data = parser.tseq != ""
if generating_test_data:
    left_path = os.path.join("data", "test_data", "my_cam", parser.tseq, "left")
    right_path = os.path.join("data", "test_data", "my_cam", parser.tseq, "right")
else:
    left_path = config["left_imgs_path"]
    right_path = config["right_imgs_path"]

utils.create_directory(left_path)
utils.create_directory(right_path)

inc, num = 0, 0
while camera.is_opened():
    left, right = camera.read(to_gray_scale=False)
    k = cv.waitKey(5)
    # press q or esc to quit
    if k == ord("q") or k == 27: 
        break
    # press s to save
    elif k == ord('s'):
        fname = f"img_{num}.png"
        if generating_test_data:
            fname = f"{inc}_ft.png"
            inc += parser.inc

        left_saved = cv.imwrite(os.path.join(left_path, fname), left)
        right_saved = cv.imwrite(os.path.join(right_path, fname), right)
        assert left_saved & right_saved, f"Did left save: [{left_saved}] --- Did right save: [{right_saved}]"
        print("Saved!!!!")
        num += 1

    cv.imshow("Left Image", left)
    cv.imshow("Right Image", right)

camera.shutdown()
cv.destroyAllWindows()