import cv2 as cv
import utils


config = utils.load_yaml_file()
left_cap = cv.VideoCapture(config["left_camera_id"])
left_cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc("M", "J", "P", "G"))
#right_cap = cv.VideoCapture(2) # for Raspberry Pi
right_cap = cv.VideoCapture(config["right_camera_id"])
right_cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc("M", "J", "P", "G"))
num = 0

while left_cap.isOpened() and right_cap.isOpened():
    succes1, left_img = left_cap.read()
    succes2, right_img = right_cap.read()
    k = cv.waitKey(5)
    if k == ord("q") or k == 27: # press q or esc to quit
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv.imwrite(f'{config["left_imgs_path"]}/img_' + str(num) + '.png', left_img)
        cv.imwrite(f'{config["right_imgs_path"]}/img_' + str(num) + '.png', right_img)
        print("images saved!")
        num += 1

    cv.imshow("Left Image", left_img)
    cv.imshow("Right Image", right_img)

left_cap.release()
right_cap.release()
cv.destroyAllWindows()