import cv2 as cv

left_cap = cv.VideoCapture(0)
right_cap = cv.VideoCapture(2)
num = 0
while left_cap.isOpened() and right_cap.isOpened():
    succes1, left_img = left_cap.read()
    succes2, right_img = right_cap.read()
    k = cv.waitKey(5)
    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv.imwrite('data/calib/left_imgs/img_' + str(num) + '.png', left_img)
        cv.imwrite('data/calib/right_imgs/img_' + str(num) + '.png', right_img)
        print("images saved!")
        num += 1

    cv.imshow("Left Image", left_img)
    cv.imshow("Right Image", right_img)

left_cap.release()
right_cap.release()
cv.destroyAllWindows()