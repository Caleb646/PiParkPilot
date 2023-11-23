import cv2 as cv

left_cap = cv.VideoCapture(2)
left_cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc("M", "J", "P", "G"))
right_cap = cv.VideoCapture(0)
right_cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc("M", "J", "P", "G"))
num = 0
while left_cap.isOpened() and right_cap.isOpened():
#while right_cap.isOpened():
    #success1 = left_cap.grab()
    #success2 = right_cap.grab()
    success1, left_img = left_cap.read()
    success2, right_img = right_cap.read()
    #print(f"Left Success: {success1} --- Right Success: {success2}")
    #left_img = left_cap.retrieve()
    #right_img = right_cap.retrieve()
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
