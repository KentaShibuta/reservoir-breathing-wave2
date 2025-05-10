import cv2
import numpy as np

def get_diff():
    I1 = cv2.imread('/root/app/data/frames/30_bin.png', cv2.IMREAD_GRAYSCALE)
    I2 = cv2.imread('/root/app/data/frames/60_bin.png', cv2.IMREAD_GRAYSCALE)
    I3 = cv2.imread('/root/app/data/frames/90_bin.png', cv2.IMREAD_GRAYSCALE)

    img_diff1 = cv2.absdiff(I2,I1)
    img_diff2 = cv2.absdiff(I3,I2)

    Im = cv2.bitwise_and(img_diff1, img_diff2)

    img_th = cv2.threshold(Im, 10, 255,cv2.THRESH_BINARY)[1]

    operator = np.ones((3,3), np.uint8)
    img_dilate = cv2.dilate(img_th, operator, iterations=4)
    img_mask = cv2.erode(img_dilate,operator,iterations=4)

    img_dst = cv2.bitwise_and(I3, img_mask)

    cv2.imwrite("./diff_1-3.png", img_dst)