import cv2 as ocv
import numpy as np
import time


previousTime = 0
currentTime = 0
cap = ocv.VideoCapture(1)

while True:
    success, image = cap.read()

    # convert the input image to the HSV color space
    img_hsv = ocv.cvtColor(image, ocv.COLOR_BGR2HSV)
    # convert the hsv to a binary image by removing any pixels that do not fall within the following HSV Min/Max values

    # img_threshold = ocv.inRange(img_hsv, (18,128,128), (28, 255, 255)) # FRC yellow cone

    img_threshold = ocv.inRange(img_hsv, (90, 50, 120), (150, 255, 255)) # plastic water bottle

    # target = ocv.bitwise_and(image,image, mask=img_threshold)

    # find contours in the new binary image
    contours, _ = ocv.findContours(img_threshold, ocv.RETR_EXTERNAL, ocv.CHAIN_APPROX_SIMPLE)

    largestContour = np.array([[]])

    # initialize an empty array of values to send back to the robot
    llpython = [0, 0, 0, 0, 0, 0, 0, 0]

    # if contours have been detected, draw them
    if len(contours) > 0:
        ocv.drawContours(image, contours, -1, 255, 2)
        # record the largest contour
        largestContour = max(contours, key = ocv.contourArea)

        # get the unrotated bounding box that surrounds the contour
        x, y, w, h = ocv.boundingRect(largestContour)

        # draw the unrotated bounding box
        ocv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # record some custom data to send back to the robot
        llpython = [1, x, y, w, h, 9, 8, 7]

    #return the largest contour for the LL crosshair, the modified image, and custom robot data
    # print("Largest Contour:"+str(largestContour))
    print("llpython:" + str(llpython))

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime
    ocv.putText(image, str(round(fps)) + " fps", (20, 70), ocv.FONT_HERSHEY_PLAIN, 1.5, (253, 253, 253), 1)  # image, text, pos (x, y), font, scale, color, thickness)
    ocv.imshow("Limelight ocv test (CPU) Preview", image)
    ocv.waitKey(1)
