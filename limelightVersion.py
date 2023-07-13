import cv2 as ocv
import numpy as np

    
# runPipeline() is called every frame by Limelight's backend.
def runPipeline(image, llrobot):
    img_hsv = ocv.cvtColor(image, ocv.COLOR_BGR2HSV)
    # convert the hsv to a binary image by removing any pixels
    # that do not fall within the following HSV Min/Max values
    img_threshold = ocv.inRange(img_hsv, (18, 128, 128), (28, 255, 255))

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
    # make sure to return a contour, an image to stream,
    # and optionally an array of up to 8 values for the "llpython"networktables array
    return largestContour, image, llpython
