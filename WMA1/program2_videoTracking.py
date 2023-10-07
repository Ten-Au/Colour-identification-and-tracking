
"""
Program 2. Red ball tracking with video (4 points).

Import the 'rgb_ball_720.mp4' video and set the condition for correct video import, e.g. using cap.read() (1 point).
Trace the red ball as above (for each frame of film) (3 points).'''
"""

import numpy as np
import cv2 as cv
cap = cv.VideoCapture('rgb_ball_720.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    img_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower_bound = np.array([0, 170, 100])
    upper_bound = np.array([10, 255, 255])

    mask = cv.inRange(img_hsv, lower_bound, upper_bound)
    kernel = np.ones((12,12), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    segmented = cv.bitwise_and(frame, frame, mask=mask)


    contours, hierarchy = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    output = cv.drawContours(frame, contours, -1, (0,0,255), 3)

    gray_image = cv.cvtColor(segmented, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray_image, 255, 255, 255)

    M = cv.moments(thresh)
    print(M)

    M = cv.moments(thresh)
    if(M["m10"] != 0.0):
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
        cv.putText(frame, "Red Ball", (cX - 25, cY - 25),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    
    cv.imshow('frame, click q to quit', frame)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
