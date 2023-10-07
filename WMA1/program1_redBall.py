'''Program 1. Find the red ball in the picture 'red_ball.jpg' (8 points).

Import libraries: cv2 (OpenCV), numpy, sys (1 point).
Import photo ball.png (1 point).
Set the condition for the correct loading of the image, e.g. using the 'sys.exit' command (1 point).
Change the image format to HSV (1 point).
Find the color using a binary operation (1 point).
Improve image quality (remove noise) through morphological operations (1 point).
Add the calculated center of gravity of the ball to the image (1 point).
Add the word "red ball" near the center of gravity (1 point).
'''

#Imports
import cv2 as cv
import numpy as np
import sys

img = cv.imread("red_ball.jpg", cv.IMREAD_COLOR)

    
#Condition of the correct image loading
if img is None:
    sys.exit("Could not read the file")
cv.imshow("original image",img)

img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
h, s, v = cv.split(img_hsv)
hsv_img = cv.merge(( h, s, v ))
cv.imshow("hsv_img", hsv_img)

lower_bound = np.array([0, 100, 50])
upper_bound = np.array([220, 255, 255])

mask = cv.inRange(hsv_img, lower_bound, upper_bound)
kernel = np.ones((20,20), np.uint8)
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
segmented = cv.bitwise_and(img, img, mask=mask)

contours, hierarchy = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contour_ball = cv.drawContours(img, contours, -1, (0,0,255), 3)
cv.imshow("contoured ball", contour_ball)

gray_image = cv.cvtColor(segmented, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray_image, 255, 255, 255)
cv.imshow("thresh", thresh)

M = cv.moments(thresh)
print(M)

cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

cv.circle(img, (cX, cY), 5, (255, 255, 255), -1)
cv.putText(img, "Red Ball", (cX - 25, cY - 25),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

cv.imshow("Image", contour_ball)

cv.waitKey(0)
cv.destroyAllWindows()
