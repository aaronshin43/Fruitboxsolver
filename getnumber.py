import cv2
import numpy as np

img = cv2.imread("game_screen.png")
#Loads the screenshot file "game_screen.png" into memory as a color image
#3D NumPy array (height × width × 3 channels: Blue, Green, Red—BGR order)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Converts the color image (img) to grayscale (gray)
#Grayscale simplifies the image to a 2D array (height × width)

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                           param1=50, param2=20, minRadius=15, maxRadius=33)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    apples = 0
    for (x, y, r) in circles:
        #print(f"Apple at ({x}, {y})")
        apples +=1
    print(apples)