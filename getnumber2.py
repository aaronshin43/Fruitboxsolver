import cv2
import numpy as np

# Load the screenshot
img = cv2.imread("game_screen.png")

# Convert to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define range for red color (adjust these values based on your apples)
lower_red1 = np.array([0, 170, 240])    # Lower bound for red (hue 0-10)
upper_red1 = np.array([10, 255, 255])  # Upper bound

# Create masks for red regions
mask = cv2.inRange(hsv, lower_red1, upper_red1)

# Optional: Clean up the mask (remove noise)
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)

# Find contours of red regions
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Process each detected apple
apples = 0
for contour in contours:
    # Skip small noise (adjust area threshold as needed)
    if cv2.contourArea(contour) > 100:
        # Get the center of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:  # Avoid division by zero
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            apples+=1
            #print(f"Apple at ({cX}, {cY})")
        # Optional: Draw the contour and center for debugging
        #cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
        #cv2.circle(img, (cX, cY), 5, (255, 0, 0), -1)
print(apples)
# Save the result to check detection
#cv2.imwrite("detected_apples.png", img)