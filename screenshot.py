import cv2
import numpy as np
import pyautogui
import time

time.sleep(2)
# Capture the full screen
screen = pyautogui.screenshot()
screen_np = np.array(screen)  # Convert to NumPy array
screen_bgr = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

# Convert to HSV for color detection
hsv = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2HSV)

# Example: Detect a green background (adjust these values for your canvas)
lower_green = np.array([35, 50, 50])   # Hue 35-85 for green
upper_green = np.array([85, 255, 255])
mask = cv2.inRange(hsv, lower_green, upper_green)

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour (assumed to be the canvas)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    # print(f"Canvas detected at: x={x}, y={y}, width={w}, height={h}")

    # Crop the canvas from the screen
    canvas = screen_bgr[y:y+h, x:x+w]

    # Save for verification
    cv2.imwrite("game_screen.png", canvas)

    # Draw rectangle on original screen for debugging
    # cv2.rectangle(screen_bgr, (x, y), (x + w, y + h), (0, 255, 0), 5)
    # cv2.imwrite("screen_with_canvas.png", screen_bgr)
else:
    print("Game Screen Not Detected!")