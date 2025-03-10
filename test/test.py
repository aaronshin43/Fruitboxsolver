import cv2
import numpy as np
import pyautogui
import time

# Define a standard reference size (width, height)
REFERENCE_SIZE = (1063, 689)  # Adjust based on game board size

def resize_canvas(canvas):
    """Resize the captured game area to a fixed reference size."""
    return cv2.resize(canvas, REFERENCE_SIZE, interpolation=cv2.INTER_LINEAR)

time.sleep(2)
# Capture the full screen
screen = pyautogui.screenshot()
screen_np = np.array(screen)  # Convert to NumPy array
screen_bgr = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

# Convert to HSV for color detection
hsv = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2HSV)

# Detect a green background
lower_green = np.array([35, 50, 50])   # Hue 35-85 for green
upper_green = np.array([85, 255, 255])
mask = cv2.inRange(hsv, lower_green, upper_green)

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour (assumed to be the canvas)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Crop the canvas from the screen
    canvas = screen_bgr[y:y+h, x:x+w]
    canvas = resize_canvas(canvas)
else:
    print("Game Screen Not Detected!")

def start_game():
    #click reset
    pyautogui.moveTo(x + 0.1 * w, y + 0.96 * h, duration=0.1)
    pyautogui.click()

    #click play
    pyautogui.moveTo(x + 0.3 * w, y + 0.55 * h, duration=0.2)
    pyautogui.click()
start_game()