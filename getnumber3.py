import cv2
import numpy as np
import pytesseract
from PIL import Image
from io import BytesIO
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
    canvas_x, canvas_y, canvas_w, canvas_h = cv2.boundingRect(largest_contour)
    #print(canvas_x, canvas_y, canvas_w, canvas_h)
    # Crop the canvas from the screen
    canvas = screen_bgr[canvas_y:canvas_y+canvas_h, canvas_x:canvas_x+canvas_w]
    #40 215 1063 689
    # Save for verification
    cv2.imwrite("game_screen.png", canvas)
else:
    print("Game Screen Not Detected!")

# Constants
rows = 10
cols = 17
spacing_x = 49
spacing_y = 49
offset_x = 110
offset_y = 115
crop_x = 20
crop_y = 26
#apple_width = 28
#apple_height = 39

apples = []
for row in range(rows):
    for col in range(cols):
        # Equation: position = canvas origin + offset + (row/col * spacing)
        x_start = offset_x + (col * spacing_x) + round(col/2)
        y_start = offset_y + (row * spacing_y) + round(row/2)
        #print(x_start, y_start)

        x_end = x_start + crop_x
        y_end = y_start + crop_y
        cropped = canvas[y_start:y_end, x_start:x_end]
        #cv2.imwrite("cropped_apple.png", cropped)

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        # sharpened = cv2.filter2D(gray, -1, kernel)
        _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
        #thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        #cv2.THRESH_BINARY_INV, 11, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        resized = cv2.resize(morphed, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC)
        
        
        cv2.imwrite("cropped_apple.png", morphed)

        number = pytesseract.image_to_string(Image.open("cropped_apple.png"), 
                                            config='--psm 10 --oem 3 -c tessedit_char_whitelist=123456789 -c tessedit_char_blacklist=0OQqg --dpi 300')
        raw_number = number.strip()
        #print(f"Raw OCR at ({col + 1}, {row + 1}): '{raw_number}'")
        if not raw_number:
            print(f"missed x:{col + 1}, y:{row + 1}")
        if raw_number.isdigit() and 1 <= int(raw_number) <= 9:
            # Store top-left corner or center
            apples.append(int(raw_number))
print(len(apples))
print(apples)