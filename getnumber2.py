import cv2
import numpy as np
import pytesseract
from PIL import Image

# Load the screenshot
img = cv2.imread("game_screen.png")

# Convert to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define range for red color (adjust these values based on your apples)
lower_red1 = np.array([0, 80, 80])    # Lower bound for red (hue 0-10)
upper_red1 = np.array([20, 255, 255])  # Upper bound
lower_red2 = np.array([150, 80, 80])
upper_red2 = np.array([180, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 + mask2

# Save initial mask for debugging
cv2.imwrite("initial_mask.png", mask)
img2 = cv2.imread("initial_mask.png")

# Find contours of red regions
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Detected {len(contours)} contours")

# Process each detected apple
apples = []
test = []
# Draw all contours for debugging
img_all_contours = img.copy()
cv2.drawContours(img_all_contours, contours, -1, (0, 255, 0), 2)
cv2.imwrite("all_contours.png", img_all_contours)


for contour in contours:
    if cv2.contourArea(contour) > 100:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        # Use top-left corner for position
        cX = x + w // 2  # Optional: Approximate center if needed
        cY = y + h // 2
        
        # Crop around this position for OCR
        crop_size = max(w, h) // 2 - 8  # Adjust based on apple size
        x_start = max(cX - crop_size, 0)
        y_start = max(cY - crop_size, 0)
        x_end = min(cX + crop_size, img.shape[1])
        y_end = min(cY + crop_size, img.shape[0])
        cropped = img2[y_start:y_end, x_start:x_end]

        #Sharpen
        #gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(cropped, -1, kernel)

        #Threshold
        #_, thresh = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY_INV)
        _, thresh = cv2.threshold(sharpened, 230, 255, cv2.THRESH_BINARY)
        thresh = cv2.resize(thresh, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite("cropped_apple.png", thresh)
        
        #cv2.imwrite("cropped_apple.png", cropped)
        #number = pytesseract.image_to_string(Image.open("cropped_apple.png"), 
                                    #config='--psm 10 --oem 3 -c tessedit_char_whitelist=123456789 -c tessedit_char_blacklist=0OQqg')
        number = pytesseract.image_to_string(Image.open("cropped_apple.png"), 
                                    config='--psm 10 --oem 3 -c tessedit_char_whitelist=123456789 -c tessedit_char_blacklist=0OQqg --dpi 300')
        number = number.strip()
        test.append(number)
        if number.isdigit() and 1 <= int(number) <= 9:
            # Store top-left corner or center
            apples.append((x, y, int(number)))  # Top-left
            # Or: apples.append((cX, cY, int(number)))  # Center
        else:
            # Fallback for 4, 5, 8, 9
            if '8' in number or 'B' in number:
                number = '8'
            elif '5' in number or 'S' in number:
                number = '5'
            elif '4' in number or 'H' in number:
                number = '4'
            elif '9' in number or 'g' in number or 'q' in number or 'Q' in number:
                number = '9'
            if number:
                apples.append((x, y, int(number)))
                print(f"Corrected apple at ({x}, {y}) with number {number}")
print(f"Kept {len(apples)} apples after filtering")

#sort apples by y then x
#apples = sorted(apples, key=lambda apple: (apple[1], apple[0]))

# Print all detected apples
#print("All detected apples:", apples)
print(test)
for (x, y, number) in apples:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imwrite("detected_apples.png", img)