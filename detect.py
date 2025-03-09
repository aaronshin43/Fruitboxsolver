import cv2
import numpy as np
import pyautogui
import time

def visualize_detections(image, detected_numbers, output_path="detected.png"):
    """
    Draw detected numbers on the image for debugging.
    
    :param image: The game board image.
    :param detected_numbers: Dictionary with {(x, y): number} detected.
    :param output_path: Path to save the debug image.
    """
    debug_img = image.copy()

    for (x, y), num in detected_numbers.items():
        cv2.putText(debug_img, str(num), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imwrite(output_path, debug_img)
    #print(f"Debug image saved at {output_path}")

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

def remove_redundant_detections(detected_numbers, distance_threshold=20):
    """
    Removes redundant detections by clustering close detections together.
    Keeps only one number per nearby cluster.
    
    :param detected_numbers: Dictionary of {(x, y): number}
    :param distance_threshold: Maximum pixel distance to consider two detections as duplicates
    :return: Cleaned dictionary with unique number positions
    """
    # Convert dict to list for processing
    detected_list = list(detected_numbers.items())

    # Sort by Y (rows) first, then X (columns)
    detected_list.sort(key=lambda item: (item[0][1], item[0][0]))

    filtered_numbers = {}
    while detected_list:
        (x, y), num = detected_list.pop(0)
        filtered_numbers[(x, y)] = num  # Keep the first occurrence
        
        # Remove any duplicates within the distance threshold
        detected_list = [
            ((x2, y2), n) for (x2, y2), n in detected_list
            if np.linalg.norm(np.array([x, y]) - np.array([x2, y2])) > distance_threshold
        ]
    
    return filtered_numbers

def sort_numbers(grid_size, detected_numbers):
    sorted_numbers = []  # 2D list to store sorted numbers

    # Sort first by Y-position (row order)
    detected_list = sorted(detected_numbers.items(), key=lambda item: item[0][1])

    # Group numbers into rows based on Y-proximity
    rows = []
    current_row = []
    prev_y = None
    y_threshold = 20  # Adjust based on vertical spacing

    for (x, y), num in detected_list:
        if prev_y is None or abs(y - prev_y) < y_threshold:
            current_row.append((x, y, num))
        else:
            rows.append(current_row)
            current_row = [(x, y, num)]
        prev_y = y
    if current_row:
        rows.append(current_row)

    # Sort numbers within each row by X-position (column order)
    for row in rows:
        row.sort(key=lambda item: item[0])  # Sort by X (left to right)
        sorted_numbers.append([num for (_, _, num) in row])

    # Convert to NumPy array for easy indexing (ensure shape is 17x10)
    return np.array(sorted_numbers, dtype=int)
sharp = 0.5

#Image preprocessing
image = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) #grayscale
_, thresh = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY_INV) #binary scale
cv2.floodFill(thresh, None, seedPoint=(0, 0), newVal=255)
blurred = cv2.GaussianBlur(thresh, (3,3), 0)
kernel = np.array([[sharp, sharp, sharp], [sharp, 9, sharp], [sharp, sharp, sharp]])
sharpened = cv2.filter2D(thresh, -1, kernel)
#cv2.imwrite("image.png", sharpened)
# Load a reference number
templates = {str(i): cv2.imread(f"data/{i}.png", 0) for i in range(1, 10)}  

# Dictionary to store detected numbers and their positions
detected_numbers = {}

# Loop through each template
for num, template in templates.items():
    res = cv2.matchTemplate(sharpened, template, cv2.TM_CCORR_NORMED)
    
    threshold = 0.97  # Adjust this based on accuracy
    loc = np.where(res >= threshold)  # Get locations where match quality is high

    for pt in zip(*loc[::-1]):  # Iterate over matching locations
        detected_numbers[pt] = num  # Store detected number with its position

# Convert NumPy int64 types to standard Python int
detected_numbers = {(int(x), int(y)): int(num) for (x, y), num in detected_numbers.items()}

# Remove redundant detections
cleaned_numbers = remove_redundant_detections(detected_numbers)

print(f"Detected {len(cleaned_numbers)} numbers")

grid_size = (17, 10)  # 17 rows, 10 columns
visualize_detections(canvas, cleaned_numbers)
#sorted_grid = sort_numbers(grid_size, cleaned_numbers)
#print(sorted_grid)