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

def get_canvas():
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
        return canvas
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

#print(f"Detected {len(cleaned_numbers)} numbers")

grid_size = (17, 10)  # 17 rows, 10 columns
sorted_grid = sort_numbers(grid_size, cleaned_numbers)
#print(sorted_grid)


# Find the position of top-left apple
# Define range for red color 
lower_red1 = np.array([0, 80, 80])    # Lower bound for red (hue 0-10)
upper_red1 = np.array([20, 255, 255])  # Upper bound
lower_red2 = np.array([150, 80, 80])
upper_red2 = np.array([180, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 + mask2

# Find contours of red regions
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter apples inside board
apples_inside_board = []
for cnt in contours:
    apple_x, apple_y, apple_w, apple_h = cv2.boundingRect(cnt)
    if x <= apple_x <= x + w and y <= apple_y <= y + h:  # Check if apple is inside the board
        apples_inside_board.append((apple_x, apple_y, apple_w, apple_h))
apple_x, apple_y, apple_w, apple_h = apples_inside_board[-1]  # Top-left apple
x1, y1, w1, h1 = apples_inside_board[0]
cell_w = round((-apple_x + x1 + w1) / 17,2)  
cell_h = round((-apple_y+y1+h1) / 10,2)


def get_start_position(row, col):
    """Convert grid coordinates to screen coordinates dynamically."""
    x = apple_x + col * cell_w - (col/2)
    y = apple_y + row * cell_h
    return x, y

def get_end_position(row, col):
    """Convert grid coordinates to screen coordinates dynamically."""
    x = apple_x + col * cell_w + cell_w + (cell_w/6) * np.log(col+1) + (cell_w/8)
    y = apple_y + row * cell_h + cell_h + (cell_h/6) * np.log(row+2) + (cell_h/8)
    return x, y

def make_move(r1, c1, r2, c2):
    """Simulate dragging a box around the best move."""
    x1, y1 = get_start_position(r1, c1)
    x2, y2 = get_end_position(r2, c2)

    # Calculate Euclidean distance
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Adjust duration based on distance
    base_speed = 0.001  # Base time per pixel (adjust as needed)
    min_duration = 0.2  # Minimum movement time
    max_duration = 0.8  # Maximum movement time

    duration = min_duration + base_speed * distance
    duration = min(max_duration, max(min_duration, duration))  # Clamp between min and max

    # Move to the first point and start dragging
    pyautogui.moveTo(x1, y1, duration=0.1)
    pyautogui.mouseDown()

    # Drag to the second point
    pyautogui.moveTo(x2, y2, duration=duration)
    pyautogui.mouseUp()

    time.sleep(0.1) 

#solve the game
score = 0
while True:
    # Compute prefix sum matrix
    prefix_sum = np.zeros((11, 18), dtype=int)  # One extra row and column for boundary conditions

    for r in range(10):
        for c in range(17):
            prefix_sum[r + 1, c + 1] = (
                sorted_grid[r, c]
                + prefix_sum[r, c + 1]
                + prefix_sum[r + 1, c]
                - prefix_sum[r, c]
            )

    # List of valid boxes found
    valid_boxes = []

    # Iterate over all possible rectangles
    for r1 in range(10):
        for c1 in range(17):
            for r2 in range(r1, 10):
                for c2 in range(c1, 17):
                    total = (
                        prefix_sum[r2 + 1, c2 + 1]
                        - prefix_sum[r1, c2 + 1]
                        - prefix_sum[r2 + 1, c1]
                        + prefix_sum[r1, c1]
                    )
                    if total == 10:
                        apples_removed = np.count_nonzero(sorted_grid[r1:r2+1, c1:c2+1])
                        valid_boxes.append(((r1, c1, r2, c2), apples_removed))

    # Sort by min apples removed
    valid_boxes.sort(key=lambda x: x[1])

    # Execute best move
    if valid_boxes:
        best_move = valid_boxes[0]
        r1, c1, r2, c2 = best_move[0]
        # Simulate the move on the screen
        make_move(r1, c1, r2, c2)

        # Update the grid and score
        sorted_grid[r1:r2+1, c1:c2+1] = 0
        score += best_move[1]
    else:
        #print(sorted_grid)
        #print(f"Final Score: {score}")
        break