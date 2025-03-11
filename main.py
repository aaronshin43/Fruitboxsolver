import cv2
import numpy as np
import pyautogui
import time
import package.solver as solver
import copy

# Define a standard reference size (width, height)
REFERENCE_SIZE = (1063, 689)  # Adjust based on game board size

def resize_canvas(canvas):
    """Resize the captured game area to a fixed reference size."""
    return cv2.resize(canvas, REFERENCE_SIZE, interpolation=cv2.INTER_LINEAR)

def get_canvas():
    """Capture game canvas in user's screen"""
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
        return x, y, w, h, hsv, canvas
    else:
        print("Game Screen Not Detected!")

def start_game(x, y, w, h):
    """Start the game by clicking reset and start button on game canvas
    
    :param x: int of x-position of top left corner of game canvas
    :param y: int of y-position of top left corner of game canvas
    :param w: int of width of game canvas
    :param h: int of height of game canvas"""
    #click reset
    pyautogui.moveTo(x + 0.1 * w, y + 0.96 * h, duration=0.1)
    pyautogui.click()

    #click play
    pyautogui.moveTo(x + 0.3 * w, y + 0.55 * h, duration=0.2)
    pyautogui.click()

    pyautogui.moveTo(x + 0.05 * w, y + 0.05 * h, duration=0.1)

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
    if len(filtered_numbers) != 170:
        print(f"Failed to detect {170-len(filtered_numbers)} numbers")
        exit(1)
    return filtered_numbers

def sort_numbers(detected_numbers):
    """Sorts the numbers in given dictionary
    
    :param detected_numbers: Cleaned dictionary with unique number positions
    :return: Sorted NumPy array with numbers"""
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

def topleft_apple(x, y, w, h, hsv):
    """Finds the screen position of top-left apple to calculate relative positions of apples
    
    :param x: int of x-position of top left corner of game canvas
    :param y: int of y-position of top left corner of game canvas
    :param w: int of width of game canvas
    :param h: int of height of game canvas
    :param hsv: MatLike image of game canvas in hsv
    :return: int values of x, y coordinates, width, and height of top-left apple"""

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
    return apple_x, apple_y, cell_w, cell_h

def get_position(row, col, row1, col1):
    """Convert grid coordinates to screen coordinates dynamically.
    
    :param row: int of grid position of starting row
    :param col: int of grid position of starting column
    :param row1: int of grid position of ending row
    :param col1: int of grid position of ending column
    :return: int of screen coordinates of starting/ending rows/columns
    """

    x = apple_x + col * (w/21.48) - w/170
    y = apple_y + row * cell_h
    x1 = apple_x + (col1+1) * (w/21.5)
    y1 = apple_y + (row1+1) * (h/13.94) + h/145
    return x, y, x1, y1

def make_move(r1, c1, r2, c2):
    """Simulate dragging a box around the best move.
    
    :param row: int of grid position of starting row
    :param col: int of grid position of starting column
    :param row1: int of grid position of ending row
    :param col1: int of grid position of ending column"""
    x1, y1, x2, y2 = get_position(r1, c1, r2, c2)

    # Move to the first point and start dragging
    pyautogui.moveTo(x1, y1, duration=0.1)
    pyautogui.mouseDown()

    # Drag to the second point
    pyautogui.moveTo(x2, y2, duration=0.5)
    time.sleep(0.05) 
    pyautogui.mouseUp()

    time.sleep(0.05) 

def simulate_strategy(grid, strategy_func):
    """Simulate the given strategy and return the final score."""
    grid_copy = copy.deepcopy(grid)  # Make a copy so original board isn't affected
    score = 0

    while True:
        best_move = strategy_func(grid_copy)

        if best_move is None:
            break  # No more valid moves

        r1, c1, r2, c2 = best_move
        apples_removed = np.count_nonzero(grid_copy[r1:r2+1, c1:c2+1])
        score += apples_removed
        grid_copy[r1:r2+1, c1:c2+1] = 0  # Apply move

    return score

def choose_best_strategy(grid):
    """Simulate all strategies and choose the best one based on final score."""
    score_min = simulate_strategy(grid, solver.min_removal_strategy)
    score_large_num = simulate_strategy(grid, solver.large_num_strategy)
    score_small_num = simulate_strategy(grid, solver.small_num_strategy)
    # Pick the strategy with the highest final score
    strategy_scores = {
        "min_removal": score_min,
        "large_num": score_large_num,
        "small_num": score_small_num
    }
    #print(strategy_scores)
    best_strategy = max(strategy_scores, key=strategy_scores.get)
    if best_strategy == "min_removal":
        print("Startegy: Target Smallest Group")
    elif best_strategy == "large_num":
        print("Startegy: Target Large Number")
    else:
        print("Startegy: Target Small Number")
    print(f"Expected score: {strategy_scores.get(best_strategy)}")
    return best_strategy

#Start the Game
time.sleep(2)
canvas_list = get_canvas()
x, y, w, h = canvas_list[0:4]
start_game(x, y, w, h)

#Get initial board
time.sleep(1)
canvas_list = get_canvas()
hsv, canvas = canvas_list[4:6]

#Get position of top-left apple
apple_x, apple_y, cell_w, cell_h = topleft_apple(x, y, w, h, hsv)

#Image preprocessing
sharp = 0.5
image = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) #grayscale
_, thresh = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY_INV) #binary scale
cv2.floodFill(thresh, None, seedPoint=(0, 0), newVal=255)
blurred = cv2.GaussianBlur(thresh, (3,3), 0)
kernel = np.array([[sharp, sharp, sharp], [sharp, 9, sharp], [sharp, sharp, sharp]])
sharpened = cv2.filter2D(thresh, -1, kernel)

# Load a template number
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

grid = sort_numbers(cleaned_numbers)
#print(sorted_grid)

# Main game loop
score = 0
best_strategy = choose_best_strategy(grid)

while True:
    if best_strategy == "min_removal":
        best_move = solver.min_removal_strategy(grid)
    elif best_strategy == "large_num":
        best_move = solver.large_num_strategy(grid)
    else:
        best_move = solver.small_num_strategy(grid)

    if best_move is None:
        #print("No valid moves found. Game Over.")
        break

    r1, c1, r2, c2 = best_move
    # Simulate the move on the screen
    make_move(r1, c1, r2, c2)
    print(grid[r1:r2+1, c1:c2+1])
    apples_removed = np.count_nonzero(grid[r1:r2+1, c1:c2+1])
    score += apples_removed
    grid[r1:r2+1, c1:c2+1] = 0  # Apply move

#print(f"Final Score: {score}")