import cv2
import numpy as np
##sensetive to size of the screen


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

#Load image as grayscale
image = cv2.imread("game_screen.png", 0)

#Image preprocessing
_, thresh = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY_INV)
cv2.floodFill(thresh, None, seedPoint=(0, 0), newVal=255)
blurred = cv2.GaussianBlur(thresh, (3,3), 0)
kernel = np.array([[sharp, sharp, sharp], [sharp, 9, sharp], [sharp, sharp, sharp]])
sharpened = cv2.filter2D(thresh, -1, kernel)

cv2.imwrite("cropped_apple.png", sharpened)

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
        #cv2.rectangle(image, pt, (pt[0] + template.shape[1], pt[1] + template.shape[0]), (0, 255, 0), 2)

# Show the detected numbers on the image for debugging
# cv2.imshow("Detected Numbers", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Convert NumPy int64 types to standard Python int
detected_numbers = {(int(x), int(y)): int(num) for (x, y), num in detected_numbers.items()}

# Remove redundant detections
cleaned_numbers = remove_redundant_detections(detected_numbers)

print(f"Detected {len(cleaned_numbers)} numbers")

grid_size = (17, 10)  # 17 rows, 10 columns
sorted_grid = sort_numbers(grid_size, cleaned_numbers)
print(sorted_grid)