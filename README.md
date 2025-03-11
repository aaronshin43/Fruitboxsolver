# Fruitbox Game Solver

This project is a Python-based automation tool that plays a [Fruitbox](https://www.gamesaien.com/game/fruit_box_a/) game where apples with numbers (1-9) must be enclosed in boxes that sum to 10. The program maximizes the score by selecting the optimal box to remove and iterates until no valid moves remain.\
<img src="https://github.com/user-attachments/assets/070c9ba5-3ab6-493c-9729-03ccec5f9471" width=400>
<img src="https://github.com/user-attachments/assets/d6a30915-2f8f-424b-98f9-6433c48bca42" width=400>

## Features
- **Automated Screen Capture**: Uses `pyautogui` to take a screenshot of the game.
- **Image Processing**: Utilizes `OpenCV` to detect the game board and recognize numbers.
- **Optimal Move Selection**: Implements a prefix sum approach to find the best strategy that gets highest score.
- **Automated Mouse Control**: Uses `pyautogui` to select and drag boxes around apples.

## Installation

Ensure you have Python installed, then install the required dependencies:

```sh
pip install opencv-python numpy pyautogui
```

## Usage

1. Start the game in your browser.
2. Run the script:

```sh
python main.py
```

3. The program will detect the game board, analyze the grid, and automatically play the game.
4. Once no valid moves are available, it will stop.

## How It Works

1. **Detect the Game Board**: Identifies the green background and crops the relevant area.
2. **Extract Numbers**: Uses template matching to recognize numbers on the grid.
3. **Find Optimal Moves**: Computes all valid boxes where the sum is 10. Simulate three strategies and pick the best one: 1. Remove the least apples in one move. 2. Remove the least apples in one move that contains large numbers (i.e. 8 or 9). 3. Remove the least apples in one move that contains small numbers (i.e. 1 or 2)
4. **Execute Moves**: Simulates mouse drag actions to select and remove the optimal box.
5. **Repeat Until No Moves Left**: Iterates the process until no more valid moves exist.

## Future Improvements
- Optimize move selection to consider alternative strategies.
- Enhance efficiency of grid processing and updating.