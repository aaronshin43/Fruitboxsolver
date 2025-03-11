import numpy as np
from package.solver import *
import copy

# Run this code on terminal

def simulate_strategy(grid, strategy_func, depth=None):
    """Simulate the given strategy and return the final score."""
    grid_copy = copy.deepcopy(grid)  # Make a copy so original board isn't affected
    score = 0

    while True:
        if strategy_func == look_ahead_strategy:
            best_move = strategy_func(grid_copy, depth)
        else:
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
    score_max = simulate_strategy(grid, max_removal_strategy)
    score_min = simulate_strategy(grid, min_removal_strategy)
    score_look_ahead = simulate_strategy(grid, look_ahead_strategy, depth=2)
    score_high_val = simulate_strategy(grid, max_value_strategy)
    score_low_val = simulate_strategy(grid, min_value_strategy)
    # Pick the strategy with the highest final score
    strategy_scores = {
        "max_removal": score_max,
        "min_removal": score_min,
        "look_ahead": score_look_ahead,
        "high_val": score_high_val,
        "low_val": score_low_val
    }
    print(strategy_scores)
    best_strategy = max(strategy_scores, key=strategy_scores.get)
    return best_strategy, strategy_scores

# Main game loop

scores_dict = {
    "max_removal": [],
    "min_removal": [],
    "look_ahead": [],
    "high_val": [],
    "low_val": []
}
#run 500 times to test which strategy to use
scores = [0,0,0,0,0]
for i in range(500):
    grid = np.random.randint(1, 10, (10, 17))
    best_strategy, strategy_scores = choose_best_strategy(grid)
    if best_strategy == "max_removal":
        scores[0] += 1
    elif best_strategy == "min_removal":
        scores[1] += 1
    elif best_strategy == "look_ahead":
        scores[2] += 1
    elif best_strategy == "high_val":
        scores[3] += 1
    else:
        scores[4] += 1
    
    scores_dict["max_removal"].append(strategy_scores.get("max_removal"))
    scores_dict["min_removal"].append(strategy_scores.get("min_removal"))
    scores_dict["look_ahead"].append(strategy_scores.get("look_ahead"))
    scores_dict["high_val"].append(strategy_scores.get("high_val"))
    scores_dict["low_val"].append(strategy_scores.get("low_val"))

strategy_stats = {}
for strategy, j in scores_dict.items():
    strategy_stats[strategy] = {
        "min": np.min(j),
        "max": np.max(j),
        "avg": np.mean(j)
    }
print(f"max: {scores[0]}, min: {scores[1]}, la: {scores[2]}, high: {scores[3]}, low: {scores[4]}")
for strategy, stats in strategy_stats.items():
    print(f"{strategy}: Min={stats['min']}, Max={stats['max']}, Avg={stats['avg']:.2f}")

# Test result:
# max: 7, min: 200, la: 9, high: 168, low: 116
# max_removal: Min=65, Max=126, Avg=97.44, winrate=1.4%
# min_removal: Min=69, Max=155, Avg=114.44, winrate=40%
# look_ahead: Min=68, Max=131, Avg=99.85, winrate=1.8%
# high_val: Min=69, Max=158, Avg=114.55, winrate=33.6%
# low_val: Min=67, Max=161, Avg=113.83, winrate=23.2%