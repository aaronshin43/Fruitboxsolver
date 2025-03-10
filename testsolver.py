import numpy as np
import solver
import copy

# Initialize game board (Example: random numbers 1-9)


def simulate_strategy(grid, strategy_func, depth=None):
    """Simulate the given strategy and return the final score."""
    grid_copy = copy.deepcopy(grid)  # Make a copy so original board isn't affected
    score = 0

    while True:
        if strategy_func == solver.look_ahead_strategy:
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
    score_max = simulate_strategy(grid, solver.max_removal_strategy)
    score_min = simulate_strategy(grid, solver.min_removal_strategy)
    score_look_ahead = simulate_strategy(grid, solver.look_ahead_strategy, depth=2)

    # Pick the strategy with the highest final score
    strategy_scores = {
        "max_removal": score_max,
        "min_removal": score_min,
        "look_ahead": score_look_ahead
    }
    print(strategy_scores)
    best_strategy = max(strategy_scores, key=strategy_scores.get)
    #print(f"Chosen Strategy: {best_strategy}, Expected Score: {strategy_scores[best_strategy]}")

    return best_strategy

# Main game loop
#score = 0

scores = [0,0,0]
for i in range(20):
    grid = np.random.randint(1, 10, (10, 17))
    best_strategy = choose_best_strategy(grid)
    if best_strategy == "look_ahead":
        scores[2] += 1
    elif best_strategy == "max_removal":
        scores[0] += 1
    else:
        scores[1] += 1

print(f"max: {scores[0]}, min: {scores[1]}, la: {scores[2]}")

#print(f"Final Score: {score}")
