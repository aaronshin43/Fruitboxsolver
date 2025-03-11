import numpy as np

def compute_prefix_sum(grid):
    """Compute prefix sum matrix for efficient region sum calculation."""
    prefix_sum = np.zeros((11, 18), dtype=int)  # One extra row/column for boundary conditions

    for r in range(10):
        for c in range(17):
            prefix_sum[r + 1, c + 1] = (
                grid[r, c] + prefix_sum[r, c + 1] + prefix_sum[r + 1, c] - prefix_sum[r, c]
            )
    return prefix_sum

def get_valid_moves(grid, prefix_sum):
    """Find all valid boxes that sum to 10 and return them with apples removed."""
    valid_moves = []

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
                        apples_removed = np.count_nonzero(grid[r1:r2+1, c1:c2+1])
                        valid_moves.append(((r1, c1, r2, c2), apples_removed))
    
    # Sort moves by the number of apples removed (highest first)
    valid_moves.sort(key=lambda x: x[1])
    return valid_moves

def max_removal_strategy(grid):
    """Strategy: Remove the most apples in one move."""
    prefix_sum = compute_prefix_sum(grid)
    valid_moves = get_valid_moves(grid, prefix_sum)
    if valid_moves:
        return max(valid_moves, key=lambda x: x[1])[0]  # Move that removes most apples
    return None

def min_removal_strategy(grid):
    """Strategy: Remove the least apples in one move."""
    prefix_sum = compute_prefix_sum(grid)
    valid_moves = get_valid_moves(grid, prefix_sum)
    if valid_moves:
        return min(valid_moves, key=lambda x: x[1])[0]  # Move that removes least apples
    return None

def look_ahead(grid, depth):
    """Simulate multiple moves ahead and pick the best initial move."""
    if depth == 0:
        return 0, None  # Base case: return score 0 if no depth left

    prefix_sum = compute_prefix_sum(grid)
    valid_moves = get_valid_moves(grid, prefix_sum)

    if not valid_moves:
        return 0, None  # No moves available

    best_score = float('-inf')
    best_move = None

    for move, apples_removed in valid_moves:
        new_grid = grid.copy()
        r1, c1, r2, c2 = move
        new_grid[r1:r2+1, c1:c2+1] = 0  # Simulate move

        future_score, _ = look_ahead(new_grid, depth - 1)  # Look deeper
        total_score = apples_removed + future_score  # Score = now + future

        if total_score > best_score:
            best_score = total_score
            best_move = move

    return best_score, best_move

# Main game loop using look-ahead strategy

def look_ahead_strategy(grid, depth=2):
    """Wrapper function to return just the best move."""
    _, best_move = look_ahead(grid, depth)
    return best_move

def max_num(grid):
    """Find the best move by prioritizing the removal of low numbers."""
    # Compute prefix sum matrix for fast sum calculations
    prefix_sum = compute_prefix_sum(grid)

    valid_moves = []

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
                        apples = grid[r1:r2+1, c1:c2+1].flatten()
                        apples = apples[apples > 0]  # Ignore zeros

                        # Compute the maximum value
                        max_num = max(apples)

                        apples_removed = len(apples)
                        valid_moves.append(((r1, c1, r2, c2), max_num, apples_removed))
    return valid_moves

def max_value_strategy(grid):
    # Sort by min apples removed, then highest number 
    valid_moves = max_num(grid)
    valid_moves.sort(key=lambda x: (x[2], x[1]))

    if valid_moves:
        return valid_moves[0][0]  # Best move (r1, c1, r2, c2)

    return None  # No valid moves found

def min_value_strategy(grid):
    valid_moves = max_num(grid)
    # Sort by min apples removed, then smallest number 
    valid_moves.sort(key=lambda x: (x[2], -x[1]))

    if valid_moves:
        return valid_moves[0][0]  # Best move (r1, c1, r2, c2)

    return None  # No valid moves found

'''
100 games
case 1:
average score: 99.11
median score: 100
lowest score:74
highest score:124
'''