import numpy as np
import itertools

#This approach removes the largest group of apple that sums 10.
scores = []

#Testing for score stats
for i in range (100):
    score = 0

    # Sample 17x10 grid (Replace this with the extracted number grid)
    grid = np.random.randint(1, 10, (10, 17))
    #print(grid)

    while True:
        # Compute prefix sum matrix
        prefix_sum = np.zeros((11, 18), dtype=int)  # One extra row and column for boundary conditions

        for r in range(10):
            for c in range(17):
                prefix_sum[r + 1, c + 1] = (
                    grid[r, c]
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
                            apples_removed = np.count_nonzero(grid[r1:r2+1, c1:c2+1])
                            valid_boxes.append(((r1, c1, r2, c2), apples_removed))

        # Sort by max apples removed
        valid_boxes.sort(key=lambda x: -x[1])

        # Execute best move
        if valid_boxes:
            best_move = valid_boxes[0]
            r1, c1, r2, c2 = best_move[0]
            grid[r1:r2+1, c1:c2+1] = 0
            score += best_move[1]
        else:
            #print("No valid moves found")
            #print(grid)
            #print(f"score: {score}")
            scores.append(score)
            break
scores.sort()
print(f"average score: {sum(scores)/len(scores)}")
print(f"median score: {scores[round(len(scores)/2)]}")
print(f"lowest score:{scores[0]}")
print(f"highest score:{scores[len(scores)-1]}")

'''
100 games
case 1:
average score: 99.16
median score: 100
lowest score:72
highest score:131

case2:
average score: 96.3
median score: 96
lowest score:62
highest score:125
'''