import pygame
import numpy as np
import heapq


def a_star(start, goal, obstacles, grid_size=(2000, 1200), cell_size=50):
    # Convert start and goal to grid coordinates
    start = tuple(np.array(start) // cell_size)
    goal = tuple(np.array(goal) // cell_size)

    # Define grid dimensions
    rows, cols = grid_size[1] // cell_size, grid_size[0] // cell_size

    # Heuristic function (Euclidean distance)
    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    # Priority queue for open set (min-heap)
    open_set = []
    heapq.heappush(open_set, (0, start))

    # Dictionaries to track the best score and the path
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    # Add vertical fence obstacles to the set
    vertical_fence = {(1000 // cell_size, y) for y in range(0 // cell_size, 950 // cell_size)}
    obstacles.update(vertical_fence)  # Add scaled fence coordinates to obstacles set

    while open_set:
        _, current = heapq.heappop(open_set)

        # Check if goal is reached
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return [(p[0] * cell_size, p[1] * cell_size) for p in path]

        # Explore neighbors
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            
            # Check if neighbor is within bounds
            if 0 <= neighbor[0] < cols and 0 <= neighbor[1] < rows:
                # Skip obstacles
                if neighbor in obstacles:
                    continue

                # Calculate tentative g-score
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # Update path and scores
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []  # No path found




# Visualize A* Path
def draw_path(screen, path, color=(0, 255, 0)):
    for i in range(len(path) - 1):
        pygame.draw.line(screen, color, path[i], path[i + 1], 5)
    #pygame.draw.circle(screen, color, path[1], 10)

def smooth_path(path, smoothing_iterations=2):
    """
    Smooths a path by averaging neighboring points.
    
    Args:
        path (list): A list of (x, y) tuples representing the path.
        smoothing_iterations (int): Number of times to apply smoothing.
    
    Returns:
        list: A smoothed path.
    """
    if not path or len(path) <= 2:
        return path  # No smoothing needed for short paths

    smoothed_path = path.copy()
    for _ in range(smoothing_iterations):
        new_path = [smoothed_path[0]]  # Keep the first point
        for i in range(1, len(smoothed_path) - 1):
            # Average the current point with its neighbors
            prev_point = smoothed_path[i - 1]
            curr_point = smoothed_path[i]
            next_point = smoothed_path[i + 1]
            new_x = (prev_point[0] + curr_point[0] + next_point[0]) / 3
            new_y = (prev_point[1] + curr_point[1] + next_point[1]) / 3
            new_path.append((new_x, new_y))
        new_path.append(smoothed_path[-1])  # Keep the last point
        smoothed_path = new_path

    return smoothed_path

