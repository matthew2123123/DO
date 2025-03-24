import math
import random
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ================================
# A* and Environment Functions
# ================================

class Obstacle3D:
    def __init__(self, base_polygon: Polygon, height: float):
        self.base = base_polygon
        self.height = height

def get_levels(obstacles, k=4):
    """
    Divide the altitude (0 to max obstacle height) into k evenly spaced levels.
    """
    max_h = max(obs.height for obs in obstacles) if obstacles else 50
    levels = [i/(k-1)*max_h for i in range(k)]
    return levels

def euclidean_distance_3d(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

def segment_collides(p1, p2, obstacles):
    """
    Simplified collision check: For each obstacle, if the average z of the segment is lower than
    the obstacle height and the 2D projection crosses the obstacle base, it's considered a collision.
    """
    avg_z = (p1[2] + p2[2]) / 2
    line = LineString([(p1[0], p1[1]), (p2[0], p2[1])])
    for obs in obstacles:
        if avg_z < obs.height and line.crosses(obs.base):
            return True
    return False

def visualize_3d(obstacles, path):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    # Visualize obstacles
    for obs in obstacles:
        x_obs, y_obs = obs.base.exterior.xy
        ax.plot(x_obs, y_obs, zs=0, color='gray', alpha=0.7)
        for i in range(len(x_obs)-1):
            ax.plot([x_obs[i], x_obs[i]], [y_obs[i], y_obs[i]], [0, obs.height], color='gray', alpha=0.5)
        verts = [list(zip(x_obs, y_obs, [obs.height]*len(x_obs)))]
        poly = Poly3DCollection(verts, facecolors='lightgray', edgecolors='gray', alpha=0.5)
        ax.add_collection3d(poly)
    # Visualize path (if exists)
    if path is not None and len(path) > 0:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        zs = [p[2] for p in path]
        ax.plot(xs, ys, zs, 'r-o', linewidth=3, markersize=8, label='PSO Path')
        ax.scatter(xs[0], ys[0], zs[0], color='green', s=100, label='Start')
        ax.scatter(xs[-1], ys[-1], zs[-1], color='blue', s=100, label='Goal')
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(0, max([obs.height for obs in obstacles])+50)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title("3D Shortest Path via PSO")
    plt.show()

def generate_random_buildings(num, x_range, y_range, min_size, max_size, max_height):
    buildings = []
    attempts = 0
    while len(buildings) < num and attempts < num * 50:
        attempts += 1
        w_b = random.uniform(min_size, max_size)
        h_b = random.uniform(min_size, max_size)
        x = random.uniform(x_range[0], x_range[1] - w_b)
        y = random.uniform(y_range[0], y_range[1] - h_b)
        poly = Polygon([(x, y), (x+w_b, y), (x+w_b, y+h_b), (x, y+h_b)])
        overlap = False
        for b in buildings:
            if poly.intersects(b.base):
                overlap = True
                break
        if not overlap:
            b_height = random.uniform(100, max_height)
            buildings.append(Obstacle3D(poly, b_height))
    return buildings

# ================================
# PSO for 3D Shortest Path: Particle and Swarm Implementation
# ================================

class Particle:
    def __init__(self, n_waypoints, bounds):
        """
        n_waypoints: Number of intermediate points (not including fixed start/goal).
        bounds: List of three tuples [(xmin, xmax), (ymin, ymax), (zmin, zmax)].
        """
        self.n_waypoints = n_waypoints
        self.position = np.array([[random.uniform(bounds[d][0], bounds[d][1]) for d in range(3)]
                                  for _ in range(n_waypoints)])
        self.velocity = np.array([[random.uniform(-1, 1) for _ in range(3)]
                                  for _ in range(n_waypoints)])
        self.best_position = np.copy(self.position)
        self.best_fitness = float('inf')

    def update_velocity(self, global_best, inertia, c1, c2):
        r1 = np.random.rand(self.n_waypoints, 3)
        r2 = np.random.rand(self.n_waypoints, 3)
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best - self.position)
        self.velocity = inertia * self.velocity + cognitive + social

    def update_position(self, bounds):
        self.position = self.position + self.velocity
        # Enforce bounds for each waypoint and coordinate
        for i in range(self.n_waypoints):
            for d in range(3):
                if self.position[i, d] < bounds[d][0]:
                    self.position[i, d] = bounds[d][0]
                elif self.position[i, d] > bounds[d][1]:
                    self.position[i, d] = bounds[d][1]

    def evaluate(self, start, goal, obstacles, base_penalty):
        """
        Build the full path and compute fitness using an adaptive collision penalty.
        """
        path = [start] + [tuple(self.position[i]) for i in range(self.n_waypoints)] + [goal]
        total_cost = adaptive_collision_penalty(path, obstacles, base_penalty)
        if total_cost < self.best_fitness:
            self.best_fitness = total_cost
            self.best_position = np.copy(self.position)
        return total_cost, path

def pso_3d_path(obstacles, start, goal, bounds, n_waypoints=5, n_particles=30,
                n_iterations=100, c1=1.5, c2=1.5, base_penalty=1e5):
    """
    PSO for 3D path planning with adaptive collision penalty, local refinement,
    and adaptive inertia.
      - obstacles: list of Obstacle3D instances.
      - start, goal: fixed (x, y, z) coordinates.
      - bounds: bounds for intermediate waypoints.
      - n_waypoints: number of intermediate waypoints.
      - n_particles: swarm size.
      - n_iterations: number of iterations.
      - c1, c2: acceleration coefficients.
      - base_penalty: base penalty for each colliding segment.
    """
    # Adaptive inertia parameters (linearly decreasing)
    w_start = 500.0
    w_end = 200.0

    swarm = [Particle(n_waypoints, bounds) for _ in range(n_particles)]
    global_best_position = None
    global_best_fitness = float('inf')
    global_best_path = None
    fitness_history = []

    for it in range(n_iterations):
        # Compute current inertia weight
        inertia = w_end + (w_start - w_end) * (n_iterations - it) / n_iterations

        for particle in swarm:
            fitness, path = particle.evaluate(start, goal, obstacles, base_penalty)
            # Apply local refinement to try to repair collisions
            refined_position, refined_cost = local_refine(particle.position, start, goal,
                                                          obstacles, base_penalty, n_waypoints)
            if refined_cost < fitness:
                particle.position = np.copy(refined_position)
                particle.best_position = np.copy(refined_position)
                particle.best_fitness = refined_cost
                fitness = refined_cost
                path = [start] + [tuple(particle.position[i]) for i in range(n_waypoints)] + [goal]

            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = np.copy(particle.position)
                global_best_path = path

        for particle in swarm:
            particle.update_velocity(global_best_position, inertia, c1, c2)
            particle.update_position(bounds)
        fitness_history.append(global_best_fitness)
        if it % 10 == 0:
            print(f"Iteration {it}: Global Best Fitness = {global_best_fitness:.2f}")

    return global_best_path, global_best_fitness, fitness_history


# ================================
# Adaptive Collision Penalty Function
# ================================

def adaptive_collision_penalty(path, obstacles, base_penalty=1e5):
    """
    Compute total cost as sum of segment lengths plus a penalty that scales with the
    number of colliding segments.
    """
    total_cost = 0.0
    collision_count = 0
    for i in range(len(path)-1):
        seg_length = euclidean_distance_3d(path[i], path[i+1])
        total_cost += seg_length
        if segment_collides(path[i], path[i+1], obstacles):
            collision_count += 1
    penalty_cost = collision_count * base_penalty
    return total_cost + penalty_cost

# ================================
# Local Refinement Function
# ================================

def local_refine(position, start, goal, obstacles, base_penalty, n_waypoints, step_size=5.0, attempts=5):
    """
    Attempt to locally repair a candidate solution by applying small random shifts to
    the waypoints involved in colliding segments.
    """
    best_position = np.copy(position)
    path = [start] + [tuple(best_position[i]) for i in range(n_waypoints)] + [goal]
    best_cost = adaptive_collision_penalty(path, obstacles, base_penalty)
    
    for _ in range(attempts):
        new_position = np.copy(best_position)
        # Identify indices where the segment collides
        colliding_indices = []
        path = [start] + [tuple(new_position[i]) for i in range(n_waypoints)] + [goal]
        for i in range(len(path)-1):
            if segment_collides(path[i], path[i+1], obstacles):
                colliding_indices.append(i)
        if not colliding_indices:
            # No collisions found: solution is feasible
            break
        # Randomly select one colliding segment to adjust
        seg_idx = random.choice(colliding_indices)
        # Adjust the corresponding intermediate waypoint (if possible)
        if seg_idx > 0 and seg_idx < n_waypoints+1:
            waypoint_idx = seg_idx - 1  # path[1] is the first waypoint
            for d in range(3):
                delta = random.uniform(-step_size, step_size)
                new_position[waypoint_idx, d] += delta
        new_path = [start] + [tuple(new_position[i]) for i in range(n_waypoints)] + [goal]
        new_cost = adaptive_collision_penalty(new_path, obstacles, base_penalty)
        if new_cost < best_cost:
            best_cost = new_cost
            best_position = np.copy(new_position)
    return best_position, best_cost

# ================================
# Main Execution: PSO vs. A* for 3D Shortest Path
# ================================
if __name__ == "__main__":
    # Define environment ranges
    x_range = (0, 1000)
    y_range = (0, 1000)

    obstacles = generate_random_buildings(num=15, x_range=x_range, y_range=y_range,
                                           min_size=100, max_size=200, max_height=500)
    if not obstacles:
        obstacles.append(Obstacle3D(Polygon([(100,100), (150,100), (150,150), (100,150)]), height=50))

    # Create altitude levels based on obstacles' max height
    levels = get_levels(obstacles, k=10)
    print("고도 레벨:", levels)

    start = (random.uniform(x_range[0]+10, x_range[0]+50),
             random.uniform(y_range[0]+10, y_range[0]+50),
             levels[0])
    goal = (random.uniform(x_range[1]-50, x_range[1]-10),
            random.uniform(y_range[1]-50, y_range[1]-10),
            levels[-2])
    print("Start:", start)
    print("Goal:", goal)

    # Define bounds for intermediate waypoints:
    bounds = [x_range, y_range, (levels[0], levels[-1])]

    # Run PSO for 3D path planning.
    best_path, best_fitness, fitness_history = pso_3d_path(
        obstacles, start, goal, bounds,
        n_waypoints=10,         # Increase for more flexibility if needed
        n_particles=100,
        n_iterations=100,
        c1=1000.0, c2=10.0,
        base_penalty=5e4
    )
    print("\nBest Path found by PSO:", best_path)
    print("Best Fitness (Path Cost):", best_fitness)

    visualize_3d(obstacles, best_path)

    # Optionally, plot PSO convergence curve
    plt.figure(figsize=(8,6))
    plt.plot(fitness_history, marker='o', linestyle='-')
    plt.xlabel("Iteration")
    plt.ylabel("Global Best Fitness")
    plt.title("PSO Convergence")
    plt.grid(True)
    plt.show()
