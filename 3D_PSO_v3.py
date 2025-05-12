import math
import random
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon, Point
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import concurrent.futures

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

def compute_collision_point(p1, p2, obstacles):
    """
    Returns (collision_point, obs) where collision_point is the first 3D entry
    into any obstacle prism, or (None, None) if no collision.
    """
    line2d    = LineString([(p1[0], p1[1]), (p2[0], p2[1])])
    total_len = line2d.length
    dz        = p2[2] - p1[2]

    best_t, best_obs, best_pt = None, None, None

    for obs in obstacles:
        # ignore if both endpoints are above the roof
        if min(p1[2], p2[2]) >= obs.height:
            continue

        # intersect with the obstacle's *boundary* for clean point results
        intr = obs.base.exterior.intersection(line2d)
        if intr.is_empty:
            continue

        # gather all entry‐points
        pts = []
        if intr.geom_type == "Point":
            pts = [intr]
        else:
            try:
                pts = [g for g in intr.geoms if g.geom_type == "Point"]
            except AttributeError:
                # e.g. a LineString overlap: take its endpoints
                pts = [Point(c) for c in intr.coords]

        for pt in pts:
            t = line2d.project(pt) / total_len
            if 0 <= t <= 1 and (best_t is None or t < best_t):
                best_t, best_obs, best_pt = t, obs, pt

    if best_t is None:
        return None, None

    # reconstruct the 3D collision point
    z_coll = p1[2] + dz * best_t
    return (best_pt.x, best_pt.y, z_coll), best_obs


def segment_collides(p1, p2, obstacles, eps=1e-9):
    """
    Returns True iff the segment actually clips under an obstacle roof.
    """
    coll_pt, obs = compute_collision_point(p1, p2, obstacles)
    if coll_pt is None:
        return False
    # strict under‐roof check
    return (coll_pt[2] < obs.height - eps)

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

from concurrent.futures import ThreadPoolExecutor, as_completed

def pso_3d_path(obstacles, start, goal, bounds,
                n_waypoints=5, n_particles=30, n_iterations=100,
                c1=2.05, c2=2.05, depth_penalty=100.0):
    """
    PSO with:
        - depth‐based penalty (no hard collisions),
        - directed local repair,
        - parallel particle evaluation,
        - standard inertia and c1,c2.
    """
    # inertia schedule: 0.9 → 0.4
    w_start, w_end = 0.9, 0.3

    # initialize swarm
    swarm = [Particle(n_waypoints, bounds) for _ in range(n_particles)]
    global_best_pos    = None
    global_best_fitness = float('inf')
    global_best_path    = None
    fitness_history     = []

    def evaluate_and_repair(idx, particle):
        # 1) baseline fitness
        fitness, path = particle.evaluate(start, goal, obstacles, depth_penalty)
        # 2) directed local repair
        new_pos, new_cost = local_refine(
            particle.position, start, goal,
            obstacles, depth_penalty, bounds
        )
        if new_cost < fitness:
            particle.position = new_pos
            particle.best_position = new_pos.copy()
            particle.best_fitness = new_cost
            fitness, path = new_cost, [start] + [tuple(new_pos[i]) for i in range(n_waypoints)] + [goal]
        return idx, particle.position, particle.best_position, particle.best_fitness, fitness, path

    for it in range(n_iterations):
        inertia = w_end + (w_start - w_end)*(n_iterations - it)/n_iterations

        # 5) Parallel evaluate + repair
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(evaluate_and_repair, i, swarm[i]): i
                       for i in range(n_particles)}
            for fut in as_completed(futures):
                i, pos, best_pos, best_fit, fit, path = fut.result()
                swarm[i].position       = pos
                swarm[i].best_position  = best_pos
                swarm[i].best_fitness   = best_fit
                if fit < global_best_fitness:
                    global_best_fitness = fit
                    global_best_pos     = best_pos.copy()
                    global_best_path    = path

        # velocity & position updates
        for particle in swarm:
            particle.update_velocity(global_best_pos, inertia, c1, c2)
            particle.update_position(bounds)

        fitness_history.append(global_best_fitness)
        if it % 10 == 0:
            print(f"Iter {it:3d}: Best fitness = {global_best_fitness:.3f}")

    return global_best_path, global_best_fitness, fitness_history

# ================================
# Adaptive Collision Penalty Function
# ================================

def adaptive_collision_penalty(path, obstacles, depth_penalty=100.0, eps=1e-9):
    """
    Sum of 3D‐Euclidean lengths plus a penalty proportional to
    how far under the roof each truly colliding segment goes.
    """
    total_cost = 0.0
    for i in range(len(path)-1):
        p1, p2 = path[i], path[i+1]
        # always add segment length
        total_cost += euclidean_distance_3d(p1, p2)

        # get the precise intersection (if any)
        coll_pt, obs = compute_collision_point(p1, p2, obstacles)
        # only penalize if it actually clips under the roof
        if coll_pt is not None and (coll_pt[2] < obs.height - eps):
            depth = obs.height - coll_pt[2]    # positive amount under the roof
            total_cost += depth_penalty * depth

    return total_cost

# ================================
# Local Refinement Function
# ================================

def local_refine(position, start, goal, obstacles,
                 depth_penalty, bounds, step_size=10.0, attempts=5):
    """
    For each colliding segment, push its waypoint directly away from the
    obstacle's centroid by `step_size` and accept if it reduces the cost.
    """
    n_waypoints = position.shape[0]
    best_pos   = position.copy()
    path       = [start] + [tuple(best_pos[i]) for i in range(n_waypoints)] + [goal]
    best_cost  = adaptive_collision_penalty(path, obstacles, depth_penalty)

    for _ in range(attempts):
        # find colliding segments
        coll_idxs = []
        path = [start] + [tuple(best_pos[i]) for i in range(n_waypoints)] + [goal]
        for i in range(len(path)-1):
            if segment_collides(path[i], path[i+1], obstacles):
                coll_idxs.append(i)

        if not coll_idxs:
            break

        # pick one to repair
        seg_i = random.choice(coll_idxs)
        # waypoint index in `best_pos`
        wp_i  = seg_i - 1
        if wp_i < 0 or wp_i >= n_waypoints:
            continue

        # compute collision details
        p1 = path[seg_i]
        p2 = path[seg_i+1]
        coll_pt, obs = compute_collision_point(p1, p2, obstacles)
        if coll_pt is None:
            continue

        # direction away from obstacle centroid (in XY plane)
        cx, cy = obs.base.centroid.x, obs.base.centroid.y
        wx, wy, wz = best_pos[wp_i]
        dx, dy = wx - cx, wy - cy
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            # random direction if too close to centroid
            theta = random.random()*2*math.pi
            dx, dy = math.cos(theta), math.sin(theta)
        else:
            dx /= norm; dy /= norm

        # propose new waypoint
        new_pos = best_pos.copy()
        new_pos[wp_i, 0] = min(max(wx + dx*step_size, bounds[0][0]), bounds[0][1])
        new_pos[wp_i, 1] = min(max(wy + dy*step_size, bounds[1][0]), bounds[1][1])
        # push z at least `step_size` above the roof
        new_z = max(new_pos[wp_i,2], obs.height + step_size)
        new_pos[wp_i, 2] = min(max(new_z, bounds[2][0]), bounds[2][1])

        # evaluate
        new_path = [start] + [tuple(new_pos[i]) for i in range(n_waypoints)] + [goal]
        new_cost = adaptive_collision_penalty(new_path, obstacles, depth_penalty)
        if new_cost < best_cost:
            best_cost  = new_cost
            best_pos   = new_pos

    return best_pos, best_cost

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
        n_waypoints=5, n_particles=30, n_iterations=100,
        c1=2.05, c2=2.05, depth_penalty=100.0
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
