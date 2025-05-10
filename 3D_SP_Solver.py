import math
import random
import os
import numpy as np
from shapely.geometry import Polygon, LineString, Point
from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# ---------- 환경 정의 ----------
class Obstacle3D:
    def __init__(self, base_polygon: Polygon, height: float):
        self.base = base_polygon
        self.height = height

def segment_collides(p1, p2, obstacles):
    """
    두 지점 사이 선분이 장애물과 충돌하는지 검사
    """
    avg_z = (p1[2] + p2[2]) / 2
    line = LineString([(p1[0], p1[1]), (p2[0], p2[1])])
    for obs in obstacles:
        if avg_z < obs.height and line.crosses(obs.base):
            return True
    return False

# ---------- 그래프 생성 ----------
def generate_random_buildings(num, x_range, y_range, min_size, max_size, max_height, start=(50, 50), safe_radius=100):
    buildings = []
    attempts = 0
    sx, sy = start[:2]

    while len(buildings) < num and attempts < num * 50:
        attempts += 1
        w = random.uniform(min_size, max_size)
        h = random.uniform(min_size, max_size)
        x = random.uniform(x_range[0], x_range[1] - w)
        y = random.uniform(y_range[0], y_range[1] - h)

        poly = Polygon([(x, y), (x+w, y), (x+w, y+h), (x, y+h)])

        # 장애물과의 충돌 검사
        center_x, center_y = x + w/2, y + h/2
        if math.hypot(center_x - sx, center_y - sy) < safe_radius:
            continue

        if not any(poly.intersects(b.base) for b in buildings):
            b_height = random.uniform(100, max_height)
            buildings.append(Obstacle3D(poly, b_height))

    return buildings

def sample_free_points(obstacles, start, goal, num_samples, x_range, y_range, z_range):
    """Free-space 랜덤 포인트 샘플링"""
    points = [tuple(start), tuple(goal)]
    while len(points) < num_samples + 2:
        pt = (
            random.uniform(*x_range),
            random.uniform(*y_range),
            random.uniform(*z_range)
        )
        # 장애물 내부(볼륨) 제외
        inside = False
        for obs in obstacles:
            if pt[2] < obs.height and Point(pt[0], pt[1]).within(obs.base):
                inside = True
                break
        if not inside:
            points.append(pt)
    return points


def build_graph(nodes, obstacles, neighbor_radius):
    """인접 노드 간 엣지 생성 및 가중치(거리) 계산"""
    edges = []
    n = len(nodes)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            p1 = nodes[i]
            p2 = nodes[j]
            # 거리 기반 이웃 판단
            dist = math.dist(p1, p2)
            if dist <= neighbor_radius and not segment_collides(p1, p2, obstacles):
                edges.append((i, j, dist))
    return edges

# ---------- 솔버 모델링 ----------
def solve_shortest_path(nodes, edges, start_idx=0, goal_idx=1):
    """MIP를 활용한 최단 경로 문제 풀기"""
    solver = pywraplp.Solver.CreateSolver('CBC')
    x = {}
    # 변수 생성: 각 엣지를 선택할지 이진 변수
    for i, j, cost in edges:
        x[(i, j)] = solver.BoolVar(f'x_{i}_{j}')
    # 유량 보존 제약
    for k in range(len(nodes)):
        inflow = []
        outflow = []
        for i, j, _ in edges:
            if j == k: inflow.append(x[(i, j)])
            if i == k: outflow.append(x[(i, j)])
        if k == start_idx:
            solver.Add(sum(outflow) - sum(inflow) == 1)
        elif k == goal_idx:
            solver.Add(sum(outflow) - sum(inflow) == -1)
        else:
            solver.Add(sum(outflow) - sum(inflow) == 0)
    # 목적 함수: 거리 합 최소화
    objective = solver.Objective()
    for i, j, cost in edges:
        objective.SetCoefficient(x[(i, j)], cost)
    objective.SetMinimization()

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        print("최적해를 찾지 못했습니다.")
        return []
    # 경로 복원
    path = [start_idx]
    current = start_idx
    visited = set()
    while current != goal_idx:
        visited.add(current)
        for i, j, _ in edges:
            if i == current and x[(i, j)].solution_value() > 0.5:
                path.append(j)
                current = j
                break
        else:
            print("경로 복원 실패")
            return []
    return [nodes[i] for i in path]

# ---------- 시각화 및 거리 계산 ----------
def visualize_path(obstacles, path, start, goal):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for obs in obstacles:
        x, y = obs.base.exterior.xy
        z = np.zeros(len(x))
        ax.plot(x, y, z, color='gray')
        ax.plot(x, y, [obs.height]*len(x), color='gray')
        for i in range(len(x)):
            ax.plot([x[i], x[i]], [y[i], y[i]], [0, obs.height], color='gray', alpha=0.3)
    if path:
        xs, ys, zs = zip(*path)
        ax.plot(xs, ys, zs, color='red', marker='o', label='Path')
        ax.scatter(start[0], start[1], start[2], color='green', s=100, label='Start')
        ax.scatter(goal[0], goal[1], goal[2], color='blue', s=100, label='Goal')
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_zlim(0, 600)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title("3D Path Planning with Solver")
    plt.show()


def compute_total_distance(path):
    total = 0.0
    for i in range(1, len(path)):
        total += math.dist(path[i-1], path[i])
    return total

# ---------- 실행 ----------
if __name__ == "__main__":
    random.seed(42)
    x_range = (0, 1000)
    y_range = (0, 1000)
    z_range = (0, 500)
    obstacles = generate_random_buildings(15, x_range, y_range, 100, 200, 500)
    start = (50.0, 50.0, 0.0)
    goal = (950.0, 950.0, 500.0)

    # 노드 샘플링 및 그래프 구축
    nodes = sample_free_points(obstacles, start, goal, num_samples=1000, x_range=x_range, y_range=y_range, z_range=z_range)
    edges = build_graph(nodes, obstacles, neighbor_radius=1000)

    # 최단 경로 해결
    start_time = time.time()
    solver_path = solve_shortest_path(nodes, edges, start_idx=0, goal_idx=1)
    end_time = time.time()
    print(f"총 노드 수: {len(nodes)}")
    print(f"총 엣지 수: {len(edges)}")

    if not solver_path:
        exit(1)

    # 결과 출력
    print(f"Solver 경로 노드 수: {len(solver_path)}")
    print(f"총 이동 거리: {compute_total_distance(solver_path):.2f}")
    print(f"Solver 실행 시간: {end_time - start_time:.2f}초")
    visualize_path(obstacles, solver_path, start, goal)