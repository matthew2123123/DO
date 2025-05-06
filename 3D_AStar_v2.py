import math, time
import random
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point
from mpl_toolkits.mplot3d import Axes3D  # 3D 플롯용
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# 3D 장애물(건물): 바닥면 다각형과 높이로 표현
class Obstacle3D:
    def __init__(self, base_polygon: Polygon, height: float):
        self.base = base_polygon   # 2D 다각형 (shapely Polygon)
        self.height = height       # 건물 높이

def get_levels(obstacles, k=4):
    """
    전체 장애물 중 최대 높이를 기준으로 k개의 고도 레벨(0부터 max까지 균등 분할)을 생성합니다.
    """
    max_h = max(obs.height for obs in obstacles) if obstacles else 50
    levels = [i/(k-1)*max_h for i in range(k)]
    return levels

def is_visible_at_level(p1, p2, effective_obs):
    """
    p1, p2: (x,y) 2D 좌표.
    effective_obs: 해당 고도에서 존재하는 장애물들 (즉, obs.height >= 현재 고도)
    두 점을 잇는 선분이 장애물 내부를 관통하면 보이지 않는 것으로 간주.
    """
    line = LineString([p1, p2])
    for obs in effective_obs:
        if line.crosses(obs.base) or line.within(obs.base):
            return False
    return True

def build_multi_layer_graph(obstacles, start, goal, levels):
    """
    각 고도 레벨(z 값)에 대해 노드를 구성합니다.
    - 각 레벨에서는 해당 고도에 있는 장애물의 바닥면 꼭짓점을 노드로 추가합니다.
    - 단, 시작(start)과 도착(goal)은 이미 (x,y,z) 좌표로 주어지므로, 해당 고도일 때만 노드로 추가합니다.
    - 같은 레벨 내에서는 두 노드 사이에 선분이 effective obstacles와 충돌하지 않으면 에지를 추가합니다.
    - 또한, 동일한 (x,y) 좌표의 노드가 인접한 레벨에 있으면 수직 에지로 연결하고,
      서로 다른 고도에 있는 노드들 간 대각선 연결도 3D 충돌 검사를 통해 추가합니다.
    """
    G = nx.Graph()
    node_positions = {}  # node_id -> (x, y, z)
    pos_to_node = {}     # (x,y,z) -> node_id
    node_counter = 0

    # 1) 각 레벨별로 노드 생성 및 같은 레벨 내 에지 추가
    for z in levels:
        effective_obs = [obs for obs in obstacles if obs.height >= z]
        
        # 시작점, 도착점이 해당 고도에 있으면 노드 추가
        for pt in [start, goal]:
            if math.isclose(pt[2], z):
                pos = (pt[0], pt[1], z)
                if pos not in pos_to_node:
                    pos_to_node[pos] = node_counter
                    node_positions[node_counter] = pos
                    G.add_node(node_counter, pos=pos)
                    node_counter += 1

        # 각 장애물의 바닥면 꼭짓점을 노드로 추가 (해당 레벨 z)
        for obs in effective_obs:
            for coord in list(obs.base.exterior.coords)[:-1]:
                pos = (coord[0], coord[1], z)
                if pos not in pos_to_node:
                    pos_to_node[pos] = node_counter
                    node_positions[node_counter] = pos
                    G.add_node(node_counter, pos=pos)
                    node_counter += 1

        # 같은 레벨 내 노드들 사이 에지 추가 (2D 가시성 검사)
        nodes_in_level = [nid for nid, pos in node_positions.items() if math.isclose(pos[2], z)]
        for i in range(len(nodes_in_level)):
            for j in range(i+1, len(nodes_in_level)):
                id1 = nodes_in_level[i]
                id2 = nodes_in_level[j]
                p1 = (node_positions[id1][0], node_positions[id1][1])
                p2 = (node_positions[id2][0], node_positions[id2][1])
                if is_visible_at_level(p1, p2, effective_obs):
                    dist = Point(p1).distance(Point(p2))  # 수평 2D 거리
                    G.add_edge(id1, id2, weight=dist)
    
    # 2) 수직(레벨 간) 연결: 동일 (x,y) 좌표의 노드를 인접 레벨끼리 연결
    for pos, nid in pos_to_node.items():
        x, y, z = pos
        for z2 in levels:
            if not math.isclose(z2, z) and z2 > z:
                pos2 = (x, y, z2)
                if pos2 in pos_to_node:
                    nid2 = pos_to_node[pos2]
                    G.add_edge(nid, nid2, weight=abs(z2 - z))
    
    # 3) 대각선 연결: 서로 다른 고도에 있는 모든 노드 쌍에 대해 3D 충돌 검사를 수행
    node_ids = list(node_positions.keys())
    for i in range(len(node_ids)):
        for j in range(i+1, len(node_ids)):
            p1 = node_positions[node_ids[i]]
            p2 = node_positions[node_ids[j]]
            if not math.isclose(p1[2], p2[2]):  # 다른 고도에 있을 때
                # 이미 수직 연결된 경우(동일 (x,y))는 건너뜀
                if math.isclose(p1[0], p2[0]) and math.isclose(p1[1], p2[1]):
                    continue
                if not segment_collides(p1, p2, obstacles):
                    weight = euclidean_distance_3d(p1, p2)
                    if not G.has_edge(node_ids[i], node_ids[j]):
                        G.add_edge(node_ids[i], node_ids[j], weight=weight)
    return G, node_positions, pos_to_node

def euclidean_distance_3d(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

def multi_level_path(obstacles, start, goal, levels):
    G, node_positions, pos_to_node = build_multi_layer_graph(
        obstacles, start, goal, levels)

    start_id = pos_to_node.get((start[0], start[1], start[2]))
    goal_id  = pos_to_node.get((goal[0],  goal[1],  goal[2]))
    if start_id is None or goal_id is None:
        raise ValueError("Start/goal not in graph")

    def h(u, v):
        return euclidean_distance_3d(node_positions[u],
                                     node_positions[v])

    path_ids = nx.astar_path(G, start_id, goal_id, heuristic=h, weight='weight')
    return G, node_positions, pos_to_node, path_ids

def compute_collision_point(p1, p2, obstacles):
    """
    Returns (collision_point, obs) where `collision_point` is the earliest
    3D point along segment p1→p2 that enters ANY obstacle prism, and `obs`
    is the obstacle it hit.  If no collision, returns (None, None).
    """
    line2d    = LineString([(p1[0], p1[1]), (p2[0], p2[1])])
    total_len = line2d.length
    dz        = p2[2] - p1[2]

    best_t    = None
    best_obs  = None
    best_pt2d = None

    for obs in obstacles:
        # only obstacles that stick above the lower endpoint
        if min(p1[2], p2[2]) >= obs.height:
            continue

        # intersect only the **exterior ring** (boundary) → guarantees Points/MultiPoint
        intr = obs.base.exterior.intersection(line2d)
        if intr.is_empty:
            continue

        # collect all candidate Points
        pts = []
        if intr.geom_type == "Point":
            pts = [intr]
        elif hasattr(intr, "geoms"):
            pts = [g for g in intr.geoms if g.geom_type == "Point"]

        for pt in pts:
            t = line2d.project(pt) / total_len
            if t < 0 or t > 1:
                continue
            if best_t is None or t < best_t:
                best_t   = t
                best_obs = obs
                best_pt2d= pt

    if best_t is None:
        return None, None

    # reconstruct the full 3D collision point
    z_coll = p1[2] + dz * best_t
    coll_pt= (best_pt2d.x, best_pt2d.y, z_coll)
    return coll_pt, best_obs


def segment_collides(p1, p2, obstacles, eps=1e-9):
    """
    Returns True iff the segment actually clips **under** an obstacle roof.
    """
    coll_pt, obs = compute_collision_point(p1, p2, obstacles)
    if coll_pt is None:
        return False

    # only a collision if the crossing altitude is strictly below the roof
    return (coll_pt[2] < obs.height - eps)

def incremental_path(obstacles, start, goal, levels, max_iters=100):
    """
    1) Build graph once.
    2) Find initial A* path.
    3) Loop:
        - Scan each segment via compute_collision_point()
        - If coll_pt is not None, that's our collision.
        - Insert that Steiner node.
        - Re-A* from coll_node to goal.
        - Splice the subpath.
       Stop when no collisions remain.
    """
    # 1) build graph & get initial path IDs
    G, node_positions, pos_to_node, path_ids = multi_level_path(
        obstacles, start, goal, levels
    )
    start_id = pos_to_node[(start[0], start[1], start[2])]
    goal_id  = pos_to_node[(goal[0],  goal[1],  goal[2])]

    def h(u, v):
        return euclidean_distance_3d(node_positions[u], node_positions[v])

    for _ in range(max_iters):
        # reconstruct full coordinate path
        coords = [node_positions[n] for n in path_ids]

        # find first collision using compute_collision_point
        collision_idx = None
        collision_pt  = None
        for i in range(len(coords)-1):
            p1, p2 = coords[i], coords[i+1]
            coll_pt, obs = compute_collision_point(p1, p2, obstacles)
            if coll_pt is not None:
                collision_idx = i
                collision_pt  = coll_pt
                break

        # if nothing collided, we’re done
        if collision_idx is None:
            break

        # we know collision_pt is a valid (x,y,z)
        coll_id = pos_to_node.get(collision_pt)
        if coll_id is None:
            # insert new Steiner node
            coll_id = len(node_positions)
            node_positions[coll_id] = collision_pt
            pos_to_node[collision_pt] = coll_id
            G.add_node(coll_id, pos=collision_pt)

            # connect horizontally at its level
            z    = collision_pt[2]
            eff  = [obs for obs in obstacles if obs.height >= z]
            layer_nodes = [
                nid for nid,p in node_positions.items()
                if math.isclose(p[2], z) and nid != coll_id
            ]
            for nid in layer_nodes:
                x,y,_ = node_positions[nid]
                if is_visible_at_level((collision_pt[0],collision_pt[1]), (x,y), eff):
                    w = Point(collision_pt[0],collision_pt[1]).distance(Point(x,y))
                    G.add_edge(coll_id, nid, weight=w)

            # vertical edges
            for lvl in levels:
                if not math.isclose(lvl, z):
                    pos2 = (collision_pt[0], collision_pt[1], lvl)
                    nid2 = pos_to_node.get(pos2)
                    if nid2 is not None:
                        G.add_edge(coll_id, nid2, weight=abs(lvl - z))

            # diagonal edges
            for nid2, p2 in node_positions.items():
                if nid2 == coll_id: continue
                if not math.isclose(p2[2], z):
                    # safe to reuse compute_collision_point
                    pt2, _ = compute_collision_point(collision_pt, p2, obstacles)
                    if pt2 is None:
                        w = euclidean_distance_3d(collision_pt, p2)
                        G.add_edge(coll_id, nid2, weight=w)

        # re-A* from collision node to goal
        sub_ids = nx.astar_path(G, coll_id, goal_id, heuristic=h, weight='weight')

        # splice: keep up to collision_idx, then append subpath (dropping duplicate)
        path_ids = path_ids[:collision_idx+1] + sub_ids

    # build the real (x,y,z) list before visualizing
    coords = [node_positions[n] for n in path_ids]
    #visualize_3d(obstacles, coords)
    # final coordinate list
    return [node_positions[n] for n in path_ids]

def visualize_3d(obstacles, path):
    """
    장애물과 최종 경로를 3D로 시각화합니다.
    - 장애물: 밑면, 기둥, 윗면 (Poly3DCollection) 모두 표시
    - 경로: 빨간색(red) 선과 마커로 표시
    """
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 장애물 시각화
    for obs in obstacles:
        x_obs, y_obs = obs.base.exterior.xy
        # 밑면 (z=0)
        ax.plot(x_obs, y_obs, zs=0, color='gray', alpha=0.7)
        # 기둥: 각 꼭짓점에서 z=0부터 z=obs.height까지 선 그리기
        for i in range(len(x_obs)-1):
            x0, y0 = x_obs[i], y_obs[i]
            ax.plot([x0, x0], [y0, y0], [0, obs.height], color='gray', alpha=0.5)
        # 윗면: 꼭짓점 좌표 (z=obs.height)
        verts = [list(zip(x_obs, y_obs, [obs.height]*len(x_obs)))]
        poly = Poly3DCollection(verts, facecolors='lightgray', edgecolors='gray', alpha=0.5)
        ax.add_collection3d(poly)
    
    # 경로 시각화 (빨간색)
    if path is not None and len(path) > 0:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        zs = [p[2] for p in path]
        ax.plot(xs, ys, zs, 'r-o', label='Path', linewidth=3, markersize=8)
        ax.scatter(xs[0], ys[0], zs[0], color='green', s=100, label='Start')
        ax.scatter(xs[-1], ys[-1], zs[-1], color='blue', s=100, label='Goal')
    
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_zlim(0, max([obs.height for obs in obstacles])+20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title("3D Visualization: Path Only")
    plt.show()

def path_length(path):
    """
    Compute total 3D‐Euclidean length of a polyline `path`,
    by summing euclidean_distance_3d over each consecutive pair.
    """
    total = 0.0
    for a, b in zip(path, path[1:]):
        total += euclidean_distance_3d(a, b)
    return total


def generate_random_buildings(num, x_range, y_range, min_size, max_size, max_height):
    """
    겹치지 않는 랜덤 건물(장애물)을 생성합니다.
    - num: 건물 수
    - x_range, y_range: 건물이 생성될 x, y 좌표 범위 (예: (0, 1000))
    - min_size, max_size: 건물 밑면(사각형)의 최소, 최대 한 변 길이
    - max_height: 건물 높이의 최대값 (높이는 100~max_height 사이 랜덤)
    """
    buildings = []
    attempts = 0
    while len(buildings) < num and attempts < num * 50:
        attempts += 1
        w = random.uniform(min_size, max_size)
        h = random.uniform(min_size, max_size)
        x = random.uniform(x_range[0], x_range[1] - w)
        y = random.uniform(y_range[0], y_range[1] - h)
        poly = Polygon([(x,y), (x+w, y), (x+w, y+h), (x, y+h)])
        overlap = False
        for b in buildings:
            if poly.intersects(b.base):
                overlap = True
                break
        if not overlap:
            b_height = random.uniform(100, max_height)  # 최소 건물 높이를 100으로 설정
            buildings.append(Obstacle3D(poly, b_height))
    return buildings

# ---------------------------
# 메인 실행
# ---------------------------
if __name__ == "__main__":
    x_range = (0, 1000)
    y_range = (0, 1000)
    
    # 랜덤 건물(장애물) 생성: 예를 들어 15개, 건물 밑면 크기는 100~200, 최대 높이는 500
    obstacles = generate_random_buildings(num=15, x_range=x_range, y_range=y_range,
                                          min_size=100, max_size=200, max_height=500)
    
    if not obstacles:
        obstacles.append(Obstacle3D(Polygon([(100,100), (150,100), (150,150), (100,150)]), height=50))
    
    # 고도 레벨 생성 (장애물 최대 높이를 기준으로 균등 분할)
    levels = get_levels(obstacles, k=10)
    print("고도 레벨:", levels)
    
    # 시작점과 도착점을 극적으로 다르게 설정:
    # 시작점: 좌측 하단 근처, 낮은 고도 (levels의 최솟값)
    start = (random.uniform(x_range[0]+10, x_range[0]+50),
             random.uniform(y_range[0]+10, y_range[0]+50),
             min(levels))
    # 도착점: 우측 상단 근처, 높은 고도 (levels의 최댓값에 가까운 값; 여기서는 levels[-2] 사용)
    goal = (random.uniform(x_range[1]-50, x_range[1]-10),
            random.uniform(y_range[1]-50, y_range[1]-10),
            levels[-2])
    print("시작 좌표:", start)
    print("도착 좌표:", goal)
    

    start_time = time.time()
    # ── Compute initial path ──
    G, node_positions, pos_to_node, path_ids = multi_level_path(
        obstacles, start, goal, levels
    )
    # build the actual list of (x,y,z) points
    path_3d = [node_positions[n] for n in path_ids]
    print("초기 3D 경로:", path_3d)

    # visualize expects a list of 3‐tuples
    dist = path_length(path_3d)
    print(f"Total path distance: {dist:.3f}")
    initial_time = time.time() - start_time
    print(f"Time taken Initial: {initial_time}")
    #visualize_3d(obstacles, path_3d)

    # ── Refinement ──
    start_time = time.time()
    final_path = incremental_path(obstacles, start, goal, levels)
    print("보정된 최종 경로:", final_path)
    dist = path_length(final_path)
    print(f"Total path distance: {dist:.3f}")
    final_time = time.time() - start_time
    print(f"Time taken Final: {final_time}")
    #visualize_3d(obstacles, final_path)