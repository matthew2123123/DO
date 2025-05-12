import math
import random
import os
import numpy as np
import gym
from gym import spaces
from shapely.geometry import Polygon, LineString, Point
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import Point
import time

# ---------- 환경 정의 ----------
class Obstacle3D:
    def __init__(self, base_polygon: Polygon, height: float):
        self.base = base_polygon
        self.height = height

def segment_collides(p1, p2, obstacles):
    avg_z = (p1[2] + p2[2]) / 2
    line = LineString([(p1[0], p1[1]), (p2[0], p2[1])])
    for obs in obstacles:
        if avg_z < obs.height and line.intersects(obs.base) and not line.touches(obs.base):
            return True
    return False

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

class PathPlanningEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, building_generator, start, goal, x_range, y_range, z_range, step_size=20):
        super(PathPlanningEnv, self).__init__()
        self.building_generator = building_generator
        self.start = np.array(start, dtype=np.float32)
        self.goal = np.array(goal, dtype=np.float32)
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.step_size = step_size
        self.max_steps = 1000

        low = np.array([x_range[0], y_range[0], z_range[0], -x_range[1], -y_range[1], -z_range[1]], dtype=np.float32)
        high = np.array([x_range[1], y_range[1], z_range[1], x_range[1], y_range[1], z_range[1]], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Discrete(6)

    def reset(self):
        # 에피소드마다 새 장애물 배치 생성
        self.obstacles = self.building_generator()
        self.agent_pos = self.start.copy()
        self.steps = 0
        return np.concatenate([self.agent_pos, self.goal - self.agent_pos])

    def step(self, action):
        # --- 위치 업데이트 ----------------------------------------------------
        move = np.zeros(3, dtype=np.float32)
        if action == 0: move[0] = self.step_size
        elif action == 1: move[0] = -self.step_size
        elif action == 2: move[1] = self.step_size
        elif action == 3: move[1] = -self.step_size
        elif action == 4: move[2] = self.step_size
        elif action == 5: move[2] = -self.step_size

        new_pos = self.agent_pos + move
        new_pos[0] = np.clip(new_pos[0], self.x_range[0], self.x_range[1])
        new_pos[1] = np.clip(new_pos[1], self.y_range[0], self.y_range[1])
        new_pos[2] = np.clip(new_pos[2], self.z_range[0], self.z_range[1])

        prev_dist = np.linalg.norm(self.agent_pos - self.goal)
        new_dist  = np.linalg.norm(new_pos - self.goal)
        collided  = segment_collides(tuple(self.agent_pos), tuple(new_pos), self.obstacles)
        # 장애물까지 최소 거리 계산
        #    new_pos[2] < obs.height일 때만 고려 (수직 충돌 가능 구간)
        dists = []
        point = Point(new_pos[0], new_pos[1])
        for obs in self.obstacles:
            if new_pos[2] < obs.height:
                dists.append(obs.base.distance(point))
        min_dist = min(dists) if dists else max(self.x_range[1], self.y_range[1])
        self.agent_pos = new_pos
        self.steps += 1

        # ---------- ①~④ 보상 설계 -------------------------------------------
        beta      = 1.0     # ③ 진행(potential-based) 보상 계수
        lambda_d  = 0.05    # ② 이동 거리 패널티 계수
        goal_eps  = 1e-3    # 목표판정 허용 오차
        gamma = 0.1     # 거리 보상 계수
        safe_radius = 50.0
        if min_dist < safe_radius:
            r_safe = - gamma * (safe_radius - min_dist)
        else:
            r_safe = 0.0
            
        # ③ 목표 쪽으로 가까워진 만큼 보상
        r_progress = beta * (prev_dist - new_dist)

        # ② 한 스텝 이동 자체에 대한 소량 패널티(경로 길이 최소화)
        r_step = -lambda_d * np.linalg.norm(move)

        reward = r_progress + r_step + r_safe
        done   = False

        # ④ 충돌 시 큰 패널티 후 에피소드 종료
        if collided:
            reward -= 50.0
            done = True

        # ① 목표 도착 보상 후 종료
        if new_dist <= goal_eps:
            reward += 50.0
            done = True

        # 최대 스텝 소진
        if self.steps >= self.max_steps:
            done = True
        # ---------------------------------------------------------------------

        obs  = np.concatenate([self.agent_pos, self.goal - self.agent_pos])
        info = {'dist_to_goal': new_dist, 'collided': collided}
        return obs, reward, done, info

    def render(self, mode='human'):
        print(f"Step {self.steps}: Pos={self.agent_pos}")

# 유틸 함수
def visualize_path(obstacles, path, start, goal):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 건물 색칠 (바닥, 옥상, 벽면)
    for obs in obstacles:
        x, y = obs.base.exterior.xy
        z0 = np.zeros(len(x))
        z1 = np.full(len(x), obs.height)
        # floor
        verts_floor = [list(zip(x, y, z0))]
        # roof
        verts_roof  = [list(zip(x, y, z1))]
        # walls: 사이사이 면 구성
        verts_walls = []
        n = len(x)
        for i in range(n - 1):
            verts_walls.append([
                (x[i],   y[i],   0),
                (x[i+1], y[i+1], 0),
                (x[i+1], y[i+1], obs.height),
                (x[i],   y[i],   obs.height)
            ])
        # 마지막 점 연결
        verts_walls.append([
            (x[-1], y[-1], 0),
            (x[0],  y[0],  0),
            (x[0],  y[0],  obs.height),
            (x[-1], y[-1], obs.height)
        ])
        # Poly3DCollection
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        collection = Poly3DCollection(verts_floor + verts_roof + verts_walls, linewidths=0.2, edgecolors='gray')
        collection.set_facecolor((0.75, 0.75, 0.75, 0.5))
        ax.add_collection3d(collection)
    # 경로 그리기
    if path:
        xs, ys, zs = zip(*path)
        ax.plot(xs, ys, zs, 'r-o', label='Path')
        ax.scatter(start[0], start[1], start[2], c='g', s=100, label='Start')
        ax.scatter(goal[0], goal[1], goal[2], c='b', s=100, label='Goal')
    ax.set_xlim(0,1000);
    ax.set_ylim(0,1000);
    ax.set_zlim(0,600)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(); plt.title("3D Path Planning")
    plt.show()

def compute_total_distance(path):
    total = 0.0
    for i in range(1, len(path)):
        total += np.linalg.norm(np.array(path[i]) - np.array(path[i-1]))
    return total

# 실행부
if __name__ == '__main__':
    x_range, y_range, z_range = (0,1000), (0,1000), (0,500)
    obstacles = generate_random_buildings(15, x_range, y_range, 100,200,500)
    start = (50.0,50.0,0.0); goal = (950.0,950.0,500.0)
    
    def building_generator():
        return generate_random_buildings(
            num=15,
            x_range=x_range,
            y_range=y_range,
            min_size=100,
            max_size=200,
            max_height=500,
            start=start,
            safe_radius=100
        )
    
    # 학습용 환경
    train_env = PathPlanningEnv(
        building_generator=building_generator,
        start=start, goal=goal,
        x_range=x_range, y_range=y_range, z_range=z_range
    )
    vec_env = DummyVecEnv([lambda: train_env])

    # learning rate scheduler
    def linear_schedule(initial_value: float):
        return lambda progress: progress * initial_value

    # PPO 모델 설정 (critic capacity 확장, lr schedule, n_steps 증가)
    policy_kwargs = dict(net_arch=[dict(pi=[128,128], vf=[256,256])])
    model_path = '3DSSP_RL_step20.zip'
    if os.path.exists(model_path):
        print("모델 로딩...")
        model = PPO.load(model_path, env=vec_env)
    else:
        print("모델 학습 시작...")
        model = PPO(
            'MlpPolicy', vec_env,
            verbose=1,
            device='cpu',
            policy_kwargs=policy_kwargs,
            learning_rate=linear_schedule(1e-4),
            n_steps=2048,
            batch_size=64,
        )
        model.learn(total_timesteps=400000, log_interval=20)
        model.save(model_path)
        print("모델 저장 완료")

    # 예측용 환경
    test_env = PathPlanningEnv(
    building_generator=building_generator,
    start=start,
    goal=goal,
    x_range=x_range,
    y_range=y_range,
    z_range=z_range
    )
    obs = test_env.reset()
    path = [obs[:3].tolist()]
    max_retries = 3
    
    start_time = time.time()
    print("모델 예측 시작...")  
    for attempt in range(max_retries):
        obs = test_env.reset()
        path = [obs[:3].tolist()]
        done = False

        while not done:
            action, _ = model.predict(obs)
            obs, _, done, info = test_env.step(action)
            path.append(obs[:3].tolist())

        if info.get('collided', False):
            print(f"충돌 발생! {attempt+1}회차 재시도합니다.")
            continue  # next retry
        else:
            print("성공적으로 목표 도착")
            break

    if info.get('collided', False):
        print("모든 시도에서 충돌 발생 — 모델/환경 재검토가 필요합니다.")
        exit(1)
    end_time = time.time()

    print(f"총 이동 거리: {compute_total_distance(path):.2f}")
    print(f"총 소요 시간: {end_time - start_time:.2f}초")
    visualize_path(obstacles, path, start, goal)