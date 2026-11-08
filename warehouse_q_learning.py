"""
Two-Robot Warehouse Coordination using Tabular Q-Learning
Run: python warehouse_q_learning.py
Dependencies: numpy, matplotlib
"""
import random
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

GRID = 5
STARTS = ((0, 0), (0, 4))
GOALS = ((4, 4), (4, 0))
OBSTACLES = {(1, 2), (2, 2), (3, 2)}
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MOVE_NAMES = ["Up", "Down", "Left", "Right"]

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

class WarehouseEnv:
    def __init__(self, max_steps=50):
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.positions = [STARTS[0], STARTS[1]]
        self.done = [False, False]
        self.steps = 0
        return self.state()

    def state(self):
        return (self.positions[0][0], self.positions[0][1],
                self.positions[1][0], self.positions[1][1],
                int(self.done[0]), int(self.done[1]))

    def step(self, action):
        self.steps += 1
        robot_id = action // 4
        dr, dc = MOVES[action % 4]
        old = self.positions[robot_id]

        if self.done[robot_id]:
            reward, invalid = -1.0, True
        else:
            new = (old[0] + dr, old[1] + dc)
            other = self.positions[1-robot_id]
            invalid = (not (0 <= new[0] < GRID and 0 <= new[1] < GRID)
                       or new in OBSTACLES or new == other)
            if invalid:
                reward = -4.0
            else:
                before = manhattan(old, GOALS[robot_id])
                after = manhattan(new, GOALS[robot_id])
                self.positions[robot_id] = new
                reward = -1.0 + 0.6 * (before-after)
                if new == GOALS[robot_id]:
                    self.done[robot_id] = True
                    reward += 12.0

        success = all(self.done)
        if success:
            reward += 30.0
        truncated = self.steps >= self.max_steps and not success
        if truncated:
            reward -= 10.0
        return self.state(), reward, success or truncated, {"success": success, "invalid": invalid}

def valid_actions(env):
    actions = []
    for robot_id in range(2):
        if env.done[robot_id]:
            continue
        for move_id, (dr, dc) in enumerate(MOVES):
            nr = env.positions[robot_id][0] + dr
            nc = env.positions[robot_id][1] + dc
            if (0 <= nr < GRID and 0 <= nc < GRID
                    and (nr, nc) not in OBSTACLES
                    and (nr, nc) != env.positions[1-robot_id]):
                actions.append(robot_id*4 + move_id)
    return actions or list(range(8))

def train(episodes=9000, alpha=0.18, gamma=0.95,
          epsilon=1.0, epsilon_min=0.04, epsilon_decay=0.9992):
    random.seed(42)
    np.random.seed(42)
    q = defaultdict(lambda: np.zeros(8, dtype=float))
    env = WarehouseEnv()
    rewards, successes = [], []

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0
        while True:
            actions = valid_actions(env)
            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                masked = np.full(8, -1e9)
                masked[actions] = 0
                action = int(np.argmax(q[state] + masked))

            next_state, reward, done, info = env.step(action)
            next_actions = valid_actions(env)
            target = reward if done else reward + gamma*np.max(q[next_state][next_actions])
            q[state][action] += alpha*(target-q[state][action])
            state = next_state
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
        successes.append(int(info["success"]))
        epsilon = max(epsilon_min, epsilon*epsilon_decay)

    return q, np.asarray(rewards), np.asarray(successes)

def evaluate(q=None, episodes=500, random_policy=False):
    env = WarehouseEnv()
    returns, successes = [], []
    for _ in range(episodes):
        state = env.reset()
        total = 0
        while True:
            actions = valid_actions(env)
            if random_policy:
                action = random.choice(actions)
            else:
                masked = np.full(8, -1e9)
                masked[actions] = 0
                action = int(np.argmax(q[state] + masked))
            state, reward, done, info = env.step(action)
            total += reward
            if done:
                break
        returns.append(total)
        successes.append(int(info["success"]))
    return np.mean(returns), np.mean(successes)*100

if __name__ == "__main__":
    q, rewards, successes = train()
    q_return, q_success = evaluate(q)
    random_return, random_success = evaluate(random_policy=True)
    print(f"Q-learning: mean return={q_return:.2f}, success={q_success:.1f}%")
    print(f"Random:     mean return={random_return:.2f}, success={random_success:.1f}%")

    window = 100
    curve = np.convolve(rewards, np.ones(window)/window, mode="valid")
    plt.plot(np.arange(len(curve))+window, curve, label="Q-learning")
    plt.axhline(random_return, linestyle="--", label="Random baseline")
    plt.xlabel("Training episode")
    plt.ylabel("Mean episode return (100-episode window)")
    plt.title("Two-Robot Warehouse: Q-learning")
    plt.legend()
    plt.tight_layout()
    plt.savefig("learning_curve.png", dpi=180)
    plt.show()
