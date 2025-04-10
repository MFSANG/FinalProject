import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import matplotlib.pyplot as plt

from env import UAVSecureEnv
from model import ActorCritic


class UAVSecureEnv:
    def __init__(self, num_waypoints=2, seed=42):
        np.random.seed(seed)
        self.np_random = np.random.default_rng(seed)

        self.dt = 1
        self.horizon_steps = 100

        self.pathloss_exp = 2.2
        self.rician_K = 6.0
        self.beta0 = 1e4
        self.sigma2 = 1.0
        self.max_power = 1.5
        self.max_speed = 15.0
        self.energy_max = 150000.0

        self.hap_pos = np.array([0.0, 0.0])
        self.eve_pos = self.np_random.uniform(-50, 50, size=(2,))
        self.x_min, self.x_max = -450.0, 450.0
        self.y_min, self.y_max = -450.0, 450.0

        self.num_waypoints = num_waypoints
        self.waypoint_radius = 100.0
        self.waypoints = []
        if self.num_waypoints >= 1:
            self.waypoints.append(self.np_random.uniform([150, 150], [200, 200]).astype(np.float32))
        if self.num_waypoints >= 2:
            self.waypoints.append(self.np_random.uniform([250, -350], [350, -250]).astype(np.float32))
        self.waypoints = self.waypoints[:num_waypoints]
        self.visited = [False for _ in self.waypoints]

        self.uav_init_pos = np.array([self.np_random.uniform(-100, -50), self.np_random.uniform(400, 450)], dtype=np.float32)
        self.reset()

    def reset(self):
        self.uav_pos = self.uav_init_pos.copy()
        self.uav_vel = np.array([0.0, 0.0])
        self.energy_remaining = self.energy_max
        self.step_count = 0
        self.current_wp_idx = 0
        self.visited = [False for _ in self.waypoints]
        self.trajectory = [self.uav_pos.copy()]
        return self._get_state()

    def _get_state(self):
        dx_h, dy_h = self.hap_pos - self.uav_pos
        dx_e, dy_e = self.eve_pos - self.uav_pos
        norm = 450.0
        vel_norm = self.max_speed
        waypoint_flags = [1.0 if v else 0.0 for v in self.visited]
        target_wp = self.waypoints[min(self.current_wp_idx, self.num_waypoints - 1)]
        wp_dx, wp_dy = target_wp - self.uav_pos
        return np.array([
            dx_h / norm, dy_h / norm,
            dx_e / norm, dy_e / norm,
            self.uav_vel[0] / vel_norm, self.uav_vel[1] / vel_norm,
            self.energy_remaining / self.energy_max,
            self.step_count / self.horizon_steps,
            wp_dx / norm, wp_dy / norm
        ] + waypoint_flags, dtype=np.float32)

    def step(self, action):
        v_x = float(action[0]) * self.max_speed
        v_y = float(action[1]) * self.max_speed
        power_frac = np.clip(float(action[2]), 0.0, 1.0)
        P_tx = power_frac * self.max_power

        speed = np.hypot(v_x, v_y)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            v_x *= scale
            v_y *= scale

        self.uav_vel = np.array([v_x, v_y])
        self.uav_pos += self.uav_vel * self.dt
        self.trajectory.append(self.uav_pos.copy())

        dist_h = np.linalg.norm(self.hap_pos - self.uav_pos) + 1e-6
        dist_e = np.linalg.norm(self.eve_pos - self.uav_pos) + 1e-6
        K = self.rician_K
        scatter_h = np.sqrt(0.5) * (np.random.normal() + 1j * np.random.normal())
        scatter_e = np.sqrt(0.5) * (np.random.normal() + 1j * np.random.normal())
        h_main = np.sqrt(self.beta0 * dist_h**(-self.pathloss_exp)) * (
            np.sqrt(K / (K + 1)) + np.sqrt(1 / (K + 1)) * scatter_h)
        h_eve = np.sqrt(self.beta0 * dist_e**(-self.pathloss_exp)) * (
            np.sqrt(K / (K + 1)) + np.sqrt(1 / (K + 1)) * scatter_e)

        gain_main = abs(h_main)**2
        gain_eve = abs(h_eve)**2
        snr_main = (P_tx * gain_main) / self.sigma2
        snr_eve = (P_tx * gain_eve) / self.sigma2
        secrecy_rate = max(np.log2(1 + snr_main) - np.log2(1 + snr_eve), 0.0)

        reward = 0.0
        secrecy_weight = 3.0
        shaped_weight = 2.0
        shaped_scale = 0.01
        direction_weight = 5.0
        direction_power = 4
        speed_weight = 1.0
        speed_scale = 0.1
        time_penalty = -1
        waypoint_hit_bonus = 20.0
        success_bonus = 200.0

        done = False

        if self.current_wp_idx < self.num_waypoints:
            current_wp = self.waypoints[self.current_wp_idx]
            dist_to_wp = np.linalg.norm(self.uav_pos - current_wp)
            if dist_to_wp <= self.waypoint_radius:
                reward += waypoint_hit_bonus
                self.visited[self.current_wp_idx] = True
                self.current_wp_idx += 1

        if self.current_wp_idx < self.num_waypoints:
            target_wp = self.waypoints[self.current_wp_idx]
        else:
            target_wp = self.waypoints[-1]

        dist_to_wp = np.linalg.norm(self.uav_pos - target_wp)

        reward += secrecy_weight * secrecy_rate
        reward += shaped_weight / (1.0 + shaped_scale * dist_to_wp)

        # 使用预测位置方向替代当前速度方向
        future_pos = self.uav_pos + self.uav_vel * self.dt
        direction_vec = target_wp - future_pos
        norm_direction = np.linalg.norm(direction_vec) + 1e-6
        direction_unit = direction_vec / norm_direction
        norm_velocity = np.linalg.norm(self.uav_vel) + 1e-6
        velocity_unit = self.uav_vel / norm_velocity
        alignment = max(np.dot(direction_unit, velocity_unit), 0.0)
        direction_reward = direction_weight * (alignment ** direction_power)
        reward += direction_reward

        ideal_speed = self.max_speed * 0.9
        reward += speed_weight * np.exp(-speed_scale * (speed - ideal_speed)**2)

        reward += time_penalty

        if self.energy_remaining <= 0 or self.step_count >= self.horizon_steps:
            done = True
            if all(self.visited):
                reward += success_bonus

        self.energy_remaining -= P_tx * self.dt
        self.step_count += 1

        return self._get_state(), float(np.clip(reward, -50000, 50000)), done, {}

    def render(self):
        traj = np.array(self.trajectory)
        current_wp_idx = min(self.current_wp_idx, self.num_waypoints)
        plt.figure(figsize=(8, 8))
        plt.plot(traj[:, 0], traj[:, 1], '-o', color='black', linewidth=1.5, markersize=4, label='UAV Trajectory')
        plt.plot(traj[-1, 0], traj[-1, 1], 'o', color='red', markersize=8, label='Current UAV Position')
        if len(traj) >= 2:
            dx, dy = traj[-1] - traj[-2]
            plt.arrow(traj[-2, 0], traj[-2, 1], dx, dy, head_width=5, head_length=8, fc='red', ec='red')

        plt.scatter(self.hap_pos[0], self.hap_pos[1], marker='D', s=100, color='blue', label='HAP')
        plt.scatter(self.eve_pos[0], self.eve_pos[1], marker='D', s=100, color='purple', label='Eve')

        fence_x = [self.x_min, self.x_max, self.x_max, self.x_min, self.x_min]
        fence_y = [self.y_min, self.y_min, self.y_max, self.y_max, self.y_min]
        plt.plot(fence_x, fence_y, '--', color='gray', linewidth=1.2, label='Geofence')

        for i, wp in enumerate(self.waypoints):
            marker_color = 'green' if self.visited[i] else 'orange'
            edge_color = 'red' if i == current_wp_idx else 'black'
            plt.scatter(wp[0], wp[1], s=80, color=marker_color,
                        edgecolors=edge_color, linewidths=1.5,
                        label=f'WP{i+1}' if i == 0 else "")
            plt.text(wp[0]+5, wp[1]+5, f'WP{i+1}', fontsize=10, color='black')

        for idx in range(0, len(traj), max(1, len(traj)//10)):
            plt.text(traj[idx, 0]+2, traj[idx, 1]+2, f'{idx}', fontsize=8, color='darkgray')

        plt.xlabel(r'$X$ (m)', fontsize=12)
        plt.ylabel(r'$Y$ (m)', fontsize=12)
        plt.title('UAV Trajectory Visualization', fontsize=14, fontweight='bold')
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right', fontsize=10)
        plt.tight_layout()
        plt.show()


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor_mean = nn.Linear(hidden_size, action_dim)
        self.actor_log_std = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x, exploration_scale=1.0):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.actor_mean(x)
        base_log_std = self.actor_log_std(x)
        # 将 exploration_scale 转为 tensor
        scale_tensor = torch.tensor(exploration_scale, dtype=base_log_std.dtype, device=base_log_std.device)
        # 通过加上 log(exploration_scale) 来调整 log_std
        log_std = base_log_std + torch.log(scale_tensor)
        # 对 log_std 进行裁剪，避免过小或过大
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std, self.critic(x)


def get_exploration_scale(iteration, max_iter, init_scale=2, final_scale=2):
    scale = init_scale - (init_scale - final_scale) * (iteration / max_iter)
    return max(scale, final_scale)

def compute_gae(traj, gamma=0.98, lam=0.95):
    returns, advantages = [], []
    gae = 0.0
    R = 0.0
    for t in reversed(range(len(traj))):
        _, _, _, r, v = traj[t]
        R = r + gamma * R
        if t + 1 < len(traj):
            next_v = traj[t + 1][4].item()
        else:
            next_v = 0.0
        delta = r + gamma * next_v - v.item()
        gae = delta + gamma * lam * gae
        returns.insert(0, R)
        advantages.insert(0, gae)
    return returns, advantages

if __name__ == '__main__':
    env = UAVSecureEnv(num_waypoints=2)
    state_dim = len(env.reset())
    action_dim = 3
    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    gamma = 0.98
    lam = 0.95
    clip_eps = 0.1
    train_iters = 1000
    episodes_per_iter = 30
    reward_history = []
    all_positions = []  # 用于记录热力图坐标

    for iter in range(train_iters):
        memory = []
        total_rewards = []
        exploration_scale = get_exploration_scale(iter, train_iters)

        for ep in range(episodes_per_iter):
            s = env.reset()
            ep_reward = 0
            traj = []
            episode_positions = []

            while True:
                s_tensor = torch.tensor(s, dtype=torch.float32)
                mean, log_std, value = model(s_tensor, exploration_scale=exploration_scale)
                dist = D.Normal(mean, torch.exp(log_std))
                raw_action = dist.sample()
                action = torch.tanh(raw_action)
                log_prob = dist.log_prob(action).sum()
                s_next, r, done, _ = env.step(action.detach().numpy())

                traj.append((s_tensor.detach(), action.detach(), log_prob.detach(), float(r), value.detach()))
                episode_positions.append(env.uav_pos.copy())

                s = s_next
                ep_reward += r
                if done:
                    break

            total_rewards.append(ep_reward)
            all_positions.extend(episode_positions)

            returns, advantages = compute_gae(traj, gamma, lam)
            for i, (s, a, logp, _, v) in enumerate(traj):
                memory.append((s, a, logp, torch.tensor(returns[i], dtype=torch.float32),
                               torch.tensor(advantages[i], dtype=torch.float32)))

        states, actions, old_logps, returns, advantages = zip(*memory)
        states = torch.stack(states)
        actions = torch.stack(actions)
        old_logps = torch.stack(old_logps)
        returns = torch.stack(returns)
        advantages = torch.stack(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(4):
            mean, log_std, values = model(states, exploration_scale=exploration_scale)
            dist = D.Normal(mean, torch.exp(log_std))
            new_logps = dist.log_prob(actions).sum(dim=1)
            ratios = torch.exp(new_logps - old_logps)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.functional.mse_loss(values.squeeze(), returns)
            loss = actor_loss + 0.5 * critic_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

        scheduler.step()

        avg_reward = np.mean(total_rewards)
        reward_history.append(avg_reward)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Iteration {iter+1}/{train_iters}, Avg Reward: {avg_reward:.2f}, Exploration Scale: {exploration_scale:.2f}, LR: {current_lr:.1e}")

    obs = env.reset()
    for _ in range(env.horizon_steps):
        s_tensor = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            mean, log_std, _ = model(s_tensor, exploration_scale=exploration_scale)
        action = mean.numpy()
        obs, _, done, _ = env.step(action)
        if done:
            break
    env.render()

    # 绘制 reward 曲线
    plt.figure()
    plt.plot(reward_history)
    plt.xlabel('Iteration')
    plt.ylabel('Average Reward')
    plt.title('Training Reward Curve')
    plt.grid(True)
    plt.savefig('reward_curve.png')
    plt.show()

    # 绘制 UAV 轨迹热力图
    all_positions_np = np.array(all_positions)
    x_vals = all_positions_np[:, 0]
    y_vals = all_positions_np[:, 1]
    plt.figure(figsize=(8, 6))
    plt.hist2d(x_vals, y_vals, bins=100, cmap='hot')
    plt.colorbar(label='Visit Frequency')
    plt.title('UAV Trajectory Heatmap')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('trajectory_heatmap.png')
    plt.show()

