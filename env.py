import numpy as np
import matplotlib.pyplot as plt

class UAVSecureEnv:
    def __init__(self, num_waypoints=2, seed=42):
        np.random.seed(seed)
        self.np_random = np.random.default_rng(seed)

        self.dt = 1
        self.horizon_steps = 80

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
