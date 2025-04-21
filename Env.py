import gymnasium
from gymnasium import spaces
import numpy as np
from Entity import *
from Channel import SimpleCommEntity, mmwave_single_channel
from Render import Render

class UAVSecureEnv(gymnasium.Env):
    def __init__(self, num_waypoints=2, seed=42):
        super(UAVSecureEnv, self).__init__()
        np.random.seed(seed)
        self.np_random = np.random.default_rng(seed)

        self.dt = 1
        self.horizon_steps = 60
        self.beta0 = 1e4
        self.sigma2 = 1.0
        self.max_power = 1.5
        self.max_speed = 15.0
        self.energy_max = 150000.0
        self.K = 6.0
        self.frequency = 28e9
        self.pathloss_exp = 2.2

        self.hap = HAP()
        self.eve = EVA(self.np_random)
        self.user_1 = User(self.np_random, region='first')
        self.user_2 = User(self.np_random, region='second')
        self.user_3 = User(self.np_random, region='third')
        self.user_4 = User(self.np_random, region='fourth')
        self.uav_init_pos = np.array([self.np_random.uniform(-100, -50), self.np_random.uniform(400, 450)], dtype=np.float32)
        self.uav = UAV(self.uav_init_pos, self.max_speed, self.energy_max)

        self.x_min, self.x_max = -450.0, 450.0
        self.y_min, self.y_max = -450.0, 450.0

        self.num_waypoints = num_waypoints
        self.waypoint_radius = 30.0
        self.waypoints = []
        if self.num_waypoints >= 1:
            self.waypoints.append(self.np_random.uniform([150, 150], [200, 200]).astype(np.float32))
        if self.num_waypoints >= 2:
            self.waypoints.append(self.np_random.uniform([250, -350], [350, -250]).astype(np.float32))
        self.waypoints = self.waypoints[:num_waypoints]
        self.visited = [False for _ in self.waypoints]

        state_dim = 10 + self.num_waypoints
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0, 0.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)

        self.renderer = Render()
        self.reset()

    def render(self):
        self.renderer.render(self)

    def reset(self, **kwargs):
        self.uav.reset()
        self.eve = EVA(self.np_random)
        self.user_1 = User(self.np_random, region='first')
        self.user_2 = User(self.np_random, region='second')
        self.user_3 = User(self.np_random, region='third')
        self.user_4 = User(self.np_random, region='fourth')
        self.step_count = 0
        self.current_wp_idx = 0
        self.visited = [False for _ in self.waypoints]
        self.trajectory = [self.uav.pos.copy()]
        self.reward_log = {k: [] for k in ['direction', 'secrecy', 'QoS', 'speed', 'shaped', 'hit', 'total']}
        return self._get_state(), {}

    def _get_state(self):
        dx_h, dy_h = self.hap.pos - self.uav.pos
        dx_e, dy_e = self.eve.pos - self.uav.pos
        norm = 450.0
        vel_norm = self.max_speed
        waypoint_flags = [1.0 if v else 0.0 for v in self.visited]
        target_wp = self.waypoints[min(self.current_wp_idx, self.num_waypoints - 1)]
        wp_dx, wp_dy = target_wp - self.uav.pos

        state = np.array([
            dx_h / norm, dy_h / norm,
            dx_e / norm, dy_e / norm,
            self.uav.vel[0] / vel_norm, self.uav.vel[1] / vel_norm,
            self.uav.energy_remaining / self.energy_max,
            self.step_count / self.horizon_steps,
            wp_dx / norm, wp_dy / norm
        ] + waypoint_flags, dtype=np.float32)

        return state

    def step(self, action):
        v_x = float(action[0]) * self.max_speed
        v_y = float(action[1]) * self.max_speed
        power_frac = np.clip(float(action[2]), 0.0, 1.0)
        P_tx = power_frac * self.max_power

        self.uav.move(np.array([v_x, v_y]), self.dt)
        self.uav.consume_energy(P_tx, self.dt)
        self.trajectory.append(self.uav.pos.copy())

        tx = SimpleCommEntity(self.uav.pos, self.uav.type, self.uav.ant_type, self.uav.ant_num)
        hap = SimpleCommEntity(self.hap.pos, self.hap.type, self.hap.ant_type, self.hap.ant_num)
        eve = SimpleCommEntity(self.eve.pos, self.eve.type, self.eve.ant_type, self.eve.ant_num)
        users = [
            SimpleCommEntity(u.pos, u.type, u.ant_type, u.ant_num)
            for u in [self.user_1, self.user_2, self.user_3, self.user_4]
        ]

        h_main, gain_main = mmwave_single_channel(tx, hap, self.frequency, self.beta0, self.K)
        h_eve, gain_eve = mmwave_single_channel(tx, eve, self.frequency, self.beta0, self.K)

        gain_dict = {'gain_main': gain_main}
        for i, user in enumerate(users):
            h_hap_user, gain_hap_user = mmwave_single_channel(hap, user, self.frequency, self.beta0, self.K)
            h_uav_user, gain_uav_user = mmwave_single_channel(tx, user, self.frequency, self.beta0, self.K)
            gain_dict[f'gain_hap_user{i+1}'] = gain_hap_user
            gain_dict[f'gain_uav_user{i+1}'] = gain_uav_user

        reward = 0.0
        secrecy_weight = 3.0
        QoS_weight = 0.5
        shaped_weight = 0.1
        direction_weight = 15.0
        speed_weight = 1.0

        snr_main = (P_tx * gain_main) / self.sigma2
        snr_eve = (P_tx * gain_eve) / self.sigma2
        secrecy_rate = max(np.log2(1 + snr_main) - np.log2(1 + snr_eve), 0.0)
        secrete_reward = secrecy_weight * secrecy_rate

        QoS_reward = QoS_weight * compute_qos_reward(P_tx, gain_dict, self.sigma2)

        hit_reward = 0.0
        waypoint_hit_bonus = 30.0
        done = False
        if self.current_wp_idx < self.num_waypoints:
            current_wp = self.waypoints[self.current_wp_idx]
            dist_to_wp = np.linalg.norm(self.uav.pos - current_wp)
            if dist_to_wp <= self.waypoint_radius:
                hit_reward = waypoint_hit_bonus
                self.visited[self.current_wp_idx] = True
                self.current_wp_idx += 1

        if self.current_wp_idx < self.num_waypoints:
            target_wp = self.waypoints[self.current_wp_idx]
        else:
            target_wp = self.waypoints[-1]

        dist_to_wp = np.linalg.norm(self.uav.pos - target_wp)
        future_pos = self.uav.pos + self.uav.vel * self.dt
        shaped_scale = 0.01
        shaped_reward = shaped_weight / (1.0 + shaped_scale * dist_to_wp)

        direction_power = 6
        direction_vec = target_wp - future_pos
        direction_unit = direction_vec / (np.linalg.norm(direction_vec) + 1e-6)
        velocity_unit = self.uav.vel / (np.linalg.norm(self.uav.vel) + 1e-6)
        alignment = max(np.dot(direction_unit, velocity_unit), 0.0)
        direction_reward = direction_weight * (alignment ** direction_power)

        speed_scale = 0.1
        ideal_speed = self.max_speed * 0.9
        speed = np.linalg.norm(self.uav.vel)
        speed_reward = speed_weight * np.exp(-speed_scale * (speed - ideal_speed) ** 2)

        success_bonus = 200.0
        if self.uav.energy_remaining <= 0 or self.step_count >= self.horizon_steps:
            done = True
            if all(self.visited):
                reward += success_bonus

        time_penalty = -1

        reward += secrete_reward
        reward += shaped_reward
        reward += hit_reward
        reward += speed_reward
        reward += time_penalty
        reward += direction_reward
        reward += QoS_reward

        self.reward_log['direction'].append(direction_reward)
        self.reward_log['secrecy'].append(secrete_reward)
        self.reward_log['QoS'].append(QoS_reward)
        self.reward_log['speed'].append(speed_reward)
        self.reward_log['shaped'].append(shaped_reward)
        self.reward_log['hit'].append(hit_reward)
        self.reward_log['total'].append(reward)

        self.step_count += 1

        return self._get_state(), float(np.clip(reward, -50000, 50000)), done, False, {}

def compute_qos_reward(P_tx, gain_dict, sigma2):
    qos_reward = 0.0
    for i in range(1, 5):
        snr_hap = (P_tx * gain_dict[f'gain_hap_user{i}']) / sigma2
        snr_uav = (P_tx * gain_dict[f'gain_uav_user{i}']) / sigma2
        snr_uav_hap = (P_tx * gain_dict['gain_main']) / sigma2
        rate_hap = np.log2(1 + snr_hap)
        rate_relay = min(np.log2(1 + snr_uav), np.log2(1 + snr_uav_hap))
        qos_reward += 0.5 * rate_hap + 1.0 * rate_relay
    return qos_reward