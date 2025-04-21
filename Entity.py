import numpy as np

class UAV(object):
    def __init__(self, init_pos, max_speed, energy_max):
        self.init_pos = np.array(init_pos, dtype=np.float32)
        self.pos = np.array(init_pos, dtype=np.float32)
        self.vel = np.array([0.0, 0.0], dtype=np.float32)
        self.max_speed = max_speed
        self.energy_max = energy_max
        self.energy_remaining = energy_max
        self.type = 'UAV'
        self.ant_num = 8  # 线性阵列，8根天线
        self.ant_type = 'ULA'

    def reset(self):
        self.pos = self.init_pos.copy()
        self.vel = np.array([0.0, 0.0], dtype=np.float32)
        self.energy_remaining = self.energy_max

    def move(self, velocity, dt):
        speed = np.linalg.norm(velocity)
        if speed > self.max_speed:
            velocity = velocity * (self.max_speed / speed)
        self.vel = velocity
        self.pos += self.vel * dt

    def consume_energy(self, power, dt):
        self.energy_remaining -= power * dt


class HAP(object):
    def __init__(self, position=(0.0, 0.0)):
        self.pos = np.array(position, dtype=np.float32)
        self.coordinate = self.pos
        self.type = 'HAP'
        self.ant_num = 16  # 平面阵列，16根天线
        self.ant_type = 'UPA'


class EVA(object):
    def __init__(self, rng, x_range=(-200, -150), y_range=(-200, 150)):
        self.pos = rng.uniform(low=[x_range[0], y_range[0]],
                               high=[x_range[1], y_range[1]]).astype(np.float32)
        self.coordinate = self.pos
        self.type = 'EVA'
        self.ant_num = 1
        self.ant_type = 'Single'


class User(object):
    def __init__(self, rng, region='first'):
        if region == 'first':
            pos = rng.uniform([200, 200], [300, 300])
        elif region == 'second':
            pos = rng.uniform([-300, 200], [-200, 300])
        elif region == 'third':
            pos = rng.uniform([-300, -300], [-200, -200])
        elif region == 'fourth':
            pos = rng.uniform([200, -300], [300, -200])
        else:
            raise ValueError("region must be one of 'first', 'second', 'third', 'fourth'")

        self.pos = pos.astype(np.float32)
        self.coordinate = self.pos
        self.ant_num = 1
        self.ant_type = 'Single'
        self.type = 'User'

    def reset(self, rng, region):
        if region == 'first':
            self.pos = rng.uniform([200, 200], [300, 300]).astype(np.float32)
        elif region == 'second':
            self.pos = rng.uniform([-300, 200], [-200, 300]).astype(np.float32)
        elif region == 'third':
            self.pos = rng.uniform([-300, -300], [-200, -200]).astype(np.float32)
        elif region == 'fourth':
            self.pos = rng.uniform([200, -300], [300, -200]).astype(np.float32)
        self.coordinate = self.pos