import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from Env import UAVSecureEnv

def test_secrecy_rate_curve(model, env, episode_len=60):
    obs, _ = env.reset()
    secrecy_rates = []
    done = False
    step = 0

    while not done and step < episode_len:
        action, _ = model.predict(obs, deterministic=True)

        # Flatten action to ensure consistent shape
        action = np.array(action).flatten()
        if len(action) < 3:
            raise ValueError(f"Expected action of length >= 3, got: {action}")
        power_ratio = float(action[2])
        P_tx = power_ratio * env.max_power

        obs, reward, done, _, _ = env.step(action)

        dist_h = np.linalg.norm(env.hap.pos - env.uav.pos) + 1e-6
        dist_e = np.linalg.norm(env.eve.pos - env.uav.pos) + 1e-6
        K = env.K
        scatter_h = np.sqrt(0.5) * (np.random.normal() + 1j * np.random.normal())
        scatter_e = np.sqrt(0.5) * (np.random.normal() + 1j * np.random.normal())
        h_main = np.sqrt(env.beta0 * dist_h ** (-env.pathloss_exp)) * (
                np.sqrt(K / (K + 1)) + np.sqrt(1 / (K + 1)) * scatter_h)
        h_eve = np.sqrt(env.beta0 * dist_e ** (-env.pathloss_exp)) * (
                np.sqrt(K / (K + 1)) + np.sqrt(1 / (K + 1)) * scatter_e)
        gain_main = abs(h_main) ** 2
        gain_eve = abs(h_eve) ** 2
        snr_main = (P_tx * gain_main) / env.sigma2
        snr_eve = (P_tx * gain_eve) / env.sigma2
        secrecy_rate = max(np.log2(1 + snr_main) - np.log2(1 + snr_eve), 0.0)


        secrecy_rates.append(secrecy_rate)
        step += 1

    plt.figure(figsize=(8, 5))
    plt.plot(secrecy_rates, marker='o', linewidth=2, label='Secrecy Rate')
    plt.xlabel('Time Step')
    plt.ylabel('Secrecy Rate (bps/Hz)')
    plt.title('Secrecy Rate per Step')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    env = UAVSecureEnv(num_waypoints=2)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=7e-4, gamma=0.98)
    total_timesteps = 100_000
    model.learn(total_timesteps=total_timesteps)
    test_secrecy_rate_curve(model, env)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.render()

if __name__ == '__main__':
    main()
