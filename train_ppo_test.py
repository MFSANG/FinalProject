import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env_ppo_test import UAVSecureEnv

def test_secrecy_rate_curve(model, env, episode_len=60):
    obs, _ = env.reset()
    secrecy_rates = []
    done = False
    step = 0

    while not done and step < episode_len:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)

        # Extract secrecy rate from reward calculation (copy same logic)
        dist_h = np.linalg.norm(env.hap_pos - env.uav_pos) + 1e-6
        dist_e = np.linalg.norm(env.eve_pos - env.uav_pos) + 1e-6
        K = env.rician_K
        scatter_h = np.sqrt(0.5) * (np.random.normal() + 1j * np.random.normal())
        scatter_e = np.sqrt(0.5) * (np.random.normal() + 1j * np.random.normal())
        h_main = np.sqrt(env.beta0 * dist_h ** (-env.pathloss_exp)) * (
                np.sqrt(K / (K + 1)) + np.sqrt(1 / (K + 1)) * scatter_h)
        h_eve = np.sqrt(env.beta0 * dist_e ** (-env.pathloss_exp)) * (
                np.sqrt(K / (K + 1)) + np.sqrt(1 / (K + 1)) * scatter_e)
        gain_main = abs(h_main) ** 2
        gain_eve = abs(h_eve) ** 2
        P_tx = float(action[2]) * env.max_power
        snr_main = (P_tx * gain_main) / env.sigma2
        snr_eve = (P_tx * gain_eve) / env.sigma2
        secrecy_rate = max(np.log2(1 + snr_main) - np.log2(1 + snr_eve), 0.0)

        secrecy_rates.append(secrecy_rate)
        step += 1

    # Plotting
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
    # 1. Create an instance of the UAV environment
    env = UAVSecureEnv(num_waypoints=2)

    # 2. Initialize the PPO model with a multilayer perceptron policy
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=7e-4, gamma=0.98)

    # 3. Train the model with the specified number of timesteps
    total_timesteps = 2_000_000  # You can adjust this depending on task complexity
    model.learn(total_timesteps=total_timesteps)

    # Test and plot secrecy rate after training
    test_secrecy_rate_curve(model, env)

    # 4. Test the trained model: run one episode and render the trajectory
    obs, _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.render()

if __name__ == '__main__':
    main()
