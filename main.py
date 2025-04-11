import numpy as np
import torch
import torch.optim as optim
import torch.distributions as D
import matplotlib.pyplot as plt
import torch.nn as nn

from env import UAVSecureEnv
from model import ActorCritic

def get_exploration_scale(iteration, max_iter, init_scale=2.5, final_scale=2.0):
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
    optimizer = optim.Adam(model.parameters(), lr=7e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    gamma = 0.98
    lam = 0.95
    clip_eps = 0.1
    train_iters = 2000
    episodes_per_iter = 30
    reward_history = []

    all_positions = []  # 记录全部轨迹点

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

    plt.figure()
    plt.plot(reward_history)
    plt.xlabel('Iteration')
    plt.ylabel('Average Reward')
    plt.title('Training Reward Curve')
    plt.grid(True)
    plt.savefig('reward_curve.png')
    plt.show()

    # === UAV 热力图 ===
    all_positions_np = np.array(all_positions)
    x_vals = all_positions_np[:, 0]
    y_vals = all_positions_np[:, 1]

    plt.figure(figsize=(8, 6))
    heatmap = plt.hist2d(x_vals, y_vals, bins=100, cmap='hot')
    plt.colorbar(label='Visit Frequency')

    # 绘制所有航路点
    for wp in env.waypoints:
        plt.scatter(wp[0], wp[1], s=100, c='cyan', edgecolors='black', label='Waypoint')

    # 绘制最后一次飞行的轨迹
    traj = np.array(env.trajectory)
    plt.plot(traj[:, 0], traj[:, 1], color='lime', linewidth=2, label='Last Trajectory')

    plt.title('UAV Trajectory Heatmap + Waypoints')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('trajectory_heatmap.png')
    plt.show()
