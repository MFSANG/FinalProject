import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor_mean = nn.Linear(hidden_size, action_dim)
        self.actor_log_std = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x, exploration_scale=1.5):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.actor_mean(x)
        base_log_std = self.actor_log_std(x)

        # 将 exploration_scale 转为 tensor，并加上一个小常数，避免出现 log(0)
        scale_tensor = torch.tensor(exploration_scale, dtype=base_log_std.dtype, device=base_log_std.device) + 1e-6

        # 计算 log_std 并裁剪，避免值过大或过小
        log_std = base_log_std + torch.log(scale_tensor)
        log_std = torch.clamp(log_std, min=-20, max=2)

        return mean, log_std, self.critic(x)

