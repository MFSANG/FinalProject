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
