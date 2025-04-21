import matplotlib.pyplot as plt
import numpy as np

class Render(object):
    def __init__(self):
        pass

    def render(self, env):
        traj = np.array(env.trajectory)
        current_wp_idx = min(env.current_wp_idx, env.num_waypoints)

        fig, ax1 = plt.subplots(figsize=(8, 6))  # 修改为单图

        # === UAV轨迹绘图 ===
        ax1.plot(traj[:, 0], traj[:, 1], '-o', color='black', linewidth=1.5, markersize=4, label='UAV Trajectory')
        ax1.plot(traj[-1, 0], traj[-1, 1], 'o', color='red', markersize=8, label='Current UAV Position')
        if len(traj) >= 2:
            dx, dy = traj[-1] - traj[-2]
            ax1.arrow(traj[-2, 0], traj[-2, 1], dx, dy, head_width=5, head_length=8, fc='red', ec='red')

        ax1.scatter(env.hap.pos[0], env.hap.pos[1], marker='D', s=100, color='blue', label='HAP')
        ax1.scatter(env.eve.pos[0], env.eve.pos[1], marker='D', s=100, color='purple', label='Eve')

        user_colors = ['orange', 'orange', 'orange', 'orange']
        for i, user in enumerate([env.user_1, env.user_2, env.user_3, env.user_4]):
            ax1.scatter(user.pos[0], user.pos[1], marker='s', s=50, color=user_colors[i], label=f'User {i + 1}')
            ax1.text(user.pos[0] + 5, user.pos[1] + 5, f'User {i + 1}', fontsize=10, color=user_colors[i])

        fence_x = [env.x_min, env.x_max, env.x_max, env.x_min, env.x_min]
        fence_y = [env.y_min, env.y_min, env.y_max, env.y_max, env.y_min]
        ax1.plot(fence_x, fence_y, '--', color='gray', linewidth=1.2, label='Geofence')

        for i, wp in enumerate(env.waypoints):
            marker_color = 'green' if env.visited[i] else 'orange'
            edge_color = 'red' if i == current_wp_idx else 'black'
            ax1.scatter(wp[0], wp[1], s=80, color=marker_color,
                        edgecolors=edge_color, linewidths=1.5)
            ax1.text(wp[0] + 5, wp[1] + 5, f'WP{i + 1}', fontsize=10, color='black')

        ax1.set_xlabel(r'$X$ (m)', fontsize=12)
        ax1.set_ylabel(r'$Y$ (m)', fontsize=12)
        ax1.set_title('UAV Trajectory Map', fontsize=13)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.axis('equal')
        ax1.legend(fontsize=8)

        plt.tight_layout()
        plt.show()
