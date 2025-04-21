import numpy as np
import math


def dB_to_normal(db_val):
    return 10 ** (db_val / 10)

def normal_to_dB(normal_val):
    return 10 * np.log10(normal_val)

def generate_rician_channel(tx_pos, rx_pos, beta0, pathloss_exp, K=6.0):
    dist = np.linalg.norm(tx_pos - rx_pos) + 1e-6
    scatter = np.sqrt(0.5) * (np.random.normal() + 1j * np.random.normal())
    h = np.sqrt(beta0 * dist ** (-pathloss_exp)) * (
        np.sqrt(K / (K + 1)) + np.sqrt(1 / (K + 1)) * scatter)
    return h

def get_channel_type_and_params(tx_type, rx_type):
    ch_type = f"{tx_type}_{rx_type}"
    if ch_type in ['UAV_HAP', 'HAP_UAV']:
        return 2.2, 3
    elif ch_type in ['UAV_EVA', 'EVA_UAV']:
        return 3.5, 3
    elif ch_type in ['UAV_User', 'User_UAV']:
        return 3.5, 3
    elif ch_type in ['HAP_User', 'User_HAP']:
        return 2.8, 3
    elif ch_type in ['HAP_EVA', 'EVA_HAP']:
        return 2.8, 3
    else:
        return 3.0, 3  # default

class SimpleCommEntity:
    def __init__(self, coordinate, entity_type='UAV', ant_type='Single', ant_num=1):
        self.coordinate = np.array(coordinate, dtype=np.float32)
        self.type = entity_type
        self.ant_type = ant_type  # 'Single', 'ULA', 'UPA'
        self.ant_num = ant_num

def generate_array_response(entity: SimpleCommEntity, theta=0.0, phi=0.0):
    N = entity.ant_num
    ant_type = entity.ant_type
    if ant_type == 'Single':
        return np.array([1], dtype=complex).reshape(-1, 1)
    elif ant_type == 'ULA':
        return np.array([
            np.exp(1j * np.pi * n * np.sin(theta)) for n in range(N)
        ], dtype=complex).reshape(-1, 1)
    elif ant_type == 'UPA':
        side = int(np.sqrt(N))
        assert side * side == N, "UPA antenna number must be a perfect square"
        return np.array([
            np.exp(1j * np.pi * (i * np.sin(theta) * np.cos(phi) + j * np.sin(theta) * np.sin(phi)))
            for i in range(side) for j in range(side)
        ], dtype=complex).reshape(-1, 1)
    else:
        raise ValueError("Unknown antenna type")

def mmwave_single_channel(tx: SimpleCommEntity,
                          rx: SimpleCommEntity,
                          frequency=28e9,
                          beta0=1e4,
                          K=6.0):
    pathloss_exp, sigma = get_channel_type_and_params(tx.type, rx.type)
    d = np.linalg.norm(tx.coordinate - rx.coordinate)
    path_loss_dB = -20 * math.log10(4 * math.pi / (3e8 / frequency)) - 10 * pathloss_exp * math.log10(d)
    shadow_fading = np.random.normal(loc=0.0, scale=sigma)
    PL = dB_to_normal(path_loss_dB + shadow_fading)

    # 基础复高斯随机变量模拟 Rician channel
    scatter = (np.random.randn(rx.ant_num, tx.ant_num) + 1j * np.random.randn(rx.ant_num, tx.ant_num)) / np.sqrt(2)
    LOS = np.ones((tx.ant_num, rx.ant_num), dtype=complex)

    # 阵列响应向量（若为 ULA 或 UPA 则计算）
    at = generate_array_response(tx)
    ar = generate_array_response(rx)

    H = np.sqrt(PL) * (
            np.sqrt(K / (K + 1)) * (ar @ at.conj().T) +
            np.sqrt(1 / (K + 1)) * scatter
    )

    # 返回任意天线对的平均增益（可用于标量评价）
    return H, np.mean(np.abs(H) ** 2)
