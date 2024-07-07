import math
import numpy as np
import torch.nn as nn
import torch


def gaussian_probability(mu, sigma, rho, data):
    """
    :param mu: 均值 [B, N, K, 3]
    :param sigma: 标准差 [B, N, K, 3]
    :param rho: 相关系数 [B, N, K, 3]
    :param data: 坐标点 [B, N, 3, m]
    :return: 坐标点 在高斯模型中的概率
    """
    data = data.to(mu.device)
    mean_x, mean_y, mean_z = torch.chunk(mu, 3, dim=-1)
    std_x, std_y, std_z = torch.chunk(sigma, 3, dim=-1)
    rho_xy, rho_xz, rho_yz = torch.chunk(rho, 3, dim=-1)

    # test
    eps = 1e-8  # 尽量避免分母为0

    x, y, z = torch.chunk(data, 3, dim=2)
    dx = x - mean_x
    dy = y - mean_y
    dz = z - mean_z

    std_xy = std_x * std_y
    std_xz = std_x * std_z
    std_yz = std_y * std_z

    Q = (dx * dx) / (std_x * std_x) + (dy * dy) / (std_y * std_y) + (dz * dz) / (std_z * std_z) \
        - (2 * rho_xy * dx * dy) / std_xy - (2 * rho_xz * dx * dz) / std_xz - (2 * rho_yz * dy * dz) / std_yz
    # print(f"Q-------max: {Q.max()}---min: {Q.min()} ---sum: {Q.sum()} ---all: {Q}")
    P = 1 - torch.pow(rho_xy, 2) - torch.pow(rho_xz, 2) - torch.pow(rho_yz, 2) + 2 * rho_xy * rho_xz * rho_yz
    # print(f"P--------max: {P.max()}---min: {P.min()} ---sum: {P.sum()} ---all: {P}")
    norm = 1 / ((2 * math.pi) ** 1.5 * std_x * std_y * std_z * torch.sqrt(P))
    # print(f"Norm-------max: {norm.max()}---min: {norm.min()} ---sum: {norm.sum()} ---all: {norm}")
    res = norm * torch.exp(0.5 * -Q / (P))
    # print(f"res --------max: {res.max()}---min: {res.min()} ---sum: {res.sum()} ---all: {res}")
    return res


def mixture_probability(pi, mu, sigma, rho, data):
    """
    混合概率密度：
    :param pi:
    :param mu:
    :param sigma:
    :param rho:
    :param data:
    :return:
    """
    pi = pi.unsqueeze(-1)  # 【B, N, K, 1】
    prob = pi * gaussian_probability(mu, sigma, rho, data)
    prob = torch.sum(prob, dim=2)
    return prob


class MDN3D(nn.Module):
    def __init__(self, input_dim, MDN_hidden_num, num_gaussians, action_map_size):
        super(MDN3D, self).__init__()
        self.input_dim = input_dim
        self.num_gaussians = num_gaussians  # 混合网络个数

        self.pi = nn.Sequential(
            nn.Linear(self.input_dim, MDN_hidden_num),
            nn.ReLU(),
            nn.Linear(MDN_hidden_num, self.num_gaussians),
            nn.Softmax(dim=-1)
        )
        self.mu = nn.Sequential(
            nn.Linear(self.input_dim, MDN_hidden_num),
            nn.ReLU(),
            nn.Linear(MDN_hidden_num, 3 * self.num_gaussians)
        )
        self.std = nn.Sequential(
            nn.Linear(self.input_dim, MDN_hidden_num),
            nn.ReLU(),
            nn.Linear(MDN_hidden_num, 3 * self.num_gaussians)
        )
        self.rho = nn.Sequential(
            nn.Linear(self.input_dim, MDN_hidden_num),
            nn.ReLU(),
            nn.Linear(MDN_hidden_num, 3 * self.num_gaussians)
        )
        self.mu[-1].bias.data.copy_(torch.rand_like(self.mu[-1].bias))

        self.action_map_size = action_map_size

        self.xyz_t = self.cal_forward_vector(self.action_map_size[0], self.action_map_size[1])

    def forward(self, x):
        pi = self.pi(x)  # k组 高斯分布 权重 [B, N, K]
        mu = torch.tanh((self.mu(x)))

        # # 计算L2范数，指定dim=-1表示最后一个维度
        # norm = torch.linalg.norm(mu, dim=-1, keepdim=True)
        # # 归一化
        # mu = mu / norm

        sigma = torch.exp(self.std(x))
        # print(sigma.mean(0))
        # sigma = torch.clamp(sigma, 0.06, 10)
        rho = torch.tanh((self.rho(x)))

        # rho = torch.clamp(self.rho(x), -0.25, 0.25)
        mu = mu.reshape(-1, mu.size(1), self.num_gaussians, 3)  # 均值 [B, N, K, 3]
        sigma = sigma.reshape(-1, sigma.size(1), self.num_gaussians, 3)  # 方差 [B, N, K, 3]
        rho = rho.reshape(-1, rho.size(1), self.num_gaussians, 3)  # 相关系数 [B, N, K, 1]

        return pi, mu, sigma, rho

    def cal_forward_vector(self, LABEL_HEI=128, LABEL_WID=256):
        img_wid, img_hei = LABEL_WID, LABEL_HEI

        x_range = np.arange(-img_wid / 2, img_wid / 2)
        y_range = np.arange(-img_hei / 2, img_hei / 2)

        tx, ty = np.meshgrid(x_range, y_range)
        lon = tx.astype(float) / img_wid * 2 * np.pi
        lat = ty.astype(float) / img_hei * np.pi

        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)

        x = torch.from_numpy(x).float().reshape(1, 1, -1)
        y = torch.from_numpy(y).float().reshape(1, 1, -1)
        z = torch.from_numpy(z).float().reshape(1, 1, -1)

        xyz_t = torch.cat([x, y, z], dim=1)
        return xyz_t

    def mixture_probability_map(self, pi, mu, sigma, rho):
        """
        :param pi:
        :param mu:
        :param sigma:
        :param rho:
        :return:
        """
        pi = pi.unsqueeze(-1)
        prob = pi * gaussian_probability(mu, sigma, rho, self.xyz_t.unsqueeze(0))
        prob = torch.sum(prob, dim=2)
        return prob

    def sample_prob(self, pis, mus, sigmas, rhos):
        """
        :param pis:
        :param mus:
        :param sigmas:
        :param rhos:
        :param length: 扫视路径长度
        :return:
        """
        B, L, K, _ = mus.shape

        pred_roi_maps = self.mixture_probability_map(pis, mus, sigmas, rhos).view(B * L, -1)
        sampled_indices = torch.multinomial(pred_roi_maps, num_samples=1, replacement=True)
        outputs = self.xyz_t[:, :, sampled_indices.view(-1)][0].permute(1, 0)
        outputs = outputs.view(B, L, -1)
        return outputs
