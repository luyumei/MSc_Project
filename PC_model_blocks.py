import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

# def timeit(tag, t):
#     print("{}: {}s".format(tag, time() - t))
#     return time()

# def pc_normalize(pc):
#     l = pc.shape[0]
#     centroid = np.mean(pc, axis=0)
#     pc = pc - centroid
#     m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
#     pc = pc / m
#     return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points. 用于计算两组点之间欧氏距离的函数

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input: 函数输入是两个点集合src和dst，它们都是三维张量（tensor）
        src: source points, [B, N, C] 是源点集，B代表批量大小（Batch Size），N代表每个批次中的源点数量，C代表每个点的坐标维度（通常是2维或3维，比如C=2表示2D点，C=3表示3D点）
        dst: target points, [B, M, C] 是目标点集，M代表每个批次中的目标点数量，C表示目标点的坐标维度（通常与src相同）
    Output:
        dist: per-point square distance, [B, N, M]  每个源点到每个目标点的「平方」欧氏距离矩阵，维度为[B, N, M]。在这个矩阵中，dist[b, i, j]代表第b个批次中的第i个源点到第j个目标点的平方欧氏距离
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1)) # 计算src和dst的点积：这一步得到的是一个[B, N, M]的矩阵，矩阵中的每个元素dist[b, i, j]等于源点src[b, i]与目标点dst[b, j]的点积
    dist += torch.sum(src ** 2, -1).view(B, N, 1)   # 计算src中每个点的平方范数：这一步得到的是一个[B, N, 1]的矩阵，矩阵中的每个元素是源点src[b, i]的平方范数，并将平方范数项加到点积矩阵上
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)   # 计算dst中每个点的平方范数：这一步得到的是一个[B, 1, M]的矩阵，矩阵中的每个元素是目标点dst[b, j]的平方范数，并将平方范数项加到点积矩阵上
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    最远点采样是一种常用的点云采样方法，旨在从原始点云中选择一部分点，以便这些点能够尽可能代表整个点云的形状
    Input:
        xyz: pointcloud data, [B, N, 3] 三维点云数据，形状为 [B, N, 3]，其中 B 表示批次大小 (batch size)，N 表示每个点云中点的数量，3 表示每个点的三个坐标 (x, y, z)/坐标维度？
        npoint: number of samples 采样后的点的数量，即要从原始点云中选择多少个点作为采样结果
    Return:
        centroids: sampled pointcloud index, [B, npoint] 存储了每个点云中选取的 npoint 个最远点的索引，形状为 [B, npoint]，表示对于每个点云，选择了哪些点作为采样结果。这些点可以作为点云的简化版本，用于加快点云处理或其他任务，同时尽可能保留原始点云的形状特征
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)# 初始化 centroids 为零矩阵，用于存储最终的采样点的索引
    distance = torch.ones(B, N).to(device) * 1e10# 始化 distance 为一个很大的数 (1e10)，用于保存每个点距离采样点的最小距离
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)# 随机选择一个点作为初始采样点，并存储在 farthest 变量中
    batch_indices = torch.arange(B, dtype=torch.long).to(device)# 保存每个点云的索引，从0到 B-1
    # 通过迭代 npoint 次来进行最远点采样：
    for i in range(npoint):
        centroids[:, i] = farthest# 将当前最远点的索引存储在 centroids 中
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)# 获取当前最远点的坐标，并扩展维度以便进行广播运算
        dist = torch.sum((xyz - centroid) ** 2, -1)# 计算每个点与当前最远点的距离，并保存在 dist 中
        mask = dist < distance# 如果 dist 中的元素小于 distance，则表达式的结果为True，否则为False
        distance[mask] = dist[mask]# 使用距离信息更新 distance，将每个点与最近采样点的距离保存在 distance 中
        farthest = torch.max(distance, -1)[1]# 选择距离最远的点作为下一个采样点，并更新 farthest 变量
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    根据给定的查询点和点云数据，将每个查询点的局部区域内的最近点索引进行分组，并返回这些索引
    Input:
        radius: local region radius 局部区域的半径，即在该半径内寻找最近的点
        nsample: max sample number in local region 每个查询点局部区域内最多的点的数量
        xyz: all points, [B, N, 3] 所有的点云数据，维度为 [B, N, 3]，其中 B 为批次大小（batch size），N 为点云中点的数量，3 代表每个点的三维坐标（x, y, z）
        new_xyz: query points, [B, S, 3] 查询点，维度为 [B, S, 3]，其中 B 为批次大小，S 为查询点的数量，3 代表每个查询点的三维坐标
    Return:
        group_idx: grouped points index, [B, S, nsample] 代表每个查询点局部区域内最近的 nsample 个点的索引。在输出 group_idx 中，对于每个查询点，最多有 nsample 个最近的点的索引被保留，其他的点会被舍弃，如果局部区域内找不到足够的点，将用相同的点进行填充。
    """
    device = xyz.device # 从 xyz 中获取设备信息，用于在 GPU 或 CPU 上进行运算
    B, N, C = xyz.shape # 获取批次大小（B）、点的数量（N）
    _, S, _ = new_xyz.shape # 获取每个查询点的数量（S）
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])    # 创建一个索引张量 group_idx，初始值为从 0 到 N-1，表示每个查询点对应的局部区域内的点的索引
    sqrdists = square_distance(new_xyz, xyz)    # 计算查询点和点云中每个点之间的欧氏距离的平方 sqrdists，这是一个维度为 [B, S, N] 的张量
    group_idx[sqrdists > radius ** 2] = N   # 将距离大于 radius 的点在 group_idx 中标记为 N，意味着这些点不在局部区域内，将在后续处理中被舍弃
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]   # 对 group_idx 进行排序，按照距离从小到大排列。注意，这里是对每个查询点的局部区域内的点按照距离排序
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])  # 保留每个查询点的最近 nsample 个点的索引，其余的点被舍弃，得到 group_idx 维度为 [B, S, nsample]
    mask = group_idx == N   # 创建一个掩码 mask，用于标记那些局部区域内找不到足够点的查询点
    group_idx[mask] = group_first[mask] # 对于那些局部区域内找不到足够点的查询点，将其索引设置为局部区域内的第一个点的索引，以填充 group_idx
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx# 最远点采样的索引
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    将输入的点云数据作为一个整体，直接将所有点作为一个局部区域进行分组；不能指定采样点的数量；不要用
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class SetAbstraction_1_branch(nn.Module):
    """
    可以选择对所有输入点进行全局区域采样或者对部分点进行局部采样，仅使用一个简单的多层卷积结构进行特征处理
    """
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        """
        :param npoint:同·采样后的点的数量
        :param radius:一个 标量，同·表示局部区域查询的球形半径
        :param nsample:同·每个局部区域中采样的点的数量
        :param in_channel:同·输入的点云的（特征的？）通道数
        :param mlp:是一个列表，其中包含了一系列整数值，表示每层MLP（多层感知器）/每个卷积层？的输出通道数。每个MLP层由一个1x1卷积和BatchNorm组成。
        :param group_all:不同·一个布尔值，如果为True，则对所有输入点（云）执行下采样（局部区域采样？），忽略npoint和radius参数。它使得整个函数可以选择对所有输入的点云进行局部区域采样，也可以选择只对一部分点进行采样，这取决于 group_all 参数？
        """
        super(SetAbstraction_1_branch, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        # 同·两个 nn.ModuleList，用于存储模型中的卷积层和批归一化层
        # 然后，在采样得到的局部区域内，通过卷积层和批归一化层对局部点云特征进行处理和聚合，得到新的点云位置数据 new_xyz 和新的点云特征数据 new_points
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N] 点云的位置数据，维度为[B, C, N]，其中B表示批量大小，C表示每个点的坐标维度（通常为3维），N表示输入点的数量
            points: input points data, [B, D, N] 点云的特征数据，维度为[B, D, N]，其中D表示每个点的特征维度
        Return: 两个tensor:
            new_xyz: sampled points position data, [B, C, S] 采样后的新点云位置数据，形状为 [B, C, S]，其中 S 是采样后的点的数量
            new_points_concat: sample points feature data, [B, D', S] 不同·采样后的新点云特征数据，形状为 [B, C+D, S]，其中 C+D 是特征数据的维度｜｜维度是特征数据维度 C+D
        """
        # 首先，将xyz和points的维度转换，将点的维度（N）放在最后，因为后续操作中需要这种维度顺序。
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        # 如果group_all为True，那么将对所有输入点执行下采样，得到新的采样点的位置数据new_xyz和采样点的特征数据new_points。
        # 如果group_all为False，将采用采样和分组的方法，通过调用sample_and_group函数，对输入点进行下采样，得到新的采样点的位置数据new_xyz和采样点的特征数据new_points。
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # 对采样点的特征数据进行处理：
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint] # 首先，将new_points的维度重新排列，将特征维度（C+D）放在第二个位置，邻居点数量（nsample）放在第三个位置，采样点的数量（npoint）放在最后一个位置，以适应后续卷积操作
        # 通过一系列的MLP层对new_points进行特征提取和转换。每个MLP层包含一个1x1卷积和BatchNorm，并使用ReLU作为激活函数
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        # 在所有MLP层处理完后，通过在特征维度上取最大值操作（max pooling），将每个采样点的特征数据缩减为单个特征向量
        new_points = torch.max(new_points, 2)[0]
        # 将新的采样点位置数据new_xyz和缩减后的特征数据new_points的维度重新转换回来，并将它们作为输出返回
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class SetAbstraction_3_branches(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):# mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        super(SetAbstraction_3_branches, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()# Use ModuleList to store shared FC layers
        self.bn_blocks = nn.ModuleList()# Use ModuleList to store shared BatchNorm1d layers
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()# Shared FC layers for each block
            bns = nn.ModuleList()# Shared BatchNorm1d layers for each block
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:# e.g. [32, 32, 64]
                convs.append(nn.Conv2d(last_channel, out_channel, 1))# Replace Linear with Conv2d
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))# Apply FC, BN, and ReLU
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


# class PointNetFeaturePropagation(nn.Module):
#     def __init__(self, in_channel, mlp):
#         super(PointNetFeaturePropagation, self).__init__()
#         self.mlp_convs = nn.ModuleList()
#         self.mlp_bns = nn.ModuleList()
#         last_channel = in_channel
#         for out_channel in mlp:
#             self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
#             self.mlp_bns.append(nn.BatchNorm1d(out_channel))
#             last_channel = out_channel
#
#     def forward(self, xyz1, xyz2, points1, points2):
#         """
#         Input:
#             xyz1: input points position data, [B, C, N]
#             xyz2: sampled input points position data, [B, C, S]
#             points1: input points data, [B, D, N]
#             points2: input points data, [B, D, S]
#         Return:
#             new_points: upsampled points data, [B, D', N]
#         """
#         xyz1 = xyz1.permute(0, 2, 1)
#         xyz2 = xyz2.permute(0, 2, 1)
#
#         points2 = points2.permute(0, 2, 1)
#         B, N, C = xyz1.shape
#         _, S, _ = xyz2.shape
#
#         if S == 1:
#             interpolated_points = points2.repeat(1, N, 1)
#         else:
#             dists = square_distance(xyz1, xyz2)
#             dists, idx = dists.sort(dim=-1)
#             dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
#
#             dist_recip = 1.0 / (dists + 1e-8)
#             norm = torch.sum(dist_recip, dim=2, keepdim=True)
#             weight = dist_recip / norm
#             interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
#
#         if points1 is not None:
#             points1 = points1.permute(0, 2, 1)
#             new_points = torch.cat([points1, interpolated_points], dim=-1)
#         else:
#             new_points = interpolated_points
#
#         new_points = new_points.permute(0, 2, 1)
#         for i, conv in enumerate(self.mlp_convs):
#             bn = self.mlp_bns[i]
#             new_points = F.relu(bn(conv(new_points)))
#         return new_points

