import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TNet(nn.Module):
    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc_bn1 = nn.BatchNorm1d(512)
        self.fc_bn2 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, dim=2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = F.relu(self.fc_bn2(self.fc2(x)))
        x = self.fc3(x)
        identity = (
            torch.eye(self.k, dtype=x.dtype, device=x.device)
            .flatten()
            .unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
        )
        x = x + identity
        x = x.view(batch_size, self.k, self.k)
        return x


class PointNetSeg(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        use_normal: bool = False,
    ):
        super().__init__()
        actual_input = input_channels + (3 if use_normal else 0)
        self.input_tnet = TNet(k=actual_input)
        self.mlp1_conv1 = nn.Conv1d(actual_input, 64, 1)
        self.mlp1_conv2 = nn.Conv1d(64, 64, 1)
        self.mlp1_bn1 = nn.BatchNorm1d(64)
        self.mlp1_bn2 = nn.BatchNorm1d(64)
        self.feature_tnet = TNet(k=64)
        self.mlp2_conv1 = nn.Conv1d(64, 128, 1)
        self.mlp2_conv2 = nn.Conv1d(128, 1024, 1)
        self.mlp2_bn1 = nn.BatchNorm1d(128)
        self.mlp2_bn2 = nn.BatchNorm1d(1024)
        self.seg_conv1 = nn.Conv1d(1088, 512, 1)
        self.seg_conv2 = nn.Conv1d(512, 256, 1)
        self.seg_conv3 = nn.Conv1d(256, num_classes, 1)
        self.seg_bn1 = nn.BatchNorm1d(512)
        self.seg_bn2 = nn.BatchNorm1d(256)
        self.num_classes = num_classes

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, n_points, n_features = x.size()
        x_transpose = x.transpose(2, 1).contiguous()
        input_transform = self.input_tnet(x_transpose)
        x = torch.bmm(input_transform, x_transpose)
        x = F.relu(self.mlp1_bn1(self.mlp1_conv1(x)))
        x = F.relu(self.mlp1_bn2(self.mlp1_conv2(x)))
        feature_transform = self.feature_tnet(x)
        x = torch.bmm(feature_transform, x)
        local_features = x
        x = F.relu(self.mlp2_bn1(self.mlp2_conv1(x)))
        x = F.relu(self.mlp2_bn2(self.mlp2_conv2(x)))
        global_features = torch.max(x, dim=2, keepdim=True)[0]
        global_features_expanded = global_features.expand(
            -1, -1, n_points
        ).contiguous()
        combined = torch.cat([local_features, global_features_expanded], dim=1)
        x = F.relu(self.seg_bn1(self.seg_conv1(combined)))
        x = F.relu(self.seg_bn2(self.seg_conv2(x)))
        x = self.seg_conv3(x)
        x = x.transpose(2, 1).contiguous()
        return x, input_transform, feature_transform


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class PositionAttention(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.query_conv = nn.Conv1d(in_channels, out_channels // 4, 1)
        self.key_conv = nn.Conv1d(in_channels, out_channels // 4, 1)
        self.value_conv = nn.Conv1d(in_channels, out_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, xyz: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, N = x.shape
        proj_query = self.query_conv(x).view(B, -1, N).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, N)
        energy = torch.bmm(proj_query, proj_key)
        if xyz is not None:
            xyz_bcn = xyz.permute(0, 2, 1)
            xyz_norm = xyz_bcn / (xyz_bcn.norm(dim=1, keepdim=True) + 1e-8)
            pos_corr = torch.bmm(xyz_norm.permute(0, 2, 1), xyz_norm)
            energy = energy + 0.1 * pos_corr
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(B, -1, N)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        return self.gamma * out + x


class MultiScaleFusion(nn.Module):
    def __init__(self, channels_list: list, out_channels: int):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(ch, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            ) for ch in channels_list
        ])
        self.attention = nn.Sequential(
            nn.Conv1d(out_channels * len(channels_list), out_channels * len(channels_list) // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels * len(channels_list) // 4, len(channels_list), 1),
            nn.Softmax(dim=1),
        )
        self.fusion_conv = nn.Conv1d(out_channels, out_channels, 1)

    def forward(self, features: list) -> torch.Tensor:
        projected = [conv(feat) for conv, feat in zip(self.convs, features)]
        concat_feats = torch.cat(projected, dim=1)
        attn_weights = self.attention(concat_feats)
        weighted_sum = sum(w.unsqueeze(-1) * feat for w, feat in zip(attn_weights.chunk(len(features), dim=1), projected))
        return self.fusion_conv(weighted_sum)


class _SetAbstraction(nn.Module):
    def __init__(self, npoint: int, radius: float, nsample: int, in_channel: int, mlp: list):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_ch = in_channel
        for out_ch in mlp:
            self.mlp_convs.append(nn.Conv2d(last_ch, out_ch, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_ch))
            last_ch = out_ch

    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B, N, C = xyz.shape
        if self.npoint is not None and self.npoint < N:
            fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)
        elif self.npoint is not None:
            new_xyz = xyz[:, :self.npoint, :]
            fps_idx = torch.arange(self.npoint, device=xyz.device).unsqueeze(0).expand(B, -1)
        else:
            fps_idx = None
            new_xyz = xyz.mean(dim=1, keepdim=True)
        k = self.nsample if self.nsample is not None else N
        dist = square_distance(new_xyz, xyz)
        _, idx = dist.topk(min(k, N), dim=-1, largest=False)
        idx = idx.clamp(max=N - 1)
        grouped_xyz = index_points(xyz, idx).permute(0, 3, 1, 2)
        relative_xyz = grouped_xyz - new_xyz.transpose(1, 2).unsqueeze(-1)
        if points is not None:
            if points.data_ptr() == xyz.data_ptr():
                grouped_pts = relative_xyz
            else:
                grouped_pts_feat = index_points(points, idx).permute(0, 3, 1, 2)
                grouped_pts = torch.cat([relative_xyz, grouped_pts_feat], dim=1)
        else:
            grouped_pts = grouped_xyz
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            grouped_pts = F.relu(bn(conv(grouped_pts)))
        new_points = torch.max(grouped_pts, dim=-1)[0]
        new_points = new_points.permute(0, 2, 1).contiguous()
        return new_xyz, new_points, fps_idx


class _FeaturePropagation(nn.Module):
    def __init__(self, in_channel: int, mlp: list):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_ch = in_channel
        for out_ch in mlp:
            self.mlp_convs.append(nn.Conv1d(last_ch, out_ch, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_ch))
            last_ch = out_ch

    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor, points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        dist, idx = three_nn(xyz1, xyz2)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=-1, keepdim=True)
        weight = dist_recip / norm
        interpolated_pts = torch.sum(index_points(points2, idx) * weight.unsqueeze(-1), dim=-2)
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_pts], dim=-1)
        else:
            new_points = interpolated_pts
        new_points = new_points.permute(0, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        return new_points.permute(0, 2, 1)


class _SetAbstractionWithAttention(nn.Module):
    def __init__(self, npoint: int, radius: float, nsample: int, in_channel: int, mlp: list, use_attention: bool = True):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.use_attention = use_attention
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_ch = in_channel
        for out_ch in mlp:
            self.mlp_convs.append(nn.Conv2d(last_ch, out_ch, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_ch))
            last_ch = out_ch
        if use_attention and len(mlp) > 0:
            self.channel_attn = ChannelAttention(mlp[-1], reduction=8)
        else:
            self.channel_attn = None

    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B, N, C = xyz.shape
        if self.npoint is not None and self.npoint < N:
            fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)
        elif self.npoint is not None:
            new_xyz = xyz[:, :self.npoint, :]
            fps_idx = torch.arange(self.npoint, device=xyz.device).unsqueeze(0).expand(B, -1)
        else:
            fps_idx = None
            new_xyz = xyz.mean(dim=1, keepdim=True)
        k = self.nsample if self.nsample is not None else N
        dist = square_distance(new_xyz, xyz)
        _, idx = dist.topk(min(k, N), dim=-1, largest=False)
        idx = idx.clamp(max=N - 1)
        grouped_xyz = index_points(xyz, idx).permute(0, 3, 1, 2)
        relative_xyz = grouped_xyz - new_xyz.transpose(1, 2).unsqueeze(-1)
        if points is not None:
            if points.data_ptr() == xyz.data_ptr():
                grouped_pts = relative_xyz
            else:
                grouped_pts_feat = index_points(points, idx).permute(0, 3, 1, 2)
                grouped_pts = torch.cat([relative_xyz, grouped_pts_feat], dim=1)
        else:
            grouped_pts = grouped_xyz
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            grouped_pts = F.relu(bn(conv(grouped_pts)))
        new_points = torch.max(grouped_pts, dim=-1)[0]
        new_points = new_points.permute(0, 2, 1).contiguous()
        if self.channel_attn is not None:
            new_points_bnc = new_points
            new_points_bcn = new_points.permute(0, 2, 1).contiguous()
            new_points_bcn = self.channel_attn(new_points_bcn)
            new_points = new_points_bcn.permute(0, 2, 1).contiguous()
        return new_xyz, new_points, fps_idx


class _FeaturePropagationWithAttention(nn.Module):
    def __init__(self, in_channel: int, mlp: list, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_ch = in_channel
        for out_ch in mlp:
            self.mlp_convs.append(nn.Conv1d(last_ch, out_ch, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_ch))
            last_ch = out_ch
        if use_attention and len(mlp) > 0:
            self.channel_attn = ChannelAttention(mlp[-1], reduction=8)
        else:
            self.channel_attn = None

    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor, points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        dist, idx = three_nn(xyz1, xyz2)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=-1, keepdim=True)
        weight = dist_recip / norm
        interpolated_pts = torch.sum(index_points(points2, idx) * weight.unsqueeze(-1), dim=-2)
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_pts], dim=-1)
        else:
            new_points = interpolated_pts
        new_points = new_points.permute(0, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        if self.channel_attn is not None:
            new_points = self.channel_attn(new_points)
        return new_points.permute(0, 2, 1)


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    B = points.shape[0]
    N = points.shape[1]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(points.device).view(view_shape).repeat(repeat_shape)
    idx_safe = idx.clamp(min=0, max=N - 1)
    new_points = points[batch_indices, idx_safe, :]
    return new_points


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    B, N, C = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.full((B, N), float('inf'), device=device)
    farthest = torch.tensor([N // 2], dtype=torch.long, device=device).expand(B)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        farthest = farthest.clamp(0, N - 1)
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest.clamp(0, N - 1), :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance, -1)
    return centroids.clamp(0, N - 1)


def three_nn(xyz1: torch.Tensor, xyz2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dist = square_distance(xyz1, xyz2)
    dist, idx = dist.sort(dim=-1)
    dist[:, :, 3:] = 0
    return dist[:, :, :3].contiguous(), idx[:, :, :3].contiguous()


class PointNetPlusPlusSeg(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        use_normal: bool = False,
    ):
        super().__init__()
        self.use_normal = use_normal
        additional_input = 6 if use_normal else 0
        in_ch = input_channels + additional_input
        self.sa1 = _SetAbstraction(512, 0.2, 32, in_ch, [64, 64, 128])
        self.sa2 = _SetAbstraction(128, 0.4, 64, 3 + 128, [128, 128, 256])
        self.sa3 = _SetAbstraction(None, None, None, 3 + 256, [256, 512, 1024])
        self.fp3 = _FeaturePropagation(256 + 1024, [256, 256])
        self.fp2 = _FeaturePropagation(128 + 256, [256, 128])
        self.fp1 = _FeaturePropagation(in_ch + 128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, C_in = x.shape
        identity_matrix = torch.eye(3, dtype=x.dtype, device=x.device).unsqueeze(0).expand(B, -1, -1)
        l0_xyz = x
        l0_points = x
        l1_xyz, l1_points, _ = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points, _ = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points, _ = self.sa3(l2_xyz, l2_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)
        feat = l0_points.permute(0, 2, 1)
        feat = F.relu(self.bn1(self.conv1(feat)))
        feat = self.drop1(feat)
        out = self.conv2(feat)
        out = out.permute(0, 2, 1).contiguous()
        return out, identity_matrix, identity_matrix


class PointNetPlusPlusAttentionSeg(nn.Module):
    def __init__(
        self,
        input_channels: int = 6,
        num_classes: int = 2,
        use_normal: bool = False,
    ):
        super().__init__()
        self.use_normal = use_normal
        additional_input = 6 if use_normal else 0
        in_ch = input_channels + additional_input
        self.sa1 = _SetAbstractionWithAttention(512, 0.2, 32, 3 + in_ch, [64, 64, 128], use_attention=True)
        self.sa2 = _SetAbstractionWithAttention(128, 0.4, 64, 3 + 128, [128, 128, 256], use_attention=True)
        self.sa3 = _SetAbstractionWithAttention(None, None, None, 3 + 256, [256, 512, 1024], use_attention=True)
        self.fp3 = _FeaturePropagationWithAttention(256 + 1024, [256, 256], use_attention=True)
        self.fp2 = _FeaturePropagationWithAttention(128 + 256, [256, 256, 128], use_attention=True)
        self.fp1 = _FeaturePropagationWithAttention(in_ch + 128, [128, 128, 128], use_attention=True)
        self.position_attn = PositionAttention(128, 128)
        self.conv1 = nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv1d(256, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.2)
        self.conv3 = nn.Conv1d(128, num_classes, 1)
        self.se_global = ChannelAttention(1024, reduction=16)
        self.se_mid = ChannelAttention(256, reduction=8)
        self.fuse_conv = nn.Conv1d(128 + 1024 + 256, 128, 1)
        self.fuse_bn = nn.BatchNorm1d(128)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, C_in = x.shape
        identity_matrix = torch.eye(3, dtype=x.dtype, device=x.device).unsqueeze(0).expand(B, -1, -1)
        l0_xyz = x[:, :, :3].contiguous()
        l0_points = x.contiguous()
        l1_xyz, l1_points, _ = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points, _ = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points, _ = self.sa3(l2_xyz, l2_points)
        l3_feat_bcn = l3_points.permute(0, 2, 1).contiguous()
        l3_feat_bcn = self.se_global(l3_feat_bcn)
        l3_points = l3_feat_bcn.permute(0, 2, 1).contiguous()
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l2_feat_bcn = l2_points.permute(0, 2, 1).contiguous()
        l2_feat_bcn = self.se_mid(l2_feat_bcn)
        l2_points = l2_feat_bcn.permute(0, 2, 1).contiguous()
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)
        l0_feat_bcn = l0_points.permute(0, 2, 1).contiguous()
        feat = self.position_attn(l0_feat_bcn, l0_xyz)
        global_feat = F.adaptive_avg_pool1d(l3_feat_bcn, 1).expand(-1, -1, N)
        mid_feat = F.interpolate(l2_feat_bcn, size=N, mode='linear', align_corners=True)
        multi_scale = torch.cat([feat, global_feat, mid_feat], dim=1)
        fused = F.relu(self.fuse_bn(self.fuse_conv(multi_scale)))
        feat = F.relu(self.bn1(self.conv1(fused)))
        feat = self.drop1(feat)
        residual = feat
        feat = F.relu(self.bn2(self.conv2(feat)))
        feat = feat + F.relu(self.bn2(self.conv2(residual)))
        feat = self.drop2(feat)
        out = self.conv3(feat)
        out = out.permute(0, 2, 1).contiguous()
        return out, identity_matrix, identity_matrix


def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    pred_flat = pred.reshape(-1, pred.size(-1))
    target_flat = target.reshape(-1)
    ce_loss = F.cross_entropy(pred_flat, target_flat, reduction='none')
    pt = torch.exp(-ce_loss)
    alpha_t = torch.ones_like(ce_loss)
    if class_weights is not None:
        alpha_t = class_weights[target_flat]
    elif isinstance(alpha, (list, tuple)):
        alpha_t = torch.tensor([alpha[t] if t < len(alpha) else alpha[-1] for t in target_flat.cpu()],
                               dtype=pred.dtype, device=pred.device)
    elif isinstance(alpha, (int, float)):
        alpha_t = torch.where(target_flat == 1, torch.tensor(alpha, dtype=pred.dtype, device=pred.device),
                              torch.tensor(1 - alpha, dtype=pred.dtype, device=pred.device))
    focal_weight = alpha_t * ((1 - pt) ** gamma)
    loss = (focal_weight * ce_loss).mean()
    return loss


def dice_loss(pred: torch.Tensor, target: torch.Tensor, num_classes: int, smooth: float = 1.0) -> torch.Tensor:
    pred_softmax = F.softmax(pred, dim=-1)
    target_one_hot = F.one_hot(target.clamp(0, num_classes - 1), num_classes).float()
    if pred_softmax.dim() > 2:
        pred_flat = pred_softmax.view(-1, num_classes)
        target_flat = target_one_hot.view(-1, num_classes)
    else:
        pred_flat = pred_softmax
        target_flat = target_one_hot
    intersection = (pred_flat * target_flat).sum(dim=0)
    cardinality_pred = pred_flat.sum(dim=0)
    cardinality_target = target_flat.sum(dim=0)
    dice_per_class = (2.0 * intersection + smooth) / (cardinality_pred + cardinality_target + smooth)
    loss = 1.0 - dice_per_class.mean()
    return loss


def lovasz_softmax_flat(probs: torch.Tensor, labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    losses = []
    for c in range(num_classes):
        fg = (labels == c).float()
        if fg.sum() == 0 or (1 - fg).sum() == 0:
            continue
        pred_c = probs[:, c]
        errors = (fg - pred_c).abs()
        sorted_idx = torch.argsort(errors, descending=True)
        sorted_fg = fg[sorted_idx]
        intersection = torch.cumsum(sorted_fg, dim=0)
        union = fg.sum() + torch.arange(1, len(fg) + 1, device=probs.device).float() - intersection
        iou_per_step = intersection / union.clamp(min=1e-8)
        losses.append(1.0 - iou_per_step.mean())
    return sum(losses) / max(len(losses), 1)


def _compute_boundary_weights(
    points: torch.Tensor,
    labels: torch.Tensor,
    k: int = 16,
    boundary_weight: float = 3.0,
) -> torch.Tensor:
    B, N, C = points.shape
    labels_flat = labels.view(-1)
    points_flat = points.view(B * N, C)
    weights = torch.ones(B * N, dtype=torch.float32, device=points.device)
    batch_size = min(4096, B * N)
    for start in range(0, B * N, batch_size):
        end = min(start + batch_size, B * N)
        chunk_pts = points_flat[start:end]
        if end - start <= k + 1:
            continue
        dists = torch.cdist(chunk_pts.unsqueeze(0), points_flat.unsqueeze(0)).squeeze(0)
        _, knn_idx = torch.topk(dists, k + 1, largest=False, dim=1)
        knn_idx = knn_idx[:, 1:]
        neighbor_labels = labels_flat[knn_idx]
        current_labels = labels_flat[start:end].unsqueeze(1)
        is_boundary = (neighbor_labels != current_labels).any(dim=1).float()
        weights[start:end] = 1.0 + (boundary_weight - 1.0) * is_boundary
    return weights


def pointnet_seg_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    input_transform: torch.Tensor,
    feature_transform: torch.Tensor,
    reg_weight: float = 0.001,
    use_focal: bool = True,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    dice_weight: float = 1.0,
    lovasz_weight: float = 2.0,
    use_boundary_aware: bool = True,
    boundary_weight: float = 3.0,
    points: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_flat = pred.view(-1, pred.size(-1))
    target_flat = target.view(-1)
    num_classes = pred.size(-1)
    class_weights = _compute_class_weights(target_flat, num_classes)
    if use_boundary_aware and points is not None:
        boundary_w = _compute_boundary_weights(points, target, k=16, boundary_weight=boundary_weight)
        boundary_w = boundary_w / boundary_w.mean().clamp(min=1e-8)
    else:
        boundary_w = None
    if use_focal:
        seg_loss = focal_loss(pred_flat, target_flat, alpha=focal_alpha, gamma=focal_gamma, class_weights=class_weights)
    else:
        seg_loss = F.cross_entropy(pred_flat, target_flat, weight=class_weights)
    if boundary_w is not None:
        ce_for_boundary = F.cross_entropy(pred_flat, target_flat, reduction='none')
        seg_loss = seg_loss + 1.0 * (boundary_w * ce_for_boundary).mean()
    dice = dice_loss(pred_flat, target_flat, num_classes=num_classes)
    seg_loss = seg_loss + dice_weight * dice
    if lovasz_weight > 0:
        probs = F.softmax(pred_flat, dim=-1)
        if boundary_w is not None:
            lovasz = (boundary_w * (probs - F.one_hot(target_flat.clamp(0, num_classes-1), num_classes).float()).pow(2).sum(dim=-1)).mean()
        else:
            lovasz = lovasz_softmax_flat(probs, target_flat, num_classes=num_classes)
        seg_loss = seg_loss + lovasz_weight * lovasz
    reg_loss = _orthogonal_regularization(feature_transform, reg_weight)
    total_loss = seg_loss + reg_loss
    return total_loss, seg_loss, reg_loss


def _compute_class_weights(
    targets: torch.Tensor, num_classes: int
) -> Optional[torch.Tensor]:
    if targets.numel() == 0:
        return None
    class_counts = torch.bincount(targets, minlength=num_classes).float()
    class_counts = torch.clamp(class_counts, min=1.0)
    weights = targets.shape[0] / (num_classes * class_counts)
    weights = weights.to(targets.device)
    return weights


class TestTimeAugmentation:
    def __init__(self, model: nn.Module, device: torch.device, n_augments: int = 8):
        self.model = model
        self.device = device
        self.n_augments = n_augments

    @torch.no_grad()
    def predict(self, points: torch.Tensor) -> torch.Tensor:
        B, N, C = points.shape
        all_probs = []
        angles = [0, 90, 180, 270]
        scales = [0.9, 1.0, 1.1]
        for angle in angles:
            for scale in scales:
                if len(all_probs) >= self.n_augments:
                    break
                aug_points = points.clone()
                rad = np.radians(angle)
                cos_a, sin_a = np.cos(rad), np.sin(rad)
                rot_mat = torch.tensor([[cos_a, -sin_a, 0, 0, 0, 0],
                                       [sin_a, cos_a, 0, 0, 0, 0],
                                       [0, 0, 1, 0, 0, 0],
                                       [0, 0, 0, 1, 0, 0],
                                       [0, 0, 0, 0, 1, 0],
                                       [0, 0, 0, 0, 0, 1]],
                                      dtype=points.dtype, device=points.device)
                aug_points[:, :, :3] = aug_points[:, :, :3] @ rot_mat[:3, :3].T * scale
                out = self.model(aug_points)
                logits = out[0] if isinstance(out, tuple) else out
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs)
        avg_probs = torch.stack(all_probs).mean(dim=0)
        return avg_probs


def _orthogonal_regularization(
    transform_matrix: torch.Tensor, weight: float
) -> torch.Tensor:
    batch_size = transform_matrix.size(0)
    identity = torch.eye(
        transform_matrix.size(1),
        dtype=transform_matrix.dtype,
        device=transform_matrix.device,
    ).unsqueeze(0).expand(batch_size, -1, -1)
    product = torch.bmm(transform_matrix, transform_matrix.transpose(2, 1))
    diff = identity - product
    loss = weight * torch.norm(diff, p="fro") ** 2
    return loss


def build_pointnet_model(config: dict) -> nn.Module:
    model_cfg = config.get("model", {})
    arch = model_cfg.get("name", "pointnet_seg")
    if arch == "pointnet_pp_attention":
        return PointNetPlusPlusAttentionSeg(
            input_channels=model_cfg.get("input_channels", 3),
            num_classes=model_cfg.get("num_classes", 2),
            use_normal=model_cfg.get("use_normal", False),
        )
    if arch == "pointnet_pp":
        return PointNetPlusPlusSeg(
            input_channels=model_cfg.get("input_channels", 3),
            num_classes=model_cfg.get("num_classes", 2),
            use_normal=model_cfg.get("use_normal", False),
        )
    return PointNetSeg(
        input_channels=model_cfg.get("input_channels", 3),
        num_classes=model_cfg.get("num_classes", 2),
        use_normal=model_cfg.get("use_normal", False),
    )
