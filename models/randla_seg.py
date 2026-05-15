import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def index_points(points, idx):
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class RandLANetSeg(nn.Module):
    def __init__(self, input_channels=3, num_classes=2, d_out=16, num_neighbors=16, decimation=4):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.decimation = decimation

        self.fc_start = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.encoder_blocks = nn.ModuleList()
        chs = [32, 64, 128, 256, 512]
        for i in range(len(chs) - 1):
            self.encoder_blocks.append(
                nn.Sequential(
                    nn.Conv1d(chs[i], chs[i + 1], kernel_size=1, bias=False),
                    nn.BatchNorm1d(chs[i + 1]),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(chs[i + 1], chs[i + 1], kernel_size=1, bias=False),
                    nn.BatchNorm1d(chs[i + 1]),
                    nn.ReLU(inplace=True),
                )
            )

        self.decoder_blocks = nn.ModuleList()
        dec_chs = [512, 256, 128, 96, 64]
        for i in range(len(dec_chs) - 1):
            in_ch = dec_chs[i] + chs[-2 - i] if i < len(chs) - 1 else dec_chs[i]
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, dec_chs[i + 1], kernel_size=1, bias=False),
                    nn.BatchNorm1d(dec_chs[i + 1]),
                    nn.ReLU(inplace=True),
                )
            )

        self.seg_head = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, num_classes, kernel_size=1),
        )

    def random_sample(self, xyz, features, npoint):
        B, N, _ = xyz.shape
        if npoint >= N:
            return xyz, features, torch.arange(N, device=xyz.device).unsqueeze(0).expand(B, -1)
        idx = torch.stack([torch.randperm(N, device=xyz.device)[:npoint] for _ in range(B)], dim=0)
        sampled_xyz = index_points(xyz, idx)
        sampled_feat = index_points(features.transpose(1, 2), idx).transpose(1, 2)
        return sampled_xyz, sampled_feat, idx

    def nearest_interpolation(self, xyz1, xyz2, feat2):
        B, N1, _ = xyz1.shape
        dists = torch.cdist(xyz1, xyz2)
        nn_idx = dists.argmin(dim=-1)
        feat2_t = feat2.transpose(1, 2)
        feat_interp = index_points(feat2_t, nn_idx).transpose(1, 2)
        return feat_interp

    def forward(self, x):
        B, N, _ = x.shape
        xyz = x[:, :, :3].contiguous()
        features = x.transpose(1, 2).contiguous()

        features = self.fc_start(features)

        enc_xyz_list = [xyz]
        enc_feat_list = [features]

        cur_xyz, cur_feat = xyz, features
        for block in self.encoder_blocks:
            npoint = max(cur_xyz.size(1) // self.decimation, 16)
            cur_xyz, cur_feat, _ = self.random_sample(cur_xyz, cur_feat, npoint)
            cur_feat = block(cur_feat)
            enc_xyz_list.append(cur_xyz)
            enc_feat_list.append(cur_feat)

        dec_feat = enc_feat_list[-1]
        for i, block in enumerate(self.decoder_blocks):
            dec_xyz = enc_xyz_list[-2 - i]
            enc_feat_skip = enc_feat_list[-2 - i]
            dec_feat = self.nearest_interpolation(dec_xyz, enc_xyz_list[-1 - i], dec_feat)
            dec_feat = torch.cat([dec_feat, enc_feat_skip], dim=1)
            dec_feat = block(dec_feat)

        dec_feat = self.nearest_interpolation(xyz, enc_xyz_list[0], dec_feat)
        seg_logits = self.seg_head(dec_feat).transpose(1, 2).contiguous()

        return seg_logits, None, None