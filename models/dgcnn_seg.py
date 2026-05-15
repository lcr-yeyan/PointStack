import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size, num_dims, num_points = x.size()
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        x = get_graph_feature(x, k=self.k)
        x = self.conv(x)
        x = x.max(dim=-1, keepdim=False)[0]
        return x


class DGCNNSeg(nn.Module):
    def __init__(self, input_channels=3, num_classes=2, k=20, dropout=0.5):
        super().__init__()
        self.k = k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels * 2, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(64 + 64 + 128 + 256, 1024, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.seg_head = nn.Sequential(
            nn.Conv1d(1024 + 64 + 64 + 128 + 256, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
            nn.Conv1d(256, num_classes, kernel_size=1),
        )

    def forward(self, x):
        batch_size, num_points, _ = x.size()
        x = x.transpose(2, 1)

        x = get_graph_feature(x, k=self.k)
        x1 = self.conv1(x)
        x1 = x1.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x2 = self.conv2(x)
        x2 = x2.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x3 = self.conv3(x)
        x3 = x3.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x4 = self.conv4(x)
        x4 = x4.max(dim=-1, keepdim=False)[0]

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        x_global = self.conv5(x_cat)
        x_global_max = F.adaptive_max_pool1d(x_global, 1).view(batch_size, -1)
        x_global_expand = x_global_max.unsqueeze(2).repeat(1, 1, num_points)

        x_seg = torch.cat((x_global_expand, x1, x2, x3, x4), dim=1)
        seg_logits = self.seg_head(x_seg)
        seg_logits = seg_logits.transpose(2, 1).contiguous()

        return seg_logits, None, None