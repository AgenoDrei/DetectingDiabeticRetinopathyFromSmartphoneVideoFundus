import torch
from torch import nn


class RetinaNet(nn.Module):
    def __init__(self, frame_stump, num_frames=20, do_avg_pooling=True):
        super(RetinaNet, self).__init__()
        self.stump = frame_stump
        self.pool_stump = do_avg_pooling
        self.num_frames = num_frames
        self.pool_params = (self.stump.last_linear.in_features, 256) if self.pool_stump else (
        98304, 1024)  # fix for higher resolutions / different networks
        self.out_stump = self.pool_params[0]

        self.avg_pooling = self.stump.avg_pool
        self.temporal_pooling = nn.MaxPool1d(self.num_frames, stride=1, padding=0, dilation=self.out_stump)

        self.after_pooling = nn.Sequential(nn.Linear(self.out_stump, self.pool_params[1]), nn.ReLU(), nn.Dropout(p=0.5),
                                           nn.Linear(self.pool_params[1], 2))
        # self.fc1 = nn.Linear(self.out_stump, 256)
        # self.fc2 = nn.Linear(256, 2)
        # self.softmax = nn.Softmax()

    def forward(self, x):
        features = []
        for idx in range(0, x.size(1)):  # Iterate over time dimension
            out = self.stump.features(x[:, idx, :, :, :])  # Shove batch trough stump
            out = self.avg_pooling(out) if self.pool_stump else out
            out = out.view(out.size(0), -1)  # Flatten results for fc
            features.append(out)  # Size: (B, c*h*w)
        out = torch.cat(features, dim=1)
        out = self.temporal_pooling(out.unsqueeze(dim=1))
        out = self.after_pooling(out.view(out.size(0), -1))
        return out


class RetinaNet2(nn.Module):
    def __init__(self, frame_stump, num_frames=20, do_avg_pooling=True):
        super(RetinaNet2, self).__init__()
        self.stump = frame_stump
        self.pool_stump = do_avg_pooling
        self.pool_params = (self.stump.last_linear.in_features, 256) if self.pool_stump else (
        98304, 1024)  # fix for higher resolutions / different networks
        self.out_stump = self.pool_params[0]

        self.avg_pooling = self.stump.avg_pool
        self.after_pooling = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.out_stump, self.pool_params[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(self.pool_params[1], 2))

    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        x = self.stump.features(x)
        h = self.avg_pooling(x) if self.pool_stump else x
        h = h.view(h.size(0), -1)  # Flatten results for fc / pooling
        h = torch.max(h, 0, keepdim=True)[0]  # Temproal pooling over 1 dim

        out = self.after_pooling(h)
        return out


class BagNet(nn.Module):
    def __init__(self, stump):
        super(BagNet, self).__init__()
        self.stump = stump
        self.L = 4096               # FC layer size of AlexNet
        self.D = 256
        self.K = 1                  # Just why, paper, whyyyy? -> Vector reasons, maybe?
        self.feature_extractor_part1 = stump.features
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))                 # Should I use you? Replace by Max? No idea...
        self.feature_extractor_part2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, self.L),                           # 256: Channel size of AlexNet features
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(self.L, self.L),
            nn.ReLU(inplace=True),
        )
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.classifier = nn.Linear(self.L * self.K, 1)

    def forward(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor_part1(x)
        H = self.avgpool(H)
        H = H.view(-1, 256 * 6 * 6)
        H = self.feature_extractor_part2(H) # N x L, Number of bag elements

        A = self.attention(H)  # N x K
        A = torch.transpose(A, 1, 0)  # K x N
        A = nn.functional.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # K x L, multiply attention weights with bag elements
        out = self.classifier(M)
        return out
