import torch
from torch import nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w

class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.drop2d = nn.Dropout2d(dropout)
        self.se = SEBlock(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.drop2d(out)
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out

class ChessModel(nn.Module):
    def __init__(self, num_res_blocks=15, channels=192, policy_size=4272, policy_channels=32, block_dropout=0.2, head_dropout=0.5):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(19, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.res_tower = nn.Sequential(
            *[ResidualBlock(channels, block_dropout) for _ in range(num_res_blocks)]
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, policy_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(policy_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Dropout(head_dropout),
            nn.Linear(policy_channels * 8 * 8, policy_size)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.res_tower(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
