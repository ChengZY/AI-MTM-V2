import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import dropout


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2, output_size=3, dropout=0.1, bidirectional=False):
        super(SimpleLSTM, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)           # out: (batch, seq_len, hidden*directions)
        last = out[:, -1, :]            # take last time-step
        return self.fc(last)            # (batch, output_size)


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=2, output_size=3, dropout=0.2):
        super(SimpleMLP, self).__init__()
        layers = []
        in_features = input_size

        for _ in range(max(1, num_layers)):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = hidden_size

        layers.append(nn.Linear(in_features, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = x.reshape(-1, x.size(-1))   # (batch*seq_len, input_size)
        y = self.net(x)                 # (batch*seq_len, output_size)
        y = y.reshape(x.size(0), -1, y.size(-1))  # (batch, seq_len, output_size) \*after reshape below
        # restore batch using original batch size
        batch_size = x.size(0) // y.size(1) if y.size(1) != 0 else 0
        y = y.reshape(batch_size, -1, y.size(-1))  # (batch, seq_len, output_size)
        return y[:, -1, :]               # (batch, output_size)


class Spectrum1DCNN(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=2, kernel_size=8, output_size=3):
        super().__init__()

        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            in_channels = input_size if layer == 0 else hidden_size
            out_channels = hidden_size*2
            self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2))
            hidden_size *= 2  # Update hidden_size for the next layer

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fcs = nn.ModuleList()
        for _ in range(num_layers):
            self.fcs.append(nn.Linear(hidden_size, hidden_size // 2))
            self.fcs.append(nn.ReLU())
            # self.fcs.append(nn.Dropout(0.2))
            hidden_size //= 2  # Halve hidden_size for the next layer
        self.fcs.append(nn.Dropout(0.1))
        self.fcs.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        x = x.transpose(1, 2)  # Ensure input shape is [batch_size, in_channels, seq_len]
        for layer in self.convs:
            x = layer(x)
        x = self.pool(x).squeeze(-1)
        for layer in self.fcs:
            x = layer(x)
        return x

# python
class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

# python
class ResNet1D(nn.Module):
    def __init__(self, block, layers, input_channels=1, base_channels=64, output_channels=3, kernel_size=3):
        """
        block: block class (e.g. BasicBlock1D)
        layers: list of 4 ints, e.g. [2,2,2,2] for ResNet18-like
        input_channels: number of input channels (features)
        base_channels: channels for the first conv (typical 64)
        output_channels: output size
        kernel_size: main conv kernel size inside residual blocks
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.inplanes = base_channels

        # initial conv: expect input shape (batch, seq_len, channels) -> transpose later
        self.conv1 = nn.Conv1d(input_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # layers: planes double each stage
        self.layer1 = self._make_layer(block, base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(block, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_channels * 8, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channels * 8 * block.expansion, output_channels)

        # weight init (follow PyTorch ResNet style)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, kernel_size=self.kernel_size))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=self.kernel_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        # expects x: (batch, seq_len, channels) -> convert to (batch, channels, seq_len)
        x = x.transpose(1, 2)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x).squeeze(-1)
        x = self.fc(x)
        return x


# Factory helpers for common depths and custom ones (including nonstandard 14 / 19)
def resnet14_1d(input_channels=1, base_channels=32, output_channels=3, kernel_size=3):
    # small / shallow variant
    return ResNet1D(BasicBlock1D, [1, 1, 1, 1], input_channels=input_channels,
                    base_channels=base_channels, output_channels=output_channels, kernel_size=kernel_size)

def resnet18_1d(input_channels=1, base_channels=64, output_channels=3, kernel_size=3):
    return ResNet1D(BasicBlock1D, [2, 2, 2, 2], input_channels=input_channels,
                    base_channels=base_channels, output_channels=output_channels, kernel_size=kernel_size)

def resnet19_1d(input_channels=1, base_channels=64, output_channels=3, kernel_size=3):
    # nonstandard depth, slightly deeper last stage
    return ResNet1D(BasicBlock1D, [2, 2, 2, 3], input_channels=input_channels,
                    base_channels=base_channels, output_channels=output_channels, kernel_size=kernel_size)

def resnet34_1d(input_channels=1, base_channels=64, output_channels=3, kernel_size=3):
    return ResNet1D(BasicBlock1D, [3, 4, 6, 3], input_channels=input_channels,
                    base_channels=base_channels, output_channels=output_channels, kernel_size=kernel_size)