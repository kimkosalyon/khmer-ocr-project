import torch.nn as nn
import torchvision.models as tv


class KhmerCRNN_BiGRU(nn.Module):
    """CRNN (ResNet34 backbone) + Bidirectional GRU model for sequence alignment."""
    def __init__(self, vocab_size, hidden=256):
        super().__init__()
        rn = tv.resnet34(weights=None)
        rn.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool, rn.layer1, rn.layer2)
        self.layer3 = rn.layer3
        self.layer4 = rn.layer4

        for b in [self.layer3, self.layer4]:
            for block in b:
                if hasattr(block, "conv1") and block.conv1.stride != (1, 1):
                    block.conv1.stride = (2, 1)
                if block.downsample is not None:
                    block.downsample[0].stride = (2, 1)
        self.bigru = nn.GRU(512, hidden, num_layers=2, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        f = self.stem(x)
        f = self.layer3(f)
        f = self.layer4(f)
        f = f.mean(dim=2)
        f = f.permute(2, 0, 1)
        out, _ = self.bigru(f)
        return self.fc(out)

