import argparse
import os
import sys

import torch.nn as nn
import torchvision.models as tv


sys.path.insert(0, os.path.dirname(__file__))
import train_200k  # noqa: E402


class KhmerCRNN_ResNet18_BiGRU(nn.Module):
    """CRNN (ResNet18 backbone) + Bidirectional GRU for faster CTC OCR training."""

    def __init__(self, vocab_size, hidden=256):
        super().__init__()
        rn = tv.resnet18(weights=None)
        rn.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool, rn.layer1, rn.layer2)
        self.layer3 = rn.layer3
        self.layer4 = rn.layer4

        # Preserve width for CTC while still reducing height.
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet18 + BiGRU OCR on 200k Khmer dataset")
    parser.add_argument("--data-dir", default="generated/training_200k_siemreap_arial", help="Output dir of generate script")
    parser.add_argument("--samples", type=int, default=None, help="Number of records to train on (None = all)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--height", type=int, default=64, help="Target crop height (e.g. 48 or 64)")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-compile", dest="compile", action="store_false", help="Disable torch.compile")
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--checkpoint-dir", default="checkpoints_200k_resnet18_bigru")
    parser.add_argument("--resume", default=None, help="Path to ResNet18 BiGRU checkpoint .safetensors to resume from")
    parser.add_argument("--lr-scheduler", default="none", choices=["none", "cosine", "plateau"], help="Learning rate scheduler")
    parser.add_argument("--max-text-len", type=int, default=199, help="Maximum text character length to prevent width/memory spikes")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.model_arch = "ResNet18_BiGRU_CTC"
    train_200k.KhmerCRNN_BiGRU = KhmerCRNN_ResNet18_BiGRU
    train_200k.run_training(args)
