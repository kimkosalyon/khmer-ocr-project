import argparse
import os
import sys

import torch.nn as nn
import torchvision.models as tv


sys.path.insert(0, os.path.dirname(__file__))
import train_200k_resnet34_bigru as train_200k  # noqa: E402


class KhmerCRNN_ResNet18_BiGRU(nn.Module):
    """
    OCR model consisting of a modified ResNet18 backbone for visual feature extraction,
    followed by a 2-layer Bidirectional GRU for sequence mapping, and a CTC output projection.
    """

    def __init__(self, vocab_size, hidden=256):
        super().__init__()
        # 1. Initialize a standard PyTorch ResNet18 model without pre-trained ImageNet weights
        rn = tv.resnet18(weights=None)
        
        # 2. Modify the first convolution layer to accept 1-channel grayscale input (instead of 3-channel RGB)
        rn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 3. Group stem layers and standard layers together (shape changes: H/4, W/8)
        self.stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool, rn.layer1, rn.layer2)
        self.layer3 = rn.layer3
        self.layer4 = rn.layer4

        # 4. Patch Layers 3 and 4: Replace vertical downsampling with horizontal width preservation.
        #    We change downsampling strides from (2, 2) to (2, 1) in Layer 3 and Layer 4.
        for b in [self.layer3, self.layer4]:
            for block in b:
                # Modify main path: change downsample convolution stride
                if hasattr(block, "conv1") and block.conv1.stride != (1, 1):
                    block.conv1.stride = (2, 1)
                # Modify shortcut path: change skip-connection projection stride to match main path
                if block.downsample is not None:
                    block.downsample[0].stride = (2, 1)

        # 5. Bidirectional GRU sequence layer (receives 512-dim features from collapsed CNN output)
        self.bigru = nn.GRU(512, hidden, num_layers=2, bidirectional=True, dropout=0.2)
        
        # 6. Final linear projection mapping hidden states (hidden * 2 due to bidirectionality) to vocab size
        self.fc = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        # Input shape: (BatchSize, 1, 64, Width)
        
        # Pass through stem and Layer1/2. Shape becomes: (BatchSize, 128, 8, Width / 8)
        f = self.stem(x)
        
        # Pass through Layer3. Stride (2, 1) changes shape to: (BatchSize, 256, 4, Width / 8)
        f = self.layer3(f)
        
        # Pass through Layer4. Stride (2, 1) changes shape to: (BatchSize, 512, 2, Width / 8)
        f = self.layer4(f)
        
        # Vertical Mean Pooling: Average values vertically over the height dimension (dim 2)
        # Output shape: (BatchSize, 512, Width / 8)
        f = f.mean(dim=2)
        
        # Permute for Recurrent Network input: (TimeSteps, BatchSize, Channels)
        # Output shape: (TimeSteps, BatchSize, 512) where TimeSteps = Width / 8
        f = f.permute(2, 0, 1)
        
        # Process sequence with bidirectional GRU. Output shape: (TimeSteps, BatchSize, hidden * 2)
        out, _ = self.bigru(f)
        
        # Project each time step to character vocabulary logits. Output shape: (TimeSteps, BatchSize, VocabSize)
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
