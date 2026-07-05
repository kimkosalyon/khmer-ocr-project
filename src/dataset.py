from io import BytesIO

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, Sampler


class KhmerImgDataset(Dataset):
    """Loads and preprocesses Khmer text images with online data augmentation, preserving aspect ratio."""
    def __init__(self, hf_ds, text_col, img_col, c2i, augment=True):
        self.ds = hf_ds
        self.text_col = text_col
        self.img_col = img_col
        self.c2i = c2i

        if augment:
            self.transform = T.Compose([
                T.RandomRotation(degrees=(-3, 3), fill=255),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.RandomApply([
                    T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5))
                ], p=0.3),
                T.Grayscale(1),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,)),
                T.RandomErasing(p=0.2, scale=(0.02, 0.1), value=0.5, inplace=False)
            ])
        else:
            self.transform = T.Compose([
                T.Grayscale(1),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,)),
            ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        img = row[self.img_col]
        if not isinstance(img, Image.Image):
            if isinstance(img, str):
                img = Image.open(img)
            else:
                img = Image.open(BytesIO(img))

        # Convert to RGB and resize preserving aspect ratio (target height = 48)
        img = img.convert("RGB")
        w, h = img.size
        new_w = max(8, int(w * (48 / h)))
        img = img.resize((new_w, 48), Image.Resampling.BILINEAR)

        x = self.transform(img)
        t = [self.c2i[c] for c in row[self.text_col] if c in self.c2i]
        return x, torch.tensor(t, dtype=torch.long)


def fast_pad_collate(batch):
    """Pads variable-width images and target label sequences inside the batch."""
    xs, ts = zip(*batch)
    
    # Pad images horizontally to the maximum width in the batch
    max_w = max(x.shape[2] for x in xs)
    padded_xs = []
    for x in xs:
        pad_w = max_w - x.shape[2]
        # Pad with 1.0, which corresponds to white background in normalized tensor space
        padded_x = torch.nn.functional.pad(x, (0, pad_w, 0, 0), value=1.0)
        padded_xs.append(padded_x)
        
    imgs = torch.stack(padded_xs)
    tgt_lens = torch.tensor([len(t) for t in ts], dtype=torch.long)
    targets = nn.utils.rnn.pad_sequence(ts, batch_first=True, padding_value=0)
    return imgs, targets, tgt_lens


class LengthGroupedBatchSampler(Sampler):
    """Groups samples of similar widths/lengths into batches to minimize padding."""
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Text length is an extremely fast proxy for visual width (0.984 correlation)
        self.lengths = [len(self.dataset.ds[i][self.dataset.text_col]) for i in range(len(self.dataset))]
        
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            # Sort indices by length + small random noise to vary batches across epochs
            import random
            noise = [random.uniform(-3, 3) for _ in range(len(indices))]
            indices.sort(key=lambda idx: self.lengths[idx] + noise[idx])
        else:
            indices.sort(key=lambda idx: self.lengths[idx])
            
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        
        if self.shuffle:
            import random
            random.shuffle(batches)
            
        return iter(batches)
        
    def __len__(self):
        import math
        return math.ceil(len(self.dataset) / self.batch_size)


