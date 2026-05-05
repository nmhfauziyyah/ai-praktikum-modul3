"""
train_efficient.py - Arsitektur 2: Custom EfficientNet-like (MobileNet-style)
Praktikum Modul 3 AI 2026 – From Scratch (NO pretrained weights)

Arsitektur: Depthwise Separable Convolution + SE blocks (EfficientNet-inspired)
"""

import os, time, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ── CONFIG ────────────────────────────────────────────────────────────────────
TRAIN_DIR  = Path("train")
OUTPUT_DIR = Path("outputs_efficient")
OUTPUT_DIR.mkdir(exist_ok=True)

CLASSES = [
    "AnnualCrop","Forest","HerbaceousVegetation","Highway",
    "Industrial","Pasture","PermanentCrop","Residential","River","SeaLake"
]
CLASS2IDX = {c:i for i,c in enumerate(CLASSES)}

IMG_SIZE    = 64
BATCH_SIZE  = 64
EPOCHS      = 40
LR          = 1e-3
WEIGHT_DECAY= 1e-4
VAL_SPLIT   = 0.15
SEED        = 42

torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ── DATASET ───────────────────────────────────────────────────────────────────
class SatelliteDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples   = []
        self.transform = transform
        for cls in CLASSES:
            for f in (root/cls).glob("*.jpg"):
                self.samples.append((str(f), CLASS2IDX[cls]))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, label

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Module-level Dataset wrapper (must be top-level for multiprocessing pickle)
class SubTF(Dataset):
    def __init__(self, sub, tf):
        self.sub = sub; self.tf = tf
    def __len__(self): return len(self.sub)
    def __getitem__(self, i):
        raw = self.sub.dataset.samples[self.sub.indices[i]]
        return self.tf(Image.open(raw[0]).convert("RGB")), raw[1]


def build_loaders():
    full_ds = SatelliteDataset(TRAIN_DIR)
    n_val   = int(len(full_ds) * VAL_SPLIT)
    gen     = torch.Generator().manual_seed(SEED)
    tr_sub, vl_sub = random_split(full_ds, [len(full_ds)-n_val, n_val], generator=gen)

    pin = DEVICE.type == "cuda"
    tr_ld = DataLoader(SubTF(tr_sub, train_tf), BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=pin)
    vl_ld = DataLoader(SubTF(vl_sub, val_tf),   BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=pin)
    print(f"Train: {len(tr_sub)}  Val: {len(vl_sub)}")
    return tr_ld, vl_ld

# ── MODEL BLOCKS ──────────────────────────────────────────────────────────────
class SqueezeExcite(nn.Module):
    """SE block for channel-wise attention."""
    def __init__(self, ch, ratio=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, ch//ratio, bias=False),
            nn.SiLU(),
            nn.Linear(ch//ratio, ch, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.se(x).view(x.size(0), -1, 1, 1)

class DepthwiseSepConv(nn.Module):
    """Depthwise Separable Convolution."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                            groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn2(self.pw(self.act(self.bn1(self.dw(x))))))

class MBConv(nn.Module):
    """Mobile Inverted Residual (MBConv) with SE."""
    def __init__(self, in_ch, out_ch, expand=4, stride=1):
        super().__init__()
        mid = in_ch * expand
        self.use_residual = (in_ch == out_ch and stride == 1)
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.SiLU()
        ) if expand != 1 else nn.Identity()
        self.dw = nn.Sequential(
            nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid), nn.SiLU()
        )
        self.se  = SqueezeExcite(mid, ratio=4)
        self.pw  = nn.Sequential(
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        out = self.expand(x)
        out = self.dw(out)
        out = self.se(out)
        out = self.pw(out)
        return out + x if self.use_residual else out

class CustomEfficientNet(nn.Module):
    """EfficientNet-inspired, built entirely from scratch."""
    def __init__(self, num_classes=10):
        super().__init__()
        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.SiLU()
        )
        # MBConv stages  (in, out, expand, stride)
        cfg = [
            (32,  16, 1, 1),
            (16,  24, 6, 1),
            (24,  40, 6, 2),
            (40,  80, 6, 2),
            (80, 112, 6, 1),
            (112,192, 6, 2),
            (192,320, 6, 1),
        ]
        layers = []
        for in_ch, out_ch, exp, st in cfg:
            layers.append(MBConv(in_ch, out_ch, expand=exp, stride=st))
        self.blocks = nn.Sequential(*layers)
        # head
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(1280, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))

# ── TRAINING LOOP ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        use_amp = DEVICE.type == "cuda"
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            out  = model(imgs)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss/total, correct/total

@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out  = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = out.argmax(1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss/total, correct/total, f1, all_preds, all_labels

def plot_history(hist):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Custom EfficientNet – Training History", fontsize=13, fontweight="bold")
    for ax, (k1, k2, ttl) in zip(axes, [
        ("train_loss","val_loss","Loss"),
        ("train_acc","val_acc","Accuracy"),
        ("val_f1", None, "Val Macro F1")
    ]):
        ax.plot(hist[k1], label="Train")
        if k2: ax.plot(hist[k2], label="Val")
        ax.set_title(ttl); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"training_history.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_confusion(labels, preds):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix – EfficientNet")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("="*55)
    print("  Train Custom EfficientNet (from scratch)")
    print(f"  Device : {DEVICE}")
    print("="*55)
    tr_ld, vl_ld = build_loaders()

    model     = CustomEfficientNet(num_classes=len(CLASSES)).to(DEVICE)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR*5, epochs=EPOCHS,
        steps_per_epoch=len(tr_ld), pct_start=0.1
    )
    scaler = torch.amp.GradScaler(enabled=DEVICE.type=="cuda")  # MPS/CPU: no-op

    hist      = {k:[] for k in ["train_loss","val_loss","train_acc","val_acc","val_f1"]}
    best_f1   = 0.
    best_path = OUTPUT_DIR/"efficient_best.pth"

    for ep in range(1, EPOCHS+1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, tr_ld, criterion, optimizer, scaler)
        vl_loss, vl_acc, vl_f1, preds, labels = eval_epoch(model, vl_ld, criterion)
        scheduler.step()

        for k,v in zip(["train_loss","val_loss","train_acc","val_acc","val_f1"],
                        [tr_loss, vl_loss, tr_acc, vl_acc, vl_f1]):
            hist[k].append(v)

        mark = "✓" if vl_f1 > best_f1 else " "
        if vl_f1 > best_f1:
            best_f1 = vl_f1
            torch.save(model.state_dict(), best_path)
        print(f"Ep {ep:>2}/{EPOCHS} | "
              f"L={tr_loss:.4f}/{vl_loss:.4f} | "
              f"Acc={tr_acc:.3f}/{vl_acc:.3f} | "
              f"F1={vl_f1:.4f} {mark} | {time.time()-t0:.1f}s")

    print(f"\n[DONE] Best Val Macro F1: {best_f1:.4f}")
    print(f"       Model saved: {best_path}")

    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    _, _, _, final_preds, final_labels = eval_epoch(model, vl_ld, criterion)
    print("\n" + classification_report(final_labels, final_preds, target_names=CLASSES, digits=4))

    plot_history(hist)
    plot_confusion(final_labels, final_preds)
    print(f"Plots saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
