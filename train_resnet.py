"""
train_resnet.py - Arsitektur 1: Custom ResNet
Praktikum Modul 3 AI 2026 – From Scratch (NO pretrained weights)

Arsitektur: ResNet-like dengan Residual Blocks
Kelas: 10 (satellite land cover)
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
TRAIN_DIR   = Path("train")
OUTPUT_DIR  = Path("outputs_resnet")
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

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

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
            cls_dir = root / cls
            if not cls_dir.exists(): continue
            for f in cls_dir.glob("*.jpg"):
                self.samples.append((str(f), CLASS2IDX[cls]))

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, label

# augmentasi train
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8,1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Module-level Dataset wrapper (must be at top-level for multiprocessing pickling)
class SubsetWithTF(Dataset):
    """Wraps a Subset and applies a transform per item."""
    def __init__(self, subset, tf):
        self.subset = subset
        self.tf     = tf
    def __len__(self): return len(self.subset)
    def __getitem__(self, i):
        raw = self.subset.dataset.samples[self.subset.indices[i]]
        img = Image.open(raw[0]).convert("RGB")
        return self.tf(img), raw[1]


def build_loaders():
    full_ds = SatelliteDataset(TRAIN_DIR, transform=None)
    n_val   = int(len(full_ds) * VAL_SPLIT)
    n_train = len(full_ds) - n_val
    gen     = torch.Generator().manual_seed(SEED)
    tr_subset, vl_subset = random_split(full_ds, [n_train, n_val], generator=gen)

    tr_ds = SubsetWithTF(tr_subset, train_tf)
    vl_ds = SubsetWithTF(vl_subset, val_tf)

    pin = DEVICE.type == "cuda"
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=2, pin_memory=pin)
    vl_loader = DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=2, pin_memory=pin)
    print(f"Train: {len(tr_ds)}  Val: {len(vl_ds)}")
    return tr_loader, vl_loader

# ── MODEL: CUSTOM ResNet ──────────────────────────────────────────────────────
class ResBlock(nn.Module):
    """Basic Residual Block."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample: identity = self.downsample(x)
        return self.relu(out + identity)

class CustomResNet(nn.Module):
    """Custom ResNet-like – dibangun dari scratch."""
    def __init__(self, num_classes=10):
        super().__init__()
        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # residual stages
        self.layer1 = nn.Sequential(ResBlock(64,  64),  ResBlock(64,  64))
        self.layer2 = nn.Sequential(ResBlock(64,  128, stride=2), ResBlock(128, 128))
        self.layer3 = nn.Sequential(ResBlock(128, 256, stride=2), ResBlock(256, 256))
        self.layer4 = nn.Sequential(ResBlock(256, 512, stride=2), ResBlock(512, 512))
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.drop   = nn.Dropout(0.4)
        self.fc     = nn.Linear(512, num_classes)

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.drop(x))

# ── TRAINING ─────────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, correct, total = 0., 0, 0
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
    total_loss, correct, total = 0., 0, 0
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

def plot_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Custom ResNet – Training History", fontsize=13, fontweight="bold")
    keys = [("train_loss","val_loss","Loss"), ("train_acc","val_acc","Accuracy"), ("val_f1",None,"Val Macro F1")]
    for ax,(k1,k2,title) in zip(axes, keys):
        ax.plot(history[k1], label="Train")
        if k2: ax.plot(history[k2], label="Val")
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"training_history.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_confusion(labels, preds):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix – ResNet")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("="*55)
    print("  Train ResNet (from scratch)")
    print(f"  Device : {DEVICE}")
    print("="*55)
    tr_loader, vl_loader = build_loaders()

    model     = CustomResNet(num_classes=len(CLASSES)).to(DEVICE)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    scaler    = torch.amp.GradScaler(enabled=DEVICE.type=="cuda")  # MPS/CPU: scaler is no-op

    history   = {k:[] for k in ["train_loss","val_loss","train_acc","val_acc","val_f1"]}
    best_f1   = 0.
    best_path = OUTPUT_DIR/"resnet_best.pth"

    for epoch in range(1, EPOCHS+1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, tr_loader, criterion, optimizer, scaler)
        vl_loss, vl_acc, vl_f1, preds, labels = eval_epoch(model, vl_loader, criterion)
        scheduler.step()

        for k,v in zip(["train_loss","val_loss","train_acc","val_acc","val_f1"],
                        [tr_loss, vl_loss, tr_acc, vl_acc, vl_f1]):
            history[k].append(v)

        mark = "✓" if vl_f1 > best_f1 else " "
        if vl_f1 > best_f1:
            best_f1 = vl_f1
            torch.save(model.state_dict(), best_path)
        print(f"Ep {epoch:>2}/{EPOCHS} | "
              f"L={tr_loss:.4f}/{vl_loss:.4f} | "
              f"Acc={tr_acc:.3f}/{vl_acc:.3f} | "
              f"F1={vl_f1:.4f} {mark} | "
              f"{time.time()-t0:.1f}s")

    print(f"\n[DONE] Best Val Macro F1: {best_f1:.4f}")
    print(f"       Model saved: {best_path}")

    # reload best & final eval
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    _, _, best_f1_check, final_preds, final_labels = eval_epoch(model, vl_loader, criterion)
    print("\n" + classification_report(final_labels, final_preds, target_names=CLASSES, digits=4))

    plot_history(history)
    plot_confusion(final_labels, final_preds)
    print(f"Plots saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
