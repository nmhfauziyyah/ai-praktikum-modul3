"""
error_analysis.py - Analisis Kesalahan Prediksi Model
Praktikum Modul 3 AI 2026
"""

import os, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    f1_score, classification_report,
    confusion_matrix, precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns

# ── CONFIG ────────────────────────────────────────────────────────────────────
TRAIN_DIR = Path("train")
OUT_DIR   = Path("error_analysis_outputs")
OUT_DIR.mkdir(exist_ok=True)

CLASSES = [
    "AnnualCrop","Forest","HerbaceousVegetation","Highway",
    "Industrial","Pasture","PermanentCrop","Residential","River","SeaLake"
]
CLASS2IDX = {c:i for i,c in enumerate(CLASSES)}
IDX2CLASS  = {i:c for c,i in CLASS2IDX.items()}

IMG_SIZE   = 64
BATCH_SIZE = 64
VAL_SPLIT  = 0.15
SEED       = 42

torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# ── TRANSFORMS ────────────────────────────────────────────────────────────────
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ── DATASETS (module-level so multiprocessing can pickle them) ─────────────────
class SatelliteDataset(Dataset):
    """Full dataset with (path, label) pairs."""
    def __init__(self, root):
        self.samples = []
        for cls in CLASSES:
            for f in (root/cls).glob("*.jpg"):
                self.samples.append((str(f), CLASS2IDX[cls]))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]   # returns (path_str, label_int)

class ValSubset(Dataset):
    """Validation subset that applies transform and returns (img, label, path)."""
    def __init__(self, samples, tf):
        self.samples = samples   # list of (path_str, label_int)
        self.tf      = tf
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.tf(img), label, path

def build_val_loader():
    full_ds = SatelliteDataset(TRAIN_DIR)
    n_val   = int(len(full_ds) * VAL_SPLIT)
    gen     = torch.Generator().manual_seed(SEED)
    n_train = len(full_ds) - n_val
    _, vl_sub = random_split(full_ds, [n_train, n_val], generator=gen)

    # extract raw samples from subset
    val_samples = [full_ds.samples[i] for i in vl_sub.indices]
    val_ds = ValSubset(val_samples, val_tf)

    pin = DEVICE.type == "cuda"
    return DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=2, pin_memory=pin)

# ── LOAD MODELS ───────────────────────────────────────────────────────────────
def load_resnet():
    from train_resnet import CustomResNet
    m = CustomResNet(num_classes=10).to(DEVICE)
    ckpt = Path("outputs_resnet/resnet_best.pth")
    if ckpt.exists():
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        print(f"  Loaded ResNet from {ckpt}")
    else:
        print(f"  [WARN] {ckpt} not found")
    m.eval()
    return m

def load_efficient():
    from train_efficient import CustomEfficientNet
    m = CustomEfficientNet(num_classes=10).to(DEVICE)
    ckpt = Path("outputs_efficient/efficient_best.pth")
    if ckpt.exists():
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        print(f"  Loaded EfficientNet from {ckpt}")
    else:
        print(f"  [WARN] {ckpt} not found")
    m.eval()
    return m

# ── INFERENCE ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def get_predictions(model, loader):
    all_preds, all_labels, all_paths, all_probs = [], [], [], []
    for imgs, labels, paths in loader:
        imgs   = imgs.to(DEVICE)
        logits = model(imgs)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_paths.extend(paths)
        all_probs.extend(probs)
    return (np.array(all_labels), np.array(all_preds),
            all_paths, np.array(all_probs))

# ── PLOT FUNCTIONS ─────────────────────────────────────────────────────────────
def plot_confusion_compare(labels, preds_r, preds_e):
    fig, axes = plt.subplots(1, 2, figsize=(22, 8))
    for ax, preds, title, cmap in zip(
        axes,
        [preds_r, preds_e],
        ["Custom ResNet","Custom EfficientNet"],
        ["Blues","Greens"]
    ):
        cm = confusion_matrix(labels, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                    xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
                    annot_kws={"size":7})
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix – {title}", fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", rotation=0,  labelsize=7)
    plt.tight_layout()
    plt.savefig(OUT_DIR/"01_confusion_compare.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> 01_confusion_compare.png")

def plot_per_class_f1(labels, preds_r, preds_e):
    _, _, f1_r, _ = precision_recall_fscore_support(
        labels, preds_r, average=None, labels=list(range(10)), zero_division=0)
    _, _, f1_e, _ = precision_recall_fscore_support(
        labels, preds_e, average=None, labels=list(range(10)), zero_division=0)

    x = np.arange(len(CLASSES)); w = 0.35
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x-w/2, f1_r, w, label="ResNet",       color="steelblue",       alpha=0.85, edgecolor="k")
    ax.bar(x+w/2, f1_e, w, label="EfficientNet",  color="mediumseagreen",  alpha=0.85, edgecolor="k")
    ax.set_xticks(x); ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score Comparison", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(OUT_DIR/"02_perclass_f1.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> 02_perclass_f1.png")

def plot_misclassified(labels, preds, paths, probs, model_name, n=16):
    wrong_idx = np.where(labels != preds)[0]
    if len(wrong_idx) == 0:
        print(f"  No misclassifications for {model_name}!"); return
    sampled = np.random.choice(wrong_idx, min(n, len(wrong_idx)), replace=False)

    cols = 4
    rows = (len(sampled) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.5, rows*3.5))
    fig.suptitle(f"Misclassified Samples – {model_name}", fontsize=13, fontweight="bold")
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax_idx, i in enumerate(sampled):
        ax = axes_flat[ax_idx]
        img = Image.open(paths[i]).convert("RGB")
        ax.imshow(img)
        true_cls = IDX2CLASS[int(labels[i])]
        pred_cls = IDX2CLASS[int(preds[i])]
        conf     = probs[i][int(preds[i])] * 100
        ax.set_title(f"True: {true_cls}\nPred: {pred_cls} ({conf:.1f}%)",
                     fontsize=7, color="red")
        ax.axis("off")
    for ax in axes_flat[len(sampled):]: ax.axis("off")
    plt.tight_layout()
    fname = f"03_misclassified_{model_name.lower().replace(' ','_')}.png"
    plt.savefig(OUT_DIR/fname, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  -> {fname}")

def plot_most_confused_pairs(labels, preds, model_name):
    cm = confusion_matrix(labels, preds)
    np.fill_diagonal(cm, 0)
    pairs = [(cm[i,j], CLASSES[i], CLASSES[j])
             for i in range(len(CLASSES)) for j in range(len(CLASSES))
             if i != j and cm[i,j] > 0]
    pairs.sort(reverse=True)
    top10 = pairs[:10]

    labs = [f"{t}→{p}" for _,t,p in top10]
    vals = [v for v,_,_ in top10]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(labs[::-1], vals[::-1], color="salmon", edgecolor="k")
    ax.set_xlabel("Count")
    ax.set_title(f"Top-10 Most Confused Pairs – {model_name}", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for b,v in zip(bars, vals[::-1]):
        ax.text(v+0.2, b.get_y()+b.get_height()/2, str(v), va="center", fontsize=9)
    plt.tight_layout()
    fname = f"04_confused_pairs_{model_name.lower().replace(' ','_')}.png"
    plt.savefig(OUT_DIR/fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {fname}")

def print_summary(labels, preds_r, preds_e):
    f1_r = f1_score(labels, preds_r, average="macro", zero_division=0)
    f1_e = f1_score(labels, preds_e, average="macro", zero_division=0)
    print(f"\n{'Model':<25} {'Macro F1':>10}")
    print("-"*37)
    print(f"{'Custom ResNet':<25} {f1_r:>10.4f}")
    print(f"{'Custom EfficientNet':<25} {f1_e:>10.4f}")
    best = "Custom ResNet" if f1_r >= f1_e else "Custom EfficientNet"
    print(f"\n>>> Best model: {best} (F1={max(f1_r,f1_e):.4f})")
    print(f"\n--- ResNet per-class report ---")
    print(classification_report(labels, preds_r, target_names=CLASSES, digits=4))
    print(f"\n--- EfficientNet per-class report ---")
    print(classification_report(labels, preds_e, target_names=CLASSES, digits=4))

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("="*55)
    print("  Error Analysis – Modul 3 AI 2026")
    print(f"  Device: {DEVICE}")
    print("="*55)

    val_loader = build_val_loader()

    print("\n[Loading models...]")
    model_r = load_resnet()
    model_e = load_efficient()

    print("\n[Running inference on validation set...]")
    labels_r, preds_r, paths_r, probs_r = get_predictions(model_r, val_loader)
    labels_e, preds_e, paths_e, probs_e = get_predictions(model_e, val_loader)

    print_summary(labels_r, preds_r, preds_e)

    print("\n[Generating plots...]")
    plot_confusion_compare(labels_r, preds_r, preds_e)
    plot_per_class_f1(labels_r, preds_r, preds_e)
    plot_misclassified(labels_r, preds_r, paths_r, probs_r, "ResNet")
    plot_misclassified(labels_e, preds_e, paths_e, probs_e, "EfficientNet")
    plot_most_confused_pairs(labels_r, preds_r, "ResNet")
    plot_most_confused_pairs(labels_e, preds_e, "EfficientNet")

    print(f"\n[DONE] Output di: {OUT_DIR}/")

if __name__ == "__main__":
    main()
