"""
eda.py - Exploratory Data Analysis
Praktikum Modul 3 AI 2026 - Satellite Image Classification
"""

import os, random
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageStat

TRAIN_DIR  = Path("train")
TEST_DIR   = Path("test")
OUTPUT_DIR = Path("eda_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

CLASSES = [
    "AnnualCrop","Forest","HerbaceousVegetation","Highway",
    "Industrial","Pasture","PermanentCrop","Residential","River","SeaLake"
]

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── helpers ──────────────────────────────────────────────────────────────────
def count_per_class(root):
    return {c: len(list((root/c).glob("*.jpg"))) for c in CLASSES if (root/c).exists()}

def load_samples(cls, n=5):
    files = list((TRAIN_DIR/cls).glob("*.jpg"))
    return [Image.open(f).convert("RGB") for f in random.sample(files, min(n,len(files)))]

def mean_brightness(cls, n=30):
    files = random.sample(list((TRAIN_DIR/cls).glob("*.jpg")), min(n, len(list((TRAIN_DIR/cls).glob("*.jpg")))))
    return float(np.mean([ImageStat.Stat(Image.open(f).convert("L")).mean[0] for f in files]))

def mean_rgb(cls, n=30):
    files = random.sample(list((TRAIN_DIR/cls).glob("*.jpg")), min(n, len(list((TRAIN_DIR/cls).glob("*.jpg")))))
    arrs  = [np.array(Image.open(f).convert("RGB")) for f in files]
    return tuple(np.mean([a[:,:,i].mean() for a in arrs]) for i in range(3))

# ── 1. distribusi kelas ──────────────────────────────────────────────────────
def plot_distribution():
    counts = count_per_class(TRAIN_DIR)
    total  = sum(counts.values())
    print(f"\n{'Kelas':<25} {'N':>6} {'%':>7}")
    print("-"*40)
    for c,n in counts.items():
        print(f"{c:<25} {n:>6} {n/total*100:>6.1f}%")
    print(f"\nTotal train: {total}  |  Test: {len(list(TEST_DIR.glob('*.jpg')))}")

    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(16,6))
    fig.suptitle("Distribusi Kelas – Train Set", fontsize=15, fontweight="bold")
    colors = plt.cm.Set3(np.linspace(0,1,len(CLASSES)))
    cls_l, cnt_l = list(counts.keys()), list(counts.values())
    bars = ax1.bar(cls_l, cnt_l, color=colors, edgecolor="black", lw=0.5)
    ax1.set_xticklabels(cls_l, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Jumlah Gambar")
    ax1.set_title("Bar Chart")
    ax1.grid(axis="y", alpha=0.3)
    for b,v in zip(bars,cnt_l):
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()+2, str(v), ha="center", fontsize=7)
    ax2.pie(cnt_l, labels=cls_l, autopct="%1.1f%%", colors=colors,
            textprops={"fontsize":7}, startangle=90)
    ax2.set_title("Pie Chart")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"01_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> 01_class_distribution.png")

# ── 2. sample images ─────────────────────────────────────────────────────────
def plot_samples():
    n = 5
    fig, axes = plt.subplots(len(CLASSES), n, figsize=(n*3, len(CLASSES)*2.8))
    fig.suptitle("Sampel Gambar per Kelas", fontsize=14, fontweight="bold")
    for r, cls in enumerate(CLASSES):
        for c, img in enumerate(load_samples(cls, n)):
            ax = axes[r][c]
            ax.imshow(img); ax.axis("off")
            if c == 0:
                ax.set_ylabel(cls, fontsize=9, rotation=0, labelpad=75, va="center")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"02_sample_images.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  -> 02_sample_images.png")

# ── 3. ukuran gambar ─────────────────────────────────────────────────────────
def plot_image_sizes():
    ws, hs = [], []
    for cls in CLASSES:
        files = list((TRAIN_DIR/cls).glob("*.jpg"))
        for f in random.sample(files, min(20, len(files))):
            w,h = Image.open(f).size
            ws.append(w); hs.append(h)
    print(f"\n  Width : min={min(ws)} max={max(ws)} mean={np.mean(ws):.1f}")
    print(f"  Height: min={min(hs)} max={max(hs)} mean={np.mean(hs):.1f}")
    uniq = Counter(zip(ws,hs))
    print(f"  Unique sizes: {len(uniq)}  (top-3: {uniq.most_common(3)})")

    fig,(a1,a2) = plt.subplots(1,2, figsize=(12,4))
    fig.suptitle("Distribusi Ukuran Gambar", fontsize=13, fontweight="bold")
    a1.hist(ws,bins=20,color="steelblue",edgecolor="black",alpha=0.8)
    a1.set(title="Width", xlabel="px", ylabel="Freq")
    a2.hist(hs,bins=20,color="salmon",edgecolor="black",alpha=0.8)
    a2.set(title="Height", xlabel="px", ylabel="Freq")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"03_image_sizes.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> 03_image_sizes.png")

# ── 4. brightness per kelas ──────────────────────────────────────────────────
def plot_brightness():
    br = {c: mean_brightness(c) for c in CLASSES}
    items = sorted(br.items(), key=lambda x:x[1])
    cls_s, val_s = zip(*items)
    colors = plt.cm.RdYlGn(np.linspace(0.2,0.9,len(CLASSES)))
    fig,ax = plt.subplots(figsize=(12,5))
    bars = ax.barh(cls_s, val_s, color=colors, edgecolor="black", lw=0.5)
    ax.axvline(np.mean(val_s), color="red", ls="--", label=f"Mean={np.mean(val_s):.1f}")
    ax.set(xlabel="Mean Brightness (0-255)", title="Brightness per Kelas")
    ax.legend(); ax.grid(axis="x", alpha=0.3)
    for b,v in zip(bars,val_s):
        ax.text(v+0.5, b.get_y()+b.get_height()/2, f"{v:.1f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"04_brightness.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> 04_brightness.png")

# ── 5. RGB mean per kelas ────────────────────────────────────────────────────
def plot_rgb():
    rgb = {c: mean_rgb(c) for c in CLASSES}
    r_v = [rgb[c][0] for c in CLASSES]
    g_v = [rgb[c][1] for c in CLASSES]
    b_v = [rgb[c][2] for c in CLASSES]
    x = np.arange(len(CLASSES)); w = 0.25
    fig,ax = plt.subplots(figsize=(14,6))
    ax.bar(x-w, r_v, w, label="Red",  color="tomato", alpha=0.85, edgecolor="k")
    ax.bar(x,   g_v, w, label="Green",color="mediumseagreen", alpha=0.85, edgecolor="k")
    ax.bar(x+w, b_v, w, label="Blue", color="cornflowerblue", alpha=0.85, edgecolor="k")
    ax.set_xticks(x); ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set(ylabel="Mean Pixel Value", title="Mean RGB per Kelas")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"05_rgb_per_class.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> 05_rgb_per_class.png")

# ── 6. kelas mirip ───────────────────────────────────────────────────────────
def plot_similar():
    pairs = [("AnnualCrop","PermanentCrop"),("Forest","HerbaceousVegetation"),
             ("River","SeaLake"),("Highway","Residential"),("Pasture","HerbaceousVegetation")]
    fig, axes = plt.subplots(len(pairs), 6, figsize=(18, len(pairs)*3))
    fig.suptitle("Perbandingan Kelas Mirip Secara Visual", fontsize=13, fontweight="bold")
    for row,(c1,c2) in enumerate(pairs):
        for col,img in enumerate(load_samples(c1,3)):
            axes[row][col].imshow(img); axes[row][col].axis("off")
            if col==0: axes[row][col].set_title(c1, fontsize=8, color="blue", fontweight="bold")
        for col,img in enumerate(load_samples(c2,3)):
            axes[row][col+3].imshow(img); axes[row][col+3].axis("off")
            if col==0: axes[row][col+3].set_title(c2, fontsize=8, color="red", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"06_similar_classes.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  -> 06_similar_classes.png")

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print("="*55)
    print("  EDA – Praktikum Modul 3 AI 2026")
    print("="*55)
    plot_distribution()
    plot_samples()
    plot_image_sizes()
    plot_brightness()
    plot_rgb()
    plot_similar()
    print(f"\n[DONE] Output di: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
