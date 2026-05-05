"""
inference.py - Inferensi Model Terbaik + Generate Submission CSV
Praktikum Modul 3 AI 2026

- Load model terbaik (ResNet atau EfficientNet, atau ensemble keduanya)
- TTA (Test-Time Augmentation)
- Output: submission.csv sesuai format Kaggle
"""

import os, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
TEST_DIR       = Path("test")
SAMPLE_CSV     = Path("sample_submission.csv")
OUTPUT_CSV     = Path("submission.csv")
RESNET_CKPT    = Path("outputs_resnet/resnet_best.pth")
EFFICIENT_CKPT = Path("outputs_efficient/efficient_best.pth")

CLASSES = [
    "AnnualCrop","Forest","HerbaceousVegetation","Highway",
    "Industrial","Pasture","PermanentCrop","Residential","River","SeaLake"
]
IDX2CLASS = {i:c for i,c in enumerate(CLASSES)}

IMG_SIZE   = 64
BATCH_SIZE = 64
SEED       = 42

torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# ── TEST DATASETS (module-level for multiprocessing pickling) ─────────────────
_test_tf_standard = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
_test_tf_hflip = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
_test_tf_vflip = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomVerticalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
_test_tf_crop = transforms.Compose([
    transforms.Resize((int(IMG_SIZE*1.1), int(IMG_SIZE*1.1))),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

TTA_TRANSFORMS = [_test_tf_standard, _test_tf_hflip, _test_tf_vflip, _test_tf_crop]

class TestDataset(Dataset):
    """Test set dataset — returns (img_tensor, filename)."""
    def __init__(self, file_list, tf):
        self.files = file_list
        self.tf    = tf
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        path = self.files[i]
        img  = Image.open(path).convert("RGB")
        return self.tf(img), path.name

# ── LOAD MODELS ───────────────────────────────────────────────────────────────
def load_resnet():
    from train_resnet import CustomResNet
    m = CustomResNet(num_classes=10).to(DEVICE)
    if RESNET_CKPT.exists():
        m.load_state_dict(torch.load(RESNET_CKPT, map_location=DEVICE))
        print(f"  [ResNet] Loaded ✓")
        return m, True
    print(f"  [ResNet] Checkpoint not found: {RESNET_CKPT}")
    return m, False

def load_efficient():
    from train_efficient import CustomEfficientNet
    m = CustomEfficientNet(num_classes=10).to(DEVICE)
    if EFFICIENT_CKPT.exists():
        m.load_state_dict(torch.load(EFFICIENT_CKPT, map_location=DEVICE))
        print(f"  [EfficientNet] Loaded ✓")
        return m, True
    print(f"  [EfficientNet] Checkpoint not found: {EFFICIENT_CKPT}")
    return m, False

# ── INFERENCE ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_with_tta(model, file_list):
    """Ensemble predictions over TTA transforms."""
    model.eval()
    pin = DEVICE.type == "cuda"
    all_probs = None

    for tf in TTA_TRANSFORMS:
        ds     = TestDataset(file_list, tf)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=pin)
        batch_probs = []
        for imgs, _ in loader:
            logits = model(imgs.to(DEVICE))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            batch_probs.append(probs)
        probs_arr = np.concatenate(batch_probs, axis=0)
        all_probs = probs_arr if all_probs is None else all_probs + probs_arr

    return all_probs / len(TTA_TRANSFORMS)

def get_filenames(file_list):
    """Run one pass to get filenames in loader order."""
    pin = DEVICE.type == "cuda"
    ds     = TestDataset(file_list, _test_tf_standard)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=pin)
    names = []
    for _, fnames in loader:
        names.extend(fnames)
    return names

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("="*55)
    print("  Inference + Submission – Modul 3 AI 2026")
    print(f"  Device: {DEVICE}")
    print("="*55)

    test_files = sorted(TEST_DIR.glob("*.jpg"))
    if not test_files:
        print(f"[ERROR] Tidak ada gambar di {TEST_DIR}"); return
    print(f"\nTest images  : {len(test_files)}")

    sample_df = pd.read_csv(SAMPLE_CSV)
    print(f"Sample rows  : {len(sample_df)}")

    resnet_ok  = RESNET_CKPT.exists()
    effnet_ok  = EFFICIENT_CKPT.exists()

    print("\n[Loading models...]")
    probs_combined = None

    if resnet_ok:
        model_r, _ = load_resnet()
        print("  [Predicting with TTA – ResNet...]")
        pr = predict_with_tta(model_r, test_files)
        probs_combined = pr

    if effnet_ok:
        model_e, _ = load_efficient()
        print("  [Predicting with TTA – EfficientNet...]")
        pe = predict_with_tta(model_e, test_files)
        probs_combined = pe if probs_combined is None else (probs_combined + pe) / 2

    if probs_combined is None:
        print("[ERROR] Tidak ada model yang berhasil dimuat!"); return

    preds     = probs_combined.argmax(axis=1)
    img_names = [f.name for f in test_files]
    pred_map  = {name: IDX2CLASS[int(p)] for name, p in zip(img_names, preds)}

    # align dengan sample_submission
    submission = sample_df.copy()
    submission["label"] = submission["image_id"].map(pred_map)
    missing = submission["label"].isna().sum()
    if missing > 0:
        print(f"[WARN] {missing} image_id tidak match – fillna dengan kelas random")
        submission["label"] = submission["label"].fillna(
            pd.Series(np.random.choice(CLASSES, size=len(submission)))
        )

    submission.to_csv(OUTPUT_CSV, index=False)

    # stats
    print(f"\n{'='*45}")
    print(f"  Submission saved : {OUTPUT_CSV}")
    print(f"  Total rows       : {len(submission)}")
    print(f"\n  Distribusi prediksi:")
    for cls, cnt in submission["label"].value_counts().items():
        bar = "█" * int(cnt / len(submission) * 30)
        print(f"    {cls:<25} {cnt:>5}  {bar}")
    print(f"\n  Preview (5 baris pertama):")
    print(submission.head().to_string(index=False))
    print("="*45)

if __name__ == "__main__":
    main()
