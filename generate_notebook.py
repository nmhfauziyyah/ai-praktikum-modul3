"""
generate_notebook.py
Membuat notebook.ipynb dari semua fase pengerjaan praktikum.
Jalankan: python generate_notebook.py
"""
import json, re
from pathlib import Path

def md(text):
    return {"cell_type":"markdown","metadata":{},"source":[text]}

def code(src):
    lines = [l + "\n" for l in src.split("\n")]
    if lines and lines[-1] == "\n":
        lines[-1] = ""
    return {
        "cell_type":"code",
        "execution_count":None,
        "metadata":{},"outputs":[],
        "source": lines
    }

def read_py(path):
    """Baca .py, hapus shebang/docstring awal & if __name__ block."""
    txt = Path(path).read_text(encoding="utf-8")
    # hapus if __name__ == "__main__": main()
    txt = re.sub(r'\nif __name__\s*==\s*["\']__main__["\']:.*', '',
                 txt, flags=re.DOTALL)
    return txt.strip()

# ── SOAL MARKDOWN ─────────────────────────────────────────────────────────────
TITLE = """# Praktikum Modul 3 AI 2026
## Satellite Image Classification — Land Cover Recognition

**Deskripsi:** Membangun model deep learning untuk mengklasifikasikan citra satelit ke dalam 10 kategori tutupan lahan menggunakan pendekatan **from scratch** (tanpa pretrained model / transfer learning).

**Kelas:** AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake

**Metrik evaluasi:** Macro F1 Score
"""

SOAL1 = """## Soal 1 — Exploratory Data Analysis (EDA)

Lakukan analisis eksploratif terhadap data (Exploratory Data Analysis), mencakup:
- Distribusi kelas pada train set
- Visualisasi sampel gambar per kelas
- Statistik ukuran gambar (width, height)
- Analisis rata-rata brightness per kelas
- Analisis rata-rata channel RGB per kelas
- Perbandingan kelas yang mirip secara visual
"""

SOAL2 = """## Soal 2 — Preprocessing & Augmentasi Data

Lakukan preprocessing dan augmentasi data yang sesuai sebagai persiapan sebelum pelatihan model:
- Resize gambar ke ukuran seragam
- Augmentasi: horizontal/vertical flip, rotasi, color jitter, random crop
- Normalisasi nilai piksel
- Stratified split train/validation
"""

SOAL3A = """## Soal 3A — Model 1: Custom ResNet (from scratch)

Bangun model klasifikasi pertama dengan arsitektur **ResNet-like** yang dibangun dari awal:
- Residual Blocks dengan skip connection
- 4 layer dengan jumlah filter bertingkat (64→128→256→512)
- AdaptiveAvgPool + Dropout + FC head
- Optimizer: AdamW + CosineAnnealingLR
- Loss: CrossEntropy dengan label smoothing
"""

SOAL3B = """## Soal 3B — Model 2: Custom EfficientNet-inspired (from scratch)

Bangun model klasifikasi kedua dengan arsitektur **EfficientNet-inspired** yang dibangun dari awal:
- MBConv blocks: Depthwise Separable Convolution + Squeeze-Excite attention
- 7 stage dengan channel progressif (32→16→24→40→80→112→192→320)
- Head: 1x1 Conv(1280) + AdaptiveAvgPool + FC
- Optimizer: AdamW + OneCycleLR
- Aktivasi: SiLU (Swish)
"""

SOAL4 = """## Soal 4 — Evaluasi & Error Analysis

Evaluasi dan bandingkan performa seluruh model yang telah dibangun menggunakan metrik yang relevan, serta lakukan analisis terhadap hasil prediksi yang salah (error analysis):
- Perbandingan Macro F1 Score kedua model
- Confusion matrix per model
- Per-class F1 score comparison
- Visualisasi sampel gambar yang salah diklasifikasi
- Top-10 most confused class pairs
"""

SOAL5 = """## Soal 5 — Inferensi & Submission

Gunakan model terbaik untuk melakukan inferensi pada data test dan simpan hasilnya dalam format submission yang telah ditentukan:
- Test-Time Augmentation (TTA) dengan 4 transform
- Ensemble kedua model (average probability)
- Output: submission.csv dengan kolom image_id, label
"""

# ── BACA SEMUA .py ────────────────────────────────────────────────────────────
BASE = Path(".")

eda_code        = read_py(BASE / "eda.py")
resnet_code     = read_py(BASE / "train_resnet.py")
efficient_code  = read_py(BASE / "train_efficient.py")
error_code      = read_py(BASE / "error_analysis.py")
inference_code  = read_py(BASE / "inference.py")

# Tambahkan pemanggilan fungsi main() di tiap cell
eda_code       += "\n\n# Jalankan EDA\nmain()"
resnet_code    += "\n\n# Jalankan training ResNet\nmain()"
efficient_code += "\n\n# Jalankan training EfficientNet\nmain()"
error_code     += "\n\n# Jalankan error analysis\nmain()"
inference_code += "\n\n# Generate submission\nmain()"

# ── SUSUN NOTEBOOK ────────────────────────────────────────────────────────────
cells = [
    md(TITLE),
    md(SOAL1),
    code(eda_code),
    md(SOAL2 + "\n\n> Preprocessing & augmentasi sudah diimplementasikan di dalam pipeline training (lihat `train_tf` dan `val_tf` pada cell berikut)."),
    md(SOAL3A),
    code(resnet_code),
    md(SOAL3B),
    code(efficient_code),
    md(SOAL4),
    code(error_code),
    md(SOAL5),
    code(inference_code),
    md("""## Hasil

| Model | Val Macro F1 |
|-------|-------------|
| Custom ResNet (from scratch) | *lihat output training* |
| Custom EfficientNet (from scratch) | *lihat output training* |
| **Ensemble (submission)** | **0.94312** |

Submission berhasil diunggah ke Kaggle dengan skor **0.94312** Macro F1 Score.
""")
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

out = BASE / "notebook_praktikum_modul3.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"[DONE] Notebook saved: {out}")
print(f"       Total cells   : {len(cells)}")
