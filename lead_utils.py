"""Fungsi pendukung: memuat model, membersihkan input, dan memberi skor lead."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "lead_scoring_model.pth"   # pipeline (joblib) hasil Colab
INFO_PATH = BASE_DIR / "model" / "model_info.json"           # metadata hasil Colab

KOSONG = "(kosong / tidak diisi)"


def load_artifacts():
    """Muat pipeline dan metadata model."""
    model = joblib.load(MODEL_PATH)
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(
            f"{type(e).__name__}: {e} | size={MODEL_PATH.stat().st_size}"
        ) from e
    info = json.loads(INFO_PATH.read_text(encoding="utf-8"))
    return model, info


def fitur_model(info):
    return info["num_cols"] + info["cat_cols"]


def bersihkan(df, info):
    """Langkah cleaning yang sama dengan notebook: 'Select' -> kosong, trim spasi,
    perbaiki penulisan 'google', lalu clipping outlier (batas IQR dari data latih)."""
    kolom = fitur_model(info)
    hilang = [c for c in kolom if c not in df.columns]
    if hilang:
        raise ValueError("Kolom wajib tidak ditemukan: " + ", ".join(hilang))

    d = df[kolom].copy()
    for c in info["cat_cols"]:
        d[c] = d[c].apply(lambda v: v.strip() if isinstance(v, str) else v)
        d[c] = d[c].replace({"Select": np.nan, KOSONG: np.nan, "": np.nan})
        d[c] = d[c].astype(object).where(d[c].notna(), np.nan)
    if "Lead Source" in d.columns:
        d["Lead Source"] = d["Lead Source"].replace({"google": "Google"})
    for c in info["num_cols"]:
        lo, hi = info["clip_bounds"][c]
        d[c] = pd.to_numeric(d[c], errors="coerce").clip(lo, hi)
    return d


def kategori(p, info):
    if p >= info["hot_cutoff"]:
        return "Hot"
    if p >= info["warm_cutoff"]:
        return "Warm"
    return "Cold"


def beri_skor(model, info, df):
    """Kembalikan DataFrame hasil: probabilitas, skor 0-100, kategori, dan prioritas."""
    proba = model.predict_proba(bersihkan(df, info))[:, 1]
    hasil = pd.DataFrame({
        "Probabilitas convert": np.round(proba, 4),
        "Skor (0-100)": np.round(proba * 100).astype(int),
        "Kategori": [kategori(p, info) for p in proba],
        "Prioritas follow-up": np.where(proba >= info["threshold"], "Ya", "Tidak"),
    }, index=df.index)
    return hasil


def cek_versi(info):
    """Peringatan jika versi library berbeda dengan saat model dilatih."""
    import sklearn
    import xgboost
    import imblearn
    now = {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
           "imbalanced-learn": imblearn.__version__}
    return [f"{k}: model dilatih dengan {info['versions'].get(k)}, terpasang {v}"
            for k, v in now.items() if info["versions"].get(k) != v]
