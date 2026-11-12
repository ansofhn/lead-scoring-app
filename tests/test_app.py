import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from lead_utils import beri_skor, load_artifacts


def test_skor_valid():
    model, info = load_artifacts()
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "sample_leads.csv")
    hasil = beri_skor(model, info, df)
    assert len(hasil) == len(df)
    assert hasil["Probabilitas convert"].between(0, 1).all()
    assert set(hasil["Kategori"]) <= {"Hot", "Warm", "Cold"}


def test_lead_sangat_aktif_lebih_tinggi_dari_tidak_aktif():
    model, info = load_artifacts()
    base = {**info["num_defaults"], **info["cat_defaults"]}
    aktif = {**base, "Total Time Spent on Website": 1500, "TotalVisits": 8, "Last Activity": "SMS Sent"}
    pasif = {**base, "Total Time Spent on Website": 0, "TotalVisits": 0, "Last Activity": "Email Bounced"}
    s = beri_skor(model, info, pd.DataFrame([aktif, pasif]))["Probabilitas convert"]
    assert s.iloc[0] > s.iloc[1]
