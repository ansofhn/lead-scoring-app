"""Aplikasi Lead Scoring - prediksi minat beli (konversi) calon pelanggan."""
import pandas as pd
import streamlit as st

from lead_utils import KOSONG, beri_skor, cek_versi, fitur_model, load_artifacts

st.set_page_config(page_title="Lead Scoring", page_icon="📈", layout="wide")


@st.cache_resource
def muat():
    return load_artifacts()


try:
    model, info = muat()
except Exception as e:  # file model hilang / versi tidak cocok
    st.error(f"Model gagal dimuat: {e}")
    st.stop()

st.title("📈 Lead Scoring")
st.caption("Memprediksi peluang sebuah lead berubah menjadi pelanggan (Converted), "
           "lalu mengelompokkannya menjadi Hot, Warm, atau Cold.")
for pesan in cek_versi(info):
    st.warning("Versi library berbeda - " + pesan)

with st.sidebar:
    st.header("Tentang model")
    st.write(f"**{info['model_name']}**")
    m = info["metrics_test"]
    c1, c2 = st.columns(2)
    c1.metric("ROC-AUC", f"{m['ROC-AUC']:.3f}")
    c2.metric("F1", f"{m['F1']:.3f}")
    c1.metric("Recall", f"{m['Recall']:.3f}")
    c2.metric("Precision", f"{m['Precision']:.3f}")
    st.caption("Metrik dihitung pada data uji 20% (lihat notebook).")
    st.divider()
    st.write("**Aturan kategori**")
    st.write(f"- Hot: probabilitas ≥ {info['hot_cutoff']:.0%}")
    st.write(f"- Warm: {info['warm_cutoff']:.0%} - {info['hot_cutoff']:.0%}")
    st.write(f"- Cold: < {info['warm_cutoff']:.0%}")
    st.write(f"- Prioritas follow-up: probabilitas ≥ {info['threshold']:.1%}")

tab1, tab2, tab3 = st.tabs(["Prediksi satu lead", "Prediksi banyak lead (CSV)", "Fitur model"])

# ---------------------------------------------------------------- tab 1
with tab1:
    UTAMA = ["Lead Origin", "Lead Source", "Last Activity",
             "What is your current occupation", "Specialization"]
    utama = [c for c in UTAMA if c in info["cat_cols"]]
    tambahan = [c for c in info["cat_cols"] if c not in utama]

    def pilih(kolom, wadah):
        opsi = info["cat_options"][kolom] + [KOSONG]
        awal = info["cat_defaults"][kolom]
        return wadah.selectbox(kolom, opsi, index=opsi.index(awal) if awal in opsi else 0, key=f"s_{kolom}")

    with st.form("form_lead"):
        st.subheader("Perilaku di website")
        n1, n2, n3 = st.columns(3)
        nilai = {}
        for wadah, kol, langkah, bantuan in [
            (n1, "TotalVisits", 1.0, "Jumlah kunjungan ke website"),
            (n2, "Total Time Spent on Website", 10.0, "Total waktu di website (detik)"),
            (n3, "Page Views Per Visit", 0.5, "Rata-rata halaman dibuka per kunjungan")]:
            nilai[kol] = wadah.number_input(kol, min_value=0.0, value=float(info["num_defaults"][kol]),
                                            step=langkah, help=bantuan, key=f"n_{kol}")
        st.subheader("Profil dan sumber lead")
        kiri, kanan = st.columns(2)
        for i, kol in enumerate(utama):
            nilai[kol] = pilih(kol, kiri if i % 2 == 0 else kanan)
        with st.expander("Data tambahan (opsional, sudah terisi nilai paling umum)"):
            a, b = st.columns(2)
            for i, kol in enumerate(tambahan):
                nilai[kol] = pilih(kol, a if i % 2 == 0 else b)
        kirim = st.form_submit_button("Hitung skor", type="primary")

    if kirim:
        hasil = beri_skor(model, info, pd.DataFrame([nilai])).iloc[0]
        p = float(hasil["Probabilitas convert"])
        k1, k2, k3 = st.columns(3)
        k1.metric("Skor lead (0-100)", int(hasil["Skor (0-100)"]))
        k2.metric("Probabilitas convert", f"{p:.1%}")
        k3.metric("Kategori", hasil["Kategori"])
        st.progress(min(max(p, 0.0), 1.0))
        if hasil["Kategori"] == "Hot":
            st.success("Lead **Hot**: prioritaskan follow-up segera oleh tim sales.")
        elif hasil["Kategori"] == "Warm":
            st.warning("Lead **Warm**: lanjutkan nurturing (email, konten, telepon ringan).")
        else:
            st.info("Lead **Cold**: peluang rendah, cukup otomasi/nurturing berbiaya rendah.")

# ---------------------------------------------------------------- tab 2
with tab2:
    st.write("Unggah CSV dengan kolom seperti `Leads.csv` (kolom lain diabaikan). "
             "Kolom wajib: " + ", ".join(f"`{c}`" for c in fitur_model(info)))
    unggah = st.file_uploader("File CSV", type="csv")
    pakai_contoh = st.button("Pakai data contoh (30 lead)")
    df = None
    if unggah is not None:
        df = pd.read_csv(unggah)
    elif pakai_contoh:
        df = pd.read_csv("data/sample_leads.csv")
    if df is not None:
        try:
            hasil = beri_skor(model, info, df)
        except ValueError as e:
            st.error(str(e))
        else:
            kunci = [c for c in ["Lead Number", "Prospect ID"] if c in df.columns][:1]
            tabel = pd.concat([df[kunci], hasil], axis=1)
            ringkas = hasil["Kategori"].value_counts().reindex(["Hot", "Warm", "Cold"], fill_value=0)
            c1, c2, c3 = st.columns(3)
            c1.metric("Hot", int(ringkas["Hot"])); c2.metric("Warm", int(ringkas["Warm"]))
            c3.metric("Cold", int(ringkas["Cold"]))
            st.bar_chart(ringkas)
            st.dataframe(tabel.sort_values("Skor (0-100)", ascending=False), width="stretch")
            st.download_button("Unduh hasil (CSV)", tabel.to_csv(index=False).encode("utf-8"),
                               "hasil_lead_scoring.csv", "text/csv")

# ---------------------------------------------------------------- tab 3
with tab3:
    st.write("Fitur numerik:", ", ".join(info["num_cols"]))
    st.write("Fitur kategorikal:", ", ".join(info["cat_cols"]))
    st.json({"versi_library_saat_training": info["versions"], "threshold": info["threshold"]})
