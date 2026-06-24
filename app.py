import streamlit as st
import joblib
import pandas as pd
from streamlit_echarts import st_echarts
import numpy as np

# ─────────────────────────────────────────────
# PREPROCESSING & FEATURE ENGINEERING
# ─────────────────────────────────────────────

def preprocess_features(df_input):
    df_copy = df_input.copy()

    # 1. Hitung ulang rasio konsisten dengan training pipeline
    df_copy['loan_percent_income'] = df_copy['loan_amnt'] / df_copy['person_income']
    df_copy['loan_to_income_ratio'] = df_copy['loan_amnt'] / df_copy['person_income']

    # 2. Membuat Bucket / Binning yang sama persis dengan masa training
    df_copy['person_age_bucket'] = pd.cut(
        df_copy['person_age'], 
        bins=[0, 25, 35, 45, 55, 65, 100], 
        labels=['18-25', '26-35', '36-45', '46-55', '56-65', '66+'],
        include_lowest=True
    ).astype(str)

    df_copy['emp_length_bucket'] = pd.cut(
        df_copy['person_emp_length'], 
        bins=[0, 1, 3, 5, 10, 20, 100], 
        labels=['<1', '1-3', '4-5', '6-10', '11-20', '20+'],
        include_lowest=True
    ).astype(str)

    # Pastikan tipe data numerik konsisten bertipe float
    numeric_cols = [
        'person_age', 'person_income', 'person_emp_length', 'loan_amnt', 
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'loan_to_income_ratio'
    ]
    for col in numeric_cols:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype(float)
            
    return df_copy


# ─────────────────────────────────────────────
# STATISTIK & LOGIKA UNTUK TAMPILAN UI
# ─────────────────────────────────────────────

def summarize_dataset_by_status(df):
    """Hitung median tiap fitur berdasarkan loan_status (0=Layak, 1=Tidak Layak)."""
    required_statuses = {0, 1}
    available = set(df['loan_status'].unique())
    missing = required_statuses - available
    if missing:
        raise ValueError(
            f"Dataset tidak memiliki loan_status: {missing}. "
            "Pastikan dataset mengandung status 0 (Layak) dan 1 (Tidak Layak)."
        )

    summary_rows = {}
    for status, label in [(0, 'Layak'), (1, 'Tidak Layak')]:
        subset = df[df['loan_status'] == status]
        default_rate = (subset['cb_person_default_on_file'] == 'Y').mean()
        summary_rows[label] = {
            'loan_amnt': subset['loan_amnt'].median(),
            'person_income': subset['person_income'].median(),
            'loan_percent_income': (subset['loan_amnt'] / subset['person_income']).median(),
            'cb_person_cred_hist_length': subset['cb_person_cred_hist_length'].median(),
            'loan_int_rate': subset['loan_int_rate'].median(),
            'default_rate': default_rate,
        }

    return pd.DataFrame(summary_rows).T  # index: 'Layak', 'Tidak Layak'


def build_decision_reasons(applicant, summary):
    good_reasons = []
    bad_reasons = []

    if applicant['cb_person_default_on_file'] == 'N':
        good_reasons.append("✅ Tidak memiliki riwayat gagal bayar.")
    else:
        bad_reasons.append("⚠️ Terdapat riwayat gagal bayar sebelumnya.")

    lpi_threshold = summary.loc['Layak', 'loan_percent_income']
    if applicant['loan_percent_income'] <= lpi_threshold:
        good_reasons.append(
            f"✅ Rasio pinjaman/pendapatan ({applicant['loan_percent_income']:.2f}) "
            f"≤ median nasabah layak ({lpi_threshold:.2f})."
        )
    else:
        bad_reasons.append(
            f"⚠️ Rasio pinjaman/pendapatan ({applicant['loan_percent_income']:.2f}) "
            f"> median nasabah layak ({lpi_threshold:.2f})."
        )

    cred_threshold = summary.loc['Layak', 'cb_person_cred_hist_length']
    if applicant['cb_person_cred_hist_length'] >= cred_threshold:
        good_reasons.append(
            f"✅ Riwayat kredit ({applicant['cb_person_cred_hist_length']} tahun) "
            f"≥ median nasabah layak ({cred_threshold:.0f} tahun)."
        )
    else:
        bad_reasons.append(
            f"⚠️ Riwayat kredit ({applicant['cb_person_cred_hist_length']} tahun) "
            f"< median nasabah layak ({cred_threshold:.0f} tahun)."
        )

    rate_threshold = summary.loc['Layak', 'loan_int_rate']
    if applicant['loan_int_rate'] <= rate_threshold:
        good_reasons.append(
            f"✅ Suku bunga ({applicant['loan_int_rate']:.1f}%) "
            f"≤ median nasabah layak ({rate_threshold:.1f}%)."
        )
    else:
        bad_reasons.append(
            f"⚠️ Suku bunga ({applicant['loan_int_rate']:.1f}%) "
            f"> median nasabah layak ({rate_threshold:.1f}%)."
        )

    amnt_threshold = summary.loc['Layak', 'loan_amnt']
    if applicant['loan_amnt'] <= amnt_threshold:
        good_reasons.append(
            f"✅ Jumlah pinjaman (${applicant['loan_amnt']:,.0f}) "
            f"≤ median nasabah layak (${amnt_threshold:,.0f})."
        )
    else:
        bad_reasons.append(
            f"⚠️ Jumlah pinjaman (${applicant['loan_amnt']:,.0f}) "
            f"> median nasabah layak (${amnt_threshold:,.0f})."
        )

    inc_threshold = summary.loc['Layak', 'person_income']
    if applicant['person_income'] >= inc_threshold:
        good_reasons.append(
            f"✅ Pendapatan (${applicant['person_income']:,.0f}) "
            f"≥ median nasabah layak (${inc_threshold:,.0f})."
        )
    else:
        bad_reasons.append(
            f"⚠️ Pendapatan (${applicant['person_income']:,.0f}) "
            f"< median nasabah layak (${inc_threshold:,.0f})."
        )

    return good_reasons, bad_reasons


def build_comparison_table(applicant, summary):
    rows = [
        {
            'Fitur': 'Jumlah Pinjaman ($)',
            'Nasabah': f"${applicant['loan_amnt']:,.0f}",
            'Median Layak': f"${summary.loc['Layak', 'loan_amnt']:,.0f}",
            'Median Tidak Layak': f"${summary.loc['Tidak Layak', 'loan_amnt']:,.0f}",
        },
        {
            'Fitur': 'Pendapatan Tahunan ($)',
            'Nasabah': f"${applicant['person_income']:,.0f}",
            'Median Layak': f"${summary.loc['Layak', 'person_income']:,.0f}",
            'Median Tidak Layak': f"${summary.loc['Tidak Layak', 'person_income']:,.0f}",
        },
        {
            'Fitur': 'Rasio Pinjaman / Pendapatan',
            'Nasabah': f"{applicant['loan_percent_income']:.3f}",
            'Median Layak': f"{summary.loc['Layak', 'loan_percent_income']:.3f}",
            'Median Tidak Layak': f"{summary.loc['Tidak Layak', 'loan_percent_income']:.3f}",
        },
        {
            'Fitur': 'Riwayat Kredit (tahun)',
            'Nasabah': f"{applicant['cb_person_cred_hist_length']}",
            'Median Layak': f"{summary.loc['Layak', 'cb_person_cred_hist_length']:.0f}",
            'Median Tidak Layak': f"{summary.loc['Tidak Layak', 'cb_person_cred_hist_length']:.0f}",
        },
        {
            'Fitur': 'Suku Bunga (%)',
            'Nasabah': f"{applicant['loan_int_rate']:.1f}%",
            'Median Layak': f"{summary.loc['Layak', 'loan_int_rate']:.1f}%",
            'Median Tidak Layak': f"{summary.loc['Tidak Layak', 'loan_int_rate']:.1f}%",
        },
        {
            'Fitur': 'Tingkat Default Historis',
            'Nasabah': '✅ Tidak' if applicant['cb_person_default_on_file'] == 'N' else '❌ Ya',
            'Median Layak': f"{summary.loc['Layak', 'default_rate']:.1%}",
            'Median Tidak Layak': f"{summary.loc['Tidak Layak', 'default_rate']:.1%}",
        },
    ]
    return pd.DataFrame(rows)


def calculate_payment_success_rate(df, loan_grade):
    subset = df[df['loan_grade'] == loan_grade]
    total = len(subset)
    if total == 0:
        return 0.0, 0, 0
    success = int((subset['loan_status'] == 0).sum())
    rate = success / total * 100
    return rate, success, total


# ─────────────────────────────────────────────
# KONFIGURASI HALAMAN STREAMLIT
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Credit Risk Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.block-container{
    padding-top: 1.5rem;
    max-width: 1400px;
}
h1,h2,h3{
    color:#1e293b;
}
[data-testid="metric-container"]{
    background:white;
    border:none;
    border-radius:16px;
    padding:20px;
    box-shadow:0 2px 12px rgba(0,0,0,.06);
}
div.stButton > button{
    width:100%;
    border-radius:12px;
    height:3.3em;
    font-weight:600;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL & DATA HISTORIS
# ─────────────────────────────────────────────

@st.cache_resource
def load_model():
    return joblib.load('models/credit_risk_model.pkl')

@st.cache_data
def load_data():
    df = pd.read_csv('data/credit_risk_dataset.csv')
    return df

model = load_model()
df_stats = load_data()
dataset_summary = summarize_dataset_by_status(df_stats)


# ─────────────────────────────────────────────
# HEADER DASHBOARD
# ─────────────────────────────────────────────

st.title("Credit Risk Analytics Dashboard")
st.markdown(
    "Sistem analisis kelayakan pinjaman berbasis Machine Learning. "
    "Isi profil nasabah di sidebar, lalu klik **Proses Analisis** untuk melihat hasil."
)
st.divider()

# ── Ringkasan Dataset ──
st.subheader("Ringkasan Data Historis")
total_nasabah = len(df_stats)
layak_pct = (df_stats['loan_status'] == 0).mean() * 100
tidak_layak_pct = 100 - layak_pct

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Nasabah", f"{total_nasabah:,}")
col2.metric("Rata-rata Pinjaman", f"${df_stats['loan_amnt'].mean():,.0f}")
col3.metric("Nasabah Layak", f"{layak_pct:.1f}%")
col4.metric("Nasabah Tidak Layak", f"{tidak_layak_pct:.1f}%")

# ── Visualisasi Dataset ──
with st.expander("Lihat Distribusi Data Historis", expanded=False):
    fig_col1, fig_col2 = st.columns(2)

    with fig_col1: 
        layak = df_stats[df_stats["loan_status"] == 0]["loan_amnt"]
        tidak_layak = df_stats[df_stats["loan_status"] == 1]["loan_amnt"]
        hist_layak, bins = np.histogram(layak, bins=20)
        hist_tidak, _ = np.histogram(tidak_layak, bins=bins)
        
        option_hist = {
            "tooltip": {"trigger": "axis"},
            "legend": {
                "data": ["Layak", "Tidak Layak"]
            },
            "xAxis": {
                "type": "category",
                "data": [str(int(x)) for x in bins[:-1]]
            },
            "yAxis": {
                "type": "value"
            },
            "series": [
                {
                    "name": "Layak",
                    "type": "bar",
                    "data": hist_layak.tolist()
                },
                {
                    "name": "Tidak Layak",
                    "type": "bar",
                    "data": hist_tidak.tolist()
                }
            ]
        }
        st_echarts(options=option_hist, height="450px")

    with fig_col2:
        grade_data = df_stats.groupby(["loan_grade", "loan_status"]).size().unstack(fill_value=0)

        option_grade = {
            "tooltip": {"trigger": "axis"},
            "legend": {
                "data": ["Layak", "Tidak Layak"]
            },
            "xAxis": {
                "type": "category",
                "data": grade_data.index.tolist()
            },
            "yAxis": {
                "type": "value"
            },
            "series": [
                {
                    "name": "Layak",
                    "type": "bar",
                    "data": grade_data[0].tolist()
                },
                {
                    "name": "Tidak Layak",
                    "type": "bar",
                    "data": grade_data[1].tolist()
                }
            ]
        }
        st_echarts(options=option_grade, height="450px")

st.divider()


# ─────────────────────────────────────────────
# SIDEBAR INPUT USER
# ─────────────────────────────────────────────

st.sidebar.header("Profil Nasabah")
st.sidebar.markdown("Isi seluruh data di bawah ini sebelum memproses analisis.")

with st.sidebar.expander("Data Pribadi", expanded=True):
    person_age = st.number_input("Usia", min_value=18, max_value=100, value=30, help="Usia nasabah dalam tahun")
    person_income = st.number_input(
        "Pendapatan Tahunan ($)", min_value=1, max_value=10_000_000, value=50_000, step=1_000,
        help="Total pendapatan tahunan nasabah sebelum pajak"
    )
    person_home_ownership = st.selectbox(
        "Kepemilikan Rumah",
        ['RENT', 'OWN', 'MORTGAGE', 'OTHER'],
        help="Status kepemilikan tempat tinggal nasabah"
    )
    person_emp_length = st.number_input(
        "Lama Bekerja (Tahun)", min_value=0.0, max_value=70.0, value=5.0, step=0.5,
        help="Durasi bekerja di pekerjaan saat ini"
    )

st.sidebar.markdown("---")


# ─────────────────────────────────────────────
# MAIN AREA: INPUT PINJAMAN & KREDIT
# ─────────────────────────────────────────────

st.subheader("Input Detail Pinjaman & Riwayat Kredit")
tab1, tab2 = st.tabs(["Detail Pinjaman", "Riwayat Kredit"])

with tab1:
    c1, c2 = st.columns(2)
    loan_intent = c1.selectbox(
        "Tujuan Pinjaman",
        ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOME_IMPROVEMENT', 'DEBT_CONSOLIDATION', 'OTHER'],
        help="Alasan utama nasabah mengajukan pinjaman"
    )
    loan_grade = c2.selectbox(
        "Grade Pinjaman",
        ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        help="Grade A = risiko terendah, G = risiko tertinggi"
    )
    loan_amnt = c1.number_input(
        "Jumlah Pinjaman ($)", min_value=1, max_value=1_000_000, value=10_000, step=500,
        help="Total jumlah pinjaman yang diajukan"
    )
    loan_int_rate = c2.number_input(
        "Suku Bunga (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1,
        help="Suku bunga tahunan yang ditawarkan"
    )

    loan_percent_income = loan_amnt / person_income if person_income > 0 else 0.0
    st.info(
        f"**Rasio Pinjaman / Pendapatan:** `{loan_percent_income:.3f}` "
        f"(dihitung otomatis dari jumlah pinjaman ÷ pendapatan tahunan)"
    )

with tab2:
    c3, c4 = st.columns(2)
    cb_person_default_on_file = c3.selectbox(
        "Riwayat Gagal Bayar",
        ['N', 'Y'],
        format_func=lambda x: "Tidak Ada (N)" if x == 'N' else "Ada (Y)",
        help="Apakah nasabah pernah tercatat gagal bayar sebelumnya?"
    )
    cb_person_cred_hist_length = c4.number_input(
        "Lama Riwayat Kredit (Tahun)", min_value=0, max_value=60, value=3,
        help="Berapa tahun nasabah sudah memiliki riwayat kredit"
    )

st.divider()


# ─────────────────────────────────────────────
# TOMBOL ANALISIS KELAYAKAN (MODIFIED LOGIC)
# ─────────────────────────────────────────────

if st.button("Proses Analisis Kelayakan"):

    # Membuat dataframe awal mentah dari input pengguna
    raw_input_data = pd.DataFrame([{
        'person_age': person_age,
        'person_income': person_income,
        'person_home_ownership': person_home_ownership,
        'person_emp_length': person_emp_length,
        'loan_intent': loan_intent,
        'loan_grade': loan_grade,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'cb_person_default_on_file': cb_person_default_on_file,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
    }])

    # Proses Preprocessing Pipeline
    input_data = preprocess_features(raw_input_data)

    # Prediksi Klasifikasi Model (0 atau 1)
    prediction = model.predict(input_data)[0]

    # Ambil probabilitas kontinu risiko untuk evaluasi margin
    risk_score = 0.0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[0][1]
        risk_score = round(proba * 100, 1)

    st.markdown("## Hasil Analisis Kelayakan")

    # KONDISI 1: JIKA MODEL MUTLAK MENOLAK (RISIKO EKSTREM)
    if prediction == 1:
        st.error(
            f"❌ **STATUS: PINJAMAN DITOLAK (Skor Risiko: {risk_score}%)** \n\n"
            "Profil risiko nasabah tergolong **Tinggi**. "
            "Sistem otomatis menolak pengajuan ini karena indikator beban finansial terlalu berbahaya."
        )
        
    # KONDISI 2: JIKA MODEL MENERIMA, TAPI MASUK MARGIN RISIKO TINGGI (ZONA WASPADA / WATCHLIST)
    elif prediction == 0 and risk_score >= 20.0:
        st.warning(
            f"⚠️ **STATUS: DISETUJUI BERSYARAT / PERLU PENINJAUAN MANUAL (Skor Risiko: {risk_score}%)** \n\n"
            "**Catatan Analis:** Meskipun model mengklasifikasikan nasabah ini 'Diterima' karena **tidak memiliki riwayat gagal bayar historis (BI Checking bersih)**, "
            f"sistem mendeteksi adanya akumulasi margin risiko yang lumayan besar ({risk_score}% dari ambang batas aman <20%).\n\n"
            "**Faktor Kerawanan Utama:**\n"
            f"* **Beban Utang Finansial Tinggi:** Rasio pinjaman terhadap pendapatan sebesar `{loan_percent_income:.2f}` telah melewati batas ideal kelompok layak, ditambah suku bunga yang tinggi ({loan_int_rate}%) memperberat cicilan bulanan.\n"
            "* **Potensi Krisis Baru:** Nasabah tipe ini berisiko menjadi kelompok gagal bayar untuk pertama kalinya (*first-time defaulter*) karena kapasitas likuiditas bulanan yang mepet.\n\n"
            "👉 *Rekomendasi: Diperlukan peninjauan manual (Manual Underwriting) oleh komite kredit sebelum pencairan dana dilakukan.*"
        )
        
    # KONDISI 3: JIKA MODEL MENERIMA DAN BENAR-BENAR AMAN
    else:
        st.success(
            f"✅ **STATUS: PINJAMAN DITERIMA MUTLAK (Skor Risiko: {risk_score}%)** \n\n"
            "Profil risiko nasabah tergolong **Sangat Layak**. "
            "Seluruh indikator keuangan sehat dan pinjaman dapat langsung diproses tanpa catatan."
        )

    # ── Statistik Grade ──
    st.markdown("### Statistik Nasabah dengan Grade yang Sama")
    success_rate, success_count, total_count = calculate_payment_success_rate(df_stats, loan_grade)

    if total_count == 0:
        st.warning(f"Tidak ada data historis untuk nasabah dengan grade '{loan_grade}'.")
    else:
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Tingkat Keberhasilan Bayar", f"{success_rate:.1f}%")
        sc2.metric("Nasabah Berhasil Bayar", f"{success_count:,}")
        sc3.metric(f"Total Nasabah Grade '{loan_grade}'", f"{total_count:,}")
        st.caption(
            f"Dari {total_count:,} nasabah dengan grade pinjaman **{loan_grade}** di dataset historis, "
            f"**{success_rate:.1f}%** berhasil melunasi pinjaman mereka."
        )

    st.divider()

    # ── Alasan Keputusan ──
    st.markdown("### Faktor Penentu Keputusan")
    applicant_features = {
        'loan_amnt': loan_amnt,
        'person_income': person_income,
        'loan_percent_income': loan_percent_income,
        'cb_person_default_on_file': cb_person_default_on_file,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'loan_int_rate': loan_int_rate,
    }
    good_reasons, bad_reasons = build_decision_reasons(applicant_features, dataset_summary)

    reason_col1, reason_col2 = st.columns(2)

    with reason_col1:
        st.markdown("**Faktor Positif**")
        if good_reasons:
            for r in good_reasons:
                st.markdown(r)
        else:
            st.markdown("_Tidak ada faktor positif yang teridentifikasi._")

    with reason_col2:
        st.markdown("**Faktor Risiko**")
        if bad_reasons:
            for r in bad_reasons:
                st.markdown(r)
        else:
            st.markdown("_Tidak ada faktor risiko yang teridentifikasi._")

    st.divider()

    # ── Tabel Komparasi ──
    st.markdown("### Perbandingan dengan Dataset Historis")
    comparison_df = build_comparison_table(applicant_features, dataset_summary)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # ── Gauge Chart Skor Risiko ──
    if hasattr(model, "predict_proba"):
        gauge_option = {
            "series": [
                {
                    "type": "gauge",
                    "radius": "90%",
                    "startAngle": 210,
                    "endAngle": -30,
                    "progress": {"show": True, "width": 18},
                    "axisLine": {"lineStyle": {"width": 18}},
                    "pointer": {"show": True},
                    "detail": {
                        "fontSize": 32,
                        "formatter": "{value}%"
                    },
                    "data": [{
                        "value": risk_score
                    }]
                }
            ]
        }

        st.markdown("### Skor Risiko Model")
        st_echarts(options=gauge_option, height="400px")