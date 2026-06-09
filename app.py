import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_features(df_input):
    df_copy = df_input.copy()

    for col in ['person_emp_length', 'loan_int_rate']:
        df_copy[col] = df_copy.groupby('loan_grade')[col].transform(
            lambda grp: grp.fillna(grp.median())
        )
        df_copy[col] = df_copy[col].fillna(df_copy[col].median())

    df_copy['loan_percent_income'] = (
        df_copy['loan_amnt'] / df_copy['person_income'].replace({0: pd.NA})
    )
    df_copy['loan_percent_income'] = (
        df_copy['loan_percent_income']
        .replace([pd.NA, float('inf'), float('-inf')], pd.NA)
        .fillna(df_copy['loan_percent_income'].median())
    )

    df_copy['person_age_bucket'] = pd.cut(
        df_copy['person_age'],
        bins=[0, 25, 35, 45, 55, 65, 100],
        labels=['18-25', '26-35', '36-45', '46-55', '56-65', '66+'],
        include_lowest=True
    ).astype(object).fillna('missing')

    df_copy['emp_length_bucket'] = pd.cut(
        df_copy['person_emp_length'],
        bins=[0, 1, 3, 5, 10, 20, 100],
        labels=['<1', '1-3', '4-5', '6-10', '11-20', '20+'],
        include_lowest=True
    ).astype(object).fillna('missing')

    df_copy['loan_int_rate_bucket'] = pd.cut(
        df_copy['loan_int_rate'],
        bins=[0, 5, 10, 15, 20, 100],
        labels=['<=5', '5-10', '10-15', '15-20', '>20'],
        include_lowest=True
    ).astype(object).fillna('missing')

    df_copy['loan_income_ratio'] = (
        df_copy['loan_amnt'] / df_copy['person_income'].replace({0: pd.NA})
    )
    df_copy['loan_income_ratio'] = (
        df_copy['loan_income_ratio']
        .replace([pd.NA, float('inf'), float('-inf')], pd.NA)
        .fillna(df_copy['loan_income_ratio'].median())
    )

    numeric_cols = [
        'person_age', 'person_income', 'person_emp_length',
        'loan_amnt', 'loan_int_rate', 'loan_percent_income',
        'cb_person_cred_hist_length', 'loan_income_ratio'
    ]
    df_copy[numeric_cols] = df_copy[numeric_cols].astype(float)
    return df_copy


# ─────────────────────────────────────────────
# STATISTIK & LOGIKA
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
            'loan_percent_income': subset['loan_percent_income'].median(),
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
            f"✅ Suku bunga ({applicant['loan_int_rate']}%) "
            f"≤ median nasabah layak ({rate_threshold:.1f}%)."
        )
    else:
        bad_reasons.append(
            f"⚠️ Suku bunga ({applicant['loan_int_rate']}%) "
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
    """
    Hitung tingkat keberhasilan pembayaran nasabah dengan loan_grade yang sama.
    Mengembalikan (success_rate_pct, success_count, total_count).
    """
    subset = df[df['loan_grade'] == loan_grade]
    total = len(subset)
    if total == 0:
        return 0.0, 0, 0
    success = int((subset['loan_status'] == 0).sum())
    rate = success / total * 100
    return rate, success, total


# ─────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Credit Risk Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Tombol utama ── */
div.stButton > button {
    width: 100%;
    border-radius: 8px;
    height: 3.2em;
    background-color: #0055b3;
    color: white;
    font-weight: 600;
    font-size: 1rem;
    border: none;
    transition: background-color 0.2s ease;
}
div.stButton > button:hover {
    background-color: #003d82;
    color: white;
}

/* ── Card metrik ── */
[data-testid="metric-container"] {
    background-color: #f0f4fa;
    border: 1px solid #d0daea;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}

/* ── Sidebar label ── */
.css-1d391kg { font-size: 0.9rem; }

/* ── Tabel lebih rapi ── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL & DATA
# ─────────────────────────────────────────────

@st.cache_resource
def load_model():
    return joblib.load('models/credit_risk_model.pkl')

@st.cache_data
def load_data():
    df = pd.read_csv('data/credit_risk_dataset.csv')
    # Pastikan kolom loan_percent_income tersedia untuk summarize
    if 'loan_percent_income' not in df.columns:
        df['loan_percent_income'] = df['loan_amnt'] / df['person_income'].replace({0: pd.NA})
        df['loan_percent_income'] = (
            df['loan_percent_income']
            .replace([float('inf'), float('-inf')], pd.NA)
            .fillna(df['loan_percent_income'].median())
        )
    return df

model = load_model()
df_stats = load_data()
dataset_summary = summarize_dataset_by_status(df_stats)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.title("🏦 Credit Risk Analytics Dashboard")
st.markdown(
    "Sistem analisis kelayakan pinjaman berbasis Machine Learning. "
    "Isi profil nasabah di sidebar, lalu klik **Proses Analisis** untuk melihat hasil."
)
st.divider()

# ── Ringkasan Dataset ──
st.subheader("📋 Ringkasan Data Historis")
total_nasabah = len(df_stats)
layak_pct = (df_stats['loan_status'] == 0).mean() * 100
tidak_layak_pct = 100 - layak_pct

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Nasabah", f"{total_nasabah:,}")
col2.metric("Rata-rata Pinjaman", f"${df_stats['loan_amnt'].mean():,.0f}")
col3.metric("Nasabah Layak", f"{layak_pct:.1f}%")
col4.metric("Nasabah Tidak Layak", f"{tidak_layak_pct:.1f}%")

# ── Visualisasi Dataset ──
with st.expander("📊 Lihat Distribusi Data Historis", expanded=False):
    fig_col1, fig_col2 = st.columns(2)

    with fig_col1:
        fig_hist = px.histogram(
            df_stats, x="loan_amnt",
            color="loan_status",
            color_discrete_map={0: "#2196F3", 1: "#F44336"},
            labels={"loan_amnt": "Jumlah Pinjaman ($)", "loan_status": "Status"},
            title="Distribusi Jumlah Pinjaman",
            nbins=40,
            barmode="overlay",
            opacity=0.75,
        )
        fig_hist.update_layout(
            legend_title_text="Status",
            legend=dict(
                itemsizing='constant',
                title_text='Status',
            )
        )
        # Ganti label legend angka jadi teks deskriptif
        fig_hist.for_each_trace(lambda t: t.update(
            name="Layak" if t.name == "0" else "Tidak Layak"
        ))
        st.plotly_chart(fig_hist, use_container_width=True)

    with fig_col2:
        grade_counts = df_stats.groupby(['loan_grade', 'loan_status']).size().reset_index(name='count')
        grade_counts['Status'] = grade_counts['loan_status'].map({0: 'Layak', 1: 'Tidak Layak'})
        fig_grade = px.bar(
            grade_counts, x='loan_grade', y='count', color='Status',
            color_discrete_map={'Layak': '#2196F3', 'Tidak Layak': '#F44336'},
            labels={'loan_grade': 'Grade', 'count': 'Jumlah Nasabah'},
            title="Distribusi Nasabah per Grade",
            barmode='group',
        )
        st.plotly_chart(fig_grade, use_container_width=True)

st.divider()


# ─────────────────────────────────────────────
# SIDEBAR INPUT
# ─────────────────────────────────────────────

st.sidebar.header("👤 Profil Nasabah")
st.sidebar.markdown("Isi seluruh data di bawah ini sebelum memproses analisis.")

with st.sidebar.expander("Data Pribadi", expanded=True):
    person_age = st.number_input("Usia", min_value=18, max_value=100, value=30, help="Usia nasabah dalam tahun")
    person_income = st.number_input(
        "Pendapatan Tahunan ($)", min_value=0, max_value=10_000_000, value=50_000, step=1_000,
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

st.subheader("📝 Input Detail Pinjaman & Riwayat Kredit")
tab1, tab2 = st.tabs(["💳 Detail Pinjaman", "📁 Riwayat Kredit"])

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
        "Jumlah Pinjaman ($)", min_value=0, max_value=1_000_000, value=10_000, step=500,
        help="Total jumlah pinjaman yang diajukan"
    )
    loan_int_rate = c2.number_input(
        "Suku Bunga (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1,
        help="Suku bunga tahunan yang ditawarkan"
    )

    loan_percent_income = loan_amnt / person_income if person_income > 0 else 0.0
    st.info(
        f"📌 **Rasio Pinjaman / Pendapatan:** `{loan_percent_income:.3f}` "
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
# TOMBOL ANALISIS
# ─────────────────────────────────────────────

if st.button("🚀 Proses Analisis Kelayakan"):

    input_data = pd.DataFrame([{
        'person_age': person_age,
        'person_income': person_income,
        'person_home_ownership': person_home_ownership,
        'person_emp_length': person_emp_length,
        'loan_intent': loan_intent,
        'loan_grade': loan_grade,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_default_on_file': cb_person_default_on_file,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
    }])

    # ── Prediksi ──
    prediction = model.predict(input_data)[0]

    st.markdown("## 🔍 Hasil Analisis Kelayakan")

    if prediction == 1:
        st.error(
            "❌ **STATUS: PINJAMAN DITOLAK**  \n"
            "Profil risiko nasabah tergolong **Tinggi**. "
            "Lihat detail faktor di bawah untuk panduan perbaikan."
        )
    else:
        st.success(
            "✅ **STATUS: PINJAMAN DITERIMA**  \n"
            "Profil risiko nasabah tergolong **Layak**. "
            "Pinjaman dapat diproses lebih lanjut."
        )

    # ── Statistik Grade ──
    st.markdown("### 📊 Statistik Nasabah dengan Grade yang Sama")
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
    st.markdown("### 📋 Faktor Penentu Keputusan")
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
    st.markdown("### 📐 Perbandingan dengan Dataset Historis")
    st.caption(
        "Nilai nasabah dibandingkan dengan **median** kelompok Layak dan Tidak Layak "
        "di dataset historis (`loan_status = 0` → Layak, `loan_status = 1` → Tidak Layak)."
    )
    comparison_df = build_comparison_table(applicant_features, dataset_summary)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # ── Gauge chart skor ──
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[0][1]  # prob kelas 1 = tidak layak
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(proba * 100, 1),
            title={'text': "Skor Risiko (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#F44336" if proba >= 0.5 else "#2196F3"},
                'steps': [
                    {'range': [0, 40], 'color': "#e8f5e9"},
                    {'range': [40, 60], 'color': "#fff8e1"},
                    {'range': [60, 100], 'color': "#ffebee"},
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.8,
                    'value': 50,
                }
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=40, b=10))
        st.markdown("### 🎯 Skor Risiko Model")
        st.caption("Skor di atas 50% mengindikasikan risiko tinggi (pinjaman cenderung ditolak).")
        st.plotly_chart(fig_gauge, use_container_width=True)