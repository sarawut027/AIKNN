from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model_utils import (
    CHEST_PAIN_MAP,
    EXERCISE_ANGINA_MAP,
    FASTING_BS_MAP,
    FEATURE_COLUMNS,
    FEATURE_THAI_NAMES,
    RESTING_ECG_MAP,
    SEX_MAP,
    ST_SLOPE_MAP,
    TARGET_COLUMN,
    TARGET_MAP,
    evaluate_k_range,
    load_data,
    predict_batch,
    predict_single,
    train_knn_model,
)

st.set_page_config(
    page_title="ระบบจำแนกความเสี่ยงโรคหัวใจ | KNN",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans Thai', sans-serif;
}

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
    max-width: 1280px;
}

.hero-banner {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 65%, #334155 100%);
    border-radius: 18px;
    padding: 1.75rem 2rem;
    color: #FFFFFF;
    margin-bottom: 1.5rem;
    border-left: 6px solid #E11D48;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.12);
}

.hero-badge {
    display: inline-block;
    background: rgba(225, 29, 72, 0.2);
    color: #FDA4AF;
    border: 1px solid rgba(253, 164, 175, 0.35);
    padding: 0.2rem 0.75rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 0.6rem;
}

.hero-title {
    font-size: 1.85rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
    line-height: 1.3;
}

.hero-subtitle {
    font-size: 0.98rem;
    color: #CBD5E1;
    margin: 0;
    line-height: 1.55;
}

.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1rem 1.15rem;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}

.kpi-label {
    font-size: 0.82rem;
    color: #64748B;
    font-weight: 500;
    margin-bottom: 0.25rem;
}

.kpi-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.2;
}

.kpi-sub {
    font-size: 0.78rem;
    color: #94A3B8;
    margin-top: 0.25rem;
}

.result-card-risk {
    background: linear-gradient(135deg, #FFF1F2 0%, #FFE4E6 100%);
    border: 1px solid #FDA4AF;
    border-left: 6px solid #E11D48;
    border-radius: 16px;
    padding: 1.4rem 1.5rem;
    color: #881337;
}

.result-card-normal {
    background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
    border: 1px solid #6EE7B7;
    border-left: 6px solid #059669;
    border-radius: 16px;
    padding: 1.4rem 1.5rem;
    color: #064E3B;
}

.section-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.2rem 1.3rem;
    margin-bottom: 1rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background-color: #FFFFFF;
    padding: 0.4rem;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 0.55rem 1rem;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background-color: #FFF1F2 !important;
    color: #E11D48 !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

DATA_PATH = Path(__file__).parent / "Heart3.csv"


@st.cache_data
def get_dataset() -> pd.DataFrame:
    return load_data(DATA_PATH)


@st.cache_resource
def get_trained_model(
    k: int,
    weights: str,
    metric: str,
    scaler_type: str,
    impute_zeros: bool,
    test_size: float,
):
    df = get_dataset()
    return train_knn_model(
        df=df,
        k=k,
        weights=weights,
        metric=metric,
        scaler_type=scaler_type,
        impute_zeros=impute_zeros,
        test_size=test_size,
        random_state=42,
    )


@st.cache_data
def get_k_sweep(
    weights: str,
    metric: str,
    scaler_type: str,
    impute_zeros: bool,
    test_size: float,
) -> pd.DataFrame:
    df = get_dataset()
    k_values = list(range(1, 32, 2))
    return evaluate_k_range(
        df=df,
        k_values=k_values,
        weights=weights,
        metric=metric,
        scaler_type=scaler_type,
        impute_zeros=impute_zeros,
        test_size=test_size,
        random_state=42,
    )


df_raw = get_dataset()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("## ⚙️ ตั้งค่าโมเดล KNN")
    st.caption("ปรับพารามิเตอร์ของอัลกอริทึม K-Nearest Neighbors เพื่อดูผลลัพธ์แบบเรียลไทม์")

    k_value = st.slider("จำนวนเพื่อนบ้าน (K)", min_value=1, max_value=31, value=9, step=2)

    weights_label = st.selectbox(
        "การถ่วงน้ำหนักระยะทาง (Weights)",
        options=["distance", "uniform"],
        format_func=lambda x: "ถ่วงน้ำหนักตามระยะทาง (distance)" if x == "distance" else "น้ำหนักเท่ากันทุกจุด (uniform)",
    )

    metric_label = st.selectbox(
        "สูตรคำนวณระยะทาง (Distance Metric)",
        options=["manhattan", "euclidean"],
        format_func=lambda x: "Manhattan Distance (L1)" if x == "manhattan" else "Euclidean Distance (L2)",
    )

    scaler_label = st.selectbox(
        "การปรับสเกลข้อมูล (Feature Scaling)",
        options=["standard", "minmax", "robust"],
        format_func=lambda x: {
            "standard": "StandardScaler (Z-score)",
            "minmax": "MinMaxScaler (0 ถึง 1)",
            "robust": "RobustScaler (ลดผลกระทบ Outlier)",
        }[x],
    )

    test_size_pct = st.slider("สัดส่วนชุดทดสอบ (Test Size %)", min_value=10, max_value=35, value=20, step=5)

    impute_zeros = st.toggle(
        "แทนที่ค่า 0 ผิดปกติด้วย Median",
        value=True,
        help="ชุดข้อมูลมี Cholesterol = 0 (172 แถว) และ RestingBP = 0 (1 แถว) การแทนที่ด้วยค่ามัธยฐานช่วยให้ KNN คำนวณระยะทางได้แม่นยำขึ้น",
    )

    st.divider()
    st.info(
        "**แหล่งข้อมูล:** `Heart3.csv`\n\n"
        f"- จำนวนทั้งหมด: **{len(df_raw):,}** แถว\n"
        f"- ฟีเจอร์: **{len(FEATURE_COLUMNS)}** ตัวแปร\n"
        "- เป้าหมาย: `HeartDisease` (0/1)"
    )

bundle = get_trained_model(
    k=k_value,
    weights=weights_label,
    metric=metric_label,
    scaler_type=scaler_label,
    impute_zeros=impute_zeros,
    test_size=test_size_pct / 100.0,
)
metrics = bundle.metrics

# ---------------- Hero Banner ----------------
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-badge">Machine Learning • K-Nearest Neighbors (KNN)</div>
        <div class="hero-title">❤️ ระบบวิเคราะห์และจำแนกความเสี่ยงโรคหัวใจ</div>
        <p class="hero-subtitle">
            เว็บแอปพลิเคชันสำหรับวิเคราะห์ชุดข้อมูล <b>Heart3.csv</b> ฝึกสอนและปรับจูนโมเดล <b>KNN</b>
            พร้อมระบบทำนายผลผู้ป่วยรายบุคคลและการทำนายชุดข้อมูลใหม่แบบไฟล์ CSV
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- Top KPI Row ----------------
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-label">ความแม่นยำ (Test Accuracy)</div>
            <div class="kpi-value">{metrics['accuracy']*100:.2f}%</div>
            <div class="kpi-sub">K = {bundle.k} • {bundle.metric}</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-label">5-Fold CV Accuracy</div>
            <div class="kpi-value">{metrics['cv_mean']*100:.2f}%</div>
            <div class="kpi-sub">± {metrics['cv_std']*100:.2f}% บนชุดฝึกสอน</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-label">F1-Score (กลุ่มเสี่ยง)</div>
            <div class="kpi-value">{metrics['f1']*100:.2f}%</div>
            <div class="kpi-sub">Recall {metrics['recall']*100:.1f}% • Precision {metrics['precision']*100:.1f}%</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-label">ROC-AUC Score</div>
            <div class="kpi-value">{metrics['roc_auc']:.3f}</div>
            <div class="kpi-sub">ความสามารถในการแยกกลุ่ม</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k5:
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-label">การจัดการค่า 0</div>
            <div class="kpi-value">{'Median' if bundle.impute_zeros else 'ค่าดิบ'}</div>
            <div class="kpi-sub">Chol: {bundle.medians['Cholesterol']:.0f} • BP: {bundle.medians['RestingBP']:.0f}</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.write("")

# ---------------- Main Tabs ----------------
tab_single, tab_batch, tab_eval, tab_eda = st.tabs(
    [
        "🩺 ทำนายข้อมูลใหม่ (รายบุคคล)",
        "📂 ทำนายชุดข้อมูลใหม่ (อัปโหลด CSV)",
        "📊 ประสิทธิภาพโมเดล KNN",
        "🔍 สำรวจข้อมูล Heart3.csv",
    ]
)

# ==================== TAB 1: SINGLE PREDICTION ====================
with tab_single:
    st.subheader("ป้อนข้อมูลสุขภาพเพื่อทำนายความเสี่ยงโรคหัวใจ")
    st.caption("กรอกค่าทั้ง 11 ตัวแปร หรือกดปุ่มตัวอย่างด้านล่างเพื่อทดสอบการทำนายได้ทันที")

    # Preset buttons
    p_col1, p_col2, p_col3 = st.columns([1, 1, 2])
    if "single_defaults" not in st.session_state:
        st.session_state.single_defaults = {
            "Age": 54,
            "Sex": 1,
            "ChestPainType": 1,
            "RestingBP": 140,
            "Cholesterol": 239,
            "FastingBS": 0,
            "RestingECG": 2,
            "MaxHR": 135,
            "ExerciseAngina": 1,
            "Oldpeak": 1.5,
            "ST_Slope": 2,
        }

    with p_col1:
        if st.button("🔴 โหลดตัวอย่างกลุ่มเสี่ยงสูง", use_container_width=True):
            st.session_state.single_defaults = {
                "Age": 60,
                "Sex": 1,
                "ChestPainType": 1,
                "RestingBP": 150,
                "Cholesterol": 280,
                "FastingBS": 1,
                "RestingECG": 3,
                "MaxHR": 112,
                "ExerciseAngina": 1,
                "Oldpeak": 2.4,
                "ST_Slope": 2,
            }
    with p_col2:
        if st.button("🟢 โหลดตัวอย่างกลุ่มปกติ", use_container_width=True):
            st.session_state.single_defaults = {
                "Age": 40,
                "Sex": 0,
                "ChestPainType": 2,
                "RestingBP": 120,
                "Cholesterol": 210,
                "FastingBS": 0,
                "RestingECG": 2,
                "MaxHR": 170,
                "ExerciseAngina": 0,
                "Oldpeak": 0.0,
                "ST_Slope": 3,
            }

    d = st.session_state.single_defaults

    with st.form("single_prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("##### 👤 ข้อมูลพื้นฐาน")
            age = st.number_input("อายุ (Age, ปี)", min_value=18, max_value=100, value=int(d["Age"]), step=1)
            sex = st.selectbox(
                "เพศ (Sex)",
                options=list(SEX_MAP.keys()),
                index=list(SEX_MAP.keys()).index(int(d["Sex"])),
                format_func=lambda x: SEX_MAP[x],
            )
            chest_pain = st.selectbox(
                "ประเภทอาการเจ็บหน้าอก (ChestPainType)",
                options=list(CHEST_PAIN_MAP.keys()),
                index=list(CHEST_PAIN_MAP.keys()).index(int(d["ChestPainType"])),
                format_func=lambda x: CHEST_PAIN_MAP[x],
            )
            exercise_angina = st.selectbox(
                "เจ็บหน้าอกขณะออกกำลังกาย (ExerciseAngina)",
                options=list(EXERCISE_ANGINA_MAP.keys()),
                index=list(EXERCISE_ANGINA_MAP.keys()).index(int(d["ExerciseAngina"])),
                format_func=lambda x: EXERCISE_ANGINA_MAP[x],
            )

        with c2:
            st.markdown("##### 🩸 สัญญาณชีพและผลเลือด")
            resting_bp = st.number_input(
                "ความดันโลหิตขณะพัก (RestingBP, mm Hg)",
                min_value=0,
                max_value=250,
                value=int(d["RestingBP"]),
                step=1,
            )
            cholesterol = st.number_input(
                "ระดับคอเลสเตอรอล (Cholesterol, mg/dl)",
                min_value=0,
                max_value=700,
                value=int(d["Cholesterol"]),
                step=1,
            )
            fasting_bs = st.selectbox(
                "น้ำตาลในเลือดหลังอดอาหาร (FastingBS)",
                options=list(FASTING_BS_MAP.keys()),
                index=list(FASTING_BS_MAP.keys()).index(int(d["FastingBS"])),
                format_func=lambda x: FASTING_BS_MAP[x],
            )
            max_hr = st.number_input(
                "อัตราการเต้นหัวใจสูงสุด (MaxHR, bpm)",
                min_value=50,
                max_value=220,
                value=int(d["MaxHR"]),
                step=1,
            )

        with c3:
            st.markdown("##### 📈 คลื่นไฟฟ้าหัวใจ (ECG & ST)")
            resting_ecg = st.selectbox(
                "ผลคลื่นไฟฟ้าหัวใจขณะพัก (RestingECG)",
                options=list(RESTING_ECG_MAP.keys()),
                index=list(RESTING_ECG_MAP.keys()).index(int(d["RestingECG"])),
                format_func=lambda x: RESTING_ECG_MAP[x],
            )
            oldpeak = st.number_input(
                "ค่า ST Depression (Oldpeak)",
                min_value=-3.0,
                max_value=7.0,
                value=float(d["Oldpeak"]),
                step=0.1,
                format="%.1f",
            )
            st_slope = st.selectbox(
                "ความชันของคลื่น ST (ST_Slope)",
                options=list(ST_SLOPE_MAP.keys()),
                index=list(ST_SLOPE_MAP.keys()).index(int(d["ST_Slope"])),
                format_func=lambda x: ST_SLOPE_MAP[x],
            )

        submitted = st.form_submit_button("🔍 ทำนายผลความเสี่ยงโรคหัวใจ", type="primary", use_container_width=True)

    if submitted:
        input_payload = {
            "Age": age,
            "Sex": sex,
            "ChestPainType": chest_pain,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "RestingECG": resting_ecg,
            "MaxHR": max_hr,
            "ExerciseAngina": exercise_angina,
            "Oldpeak": oldpeak,
            "ST_Slope": st_slope,
        }
        res = predict_single(bundle, input_payload)

        st.divider()
        r_col1, r_col2 = st.columns([1.2, 1])

        with r_col1:
            prob_pct = res["prob_disease"] * 100
            if res["prediction"] == 1:
                st.markdown(
                    f"""
                    <div class="result-card-risk">
                        <h3 style="margin:0 0 0.4rem 0;">⚠️ ผลการทำนาย: {res['label_th']}</h3>
                        <p style="margin:0; font-size:1.05rem;">
                            ความน่าจะเป็นในการเกิดโรคหัวใจ: <b>{prob_pct:.1f}%</b>
                            (อ้างอิงจากเพื่อนบ้านที่ใกล้เคียงที่สุด {bundle.k} ราย)
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-card-normal">
                        <h3 style="margin:0 0 0.4rem 0;">✅ ผลการทำนาย: {res['label_th']}</h3>
                        <p style="margin:0; font-size:1.05rem;">
                            ความน่าจะเป็นในการเกิดโรคหัวใจ: <b>{prob_pct:.1f}%</b>
                            (ความน่าจะเป็นปกติ: <b>{res['prob_normal']*100:.1f}%</b>)
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if bundle.impute_zeros and (resting_bp == 0 or cholesterol == 0):
                st.warning(
                    f"ระบบตรวจพบค่า 0 และได้แทนที่ด้วยค่ามัธยฐานอัตโนมัติก่อนทำนาย "
                    f"(RestingBP = {res['imputed_values']['RestingBP']:.0f}, "
                    f"Cholesterol = {res['imputed_values']['Cholesterol']:.0f})"
                )

            st.caption("หมายเหตุ: ผลลัพธ์นี้เป็นการประเมินเชิงสถิติจากโมเดล Machine Learning เท่านั้น ไม่ใช่การวินิจฉัยทางการแพทย์")

        with r_col2:
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob_pct,
                    number={"suffix": "%", "font": {"size": 34}},
                    title={"text": "ระดับความเสี่ยงโรคหัวใจ", "font": {"size": 16}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#E11D48" if res["prediction"] == 1 else "#059669"},
                        "steps": [
                            {"range": [0, 40], "color": "#ECFDF5"},
                            {"range": [40, 70], "color": "#FEF3C7"},
                            {"range": [70, 100], "color": "#FFE4E6"},
                        ],
                        "threshold": {"line": {"color": "#0F172A", "width": 3}, "thickness": 0.75, "value": 50},
                    },
                )
            )
            fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with st.expander(f"🔎 ดูข้อมูลผู้ป่วยที่ใกล้เคียงที่สุด {bundle.k} ราย (K-Nearest Neighbors)", expanded=True):
            st.dataframe(res["neighbors"], use_container_width=True)

# ==================== TAB 2: BATCH CSV PREDICTION ====================
with tab_batch:
    st.subheader("ทำนายชุดข้อมูลใหม่จากไฟล์ CSV")
    st.write(
        "อัปโหลดไฟล์ `.csv` ที่มีคอลัมน์ฟีเจอร์ครบทั้ง 11 คอลัมน์ ระบบจะประมวลผลด้วยโมเดล KNN ปัจจุบัน "
        "และให้ดาวน์โหลดผลลัพธ์พร้อมค่าความน่าจะเป็นกลับไปใช้งานได้ทันที"
    )

    sample_batch = df_raw[FEATURE_COLUMNS].head(10)
    st.download_button(
        label="📥 ดาวน์โหลดไฟล์ตัวอย่างสำหรับกรอกข้อมูล (sample_new_patients.csv)",
        data=sample_batch.to_csv(index=False).encode("utf-8-sig"),
        file_name="sample_new_patients.csv",
        mime="text/csv",
    )

    uploaded_csv = st.file_uploader("เลือกไฟล์ CSV ชุดข้อมูลใหม่", type=["csv"])

    if uploaded_csv is not None:
        try:
            new_df = pd.read_csv(uploaded_csv)
            st.write(f"อ่านไฟล์สำเร็จ: พบข้อมูลทั้งหมด **{len(new_df):,}** แถว")
            pred_df = predict_batch(bundle, new_df)

            risk_count = int((pred_df["ผลการทำนาย (HeartDisease)"] == 1).sum())
            normal_count = len(pred_df) - risk_count

            b1, b2, b3 = st.columns(3)
            b1.metric("จำนวนผู้ป่วยทั้งหมด", f"{len(pred_df):,} ราย")
            b2.metric("พบความเสี่ยงโรคหัวใจ (1)", f"{risk_count:,} ราย", f"{risk_count/len(pred_df)*100:.1f}%")
            b3.metric("ปกติ (0)", f"{normal_count:,} ราย", f"{normal_count/len(pred_df)*100:.1f}%")

            fig_batch = px.pie(
                names=["มีความเสี่ยงโรคหัวใจ (1)", "ปกติ (0)"],
                values=[risk_count, normal_count],
                color_discrete_sequence=["#E11D48", "#10B981"],
                hole=0.48,
                title="สัดส่วนผลการทำนายชุดข้อมูลใหม่",
            )
            fig_batch.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_batch, use_container_width=True)

            st.dataframe(pred_df, use_container_width=True)

            st.download_button(
                label="💾 ดาวน์โหลดผลลัพธ์การทำนายทั้งหมด (.csv)",
                data=pred_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="heart_knn_predictions.csv",
                mime="text/csv",
                type="primary",
            )
        except Exception as exc:
            st.error(f"ไม่สามารถประมวลผลไฟล์ได้: {exc}")

# ==================== TAB 3: MODEL EVALUATION ====================
with tab_eval:
    st.subheader("การประเมินประสิทธิภาพและเลือกค่า K ที่เหมาะสม")

    sweep_df = get_k_sweep(
        weights=weights_label,
        metric=metric_label,
        scaler_type=scaler_label,
        impute_zeros=impute_zeros,
        test_size=test_size_pct / 100.0,
    )
    best_row = sweep_df.loc[sweep_df["5-Fold CV Accuracy"].idxmax()]

    st.info(
        f"จากการทดสอบค่า K ตั้งแต่ 1 ถึง 31 พบว่าค่า K ที่ให้ **5-Fold CV Accuracy** สูงสุดคือ "
        f"**K = {int(best_row['K'])}** (CV Accuracy = **{best_row['5-Fold CV Accuracy']*100:.2f}%**, "
        f"Test Accuracy = **{best_row['Test Accuracy']*100:.2f}%**)"
    )

    fig_k = go.Figure()
    fig_k.add_trace(
        go.Scatter(
            x=sweep_df["K"],
            y=sweep_df["Test Accuracy"] * 100,
            mode="lines+markers",
            name="Test Accuracy (%)",
            line=dict(color="#E11D48", width=3),
        )
    )
    fig_k.add_trace(
        go.Scatter(
            x=sweep_df["K"],
            y=sweep_df["5-Fold CV Accuracy"] * 100,
            mode="lines+markers",
            name="5-Fold CV Accuracy (%)",
            line=dict(color="#0F172A", width=2.5, dash="dash"),
        )
    )
    fig_k.add_vline(x=bundle.k, line_dash="dot", line_color="#2563EB", annotation_text=f"K ปัจจุบัน = {bundle.k}")
    fig_k.update_layout(
        title="เปรียบเทียบความแม่นยำของโมเดลตามค่า K",
        xaxis_title="จำนวนเพื่อนบ้าน (K)",
        yaxis_title="ความแม่นยำ (%)",
        height=360,
        hovermode="x unified",
    )
    st.plotly_chart(fig_k, use_container_width=True)

    ev_col1, ev_col2 = st.columns(2)
    with ev_col1:
        cm = metrics["confusion_matrix"]
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=["ทำนาย: ปกติ (0)", "ทำนาย: เสี่ยง (1)"],
            y=["จริง: ปกติ (0)", "จริง: เสี่ยง (1)"],
            color_continuous_scale="Blues",
            title="Confusion Matrix (ชุดทดสอบ)",
        )
        fig_cm.update_layout(height=360, coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

    with ev_col2:
        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(
                x=metrics["fpr"],
                y=metrics["tpr"],
                mode="lines",
                name=f"KNN (AUC = {metrics['roc_auc']:.3f})",
                line=dict(color="#E11D48", width=3),
            )
        )
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="สุ่ม (AUC = 0.500)",
                line=dict(color="#94A3B8", dash="dash"),
            )
        )
        fig_roc.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=360,
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("##### ตารางสรุป Classification Report")
    rep_df = pd.DataFrame(metrics["classification_report"]).T
    st.dataframe(rep_df.style.format("{:.3f}"), use_container_width=True)

# ==================== TAB 4: EDA & DATA DICTIONARY ====================
with tab_eda:
    st.subheader("ภาพรวมและโครงสร้างข้อมูล Heart3.csv")

    eda1, eda2, eda3, eda4 = st.columns(4)
    eda1.metric("จำนวนแถวทั้งหมด", f"{len(df_raw):,}")
    eda2.metric("ป่วยเป็นโรคหัวใจ (1)", f"{int((df_raw[TARGET_COLUMN] == 1).sum()):,}", "55.3%")
    eda3.metric("Cholesterol = 0", f"{int((df_raw['Cholesterol'] == 0).sum()):,} แถว")
    eda4.metric("RestingBP = 0", f"{int((df_raw['RestingBP'] == 0).sum()):,} แถว")

    c_eda1, c_eda2 = st.columns(2)
    with c_eda1:
        df_plot = df_raw.copy()
        df_plot["สถานะ"] = df_plot[TARGET_COLUMN].map(TARGET_MAP)
        fig_age = px.histogram(
            df_plot,
            x="Age",
            color="สถานะ",
            barmode="overlay",
            nbins=25,
            color_discrete_map={TARGET_MAP[0]: "#10B981", TARGET_MAP[1]: "#E11D48"},
            title="การกระจายตัวของอายุแยกตามกลุ่มโรคหัวใจ",
            labels={"Age": "อายุ (ปี)", "count": "จำนวนผู้ป่วย"},
        )
        fig_age.update_layout(height=340)
        st.plotly_chart(fig_age, use_container_width=True)

    with c_eda2:
        cp_df = df_raw.copy()
        cp_df["อาการเจ็บหน้าอก"] = cp_df["ChestPainType"].map(CHEST_PAIN_MAP)
        cp_df["สถานะ"] = cp_df[TARGET_COLUMN].map(TARGET_MAP)
        cp_grouped = cp_df.groupby(["อาการเจ็บหน้าอก", "สถานะ"]).size().reset_index(name="จำนวน")
        fig_cp = px.bar(
            cp_grouped,
            x="อาการเจ็บหน้าอก",
            y="จำนวน",
            color="สถานะ",
            barmode="group",
            color_discrete_map={TARGET_MAP[0]: "#10B981", TARGET_MAP[1]: "#E11D48"},
            title="จำนวนผู้ป่วยตามประเภทอาการเจ็บหน้าอก",
        )
        fig_cp.update_layout(height=340)
        st.plotly_chart(fig_cp, use_container_width=True)

    st.markdown("##### พจนานุกรมข้อมูล (Data Dictionary)")
    dict_rows = [
        {"คอลัมน์": col, "ความหมายภาษาไทย": FEATURE_THAI_NAMES[col], "ช่วงค่า / รหัสตัวเลข": desc}
        for col, desc in [
            ("Age", "28 – 77 ปี"),
            ("Sex", "0 = หญิง, 1 = ชาย"),
            ("ChestPainType", "1 = ASY, 2 = ATA, 3 = NAP, 4 = TA"),
            ("RestingBP", "0 – 200 mm Hg (มีค่า 0 จำนวน 1 แถว)"),
            ("Cholesterol", "0 – 603 mg/dl (มีค่า 0 จำนวน 172 แถว)"),
            ("FastingBS", "0 = ≤ 120 mg/dl, 1 = > 120 mg/dl"),
            ("RestingECG", "1 = LVH, 2 = Normal, 3 = ST"),
            ("MaxHR", "60 – 202 bpm"),
            ("ExerciseAngina", "0 = ไม่มี, 1 = มีอาการเจ็บหน้าอก"),
            ("Oldpeak", "-2.6 – 6.2"),
            ("ST_Slope", "1 = Down, 2 = Flat, 3 = Up"),
        ]
    ]
    st.dataframe(pd.DataFrame(dict_rows), use_container_width=True, hide_index=True)

    with st.expander("ดูตัวอย่างข้อมูลดิบ 20 แถวแรก"):
        st.dataframe(df_raw.head(20), use_container_width=True)
