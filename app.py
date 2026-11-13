import os
import re
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from google import genai

# ==========================================
# 1. إعداد الصفحة والهوية البصرية (Dark Navy & Gold)
# ==========================================
st.set_page_config(
    page_title="AI Data Analyst Pro | Mohamed Gomaa",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Cairo', sans-serif;
    }

    .stApp {
        background-color: #080d1a;
        color: #f3f4f6;
    }

    section[data-testid="stSidebar"] {
        background-color: #0b1120;
        border-right: 1px solid #1e293b;
    }

    .hero-badge {
        display: inline-block;
        background: rgba(201, 162, 39, 0.12);
        border: 1px solid #c9a227;
        color: #e5c158;
        padding: 5px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .hero-title {
        color: #e5c158;
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.3;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 6px;
        margin-bottom: 22px;
    }

    .kpi-card {
        background: linear-gradient(145deg, #0d1426, #111a30);
        border: 1px solid #1e293b;
        border-top: 3px solid #c9a227;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.35);
    }

    .kpi-label {
        color: #94a3b8;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 6px;
    }

    .kpi-value {
        color: #e5c158;
        font-size: 1.75rem;
        font-weight: 800;
    }

    .section-header {
        color: #ffffff;
        font-size: 1.35rem;
        font-weight: 700;
        border-right: 4px solid #c9a227;
        padding-right: 12px;
        margin: 18px 0 12px 0;
        direction: rtl;
        text-align: right;
    }

    .callout-box {
        background-color: #f8fafc;
        color: #0f172a;
        border-right: 5px solid #c9a227;
        border-radius: 8px;
        padding: 12px 16px;
        font-weight: 700;
        font-size: 0.95rem;
        direction: rtl;
        text-align: right;
        margin: 10px 0;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #c9a227, #e5c158);
        color: #080d1a;
        font-weight: 800;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1.2rem;
    }

    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(90deg, #e5c158, #f3d274);
        color: #080d1a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 2. دوال مساعدة (البيانات التجريبية، التنظيف، التنسيق، AI)
# ==========================================
def generate_sample_data() -> pd.DataFrame:
    np.random.seed(42)
    n = 120
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    branches = ["أسوان", "الإسكندرية", "المنصورة", "القاهرة"]
    categories = ["أجهزة منزلية", "إلكترونيات", "إكسسوارات", "برمجيات"]
    payments = ["إنستاباي", "بطاقة بنكية", "كاش"]

    qty = np.random.randint(1, 15, size=n)
    price = np.random.choice([250, 500, 1200, 2500, 3500], size=n)
    total_sales = qty * price
    profit = np.round(total_sales * np.random.uniform(0.18, 0.35, size=n), 2)

    df = pd.DataFrame(
        {
            "order_date": dates,
            "branch": np.random.choice(branches, size=n),
            "category": np.random.choice(categories, size=n),
            "payment_method": np.random.choice(payments, size=n),
            "quantity": qty,
            "unit_price": price,
            "total_sales": total_sales,
            "profit": profit,
        }
    )
    return df


def style_plotly_fig(fig: go.Figure, title: str = "") -> go.Figure:
    """توحيد ثيم الرسوم البيانية بالألوان الداكنة مع توضيح النصوص والـ Legend."""
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e5c158", size=16), x=0.5, xanchor="center"),
        paper_bgcolor="#0d1426",
        plot_bgcolor="#080d1a",
        font=dict(color="#f3f4f6", family="Cairo, sans-serif", size=12),
        legend=dict(
            font=dict(color="#ffffff", size=12),
            bgcolor="rgba(15, 23, 42, 0.85)",
            bordercolor="#334155",
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis=dict(
            gridcolor="rgba(148, 163, 184, 0.15)",
            zerolinecolor="rgba(148, 163, 184, 0.25)",
            tickfont=dict(color="#cbd5e1"),
            title_font=dict(color="#e5c158"),
        ),
        yaxis=dict(
            gridcolor="rgba(148, 163, 184, 0.15)",
            zerolinecolor="rgba(148, 163, 184, 0.25)",
            tickfont=dict(color="#cbd5e1"),
            title_font=dict(color="#e5c158"),
        ),
        margin=dict(l=30, r=30, t=60, b=40),
    )
    return fig


def calculate_health_score(df: pd.DataFrame) -> tuple[int, int, int]:
    total_cells = max(df.shape[0] * df.shape[1], 1)
    nulls = int(df.isna().sum().sum())
    dups = int(df.duplicated().sum())
    null_penalty = (nulls / total_cells) * 60
    dup_penalty = (dups / max(len(df), 1)) * 40
    score = int(max(0, min(100, round(100 - null_penalty - dup_penalty))))
    return score, dups, nulls


def auto_clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cleaned = df.copy()
    logs = []

    # 1. توحيد أسماء الأعمدة
    old_cols = list(cleaned.columns)
    cleaned.columns = [
        re.sub(r"\s+", "_", str(c).strip().lower()) for c in cleaned.columns
    ]
    if list(cleaned.columns) != old_cols:
        logs.append("تم توحيد أسماء الأعمدة وإزالة المسافات الزائدة.")

    # 2. إزالة الصفوف المكررة
    dup_count = int(cleaned.duplicated().sum())
    if dup_count > 0:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        logs.append(f"تم حذف {dup_count} صف مكرر.")

    # 3. تنظيف النصوص وتحويل التواريخ والأرقام
    for col in cleaned.columns:
        if cleaned[col].dtype == "object":
            cleaned[col] = cleaned[col].astype(str).str.strip()
            cleaned[col] = cleaned[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})

            if any(k in col for k in ["date", "time", "تاريخ"]):
                parsed = pd.to_datetime(cleaned[col], errors="coerce")
                if parsed.notna().sum() > len(cleaned) * 0.6:
                    cleaned[col] = parsed
                    logs.append(f"تم تحويل العمود ({col}) إلى صيغة تاريخ.")
                    continue

            numeric_try = pd.to_numeric(
                cleaned[col].astype(str).str.replace(",", ""), errors="coerce"
            )
            if numeric_try.notna().sum() > len(cleaned) * 0.8 and cleaned[col].notna().sum() > 0:
                cleaned[col] = numeric_try
                logs.append(f"تم تصحيح نوع العمود ({col}) إلى رقمي.")

    # 4. معالجة القيم المفقودة
    null_before = int(cleaned.isna().sum().sum())
    if null_before > 0:
        for col in cleaned.columns:
            if cleaned[col].isna().sum() > 0:
                if pd.api.types.is_numeric_dtype(cleaned[col]):
                    cleaned[col] = cleaned[col].fillna(cleaned[col].median())
                elif pd.api.types.is_datetime64_any_dtype(cleaned[col]):
                    cleaned[col] = cleaned[col].ffill().bfill()
                else:
                    mode_val = cleaned[col].mode()
                    fill_val = mode_val.iloc[0] if not mode_val.empty else "غير محدد"
                    cleaned[col] = cleaned[col].fillna(fill_val)
        logs.append(f"تمت معالجة {null_before} قيمة مفقودة تلقائياً.")

    if not logs:
        logs.append("تنظيف ذكي شامل: البيانات نظيفة بالفعل ولا توجد أخطاء شكلية!")

    return cleaned, logs


def get_gemini_client():
    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def ask_gemini_with_fallback(prompt: str) -> str:
    client = get_gemini_client()
    if not client:
        return "⚠️ لم يتم العثور على مفتاح `GEMINI_API_KEY` في إعدادات التطبيق (Secrets)."

    models = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]
    last_err = ""
    for model_name in models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            if response and response.text:
                return response.text
        except Exception as e:
            last_err = str(e)
            continue
    return f"⚠️ تعذر الاتصال بخدمة الذكاء الاصطناعي حالياً. التفاصيل: {last_err}"


# ==========================================
# 3. إدارة الحالة (Session State) والقائمة الجانبية
# ==========================================
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "df" not in st.session_state:
    st.session_state.df = None
if "clean_logs" not in st.session_state:
    st.session_state.clean_logs = []
if "file_signature" not in st.session_state:
    st.session_state.file_signature = None

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 10px 0 18px 0;">
            <div style="font-size: 1.55rem; font-weight: 800; color: #e5c158;">AI Data Analyst ⚡</div>
            <div style="font-size: 0.82rem; color: #94a3b8;">منصة التحليل الذكي وتنظيف البيانات</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "📂 ارفع ملف البيانات هنا",
        type=["csv", "xlsx"],
        help="يدعم ملفات CSV و Excel حتى 200MB",
    )

    if uploaded_file is not None:
        sig = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.file_signature != sig:
            try:
                if uploaded_file.name.lower().endswith(".csv"):
                    loaded = pd.read_csv(uploaded_file)
                else:
                    loaded = pd.read_excel(uploaded_file)
                st.session_state.raw_df = loaded.copy()
                cleaned_df, logs = auto_clean_dataframe(loaded)
                st.session_state.df = cleaned_df
                st.session_state.clean_logs = logs
                st.session_state.file_signature = sig
            except Exception as e:
                st.error(f"حدث خطأ أثناء قراءة الملف: {e}")

    st.markdown("<div style='text-align:center; color:#64748b; margin:6px 0;'>— أو —</div>", unsafe_allow_html=True)

    if st.button("🚀 جرب ببيانات مبيعات جاهزة", use_container_width=True):
        demo = generate_sample_data()
        st.session_state.raw_df = demo.copy()
        st.session_state.df = demo.copy()
        st.session_state.clean_logs = ["تم تحميل بيانات المبيعات التجريبية بنجاح (120 صف)."]
        st.session_state.file_signature = "demo_sales_data"

    if st.session_state.clean_logs:
        st.markdown("---")
        st.markdown("#### 📜 سجل خطوات التنظيف")
        for item in st.session_state.clean_logs:
            st.caption(f"✅ {item}")

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center; font-size:0.82rem; color:#94a3b8; padding-top:8px;">
            Built with ❤️ by <strong style="color:#e5c158;">Mohamed Gomaa</strong><br>
            <span style="font-size:0.75rem;">Data Analyst & Software Engineer</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# 4. الترويسة الرئيسية
# ==========================================
st.markdown(
    """
    <div style="text-align:center;">
        <div class="hero-badge">🏆 Engineered & Designed by Mohamed Gomaa</div>
        <h1 class="hero-title">📊 لوحة تحليل البيانات الذكية</h1>
        <p class="hero-subtitle">ارفع بياناتك، نظّفها بضغطة زر، استشرف التوقعات، وأصدر تقارير تنفيذية متكاملة</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.df is None:
    st.markdown(
        """
        <div style="background:#0d1426; border:1px dashed #c9a227; border-radius:14px; padding:36px; text-align:center; margin-top:20px;">
            <h3 style="color:#e5c158; margin-bottom:10px;">👋 أهلاً بك في مساحة العمل الذكية</h3>
            <p style="color:#cbd5e1; font-size:1rem;">
                ابدأ برفع ملف البيانات (CSV أو Excel) أو اضغط على زر <strong>🚀 جرب ببيانات مبيعات جاهزة</strong> من القائمة الجانبية لتفعيل التنظيف، المحاكاة، والتقرير التنفيذي الذكي.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

df = st.session_state.df
health_score, dup_rows, missing_vals = calculate_health_score(df)

# ==========================================
# 5. بطاقات مؤشرات الجودة (5 KPI Cards)
# ==========================================
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">إجمالي الصفوف</div><div class="kpi-value">{len(df):,}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">عدد الأعمدة</div><div class="kpi-value">{df.shape[1]}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">القيم المفقودة</div><div class="kpi-value">{missing_vals:,}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">الصفوف المكررة</div><div class="kpi-value">{dup_rows:,}</div></div>', unsafe_allow_html=True)
with c5:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">جودة البيانات</div><div class="kpi-value">{health_score}%</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 6. التبويبات الرئيسية الأربعة
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🧹 معاينة وتنظيف البيانات",
        "📈 التحليل الاستكشافي (EDA)",
        "🔮 التنبؤ ومحاكاة السيناريوهات",
        "🤖 الذكاء الاصطناعي والتقرير التنفيذي",
    ]
)

# ------------------------------------------
# التبويب الأول: معاينة وتنظيف البيانات
# ------------------------------------------
with tab1:
    if st.session_state.clean_logs:
        st.markdown(
            f'<div class="callout-box">📋 تقرير المعالجة الأخيرة: {st.session_state.clean_logs[-1]}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-header">🛠️ أدوات التنظيف الاحترافي</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        if st.button("⚡ التنظيف الذكي الشامل (Auto-Clean All)", type="primary", use_container_width=True):
            cleaned_df, logs = auto_clean_dataframe(st.session_state.df)
            st.session_state.df = cleaned_df
            st.session_state.clean_logs.extend(logs)
            st.rerun()
    with col_b:
        if st.button("🔄 استرجاع البيانات الأصلية", use_container_width=True):
            if st.session_state.raw_df is not None:
                st.session_state.df = st.session_state.raw_df.copy()
                st.session_state.clean_logs = ["تم استرجاع النسخة الأصلية من البيانات."]
                st.rerun()

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("🗑️ حذف المكرر", use_container_width=True):
            before = len(st.session_state.df)
            st.session_state.df = st.session_state.df.drop_duplicates().reset_index(drop=True)
            removed = before - len(st.session_state.df)
            st.session_state.clean_logs.append(f"تم حذف {removed} صف مكرر يدوياً.")
            st.rerun()
    with b2:
        if st.button("✨ توحيد النصوص", use_container_width=True):
            for col in st.session_state.df.select_dtypes(include="object").columns:
                st.session_state.df[col] = st.session_state.df[col].astype(str).str.strip()
            st.session_state.clean_logs.append("تم تنظيف وتوحيد المسافات في جميع الأعمدة النصية.")
            st.rerun()
    with b3:
        if st.button("🔢 تصحيح الأنواع", use_container_width=True):
            cleaned_df, logs = auto_clean_dataframe(st.session_state.df)
            st.session_state.df = cleaned_df
            st.session_state.clean_logs.append("تم فحص وتصحيح أنواع البيانات.")
            st.rerun()
    with b4:
        if st.button("🏷️ توحيد أسماء الأعمدة", use_container_width=True):
            st.session_state.df.columns = [
                re.sub(r"\s+", "_", str(c).strip().lower()) for c in st.session_state.df.columns
            ]
            st.session_state.clean_logs.append("تم توحيد أسماء الأعمدة.")
            st.rerun()

    exp1, exp2 = st.columns(2)
    with exp1:
        with st.expander("🩹 معالجة القيم المفقودة (Missing Values)"):
            null_cols = [c for c in df.columns if df[c].isna().sum() > 0]
            if not null_cols:
                st.success("لا توجد أي قيم مفقودة في الجدول حالياً.")
            else:
                target_null_col = st.selectbox("اختر العمود:", null_cols)
                method = st.radio("طريقة المعالجة:", ["التعويض بالمتوسط/الوسيط", "حذف الصفوف الفارغة"], horizontal=True)
                if st.button("تطبيق على القيم المفقودة"):
                    if method == "حذف الصفوف الفارغة":
                        st.session_state.df = st.session_state.df.dropna(subset=[target_null_col]).reset_index(drop=True)
                    else:
                        if pd.api.types.is_numeric_dtype(st.session_state.df[target_null_col]):
                            st.session_state.df[target_null_col] = st.session_state.df[target_null_col].fillna(
                                st.session_state.df[target_null_col].median()
                            )
                        else:
                            st.session_state.df[target_null_col] = st.session_state.df[target_null_col].fillna("غير محدد")
                    st.session_state.clean_logs.append(f"تمت معالجة القيم المفقودة في عمود {target_null_col}.")
                    st.rerun()

        with st.expander("🎯 معالجة القيم الشاذة (Outliers - IQR)"):
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if num_cols:
                out_col = st.selectbox("اختر العمود الرقمي:", num_cols, key="iqr_col")
                if st.button("ضبط القيم الشاذة (Winsorize IQR)"):
                    q1 = st.session_state.df[out_col].quantile(0.25)
                    q3 = st.session_state.df[out_col].quantile(0.75)
                    iqr = q3 - q1
                    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    st.session_state.df[out_col] = np.clip(st.session_state.df[out_col], low, high)
                    st.session_state.clean_logs.append(f"تم ضبط القيم الشاذة في ({out_col}) باستخدام IQR.")
                    st.rerun()

    with exp2:
        with st.expander("✂️ حذف الأعمدة غير المفيدة (Drop Columns)"):
            cols_to_drop = st.multiselect("حدد الأعمدة المراد حذفها:", df.columns.tolist())
            if st.button("حذف الأعمدة المحددة") and cols_to_drop:
                st.session_state.df = st.session_state.df.drop(columns=cols_to_drop)
                st.session_state.clean_logs.append(f"تم حذف الأعمدة: {', '.join(cols_to_drop)}")
                st.rerun()

        with st.expander("🔍 البحث والاستبدال (Find & Replace)"):
            rep_col = st.selectbox("العمود المستهدف:", df.columns.tolist(), key="rep_col")
            old_val = st.text_input("القيمة القديمة:")
            new_val = st.text_input("القيمة الجديدة:")
            if st.button("تنفيذ الاستبدال") and old_val:
                st.session_state.df[rep_col] = (
                    st.session_state.df[rep_col].astype(str).replace(old_val, new_val)
                )
                st.session_state.clean_logs.append(f"تم استبدال ({old_val}) بـ ({new_val}) في {rep_col}.")
                st.rerun()

    st.markdown('<div class="section-header">📋 جدول البيانات بعد التحديث</div>', unsafe_allow_html=True)
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 تحميل الداتا النظيفة (CSV)",
        data=csv_bytes,
        file_name="cleaned_dataset.csv",
        mime="text/csv",
    )
    st.dataframe(df, use_container_width=True, height=340)


# ------------------------------------------
# التبويب الثاني: التحليل الاستكشافي (EDA)
# ------------------------------------------
with tab2:
    st.markdown('<div class="section-header">🔍 فحص هيكل البيانات (Data Schema)</div>', unsafe_allow_html=True)
    schema_df = pd.DataFrame(
        {
            "اسم العمود": df.columns,
            "نوع البيانات": [str(dtype) for dtype in df.dtypes],
            "القيم الفريدة (Unique)": [df[col].nunique() for col in df.columns],
            "القيم المفقودة": [int(df[col].isna().sum()) for col in df.columns],
        }
    )
    st.dataframe(schema_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">📊 الإحصائيات الوصفية (Descriptive Statistics)</div>', unsafe_allow_html=True)
    stat_c1, stat_c2 = st.columns(2)
    num_df = df.select_dtypes(include=np.number)
    cat_df = df.select_dtypes(exclude=[np.number, "datetime64[ns]", "datetime64[us]"])

    with stat_c1:
        st.caption("الأعمدة الرقمية:")
        if not num_df.empty:
            st.dataframe(num_df.describe().T, use_container_width=True)
        else:
            st.info("لا توجد أعمدة رقمية.")

    with stat_c2:
        st.caption("الأعمدة النصية / التصنيفية:")
        if not cat_df.empty:
            st.dataframe(cat_df.describe().T, use_container_width=True)
        else:
            st.info("لا توجد أعمدة تصنيفية.")

    st.markdown('<div class="section-header">📉 استكشاف التوزيعات والعلاقات تلقائياً</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        if not num_df.empty:
            sel_num = st.selectbox("اختر عمود رقمي لعرض توزيعه:", num_df.columns, key="eda_num")
            fig_box = px.box(
                df,
                x=sel_num,
                color_discrete_sequence=["#e5c158"],
                orientation="h",
            )
            st.plotly_chart(style_plotly_fig(fig_box, f"توزيع قيم {sel_num}"), use_container_width=True)

    with d2:
        if not cat_df.empty:
            sel_cat = st.selectbox("اختر عمود تصنيفي لعرض أعلى الفئات:", cat_df.columns, key="eda_cat")
            top_counts = df[sel_cat].value_counts().head(10).reset_index()
            top_counts.columns = [sel_cat, "العدد"]
            fig_bar = px.bar(
                top_counts,
                x="العدد",
                y=sel_cat,
                orientation="h",
                color_discrete_sequence=["#38bdf8"],
            )
            st.plotly_chart(style_plotly_fig(fig_bar, f"أعلى الفئات تكراراً في {sel_cat}"), use_container_width=True)

    # تحليل الاتجاه الزمني مع تجميع ذكي للبيانات الكثيفة
    date_cols = [
        c for c in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[c]) or any(k in c.lower() for k in ["date", "time", "تاريخ"])
    ]
    if date_cols and not num_df.empty:
        st.markdown('<div class="section-header">⏳ تحليل الاتجاه الزمني (Time Series)</div>', unsafe_allow_html=True)
        ts_c1, ts_c2, ts_c3 = st.columns(3)
        with ts_c1:
            ts_date = st.selectbox("عمود التاريخ:", date_cols, key="ts_date")
        with ts_c2:
            ts_val = st.selectbox("المتغير الرقمي:", num_df.columns, key="ts_val")
        with ts_c3:
            ts_freq = st.selectbox(
                "مستوى التجميع الزمني:",
                ["تلقائي (ذكي)", "يومي (Daily)", "أسبوعي (Weekly)", "شهري (Monthly)"],
                key="ts_freq",
            )

        ts_temp = df[[ts_date, ts_val]].copy()
        ts_temp[ts_date] = pd.to_datetime(ts_temp[ts_date], errors="coerce")
        ts_temp = ts_temp.dropna().sort_values(ts_date)

        if not ts_temp.empty:
            unique_days = ts_temp[ts_date].nunique()
            if ts_freq == "تلقائي (ذكي)":
                rule = "ME" if unique_days > 180 else ("W" if unique_days > 45 else "D")
            elif ts_freq == "شهري (Monthly)":
                rule = "ME"
            elif ts_freq == "أسبوعي (Weekly)":
                rule = "W"
            else:
                rule = "D"

            ts_grouped = ts_temp.set_index(ts_date)[ts_val].resample(rule).sum().reset_index()
            fig_ts = px.line(
                ts_grouped,
                x=ts_date,
                y=ts_val,
                markers=True,
                color_discrete_sequence=["#10b981"],
            )
            fig_ts.update_traces(line=dict(width=2.8), marker=dict(size=6))
            st.plotly_chart(style_plotly_fig(fig_ts, f"تطور {ts_val} عبر {ts_date}"), use_container_width=True)

    if num_df.shape[1] >= 2:
        st.markdown('<div class="section-header">🔥 مصفوفة الارتباط (Correlation Heatmap)</div>', unsafe_allow_html=True)
        corr = num_df.corr().round(2)
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            aspect="auto",
        )
        st.plotly_chart(style_plotly_fig(fig_corr, "مصفوفة الارتباط بين المتغيرات الرقمية"), use_container_width=True)


# ------------------------------------------
# التبويب الثالث: التنبؤ ومحاكاة السيناريوهات
# ------------------------------------------
with tab3:
    st.markdown('<div class="section-header">🎛️ محاكي السيناريوهات واتخاذ القرار (What-If Decision Simulator)</div>', unsafe_allow_html=True)
    st.caption("اختبر تأثير التغييرات المستقبلية (زيادة المبيعات، خفض التكاليف أو الهدر) على إجمالي الأداء فوراً.")

    num_cols_sim = df.select_dtypes(include=np.number).columns.tolist()
    if num_cols_sim:
        s1, s2, s3 = st.columns(3)
        with s1:
            sim_metric = st.selectbox("اختر المؤشر الرئيسي (مثل المبيعات / الإيرادات):", num_cols_sim)
        with s2:
            growth_pct = st.slider("نسبة النمو المتوقعة في المؤشر (%):", -50, 100, 20, step=5)
        with s3:
            cost_save_pct = st.slider("نسبة تحسين الهامش / خفض الهدر (%):", 0, 50, 15, step=5)

        current_total = float(df[sim_metric].sum())
        projected_total = current_total * (1 + growth_pct / 100.0)
        current_net = current_total * 0.75
        projected_net = projected_total * (0.75 + (cost_save_pct / 100.0) * 0.25)
        net_diff = projected_net - current_net

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">الإجمالي الحالي</div><div class="kpi-value">{current_total:,.2f}</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">الإجمالي المتوقع (بعد السيناريو)</div><div class="kpi-value">{projected_total:,.2f}</div><div style="color:#10b981; font-size:0.8rem; font-weight:700;">{growth_pct:+d}%</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">صافي العائد الحالي</div><div class="kpi-value">{current_net:,.2f}</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">صافي العائد المتوقع</div><div class="kpi-value">{projected_net:,.2f}</div><div style="color:#10b981; font-size:0.8rem; font-weight:700;">{net_diff:+,.2f}</div></div>', unsafe_allow_html=True)

        sim_chart_df = pd.DataFrame(
            {
                "الحالة": ["الوضع الحالي", "الوضع المتوقع (السيناريو)"],
                "الإجمالي": [current_total, projected_total],
                "صافي العائد": [current_net, projected_net],
            }
        )
        fig_sim = go.Figure(
            data=[
                go.Bar(name="الإجمالي", x=sim_chart_df["الحالة"], y=sim_chart_df["الإجمالي"], marker_color="#e5c158"),
                go.Bar(name="صافي العائد", x=sim_chart_df["الحالة"], y=sim_chart_df["صافي العائد"], marker_color="#10b981"),
            ]
        )
        fig_sim.update_layout(barmode="group")
        st.plotly_chart(style_plotly_fig(fig_sim, f"مقارنة الأداء المالي ({sim_metric})"), use_container_width=True)

    # ------------------------------------------
    # التوقع الزمني المستقبلي المحسّن (Smooth Resampled Forecasting)
    # ------------------------------------------
    st.markdown('<div class="section-header">📈 التوقع الزمني المستقبلي (Linear Trend Forecasting)</div>', unsafe_allow_html=True)
    if date_cols and num_cols_sim:
        f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.2, 1])
        with f1:
            fc_date = st.selectbox("عمود التاريخ:", date_cols, key="fc_date")
        with f2:
            fc_target = st.selectbox("المتغير المراد توقعه:", num_cols_sim, index=len(num_cols_sim) - 1, key="fc_target")
        with f3:
            fc_granularity = st.selectbox(
                "تجميع الفترات (لتنعيم المنحنى):",
                ["تلقائي (ذكي)", "أسبوعي (Weekly)", "شهري (Monthly)", "يومي (Daily)"],
                key="fc_gran",
            )
        with f4:
            n_periods = st.slider("عدد الفترات القادمة:", 3, 15, 7)

        fc_df = df[[fc_date, fc_target]].copy()
        fc_df[fc_date] = pd.to_datetime(fc_df[fc_date], errors="coerce")
        fc_df = fc_df.dropna().sort_values(fc_date)

        if len(fc_df) >= 4:
            unique_days = fc_df[fc_date].nunique()
            if fc_granularity == "تلقائي (ذكي)":
                if unique_days > 150:
                    freq_code, freq_label, pd_offset = "W", "أسبوع", pd.Timedelta(weeks=1)
                elif unique_days > 45:
                    freq_code, freq_label, pd_offset = "W", "أسبوع", pd.Timedelta(weeks=1)
                else:
                    freq_code, freq_label, pd_offset = "D", "يوم", pd.Timedelta(days=1)
            elif fc_granularity == "شهري (Monthly)":
                freq_code, freq_label, pd_offset = "ME", "شهر", pd.DateOffset(months=1)
            elif fc_granularity == "أسبوعي (Weekly)":
                freq_code, freq_label, pd_offset = "W", "أسبوع", pd.Timedelta(weeks=1)
            else:
                freq_code, freq_label, pd_offset = "D", "يوم", pd.Timedelta(days=1)

            hist = fc_df.set_index(fc_date)[fc_target].resample(freq_code).sum().reset_index()

            # إزالة آخر فترة غير مكتملة إذا كانت صفراً تماماً
            if len(hist) > 4 and hist[fc_target].iloc[-1] == 0:
                hist = hist.iloc[:-1]

            x_idx = np.arange(len(hist))
            y_vals = hist[fc_target].values.astype(float)
            slope, intercept = np.polyfit(x_idx, y_vals, 1)

            future_x = np.arange(len(hist), len(hist) + n_periods)
            future_y = np.maximum(0, slope * future_x + intercept)
            last_date = hist[fc_date].iloc[-1]
            future_dates = [last_date + (i + 1) * pd_offset for i in range(n_periods)]

            # ربط آخر نقطة تاريخية بأول نقطة متوقعة ليكون الخط متصلاً بصرياً
            conn_dates = [last_date] + future_dates
            conn_vals = [y_vals[-1]] + list(future_y)

            fig_fc = go.Figure()
            fig_fc.add_trace(
                go.Scatter(
                    x=hist[fc_date],
                    y=hist[fc_target],
                    mode="lines+markers",
                    name=f"البيانات التاريخية ({freq_label}ي)",
                    line=dict(color="#38bdf8", width=2.8, shape="spline", smoothing=0.45),
                    marker=dict(size=6, color="#38bdf8"),
                )
            )
            fig_fc.add_trace(
                go.Scatter(
                    x=conn_dates,
                    y=conn_vals,
                    mode="lines+markers",
                    name=f"التوقع المستقبلي ({n_periods} {freq_label})",
                    line=dict(color="#e5c158", width=3.2, dash="dash"),
                    marker=dict(size=8, color="#e5c158", symbol="diamond"),
                )
            )

            st.plotly_chart(
                style_plotly_fig(fig_fc, f"التوقع المستقبلي لـ ({fc_target}) للـ {n_periods} فترات القادمة"),
                use_container_width=True,
            )

            trend_word = "تصاعدي 📈" if slope >= 0 else "تنازلي 📉"
            total_forecast = float(np.sum(future_y))
            st.markdown(
                f'<div class="callout-box">💡 مؤشر الاتجاه العام: الاتجاه التاريخي للبيانات <strong>{trend_word}</strong> بمعدل تغير <strong>{slope:+.2f}</strong> لكل {freq_label}، وإجمالي التوقع للفترات القادمة يبلغ <strong>{total_forecast:,.2f}</strong>.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning("البيانات الزمنية غير كافية لبناء نموذج التوقع.")
    else:
        st.info("يتطلب التوقع الزمني وجود عمود تاريخ وعمود رقمي واحد على الأقل.")


# ------------------------------------------
# التبويب الرابع: الذكاء الاصطناعي والتقرير التنفيذي
# ------------------------------------------
with tab4:
    st.markdown('<div class="section-header">📑 مولّد التقرير التنفيذي والاستراتيجية الشاملة (One-Click Executive Report)</div>', unsafe_allow_html=True)
    st.caption("يقوم النظام بتحليل كامل الملف إحصائياً واستخلاص أعمق الرؤى، المخاطر، وخطة العمل الاستراتيجية في تقرير متكامل قابل للتحميل.")

    if st.button("🪄 توليد التقرير التنفيذي والاستراتيجية الشاملة الآن", type="primary", use_container_width=True):
        with st.spinner("جاري تحليل البيانات وصياغة التقرير التنفيذي بالذكاء الاصطناعي..."):
            summary_stats = df.describe(include="all").fillna("").to_string()
            sample_rows = df.head(5).to_string()
            exec_prompt = f"""
            أنت مستشار تحليل بيانات واستراتيجية أعمال (Senior BI & Strategy Consultant).
            حلل ملخص البيانات التالي واكتب تقريراً تنفيذياً احترافياً باللغة العربية (مع المصطلحات الإنجليزية المهمة) منظم في الأقسام التالية:
            1. الملخص التنفيذي (Executive Summary)
            2. أبرز الرؤى والأنماط المكتشفة (Key Insights)
            3. المخاطر ونقاط الضعف المحتملة (Risks & Anomalies)
            4. خطة العمل والتوصيات الاستراتيجية القابلة للتنفيذ (Actionable Recommendations)

            إحصائيات البيانات (عدد الصفوف {len(df)}):
            {summary_stats[:3500]}

            عينة من الصفوف:
            {sample_rows[:1200]}
            """
            report_md = ask_gemini_with_fallback(exec_prompt)
            st.session_state.exec_report = report_md

    if "exec_report" in st.session_state and st.session_state.exec_report:
        st.markdown(st.session_state.exec_report)
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "📥 تحميل التقرير (Markdown)",
                data=st.session_state.exec_report.encode("utf-8"),
                file_name="Executive_Strategy_Report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with dl2:
            html_report = f"""
            <html dir="rtl" lang="ar">
            <head><meta charset="utf-8"><style>
            body {{ font-family: Arial, sans-serif; background:#0d1426; color:#f3f4f6; padding:30px; line-height:1.7; }}
            h1, h2, h3 {{ color:#e5c158; }}
            </style></head>
            <body><h1>التقرير التنفيذي الشامل — AI Data Analyst Pro</h1><pre style="white-space:pre-wrap; font-family:inherit;">{st.session_state.exec_report}</pre></body>
            </html>
            """
            st.download_button(
                "🌐 تحميل التقرير (HTML)",
                data=html_report.encode("utf-8"),
                file_name="Executive_Strategy_Report.html",
                mime="text/html",
                use_container_width=True,
            )

    st.markdown("---")
    st.markdown('<div class="section-header">💬 اسأل المحلل الذكي (Interactive AI Chart Builder)</div>', unsafe_allow_html=True)
    st.caption("اكتب أي سؤال مخصص وسيقوم الذكاء الاصطناعي بتحليل البيانات والإجابة ورسم الشارت المناسب.")

    user_q = st.text_input("...اسأل أي سؤال عن بياناتك (مثال: قارن إجمالي المبيعات حسب الفرع)")
    if st.button("🚀 تحليل ورسم الشارت") and user_q:
        with st.spinner("جاري التحليل..."):
            cols_info = {c: str(df[c].dtype) for c in df.columns}
            chat_prompt = f"""
            لديك DataFrame باسم df أعمدته وأنواعها: {cols_info}.
            أجب عن سؤال المستخدم باختصار ووضوح بالعربية، وإذا كان السؤال يناسبه رسم بياني، أرفق في نهاية ردك كود JSON فقط داخل ```json ... ``` بالشكل التالي:
            {{"chart_type": "bar|line|pie", "x": "اسم_عمود_x", "y": "اسم_عمود_y", "agg": "sum|mean|count", "title": "عنوان الرسم"}}
            سؤال المستخدم: {user_q}
            """
            ans = ask_gemini_with_fallback(chat_prompt)
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", ans, re.DOTALL)
            clean_text = re.sub(r"```json.*?```", "", ans, flags=re.DOTALL).strip()
            st.markdown(clean_text)

            if json_match:
                try:
                    spec = json.loads(json_match.group(1))
                    x_col, y_col = spec.get("x"), spec.get("y")
                    agg_fn = spec.get("agg", "sum")
                    if x_col in df.columns and y_col in df.columns:
                        if agg_fn == "mean":
                            plot_df = df.groupby(x_col, as_index=False)[y_col].mean()
                        elif agg_fn == "count":
                            plot_df = df.groupby(x_col, as_index=False)[y_col].count()
                        else:
                            plot_df = df.groupby(x_col, as_index=False)[y_col].sum()

                        ctype = spec.get("chart_type", "bar")
                        if ctype == "line":
                            fig_ai = px.line(plot_df, x=x_col, y=y_col, markers=True, color_discrete_sequence=["#e5c158"])
                        elif ctype == "pie":
                            fig_ai = px.pie(plot_df, names=x_col, values=y_col)
                        else:
                            fig_ai = px.bar(plot_df, x=x_col, y=y_col, color_discrete_sequence=["#e5c158"])

                        st.plotly_chart(style_plotly_fig(fig_ai, spec.get("title", "")), use_container_width=True)
                except Exception:
                    pass
