import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from google import genai

# 1. إعدادات الصفحة الأساسية
st.set_page_config(
    page_title="AI Data Analyst Pro | Mohamed Gomaa",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. حقن كود CSS احترافي متكامل (Dark Navy, Gold Accents, Glassmorphism & RTL)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');

    /* الخط العام والخلفية الرئيسية */
    html, body, [class*="css"] {
        font-family: 'Cairo', sans-serif;
    }
    .stApp {
        background: radial-gradient(circle at top right, #131c31 0%, #0a0f1c 60%, #060911 100%);
        color: #f3f4f6;
    }

    /* إخفاء عناصر Streamlit الافتراضية */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* محاذاة النصوص العربية RTL */
    h1, h2, h3, h4, p, .stMarkdown, .stAlert, .stCaption {
        direction: rtl;
        text-align: right;
    }
    /* الحفاظ على الجداول والأكواد LTR */
    [data-testid="stDataFrame"], code, pre {
        direction: ltr;
    }

    /* تنسيق الشريط الجانبي Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1323 0%, #090d18 100%);
        border-right: 1px solid rgba(212, 175, 55, 0.25);
    }

    /* تصميم كروت الأرقام (Metrics) بتأثير Glassmorphism */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(212, 175, 55, 0.28);
        border-radius: 14px;
        padding: 15px 14px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        transition: all 0.25s ease;
        direction: rtl;
        text-align: right;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: #d4af37;
        box-shadow: 0 10px 28px rgba(212, 175, 55, 0.15);
    }
    [data-testid="stMetricLabel"] {
        color: #9ca3af !important;
        font-size: 0.9rem !important;
    }
    [data-testid="stMetricValue"] {
        color: #d4af37 !important;
        font-weight: 700 !important;
        font-size: 1.55rem !important;
    }

    /* تخصيص الأزرار: الزر الرئيسي بتدرج ذهبي فخم */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #d4af37 0%, #f3d06b 50%, #b89225 100%) !important;
        color: #0a0f1c !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.2rem !important;
        box-shadow: 0 4px 15px rgba(212, 175, 55, 0.3) !important;
        transition: all 0.25s ease !important;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 22px rgba(212, 175, 55, 0.5) !important;
    }

    /* الأزرار الفرعية الزجاجية */
    div.stButton > button[kind="secondary"], div.stDownloadButton > button {
        background: rgba(255, 255, 255, 0.04) !important;
        color: #f3f4f6 !important;
        border: 1px solid rgba(212, 175, 55, 0.3) !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button[kind="secondary"]:hover, div.stDownloadButton > button:hover {
        border-color: #d4af37 !important;
        background: rgba(212, 175, 55, 0.12) !important;
        color: #d4af37 !important;
    }

    /* تنسيق التبويبات Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(255, 255, 255, 0.02);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        direction: rtl;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 18px;
        color: #9ca3af;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.2), rgba(212, 175, 55, 0.08)) !important;
        color: #d4af37 !important;
        border: 1px solid rgba(212, 175, 55, 0.45);
    }

    /* تنسيق الجداول والقوائم المنسدلة وإصلاح تباين الألوان */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(212, 175, 55, 0.2);
        border-radius: 12px;
        overflow: hidden;
    }
    [data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.85) !important;
        border: 1px solid rgba(212, 175, 55, 0.28) !important;
        border-radius: 12px;
        direction: rtl;
        color: #f3f4f6 !important;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] label,
    [data-testid="stExpander"] span {
        color: #f3f4f6 !important;
    }
    [data-testid="stExpander"] details {
        background-color: transparent !important;
    }

    /* شريط التقدم الذهبي لمؤشر جودة البيانات */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #b89225, #d4af37, #f3d06b);
    }
</style>
""", unsafe_allow_html=True)

# قراءة مفتاح Gemini API تلقائياً من secrets.toml أو متغيرات البيئة
try:
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
except Exception:
    api_key = os.environ.get("GEMINI_API_KEY", "")

# دالة إنشاء بيانات مبيعات تجريبية جاهزة لاختبار الزوار الفوري
def load_sample_dataset():
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=120, freq="D")
    branches = ["القاهرة", "الإسكندرية", "أسوان", "المنصورة"]
    categories = ["إلكترونيات", "أجهزة منزلية", "إكسسوارات", "برمجيات"]
    payment_methods = ["كاش", "إنستاباي", "بطاقة بنكية"]
    
    qty = np.random.randint(1, 15, size=120)
    unit_price = np.random.choice([250, 500, 1200, 3500, 7500], size=120)
    
    return pd.DataFrame({
        "order_date": dates,
        "branch": np.random.choice(branches, size=120),
        "category": np.random.choice(categories, size=120),
        "payment_method": np.random.choice(payment_methods, size=120),
        "quantity": qty,
        "unit_price": unit_price,
        "total_sales": qty * unit_price
    })

# دوال التنظيف الاحترافية مع إرجاع تقرير تفصيلي
def log_action(message):
    if "cleaning_log" not in st.session_state:
        st.session_state.cleaning_log = []
    st.session_state.cleaning_log.append(f"✔ {message}")
    st.session_state.last_report = message

def standardize_col_names(dataframe):
    df_copy = dataframe.copy()
    old_cols = list(df_copy.columns)
    new_cols = []
    seen = {}
    for col in old_cols:
        clean_col = re.sub(r'[^\w\s]', '', str(col)).strip().lower()
        clean_col = re.sub(r'\s+', '_', clean_col)
        if not clean_col:
            clean_col = "unnamed_col"
        if clean_col in seen:
            seen[clean_col] += 1
            clean_col = f"{clean_col}_{seen[clean_col]}"
        else:
            seen[clean_col] = 0
        new_cols.append(clean_col)
    df_copy.columns = new_cols
    changed = sum(1 for o, n in zip(old_cols, new_cols) if str(o) != n)
    return df_copy, changed

def normalize_hidden_nulls(dataframe):
    df_copy = dataframe.copy()
    null_tokens = ["n/a", "na", "null", "none", "-", "--", "?", "nan", ""]
    before_nulls = df_copy.isnull().sum().sum()
    for col in df_copy.select_dtypes(include=['object']).columns:
        df_copy[col] = df_copy[col].apply(
            lambda x: np.nan if (isinstance(x, str) and x.strip().lower() in null_tokens) else x
        )
    after_nulls = df_copy.isnull().sum().sum()
    return df_copy, int(after_nulls - before_nulls)

def promote_header_if_needed(dataframe):
    df_copy = dataframe.copy()
    unnamed_ratio = sum(1 for col in df_copy.columns if 'unnamed' in str(col).lower()) / max(len(df_copy.columns), 1)
    if unnamed_ratio >= 0.4 and len(df_copy) > 0:
        for idx in range(min(8, len(df_copy))):
            row_values = df_copy.iloc[idx]
            non_null_ratio = row_values.notna().mean()
            str_ratio = sum(1 for v in row_values.dropna() if isinstance(v, str) and not v.strip().replace('.', '').isdigit()) / max(row_values.notna().sum(), 1)
            if non_null_ratio >= 0.75 and str_ratio >= 0.6:
                new_headers = [str(v).strip() if pd.notna(v) else f"col_{i}" for i, v in enumerate(row_values)]
                df_copy.columns = new_headers
                df_copy = df_copy.iloc[idx + 1:].reset_index(drop=True)
                return df_copy, True
    return df_copy, False

def smart_fix_types(dataframe):
    df_fixed = dataframe.copy()
    converted_cols = []
    protected_keywords = ['id', 'code', 'phone', 'mobile', 'tel', 'sku', 'serial', 'zip', 'postal', 'كود', 'هاتف', 'تليفون', 'رقم']
    
    for col in df_fixed.select_dtypes(include=['object']).columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in protected_keywords):
            continue
        non_null_series = df_fixed[col].dropna().astype(str)
        if non_null_series.empty:
            continue
        if non_null_series.str.startswith('0').mean() > 0.3 and not non_null_series.str.startswith('0.').any():
            continue

        cleaned_series = df_fixed[col].astype(str).str.replace(r'[^\d.\-]', '', regex=True)
        numeric_converted = pd.to_numeric(cleaned_series, errors='coerce')
        if (numeric_converted.notnull().sum() / len(non_null_series)) >= 0.8:
            df_fixed[col] = numeric_converted
            converted_cols.append(f"{col} (رقمي)")
            continue
        
        if any(word in col_lower for word in ['date', 'time', 'day', 'month', 'year', 'تاريخ']):
            try:
                date_converted = pd.to_datetime(df_fixed[col], errors='coerce', dayfirst=True)
                if (date_converted.notnull().sum() / len(non_null_series)) >= 0.8:
                    df_fixed[col] = date_converted
                    converted_cols.append(f"{col} (تاريخ)")
            except Exception:
                pass
    return df_fixed, converted_cols

def calculate_health_score(dataframe):
    total_cells = dataframe.size
    if total_cells == 0:
        return 100
    null_cells = dataframe.isnull().sum().sum()
    dup_rows = dataframe.duplicated().sum() * dataframe.shape[1]
    penalty = ((null_cells + dup_rows) / total_cells) * 100
    return max(0, min(100, round(100 - penalty, 1)))

def call_gemini_api(prompt_text):
    client = genai.Client(api_key=api_key)
    models_to_try = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]
    try:
        for m in client.models.list():
            m_id = m.name.replace("models/", "")
            if "gemini" in m_id and m_id not in models_to_try:
                models_to_try.append(m_id)
    except Exception:
        pass

    last_err = None
    for m_name in models_to_try:
        try:
            return client.models.generate_content(model=m_name, contents=prompt_text)
        except Exception as e:
            last_err = e
            continue
    raise last_err

def build_html_report(report_md, file_name, rows_count, cols_count, health):
    html_body = report_md.replace("\n", "<br>")
    return f"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<title>Executive Data Report - {file_name}</title>
<style>
    body {{
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #0a0f1c;
        color: #f3f4f6;
        margin: 0;
        padding: 35px;
        line-height: 1.8;
    }}
    .container {{
        max-width: 900px;
        margin: auto;
        background: #111827;
        border: 1px solid #d4af37;
        border-radius: 16px;
        padding: 32px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }}
    .header {{
        border-bottom: 2px solid #d4af37;
        padding-bottom: 16px;
        margin-bottom: 24px;
        text-align: center;
    }}
    .header h1 {{
        color: #d4af37;
        margin: 0;
        font-size: 26px;
    }}
    .meta {{
        display: flex;
        justify-content: space-around;
        background: rgba(212, 175, 55, 0.08);
        border: 1px solid rgba(212, 175, 55, 0.3);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 24px;
        font-weight: bold;
        color: #f3d06b;
    }}
    .content {{
        font-size: 15px;
        color: #e5e7eb;
    }}
    .footer {{
        margin-top: 30px;
        padding-top: 12px;
        border-top: 1px solid #374151;
        text-align: center;
        font-size: 12px;
        color: #9ca3af;
    }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>📊 التقرير التنفيذي والاستراتيجية التحليلية</h1>
        <p style="color:#9ca3af; margin:5px 0 0 0;">ملف البيانات: {file_name}</p>
    </div>
    <div class="meta">
        <span>إجمالي الصفوف: {rows_count:,}</span>
        <span>عدد الأعمدة: {cols_count}</span>
        <span>جودة البيانات: {health}%</span>
    </div>
    <div class="content">{html_body}</div>
    <div class="footer">Generated by AI Data Analyst Pro | Engineered by Mohamed Gomaa</div>
</div>
</body>
</html>"""

# 3. الشريط الجانبي (Sidebar)
with st.sidebar:
    st.markdown("<h2 style='color: #d4af37; text-align: center;'>⚡ AI Data Analyst</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #9ca3af; font-size: 0.85rem;'>منصة التحليل الذكي وتنظيف البيانات</p>", unsafe_allow_html=True)
    st.divider()
    
    uploaded_file = st.file_uploader(
        "📂 ارفع ملف البيانات هنا", 
        type=["csv", "xlsx"],
        help="يدعم ملفات CSV و Excel حتى 200 ميجابايت"
    )
    
    st.markdown("<p style='text-align: center; color: #9ca3af; font-size: 0.8rem; margin: 4px 0;'>— أو —</p>", unsafe_allow_html=True)
    if st.button("🚀 جرب ببيانات مبيعات جاهزة", use_container_width=True):
        sample_df = load_sample_dataset()
        st.session_state.current_file = "Sample_Sales_Data.xlsx"
        st.session_state.raw_df = sample_df.copy()
        st.session_state.df = sample_df.copy()
        st.session_state.cleaning_log = ["تم تحميل بيانات المبيعات التجريبية بنجاح"]
        st.session_state.last_report = None
        st.session_state.exec_report_md = None
        st.rerun()

    if "cleaning_log" in st.session_state and st.session_state.cleaning_log:
        st.divider()
        st.markdown("### 📜 سجل خطوات التنظيف")
        for item in st.session_state.cleaning_log:
            st.caption(item)

    st.divider()
    st.markdown("""
    <div style='text-align: center; direction: ltr; padding: 10px; background: rgba(255,255,255,0.02); border-radius: 10px; border: 1px solid rgba(212, 175, 55, 0.15);'>
        <p style='text-align: center; color: #9ca3af; font-size: 0.78rem; margin: 0;'>Built with ❤️ by</p>
        <p style='text-align: center; color: #d4af37; font-size: 0.9rem; font-weight: 700; margin: 2px 0;'>Mohamed Gomaa</p>
        <p style='text-align: center; color: #6b7280; font-size: 0.72rem; margin: 0;'>Data Analyst & Software Engineer</p>
    </div>
    """, unsafe_allow_html=True)

# 4. المساحة الرئيسية
st.markdown("""
<div style='text-align: center; padding: 10px 0 20px 0;'>
    <div style='display: inline-block; background: rgba(212, 175, 55, 0.12); border: 1px solid rgba(212, 175, 55, 0.4); border-radius: 20px; padding: 5px 16px; margin-bottom: 12px; direction: ltr;'>
        <span style='color: #d4af37; font-size: 0.85rem; font-weight: 600; letter-spacing: 0.5px;'>
            👨‍💻 Engineered & Designed by <a href="https://github.com/MohamedGomaa11111" target="_blank" style="color: #f3d06b; text-decoration: none; font-weight: 700;">Mohamed Gomaa</a>
        </span>
    </div>
    <h1 style='text-align: center; color: #d4af37; font-size: 2.5rem; margin-bottom: 5px;'>📊 لوحة تحليل البيانات الذكية</h1>
    <p style='text-align: center; color: #9ca3af; font-size: 1.05rem;'>ارفع بياناتك، نظّفها بضغطة زر، استشرف التوقعات، وأصدر تقارير تنفيذية متكاملة</p>
</div>
""", unsafe_allow_html=True)

if uploaded_file is not None or "df" in st.session_state:
    if uploaded_file is not None and ("current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name):
        if uploaded_file.name.endswith('.csv'):
            raw_df = pd.read_csv(uploaded_file)
        else:
            raw_df = pd.read_excel(uploaded_file)
        st.session_state.current_file = uploaded_file.name
        st.session_state.raw_df = raw_df.copy()
        st.session_state.df = raw_df.copy()
        st.session_state.cleaning_log = ["تم تحميل الملف الأصلي بنجاح"]
        st.session_state.last_report = None
        st.session_state.exec_report_md = None

    active_file_name = uploaded_file.name if uploaded_file is not None else st.session_state.get("current_file", "Sample_Sales_Data.xlsx")
    df = st.session_state.df
    health_score = calculate_health_score(df)

    # كروت الملخص السريع + مؤشر جودة البيانات
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("إجمالي الصفوف", f"{df.shape[0]:,}")
    c2.metric("عدد الأعمدة", df.shape[1])
    c3.metric("القيم المفقودة", f"{df.isnull().sum().sum():,}")
    c4.metric("الصفوف المكررة", f"{df.duplicated().sum():,}")
    c5.metric("جودة البيانات", f"{health_score}%")
    
    st.progress(health_score / 100)
    st.write("")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧹 معاينة وتنظيف البيانات",
        "📈 التحليل الاستكشافي (EDA)",
        "🔮 التنبؤ ومحاكاة السيناريوهات",
        "🤖 الذكاء الاصطناعي والتقرير التنفيذي"
    ])
    
    # ==================== TAB 1: التنظيف ====================
    with tab1:
        if st.session_state.get("last_report"):
            st.success(f"**📋 تقرير المعالجة الأخيرة:** {st.session_state.last_report}")

        st.subheader("🛠️ أدوات التنظيف الاحترافي")
        
        top_btn1, top_btn2 = st.columns([2, 1])
        with top_btn1:
            if st.button("⚡ التنظيف الذكي الشامل (Auto-Clean All)", type="primary", use_container_width=True):
                temp_df = st.session_state.df.copy()
                report_details = []

                temp_df, hidden_nulls = normalize_hidden_nulls(temp_df)
                if hidden_nulls > 0:
                    report_details.append(f"اكتشاف {hidden_nulls} قيمة فارغة مخفية")

                cols_before_all = temp_df.shape[1]
                rows_before_all = temp_df.shape[0]
                temp_df = temp_df.dropna(axis=1, how='all').dropna(axis=0, how='all').reset_index(drop=True)
                if temp_df.shape[1] < cols_before_all:
                    report_details.append(f"حذف {cols_before_all - temp_df.shape[1]} أعمدة فارغة تماماً")
                if temp_df.shape[0] < rows_before_all:
                    report_details.append(f"حذف {rows_before_all - temp_df.shape[0]} صفوف فارغة تماماً")

                if len(temp_df) > 0:
                    cols_before_sparse = temp_df.shape[1]
                    min_valid = max(2, int(len(temp_df) * 0.25))
                    temp_df = temp_df.dropna(axis=1, thresh=min_valid)
                    if temp_df.shape[1] < cols_before_sparse:
                        report_details.append(f"حذف {cols_before_sparse - temp_df.shape[1]} أعمدة جانبية مهملة")

                temp_df, header_promoted = promote_header_if_needed(temp_df)
                if header_promoted:
                    report_details.append("اكتشاف صف العناوين وتعيينه كـ Header تلقائياً")

                rows_before_na = len(temp_df)
                temp_df = temp_df.dropna(how='any').reset_index(drop=True)
                if len(temp_df) < rows_before_na:
                    report_details.append(f"حذف {rows_before_na - len(temp_df)} صفوف ناقصة")

                temp_df, cols_changed = standardize_col_names(temp_df)
                if cols_changed > 0:
                    report_details.append(f"توحيد {cols_changed} اسم عمود")

                before_dup = len(temp_df)
                temp_df = temp_df.drop_duplicates().reset_index(drop=True)
                removed_dup = before_dup - len(temp_df)
                if removed_dup > 0:
                    report_details.append(f"حذف {removed_dup} صف مكرر")

                text_cols_cleaned = 0
                for col in temp_df.select_dtypes(include=['object']).columns:
                    orig_col = temp_df[col].copy()
                    temp_df[col] = temp_df[col].apply(lambda x: x.strip().title() if isinstance(x, str) else x)
                    if not orig_col.equals(temp_df[col]):
                        text_cols_cleaned += 1
                if text_cols_cleaned > 0:
                    report_details.append(f"تنظيف النصوص في {text_cols_cleaned} عمود")

                temp_df, fixed_types = smart_fix_types(temp_df)
                if fixed_types:
                    report_details.append(f"تحويل الأنواع: {', '.join(fixed_types)}")

                st.session_state.df = temp_df
                summary_msg = " | ".join(report_details) if report_details else "البيانات نظيفة بالفعل ولا توجد أخطاء شكلية!"
                log_action(f"تنظيف ذكي شامل: {summary_msg}")
                st.rerun()

        with top_btn2:
            if st.button("🔄 استرجاع البيانات الأصلية", use_container_width=True):
                st.session_state.df = st.session_state.raw_df.copy()
                st.session_state.cleaning_log = ["تمت استعادة النسخة الأصلية للملف"]
                st.session_state.last_report = "تمت استعادة البيانات الأصلية بنجاح."
                st.rerun()

        st.write("")

        col_clean1, col_clean2, col_clean3, col_clean4 = st.columns(4)
        with col_clean1:
            if st.button("🗑️ حذف المكرر", use_container_width=True):
                before = len(st.session_state.df)
                st.session_state.df = st.session_state.df.drop_duplicates().reset_index(drop=True)
                removed = before - len(st.session_state.df)
                log_action(f"حذف {removed} صف مكرر وإعادة ترتيب الفهرس")
                st.rerun()

        with col_clean2:
            if st.button("✨ توحيد النصوص", use_container_width=True):
                temp_df, hidden = normalize_hidden_nulls(st.session_state.df)
                for col in temp_df.select_dtypes(include=['object']).columns:
                    temp_df[col] = temp_df[col].apply(lambda x: x.strip().title() if isinstance(x, str) else x)
                st.session_state.df = temp_df
                log_action(f"تنظيف المسافات وتوحيد النصوص (واكتشاف {hidden} قيمة فارغة مخفية)")
                st.rerun()

        with col_clean3:
            if st.button("🔢 تصحيح الأنواع", use_container_width=True):
                st.session_state.df, fixed = smart_fix_types(st.session_state.df)
                msg = f"تم تحويل: {', '.join(fixed)}" if fixed else "جميع أنواع الأعمدة صحيحة بالفعل"
                log_action(msg)
                st.rerun()

        with col_clean4:
            if st.button("🏷️ توحيد أسماء الأعمدة", use_container_width=True):
                st.session_state.df, changed = standardize_col_names(st.session_state.df)
                log_action(f"توحيد {changed} اسم عمود لصيغة snake_case")
                st.rerun()

        st.write("")

        adv_col1, adv_col2 = st.columns(2)
        with adv_col1:
            with st.expander("🩹 معالجة القيم المفقودة (Missing Values)", expanded=False):
                missing_method = st.selectbox(
                    "اختر طريقة المعالجة:",
                    [
                        "حذف الصفوف التي تحتوي على قيم فارغة (Drop Rows)",
                        "تعويض الأرقام بالمتوسط (Mean) والنصوص بالأكثر تكراراً (Mode)",
                        "تعويض الأرقام بالوسيط (Median) والنصوص بالأكثر تكراراً (Mode)"
                    ]
                )
                if st.button("تطبيق معالجة القيم الفارغة", use_container_width=True):
                    temp_df, _ = normalize_hidden_nulls(st.session_state.df)
                    nulls_before = temp_df.isnull().sum().sum()
                    if nulls_before == 0:
                        log_action("فحص القيم المفقودة: لا توجد أي قيم فارغة في الجدول")
                    elif "حذف" in missing_method:
                        rows_before = len(temp_df)
                        temp_df = temp_df.dropna().reset_index(drop=True)
                        log_action(f"حذف {rows_before - len(temp_df)} صف يحتوي على قيم مفقودة")
                    else:
                        for col in temp_df.columns:
                            if temp_df[col].isnull().sum() > 0:
                                if pd.api.types.is_numeric_dtype(temp_df[col]):
                                    fill_val = temp_df[col].mean() if "Mean" in missing_method else temp_df[col].median()
                                    temp_df[col] = temp_df[col].fillna(fill_val)
                                else:
                                    mode_val = temp_df[col].mode()[0] if not temp_df[col].mode().empty else "Unknown"
                                    temp_df[col] = temp_df[col].fillna(mode_val)
                        log_action(f"تعويض {nulls_before} خلية فارغة باستخدام ({missing_method.split('(')[1].split(')')[0]})")
                    st.session_state.df = temp_df
                    st.rerun()

        with adv_col2:
            with st.expander("✂️ حذف الأعمدة غير المفيدة (Drop Columns)", expanded=False):
                cols_to_drop = st.multiselect(
                    "اختر الأعمدة التي تريد حذفها:",
                    options=st.session_state.df.columns.tolist()
                )
                if st.button("حذف الأعمدة المحددة", use_container_width=True):
                    if cols_to_drop:
                        st.session_state.df = st.session_state.df.drop(columns=cols_to_drop)
                        log_action(f"حذف الأعمدة: {', '.join(cols_to_drop)}")
                        st.rerun()
                    else:
                        st.warning("اختر عموداً واحداً على الأقل أولاً.")

        adv_col3, adv_col4 = st.columns(2)
        with adv_col3:
            with st.expander("🎯 معالجة القيم الشاذة (Outliers - IQR)", expanded=False):
                num_cols_out = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
                if num_cols_out:
                    target_num_col = st.selectbox("اختر العمود الرقمي:", num_cols_out, key="outlier_col")
                    Q1 = st.session_state.df[target_num_col].quantile(0.25)
                    Q3 = st.session_state.df[target_num_col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers_count = ((st.session_state.df[target_num_col] < lower_bound) | (st.session_state.df[target_num_col] > upper_bound)).sum()
                    
                    st.caption(f"عدد القيم الشاذة المكتشفة في ({target_num_col}): **{outliers_count}**")
                    outlier_action = st.radio("الإجراء:", ["حصر القيم عند الحدود الطبيعية (Capping)", "حذف الصفوف الشاذة (Remove)"], horizontal=True)
                    
                    if st.button("تطبيق معالجة القيم الشاذة", use_container_width=True):
                        temp_df = st.session_state.df.copy()
                        if outliers_count == 0:
                            log_action(f"فحص القيم الشاذة في ({target_num_col}): لا توجد قيم شاذة")
                        elif "Capping" in outlier_action:
                            temp_df[target_num_col] = np.clip(temp_df[target_num_col], lower_bound, upper_bound)
                            log_action(f"حصر {outliers_count} قيمة شاذة في عمود ({target_num_col}) عند [{lower_bound:.1f}, {upper_bound:.1f}]")
                        else:
                            temp_df = temp_df[(temp_df[target_num_col] >= lower_bound) & (temp_df[target_num_col] <= upper_bound)].reset_index(drop=True)
                            log_action(f"حذف {outliers_count} صف شاذ من عمود ({target_num_col})")
                        st.session_state.df = temp_df
                        st.rerun()
                else:
                    st.info("لا توجد أعمدة رقمية لفحص القيم الشاذة.")

        with adv_col4:
            with st.expander("🔄 البحث والاستبدال (Find & Replace)", expanded=False):
                rep_col = st.selectbox("اختر العمود:", st.session_state.df.columns.tolist(), key="rep_col")
                c_find, c_rep = st.columns(2)
                find_val = c_find.text_input("القيمة القديمة:")
                replace_val = c_rep.text_input("القيمة الجديدة:")
                if st.button("تنفيذ الاستبدال", use_container_width=True):
                    if find_val != "":
                        temp_df = st.session_state.df.copy()
                        if pd.api.types.is_numeric_dtype(temp_df[rep_col]):
                            try:
                                f_v, r_v = float(find_val), float(replace_val)
                                count_rep = (temp_df[rep_col] == f_v).sum()
                                temp_df[rep_col] = temp_df[rep_col].replace(f_v, r_v)
                                st.session_state.df = temp_df
                                log_action(f"استبدال {count_rep} خلية من ({find_val}) إلى ({replace_val}) في عمود {rep_col}")
                                st.rerun()
                            except ValueError:
                                st.error("العمود رقمي، أدخل أرقاماً صحيحة.")
                        else:
                            count_rep = (temp_df[rep_col].astype(str) == find_val).sum()
                            temp_df[rep_col] = temp_df[rep_col].astype(str).replace(find_val, replace_val)
                            st.session_state.df = temp_df
                            log_action(f"استبدال {count_rep} خلية من ('{find_val}') إلى ('{replace_val}') في عمود {rep_col}")
                            st.rerun()

        st.divider()
        
        head_col1, head_col2 = st.columns([3, 1])
        with head_col1:
            st.subheader("📋 جدول البيانات بعد التحديث")
        with head_col2:
            csv_data = st.session_state.df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 تحميل الداتا النظيفة (CSV)",
                data=csv_data,
                file_name=f"cleaned_{active_file_name.split('.')[0]}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        st.dataframe(st.session_state.df, use_container_width=True, height=350)
        
    # ==================== TAB 2: التحليل الاستكشافي ====================
    with tab2:
        st.subheader("🔍 فحص هيكل البيانات (Data Schema)")
        schema_df = pd.DataFrame({
            "اسم العمود": df.columns,
            "نوع البيانات": df.dtypes.astype(str),
            "القيم الفريدة (Unique)": df.nunique().values,
            "القيم المفقودة": df.isnull().sum().values
        })
        st.dataframe(schema_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("📊 الإحصائيات الوصفية (Descriptive Statistics)")
        
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime']).columns.tolist()

        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            st.markdown("**الأعمدة الرقمية:**")
            if num_cols:
                st.dataframe(df[num_cols].describe().T, use_container_width=True)
            else:
                st.info("لا توجد أعمدة رقمية في هذا الملف.")

        with stat_col2:
            st.markdown("**الأعمدة النصية / التصنيفية:**")
            if cat_cols:
                st.dataframe(df[cat_cols].astype(str).describe().T, use_container_width=True)
            else:
                st.info("لا توجد أعمدة نصية في هذا الملف.")

        st.divider()
        st.subheader("📈 استكشاف التوزيعات والعلاقات تلقائياً")

        useful_num_cols = [c for c in num_cols if not any(k in c.lower() for k in ['id', 'code', 'zip', 'phone'])] or num_cols
        useful_cat_cols = [c for c in cat_cols if not any(k in c.lower() for k in ['id', 'code', 'sku', 'serial']) and df[c].nunique() < len(df) * 0.8] or cat_cols

        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            if useful_num_cols:
                selected_num = st.selectbox("اختر عمود رقمي لعرض توزيعه:", useful_num_cols)
                fig_hist = px.histogram(
                    df, x=selected_num, marginal="box",
                    title=f"توزيع قيم: {selected_num}",
                    template="plotly_dark",
                    color_discrete_sequence=["#d4af37"]
                )
                fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_hist, use_container_width=True)

        with viz_col2:
            if useful_cat_cols:
                selected_cat = st.selectbox("اختر عمود تصنيفي لعرض أعلى الفئات:", useful_cat_cols)
                top_cats = df[selected_cat].value_counts().head(10).reset_index()
                top_cats.columns = [selected_cat, "العدد"]
                fig_bar = px.bar(
                    top_cats, x=selected_cat, y="العدد",
                    title=f"أعلى الفئات تكراراً في: {selected_cat}",
                    template="plotly_dark",
                    color_discrete_sequence=["#38bdf8"]
                )
                fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_bar, use_container_width=True)

        if date_cols and useful_num_cols:
            st.subheader("📅 تحليل الاتجاه الزمني (Time Series)")
            ts_col1, ts_col2 = st.columns(2)
            selected_date = ts_col1.selectbox("عمود التاريخ:", date_cols)
            selected_ts_num = ts_col2.selectbox("المتغير الرقمي:", useful_num_cols, key="ts_num")
            
            ts_df = df.groupby(selected_date)[selected_ts_num].sum().reset_index()
            fig_line = px.line(
                ts_df, x=selected_date, y=selected_ts_num, markers=True,
                title=f"تطور {selected_ts_num} عبر {selected_date}",
                template="plotly_dark",
                color_discrete_sequence=["#10b981"]
            )
            fig_line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_line, use_container_width=True)

        if len(useful_num_cols) >= 2:
            st.subheader("🔥 مصفوفة الارتباط (Correlation Heatmap)")
            corr_matrix = df[useful_num_cols].corr()
            fig_corr = px.imshow(
                corr_matrix, text_auto=".2f", aspect="auto",
                color_continuous_scale="RdBu_r",
                template="plotly_dark",
                title="قوة الارتباط بين المتغيرات الرقمية"
            )
            fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_corr, use_container_width=True)

    # ==================== TAB 3: التنبؤ ومحاكاة السيناريوهات ====================
    with tab3:
        st.subheader("🎛️ محاكي السيناريوهات واتخاذ القرار (What-If Decision Simulator)")
        st.caption("اختبر تأثير التغييرات المستقبلية (زيادة المبيعات، خفض التكاليف أو الهدر) على إجمالي الأداء فوراً.")

        sim_num_cols = [c for c in df.select_dtypes(include=['number']).columns if not any(k in c.lower() for k in ['id', 'code', 'zip', 'phone'])]
        if sim_num_cols:
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                target_metric = st.selectbox("اختر المؤشر الرئيسي (مثل المبيعات / الإيرادات):", sim_num_cols, key="sim_metric")
            with sc2:
                growth_pct = st.slider("نسبة التغير المتوقعة في المؤشر (%):", min_value=-50, max_value=100, value=15, step=5)
            with sc3:
                cost_ratio = st.slider("نسبة التكاليف / الهدر التقديرية (%):", min_value=0, max_value=90, value=30, step=5)

            current_total = float(df[target_metric].sum())
            projected_total = current_total * (1 + growth_pct / 100.0)
            current_net = current_total * (1 - cost_ratio / 100.0)
            projected_net = projected_total * (1 - cost_ratio / 100.0)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("الإجمالي الحالي", f"{current_total:,.2f}")
            m2.metric("الإجمالي المتوقع (بعد السيناريو)", f"{projected_total:,.2f}", delta=f"{growth_pct:+d}%")
            m3.metric("صافي العائد الحالي", f"{current_net:,.2f}")
            m4.metric("صافي العائد المتوقع", f"{projected_net:,.2f}", delta=f"{(projected_net - current_net):+,.2f}")

            sim_chart_df = pd.DataFrame({
                "الحالة": ["الوضع الحالي", "الوضع المتوقع (السيناريو)"],
                "الإجمالي الكلي": [current_total, projected_total],
                "صافي العائد": [current_net, projected_net]
            })
            fig_sim = go.Figure(data=[
                go.Bar(name="الإجمالي الكلي", x=sim_chart_df["الحالة"], y=sim_chart_df["الإجمالي الكلي"], marker_color="#d4af37"),
                go.Bar(name="صافي العائد", x=sim_chart_df["الحالة"], y=sim_chart_df["صافي العائد"], marker_color="#10b981")
            ])
            fig_sim.update_layout(
                barmode="group",
                title=f"مقارنة الأداء المالي لـ ({target_metric}) عند تغير بنسبة {growth_pct:+d}%",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=380
            )
            st.plotly_chart(fig_sim, use_container_width=True)
        else:
            st.info("يرجى التأكد من وجود أعمدة رقمية صالحة لتفعيل محاكي السيناريوهات.")

        st.divider()
        st.subheader("📈 التوقع الزمني المستقبلي (Linear Trend Forecasting)")
        dt_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        if dt_cols and sim_num_cols:
            fc1, fc2, fc3 = st.columns(3)
            f_date_col = fc1.selectbox("عمود التاريخ:", dt_cols, key="fc_date")
            f_val_col = fc2.selectbox("المتغير المراد توقعه:", sim_num_cols, key="fc_val")
            periods_ahead = fc3.slider("عدد الفترات المستقبلية للتوقع:", min_value=3, max_value=30, value=7)

            ts_grouped = df.groupby(f_date_col)[f_val_col].sum().reset_index().sort_values(f_date_col)
            if len(ts_grouped) >= 3:
                ts_grouped["t"] = np.arange(len(ts_grouped))
                slope, intercept = np.polyfit(ts_grouped["t"], ts_grouped[f_val_col], 1)

                avg_step = ts_grouped[f_date_col].diff().dropna().median()
                if pd.isna(avg_step) or avg_step == pd.Timedelta(0):
                    avg_step = pd.Timedelta(days=1)

                last_t = ts_grouped["t"].iloc[-1]
                last_date = ts_grouped[f_date_col].iloc[-1]

                future_t = np.arange(last_t + 1, last_t + 1 + periods_ahead)
                future_dates = [last_date + avg_step * i for i in range(1, periods_ahead + 1)]
                future_preds = np.maximum(0, slope * future_t + intercept)

                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(
                    x=ts_grouped[f_date_col], y=ts_grouped[f_val_col],
                    mode="lines+markers", name="البيانات الفعلية",
                    line=dict(color="#38bdf8", width=2.5)
                ))
                fig_fc.add_trace(go.Scatter(
                    x=future_dates, y=future_preds,
                    mode="lines+markers", name="التوقع المستقبلي (Forecast)",
                    line=dict(color="#d4af37", width=3, dash="dash")
                ))
                fig_fc.update_layout(
                    title=f"التوقع المستقبلي لـ ({f_val_col}) للـ {periods_ahead} فترات القادمة",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=400
                )
                st.plotly_chart(fig_fc, use_container_width=True)
                trend_dir = "تصاعدي 📈" if slope > 0 else ("تنازلي 📉" if slope < 0 else "مستقر ➖")
                st.info(f"**💡 مؤشر الاتجاه العام:** الاتجاه التاريخي للبيانات **{trend_dir}** بمعدل تغير **{slope:+,.2f}** لكل فترة زمنية، وإجمالي التوقع للفترات القادمة يبلغ **{future_preds.sum():,.2f}**.")
            else:
                st.warning("تحتاج البيانات إلى 3 نقاط زمنية على الأقل لحساب التوقع المستقبلي.")
        else:
            st.info("💡 لتفعيل التوقع الزمني التلقائي، تأكد من وجود عمود تاريخ (Date) في ملفك واضغط على زر (التنظيف الذكي الشامل) لتحويله.")

    # ==================== TAB 4: الذكاء الاصطناعي والتقرير التنفيذي ====================
    with tab4:
        st.subheader("📑 مولّد التقرير التنفيذي والاستراتيجية الشاملة (One-Click Executive Report)")
        st.caption("يقوم النظام بتحليل كامل الملف إحصائياً واستخلاص أعمق الرؤى، المخاطر، وخطة العمل الاستراتيجية في تقرير متكامل قابل للتحميل.")

        if not api_key:
            st.error("⚠️ لم يتم العثور على مفتاح `GEMINI_API_KEY` في ملف `.streamlit/secrets.toml`.")
        else:
            if st.button("🚀 توليد التقرير التنفيذي والاستراتيجية الشاملة الآن", type="primary", use_container_width=True):
                with st.spinner("🧠 جاري فحص كامل البيانات واستخلاص المؤشرات وبناء التقرير التنفيذي..."):
                    try:
                        num_summary = df.describe().to_string() if len(df.select_dtypes(include=['number']).columns) > 0 else "لا توجد أعمدة رقمية"
                        
                        corr_insights = []
                        valid_nums = [c for c in df.select_dtypes(include=['number']).columns if not any(k in c.lower() for k in ['id', 'code'])]
                        if len(valid_nums) >= 2:
                            corr_mat = df[valid_nums].corr()
                            for i in range(len(valid_nums)):
                                for j in range(i + 1, len(valid_nums)):
                                    val = corr_mat.iloc[i, j]
                                    if abs(val) >= 0.4:
                                        corr_insights.append(f"- ارتباط بين ({valid_nums[i]}) و ({valid_nums[j]}): {val:.2f}")

                        top_cat_insights = []
                        valid_cats = [c for c in df.select_dtypes(include=['object', 'category']).columns if df[c].nunique() <= 30]
                        for c in valid_cats[:4]:
                            top_val = df[c].value_counts().index[0]
                            top_pct = (df[c].value_counts().iloc[0] / len(df)) * 100
                            top_cat_insights.append(f"- في عمود ({c}): الفئة الأعلى هي '{top_val}' بنسبة {top_pct:.1f}%")

                        report_prompt = f"""
أنت كبير محللي البيانات ومستشار استراتيجي للأعمال (Chief Data Strategist).
لديك ملف بيانات باسم: "{active_file_name}"
- عدد الصفوف: {df.shape[0]:,}
- عدد الأعمدة: {df.shape[1]}
- نسبة جودة البيانات: {health_score}%

الملخص الإحصائي لكامل البيانات:
{num_summary}

أبرز علاقات الارتباط المكتشفة:
{chr(10).join(corr_insights) if corr_insights else "لا توجد ارتباطات خطية قوية"}

توزيع الفئات الرئيسية:
{chr(10).join(top_cat_insights) if top_cat_insights else "لا توجد أعمدة تصنيفية محدودة"}

المطلوب: اكتب تقريراً تنفيذياً احترافياً باللغة العربية موجهاً للإدارة العليا، منسقاً بعناوين واضحة ونقاط محددة، ويتضمن الأقسام الأربعة التالية بدقة:
1. **📌 الملخص التنفيذي (Executive Summary):** نظرة شاملة على حجم البيانات وأهم المؤشرات الرقمية.
2. **💎 أعمق الرؤى والأنماط المكتشفة (Key Insights):** تحليل الأرقام، الفئات المسيطرة، والعلاقات المؤثرة.
3. **⚠️ المخاطر ونقاط الضعف المرصودة (Risks & Anomalies):** أي تشتت عالٍ، تركز مفرط في فئة واحدة، أو فجوات في الأداء.
4. **🎯 خطة العمل والتوصيات الاستراتيجية (Actionable Strategy):** 4 خطوات عملية ومحددة قابلة للتنفيذ فوراً لتحسين الأداء وزيادة العائد.
"""
                        rep_response = call_gemini_api(report_prompt)
                        st.session_state.exec_report_md = rep_response.text
                    except Exception as e:
                        st.error(f"حدث خطأ أثناء توليد التقرير: {str(e)}")

            if st.session_state.get("exec_report_md"):
                st.markdown("---")
                st.markdown(st.session_state.exec_report_md)
                
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    st.download_button(
                        label="📄 تحميل التقرير التنفيذي (HTML جاهز للطباعة)",
                        data=build_html_report(
                            st.session_state.exec_report_md,
                            active_file_name,
                            df.shape[0],
                            df.shape[1],
                            health_score
                        ).encode("utf-8"),
                        file_name=f"Executive_Report_{active_file_name.split('.')[0]}.html",
                        mime="text/html",
                        use_container_width=True
                    )
                with dl_col2:
                    st.download_button(
                        label="📝 تحميل التقرير كنص منسق (Markdown)",
                        data=st.session_state.exec_report_md.encode("utf-8"),
                        file_name=f"Executive_Report_{active_file_name.split('.')[0]}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )

            st.divider()
            st.subheader("💬 اسأل المحلل الذكي (Interactive AI Chart Builder)")
            st.caption("اكتب أي سؤال مخصص وسيقوم الذكاء الاصطناعي بكتابة الكود، حساب النتيجة من كامل الملف، ورسم الشارت المناسب.")

            if "chat_history" not in st.session_state or st.session_state.get("chat_file") != active_file_name:
                st.session_state.chat_history = []
                st.session_state.chat_file = active_file_name

            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("fig") is not None:
                        st.plotly_chart(msg["fig"], use_container_width=True)

            user_question = st.chat_input("اسأل أي سؤال عن بياناتك (مثال: ارسم العلاقة بين المبيعات والعملاء)...")

            if user_question:
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.markdown(user_question)

                schema_info = "\n".join([f"- {col} ({dtype})" for col, dtype in df.dtypes.items()])
                sample_data = df.head(3).to_string()

                prompt = f"""
أنت محلل بيانات خبير ومبرمج Python محترف.
لديك DataFrame جاهز باسم `df` يحتوي على الأعمدة التالية وأنواعها:
{schema_info}

عينة من أول 3 صفوف:
{sample_data}

سؤال المستخدم: "{user_question}"

المطلوب منك:
1. كتابة كود Python نظيف ومباشر يستخدم مكتبة `plotly.express as px` (أو `pandas`) للإجابة على السؤال ورسم الشارت الأنسب.
2. يجب أن تخزن الرسم البياني النهائي في متغير اسمه `fig` بالضبط.
3. اجعل تصميم الرسم البياني احترافياً ومتناسقاً مع الثيم الداكن باستخدام:
   `fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")`
   واستخدم ألواناً متناسقة مثل `#d4af37` أو `#38bdf8`.
4. لا تستخدم `fig.show()` ولا `st.plotly_chart()` داخل الكود.
5. أرجع الرد بالصيغة التالية فقط:
   - أولاً: كود البايثون محصوراً بين ```python و ```
   - ثانياً: بعد الكود، اكتب فقرة قصيرة ومفيدة باللغة العربية بعنوان **💡 الاستنتاج التحليلي:** تشرح فيها ما يوضحه التحليل أو كيف نقرأ الرسمة.
"""

                with st.chat_message("assistant"):
                    with st.spinner("🧠 جاري تحليل البيانات وبناء الرسم البياني..."):
                        try:
                            response = call_gemini_api(prompt)
                            ai_reply = response.text

                            code_match = re.search(r"```python\n(.*?)```", ai_reply, re.DOTALL)
                            explanation = re.sub(r"```python\n.*?```", "", ai_reply, flags=re.DOTALL).strip()

                            fig_result = None
                            if code_match:
                                generated_code = code_match.group(1)
                                local_vars = {"df": df.copy(), "pd": pd, "np": np, "px": px}
                                exec(generated_code, {}, local_vars)
                                fig_result = local_vars.get("fig")

                            if explanation:
                                st.markdown(explanation)
                            if fig_result is not None:
                                st.plotly_chart(fig_result, use_container_width=True)

                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": explanation,
                                "fig": fig_result
                            })

                        except Exception as e:
                            st.error(f"حدث خطأ أثناء معالجة الطلب: {str(e)}")
else:
    st.markdown("""
    <div style='text-align: center; padding: 60px; background: rgba(255,255,255,0.02); border-radius: 16px; border: 1px dashed rgba(212, 175, 55, 0.45); margin-top: 20px;'>
        <h3 style='text-align: center; color: #d4af37;'>👋 أهلاً بك في مساحة العمل الذكية</h3>
        <p style='text-align: center; color: #9ca3af;'>ابدأ برفع ملف البيانات (CSV أو Excel) أو اضغط على زر <b>🚀 جرب ببيانات مبيعات جاهزة</b> من القائمة الجانبية لتفعيل التنظيف، المحاكاة، والتقرير التنفيذي الذكي.</p>
    </div>
    """, unsafe_allow_html=True)
