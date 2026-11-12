import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
        gap: 12px;
        background-color: rgba(255, 255, 255, 0.02);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        direction: rtl;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        color: #9ca3af;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.2), rgba(212, 175, 55, 0.08)) !important;
        color: #d4af37 !important;
        border: 1px solid rgba(212, 175, 55, 0.45);
    }

    /* تنسيق الجداول والقوائم المنسدلة */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(212, 175, 55, 0.2);
        border-radius: 12px;
        overflow: hidden;
    }
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(212, 175, 55, 0.2);
        border-radius: 12px;
        direction: rtl;
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
    for col in old_cols:
        clean_col = re.sub(r'[^\w\s]', '', str(col)).strip().lower()
        clean_col = re.sub(r'\s+', '_', clean_col)
        new_cols.append(clean_col if clean_col else "unnamed_col")
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
                date_converted = pd.to_datetime(df_fixed[col], errors='coerce')
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
    <p style='text-align: center; color: #9ca3af; font-size: 1.05rem;'>ارفع بياناتك، نظّفها بضغطة زر، واسأل الذكاء الاصطناعي لرسم العلاقات فوراً</p>
</div>
""", unsafe_allow_html=True)

if uploaded_file is not None:
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        if uploaded_file.name.endswith('.csv'):
            raw_df = pd.read_csv(uploaded_file)
        else:
            raw_df = pd.read_excel(uploaded_file)
        st.session_state.current_file = uploaded_file.name
        st.session_state.raw_df = raw_df.copy()
        st.session_state.df = raw_df.copy()
        st.session_state.cleaning_log = ["تم تحميل الملف الأصلي بنجاح"]
        st.session_state.last_report = None

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
    
    tab1, tab2, tab3 = st.tabs(["🧹 معاينة وتنظيف البيانات", "📈 التحليل الاستكشافي (EDA)", "🤖 اسأل الذكاء الاصطناعي"])
    
    with tab1:
        if st.session_state.get("last_report"):
            st.success(f"**📋 تقرير المعالجة الأخيرة:** {st.session_state.last_report}")

        st.subheader("🛠️ أدوات التنظيف الاحترافي")
        
        top_btn1, top_btn2 = st.columns([2, 1])
        with top_btn1:
            if st.button("⚡ التنظيف الذكي الشامل (Auto-Clean All)", type="primary", use_container_width=True):
                temp_df = st.session_state.df.copy()
                report_details = []
                
                temp_df, cols_changed = standardize_col_names(temp_df)
                if cols_changed > 0:
                    report_details.append(f"توحيد {cols_changed} اسم عمود")
                
                temp_df, hidden_nulls = normalize_hidden_nulls(temp_df)
                if hidden_nulls > 0:
                    report_details.append(f"اكتشاف {hidden_nulls} قيمة فارغة مخفية")
                
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
                    report_details.append(f"تحويل أنواع الأعمدة: {', '.join(fixed_types)}")
                
                st.session_state.df = temp_df
                summary_msg = " | ".join(report_details) if report_details else "البيانات نظيفة بالفعل ولا توجد أخطاء شكلية!"
                log_action(f"تنظيف شامل: {summary_msg}")
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
                num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
                if num_cols:
                    target_num_col = st.selectbox("اختر العمود الرقمي:", num_cols, key="outlier_col")
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
                file_name=f"cleaned_{uploaded_file.name.split('.')[0]}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        st.dataframe(st.session_state.df, use_container_width=True, height=350)
        
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

        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            if num_cols:
                selected_num = st.selectbox("اختر عمود رقمي لعرض توزيعه:", num_cols)
                fig_hist = px.histogram(
                    df, x=selected_num, marginal="box",
                    title=f"توزيع قيم: {selected_num}",
                    template="plotly_dark",
                    color_discrete_sequence=["#d4af37"]
                )
                fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_hist, use_container_width=True)

        with viz_col2:
            if cat_cols:
                selected_cat = st.selectbox("اختر عمود تصنيفي لعرض أعلى الفئات:", cat_cols)
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

        if date_cols and num_cols:
            st.subheader("📅 تحليل الاتجاه الزمني (Time Series)")
            ts_col1, ts_col2 = st.columns(2)
            selected_date = ts_col1.selectbox("عمود التاريخ:", date_cols)
            selected_ts_num = ts_col2.selectbox("المتغير الرقمي:", num_cols, key="ts_num")
            
            ts_df = df.groupby(selected_date)[selected_ts_num].sum().reset_index()
            fig_line = px.line(
                ts_df, x=selected_date, y=selected_ts_num, markers=True,
                title=f"تطور {selected_ts_num} عبر {selected_date}",
                template="plotly_dark",
                color_discrete_sequence=["#10b981"]
            )
            fig_line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_line, use_container_width=True)

        if len(num_cols) >= 2:
            st.subheader("🔥 مصفوفة الارتباط (Correlation Heatmap)")
            corr_matrix = df[num_cols].corr()
            fig_corr = px.imshow(
                corr_matrix, text_auto=".2f", aspect="auto",
                color_continuous_scale="RdBu_r",
                template="plotly_dark",
                title="قوة الارتباط بين المتغيرات الرقمية"
            )
            fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_corr, use_container_width=True)
        
    with tab3:
        st.subheader("🤖 اسأل الذكاء الاصطناعي عن بياناتك")
        st.caption("اكتب سؤالك بالعربي أو الإنجليزي، وسيقوم الذكاء الاصطناعي بتحليل البيانات ورسم الشارت المناسب تلقائياً.")

        if not api_key:
            st.error("⚠️ لم يتم العثور على مفتاح `GEMINI_API_KEY` في ملف `.streamlit/secrets.toml`.")
        else:
            st.markdown("**💡 جرب أسئلة مثل:** *ما هي العلاقة بين المتغيرات؟* • *اعرض توزيع القيم كـ Bar Chart* • *ما هو الاتجاه العام بمرور الوقت؟*")
            
            if "chat_history" not in st.session_state or st.session_state.get("chat_file") != uploaded_file.name:
                st.session_state.chat_history = []
                st.session_state.chat_file = uploaded_file.name

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
                            client = genai.Client(api_key=api_key)
                            models_to_try = ["gemini-3.8-flash", "gemini-3.1-pro-preview"]
                            
                            # اكتشاف أي موديلات متاحة في حسابك تلقائياً كخطة بديلة
                            try:
                                for m in client.models.list():
                                    m_id = m.name.replace("models/", "")
                                    if "gemini" in m_id and m_id not in models_to_try:
                                        models_to_try.append(m_id)
                            except Exception:
                                pass

                            response = None
                            last_err = None

                            for m_name in models_to_try:
                                try:
                                    response = client.models.generate_content(
                                        model=m_name,
                                        contents=prompt
                                    )
                                    break
                                except Exception as e:
                                    last_err = e
                                    continue

                            if response is None:
                                raise last_err

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
        <p style='text-align: center; color: #9ca3af;'>ابدأ برفع ملف البيانات (CSV أو Excel) من القائمة الجانبية على اليسار لتفعيل التنظيف والتحليل الذكي.</p>
    </div>
    """, unsafe_allow_html=True)