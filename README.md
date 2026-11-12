<div align="center">

# ⚡ AI Data Analyst Pro
### Intelligent Data Cleaning, Automated EDA & Text-to-Chart Analytics Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-AI_Powered-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive_Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)

**Engineered & Designed by [Mohamed Gomaa](https://github.com/MohamedGomaa11111)**  
*Data Analyst & Software Engineer*

</div>

---

## 📌 Overview

**AI Data Analyst Pro** is an end-to-end, interactive web application built to bridge the gap between raw, messy datasets and executive-ready visual insights. Combining deterministic data engineering pipelines with generative AI (**Google Gemini**), the platform empowers analysts and business users to audit, clean, explore, and query CSV/Excel datasets in natural language (Arabic & English) with zero boilerplate code.

---

## 🚀 Core Capabilities

### 1. 🧹 Smart Data Cleaning & Governance Pipeline
* **One-Click Auto-Clean:** Executes a full preprocessing workflow including `snake_case` header standardization, hidden null normalization (`N/A`, `-`, `?`, `null`), exact duplicate removal, and whitespace/case trimming.
* **Domain-Aware Type Inference:** Automatically casts string-encoded numbers and dates to their proper types while protecting sensitive identifiers (`ID`, `SKU`, `Phone`, `Postal Code`, leading-zero strings).
* **Statistical Outlier Management (IQR):** Detects anomalies using the Interquartile Range ($1.5 \times \text{IQR}$) and supports non-destructive **Winsorization (Capping)** or row removal.
* **Live Data Quality Health Score & Audit Log:** Dynamically recalculates dataset completeness/uniqueness (`0%–100%`) and logs every transformation step for full reproducibility and rollback support.

### 2. 📈 Automated Exploratory Data Analysis (`Auto-EDA`)
* **Schema & Cardinality Profiling:** Instant breakdown of column data types, unique counts, and missing value distributions.
* **Descriptive Statistics:** Separate statistical summaries for numerical metrics (`mean`, `std`, quartiles) and categorical dimensions (`top`, `freq`, `unique`).
* **Interactive Visualizations:** Auto-generated Plotly distributions (Histograms with marginal Box Plots), Top-N categorical bar charts, Time-Series trend lines, and Pearson Correlation Heatmaps.

### 3. 🤖 Conversational AI Analyst (`Text-to-Chart`)
* **Natural Language Querying:** Ask questions in **Arabic** or **English** (e.g., *"Compare sales across customers using a descending bar chart"*).
* **Dynamic Code Generation & Execution:** Uses Gemini AI to generate and execute tailored `Plotly Express` code directly against the active DataFrame, returning both an interactive dark-themed chart and an executive analytical takeaway.
* **Resilient Multi-Model Fallback:** Automatically switches across available Gemini models (`gemini-3.8-flash` / `gemini-3.1-pro-preview`) to ensure uninterrupted availability during traffic spikes.

### 4. 🎨 Executive UI/UX
* Custom **Dark Navy & Gold** glassmorphism theme (`Cairo` typography) with full **Arabic RTL** layout support alongside **LTR** data grids and code blocks.

---

## 🏗️ Project Structure

```text
ai-data-analyst/
├── .streamlit/
│   └── secrets.toml        # Local API credentials (excluded from Git)
├── app.py                  # Main Streamlit application & UI logic
├── requirements.txt        # Python dependencies
├── .gitignore              # Git exclusion rules
└── README.md               # Project documentation