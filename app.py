import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import warnings
warnings.filterwarnings("ignore")

# ── sklearn imports ──────────────────────────────────────────────
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error,
    roc_auc_score, roc_curve
)

st.set_page_config(
    page_title="ML Pipeline Studio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

/* ── Root palette ── */
:root {
    --bg:         #0d0f1a;
    --surface:    #131626;
    --panel:      #1a1e35;
    --border:     #2a2f50;
    --accent1:    #6c63ff;
    --accent2:    #00d4aa;
    --accent3:    #ff6b6b;
    --accent4:    #ffd166;
    --text:       #e8eaf6;
    --muted:      #8890b5;
    --mono:       'Space Mono', monospace;
    --sans:       'DM Sans', sans-serif;
    --radius:     14px;
    --glow1:      0 0 24px rgba(108,99,255,.35);
    --glow2:      0 0 24px rgba(0,212,170,.35);
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: var(--sans);
    background: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg) !important; }
.block-container { padding: 1.5rem 2rem 3rem !important; max-width: 1400px; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--accent1); border-radius:3px; }

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #1a1e35 0%, #0d1535 50%, #151028 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content:'';
    position:absolute; inset:0;
    background: radial-gradient(ellipse at 20% 50%, rgba(108,99,255,.12) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 50%, rgba(0,212,170,.08) 0%, transparent 60%);
    pointer-events:none;
}
.hero-title {
    font-family: var(--mono);
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #6c63ff, #00d4aa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -1px;
}
.hero-sub { color: var(--muted); margin: .4rem 0 0; font-size: .95rem; }

/* ── Step pipeline bar ── */
.pipeline-bar {
    display: flex;
    align-items: center;
    gap: 0;
    margin-bottom: 2rem;
    overflow-x: auto;
    padding: 1rem 0;
}
.step-pill {
    display: flex;
    align-items: center;
    gap: .5rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 50px;
    padding: .45rem 1rem .45rem .65rem;
    white-space: nowrap;
    font-size: .78rem;
    font-family: var(--mono);
    color: var(--muted);
    transition: all .2s;
    position: relative;
    cursor: default;
}
.step-pill.done {
    border-color: var(--accent2);
    color: var(--accent2);
    background: rgba(0,212,170,.07);
    box-shadow: var(--glow2);
}
.step-pill.active {
    border-color: var(--accent1);
    color: #fff;
    background: rgba(108,99,255,.18);
    box-shadow: var(--glow1);
}
.step-icon { font-size: 1rem; }
.step-connector {
    width: 32px; height: 2px;
    background: var(--border);
    flex-shrink: 0;
}
.step-connector.done { background: var(--accent2); }

/* ── Section card ── */
.section-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.5rem;
    position: relative;
}
.section-header {
    display: flex;
    align-items: center;
    gap: .75rem;
    margin-bottom: 1.2rem;
    padding-bottom: .8rem;
    border-bottom: 1px solid var(--border);
}
.section-badge {
    background: linear-gradient(135deg, var(--accent1), #9c59d1);
    color: #fff;
    font-family: var(--mono);
    font-size: .7rem;
    font-weight: 700;
    padding: .2rem .6rem;
    border-radius: 50px;
    letter-spacing: 1px;
}
.section-title {
    font-family: var(--mono);
    font-size: 1.1rem;
    font-weight: 700;
    color: #fff;
    margin: 0;
}

/* ── Metric cards ── */
.metric-row { display:flex; gap:1rem; flex-wrap:wrap; margin:1rem 0; }
.metric-card {
    flex:1; min-width:140px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-val {
    font-family: var(--mono);
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent2);
}
.metric-lbl { font-size: .72rem; color: var(--muted); margin-top:.2rem; text-transform: uppercase; letter-spacing:.5px; }

/* ── Stat pill ── */
.stat-pill {
    display:inline-flex; align-items:center; gap:.4rem;
    background: var(--panel);
    border:1px solid var(--border);
    border-radius:8px;
    padding:.35rem .75rem;
    font-size:.82rem;
    color: var(--text);
    margin:.2rem;
}
.stat-dot { width:8px;height:8px;border-radius:50%; }

/* ── Buttons ── */
.stButton>button {
    background: linear-gradient(135deg, var(--accent1), #9c59d1) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: var(--sans) !important;
    font-weight: 600 !important;
    padding: .55rem 1.4rem !important;
    transition: all .2s !important;
    box-shadow: 0 4px 14px rgba(108,99,255,.3) !important;
}
.stButton>button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(108,99,255,.45) !important;
}

/* ── Select / Input ── */
.stSelectbox>div>div, .stMultiSelect>div>div {
    background: var(--panel) !important;
    border-color: var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
.stNumberInput>div>div>input, .stTextInput>div>div>input {
    background: var(--panel) !important;
    border-color: var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
.stSlider>div>div { color: var(--text) !important; }
.stRadio>div { gap:.5rem !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 12px !important;
    padding: .25rem !important;
    gap:.3rem !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 10px !important;
    color: var(--muted) !important;
    font-weight: 500 !important;
    font-family: var(--sans) !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent1) !important;
    color: #fff !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--panel) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-weight: 600 !important;
}

/* ── Dataframe ── */
.stDataFrame { border-radius:10px; overflow:hidden; }

/* ── Alert boxes ── */
.stSuccess, .stInfo, .stWarning, .stError {
    border-radius: 10px !important;
}

/* ── Upload widget ── */
.stFileUploader>div {
    background: var(--panel) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
}
.stFileUploader>div:hover {
    border-color: var(--accent1) !important;
}

/* ── Checkbox ── */
.stCheckbox>label { color: var(--text) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Tag-style text ── */
.tag {
    display: inline-block;
    background: rgba(108,99,255,.15);
    border: 1px solid rgba(108,99,255,.4);
    border-radius: 6px;
    padding: .15rem .5rem;
    font-size: .78rem;
    font-family: var(--mono);
    color: var(--accent1);
    margin: .15rem;
}
.tag-green {
    background: rgba(0,212,170,.1);
    border-color: rgba(0,212,170,.4);
    color: var(--accent2);
}
.tag-red {
    background: rgba(255,107,107,.1);
    border-color: rgba(255,107,107,.4);
    color: var(--accent3);
}
.tag-yellow {
    background: rgba(255,209,102,.1);
    border-color: rgba(255,209,102,.4);
    color: var(--accent4);
}

/* ── Plotly chart bg ── */
.js-plotly-plot .plotly { border-radius: 12px; }

/* ── Progress indicator ── */
.stProgress>div>div { background: var(--accent1) !important; border-radius:8px !important; }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PLOTLY THEME HELPER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(26,30,53,0.6)",
    font=dict(color="#e8eaf6", family="DM Sans"),
    margin=dict(l=40, r=20, t=40, b=40),
    colorway=["#6c63ff","#00d4aa","#ff6b6b","#ffd166","#a29bfe","#74b9ff","#fd79a8"],
    xaxis=dict(gridcolor="#2a2f50", zerolinecolor="#2a2f50"),
    yaxis=dict(gridcolor="#2a2f50", zerolinecolor="#2a2f50"),
)

def apply_theme(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SESSION STATE INIT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEFAULTS = dict(
    step=0, problem_type=None,
    raw_df=None, df=None, target=None, feature_cols=None,
    encoders={}, scaler=None,
    outlier_indices=[], remove_outliers=False,
    selected_features=None,
    X_train=None, X_test=None, y_train=None, y_test=None,
    model=None, model_name=None, kernel="rbf",
    k_folds=5,
    cv_scores=None, trained=False,
    y_pred=None, y_pred_train=None,
    tuning_done=False, best_params=None,
)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

S = st.session_state   # shorthand

STEPS = [
    ("🧩","Problem"),
    ("📂","Data"),
    ("🔍","EDA"),
    ("🔧","Engineering"),
    ("🎯","Features"),
    ("✂️","Split"),
    ("🤖","Model"),
    ("🏋️","Training"),
    ("📊","Metrics"),
    ("⚙️","Tuning"),
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HERO + PIPELINE BAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<div class="hero">
  <p class="hero-title">🧬 ML Pipeline </p>
  <p class="hero-sub">End-to-end machine learning — from raw data to tuned models — in one visual pipeline.</p>
</div>
""", unsafe_allow_html=True)

# Build pipeline bar HTML
bar_html = '<div class="pipeline-bar">'
for i, (icon, label) in enumerate(STEPS):
    cls = "done" if i < S.step else ("active" if i == S.step else "")
    bar_html += f'<div class="step-pill {cls}"><span class="step-icon">{icon}</span>{label}</div>'
    if i < len(STEPS)-1:
        conn_cls = "done" if i < S.step else ""
        bar_html += f'<div class="step-connector {conn_cls}"></div>'
bar_html += '</div>'
st.markdown(bar_html, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HELPER: section card wrapper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def section(badge, title):
    st.markdown(f"""
    <div class="section-header">
      <span class="section-badge">{badge}</span>
      <p class="section-title">{title}</p>
    </div>""", unsafe_allow_html=True)

def metric_cards(items):
    """items: list of (label, value, color) tuples"""
    cols = st.columns(len(items))
    for col, (lbl, val, color) in zip(cols, items):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-val" style="color:{color}">{val}</div>
              <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

def next_step_fn():
    S.step += 1

def prev_step_fn():
    S.step -= 1

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 0 — PROBLEM TYPE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if S.step == 0:
    with st.container():
        section("STEP 01", "🧩 Define Your Problem")
        col1, col2 = st.columns([1.6, 1])
        with col1:
            st.markdown("#### What type of ML problem are you solving?")
            st.markdown('<span class="tag">Supervised</span> <span class="tag">Binary / Multi-class</span> <span class="tag-yellow">Regression</span>', unsafe_allow_html=True)
            st.markdown("")
            choice = st.radio(
                "Select problem type:",
                ["Classification", "Regression"],
                horizontal=True,
                index=0 if S.problem_type != "Regression" else 1,
            )
            if st.button("✅ Confirm & Proceed", key="btn_problem"):
                S.problem_type = choice
                next_step_fn()
                st.rerun()
        with col2:
            icon = "🏷️" if S.problem_type == "Classification" else "📈"
            color = "#6c63ff" if S.problem_type != "Regression" else "#00d4aa"
            st.markdown(f"""
            <div class="metric-card" style="margin-top:1rem; border-color:{color}; background:rgba(0,0,0,.2)">
              <div style="font-size:2.5rem; margin-bottom:.5rem">{icon}</div>
              <div class="metric-val" style="color:{color}; font-size:1.2rem">{S.problem_type or '—'}</div>
              <div class="metric-lbl">Selected Type</div>
            </div>""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 1 — DATA INPUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif S.step == 1:
    section("STEP 02", "📂 Data Input & PCA Projection")

    upload = st.file_uploader("Upload your CSV dataset", type=["csv"], key="uploader")

    if upload:
        raw = pd.read_csv(upload)
        S.raw_df = raw.copy()
        S.df = raw.copy()

    if S.df is not None:
        df = S.df
        st.markdown("#### 🎯 Select Target Column")
        target = st.selectbox("Target feature:", df.columns.tolist(), key="target_sel", index=len(df.columns)-1)
        S.target = target

        feat_cols = [c for c in df.columns if c != target]
        sel_feats = st.multiselect("Features for PCA:", feat_cols, default=feat_cols, key="feat_sel_pca")

        if sel_feats:
            st.markdown("#### 🔮 PCA Projection")
            tab_pca2d, tab_pca3d = st.tabs(["PCA 2D", "PCA 3D"])

            tmp = df[sel_feats + [target]].copy()
            for col in tmp.select_dtypes(include="object").columns:
                le = LabelEncoder()
                tmp[col] = le.fit_transform(tmp[col].astype(str))
            tmp = tmp.dropna()

            scaler_pca = StandardScaler()
            X_pca = scaler_pca.fit_transform(tmp[sel_feats])
            pca_obj = PCA(n_components=min(3, len(sel_feats)))
            comps = pca_obj.fit_transform(X_pca)
            var_exp = pca_obj.explained_variance_ratio_

            with tab_pca2d:
                pca_df = pd.DataFrame({"PC1": comps[:,0], "PC2": comps[:,1], target: tmp[target].values})
                fig = px.scatter(pca_df, x="PC1", y="PC2", color=target,
                                 title=f"PCA 2D — {var_exp[0]*100:.1f}% + {var_exp[1]*100:.1f}% variance",
                                 color_continuous_scale="Viridis", template="plotly_dark")
                apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

            with tab_pca3d:
                if comps.shape[1] >= 3:
                    pca_df3 = pd.DataFrame({"PC1": comps[:,0], "PC2": comps[:,1], "PC3": comps[:,2], target: tmp[target].values})
                    fig3 = px.scatter_3d(pca_df3, x="PC1", y="PC2", z="PC3", color=target, title="PCA 3D",
                                         color_continuous_scale="Plasma", template="plotly_dark")
                    apply_theme(fig3)
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("Need ≥ 3 features for 3D PCA.")

        c1, c2 = st.columns(2)
        with c1: st.button("← Back", on_click=prev_step_fn)
        with c2: 
            if st.button("▶ Proceed to EDA", key="btn_data"):
                S.feature_cols = [c for c in df.columns if c != S.target]
                for col in S.df.select_dtypes(include="object").columns:
                    le = LabelEncoder()
                    S.df[col] = le.fit_transform(S.df[col].astype(str))
                    S.encoders[col] = le
                next_step_fn()
                st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 2 — EDA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif S.step == 2:
    section("STEP 03", "🔍 Exploratory Data Analysis")
    df = S.df

    tab_stat, tab_corr, tab_miss = st.tabs(["📋 Statistics", "🔗 Correlation", "❓ Missing Values"])

    with tab_stat:
        desc = df.describe(include="all").T
        st.dataframe(desc, use_container_width=True)        
        metric_cards([
            ("Rows", f"{df.shape[0]:,}", "#6c63ff"),
            ("Features", f"{df.shape[1]}", "#00d4aa"),
            ("Numeric", f"{df.select_dtypes(include='number').shape[1]}", "#ffd166"),
            ("Missing", f"{df.isnull().sum().sum()}", "#ff6b6b"),
        ])

    with tab_corr:
        corr = df[df.select_dtypes(include="number").columns].corr()
        fig = px.imshow(corr, color_continuous_scale="RdBu_r", aspect="auto",
                        title="Correlation Heatmap", template="plotly_dark",
                        zmin=-1, zmax=1, text_auto=".2f")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with tab_miss:
        miss = df.isnull().sum().reset_index()
        miss.columns = ["Feature", "Missing"]
        miss = miss[miss["Missing"] > 0]
        if miss.empty:
            st.success("✅ No missing values detected!")
        else:
            fig = px.bar(miss, x="Feature", y="Missing", color="Missing",
                         color_continuous_scale="Reds", title="Missing Values", template="plotly_dark")
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

   # [1, 3, 1] creates a small left col, a large empty middle, and a small right col
    c1, mid, c2 = st.columns([1, 3, 1]) 

    with c1: 
        st.button("← Back", on_click=prev_step_fn, use_container_width=True)

    with c2:
    # Adding use_container_width=True prevents the overlap
        if st.button("Proceed to Engineering →", key="btn_eda_nav", use_container_width=True):
            next_step_fn()
            st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 3 — DATA ENGINEERING & CLEANING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif S.step == 3:
    section("STEP 04", "🔧 Data Engineering & Cleaning")
    df = S.df

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🩹 Missing Values")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        impute_method = st.selectbox("Imputation method:", ["Mean","Median","Mode","None"], key="impute")
        impute_cols = st.multiselect("Columns:", num_cols, default=num_cols, key="impute_cols")

        if st.button("Apply Imputation", key="btn_impute"):
            for col in impute_cols:
                if impute_method == "Mean": S.df[col].fillna(S.df[col].mean(), inplace=True)
                elif impute_method == "Median": S.df[col].fillna(S.df[col].median(), inplace=True)
                elif impute_method == "Mode": S.df[col].fillna(S.df[col].mode()[0], inplace=True)
            st.success("✅ Imputation applied.")
            df = S.df

    with col2:
        st.markdown("#### 🔎 Outliers")
        outlier_method = st.selectbox("Method:", ["IQR","Isolation Forest","DBSCAN","OPTICS"], key="out_method")
        outlier_cols = st.multiselect("Features:", num_cols, default=num_cols[:4], key="out_cols")

        if st.button("🔍 Detect Outliers", key="btn_outlier"):
            if outlier_cols:
                X_out = df[outlier_cols].dropna()
                if outlier_method == "IQR":
                    mask = pd.Series([False]*len(X_out), index=X_out.index)
                    for col in outlier_cols:
                        Q1, Q3 = X_out[col].quantile(.25), X_out[col].quantile(.75)
                        IQR = Q3 - Q1
                        mask |= (X_out[col] < Q1-1.5*IQR) | (X_out[col] > Q3+1.5*IQR)
                    S.outlier_indices = X_out[mask].index.tolist()
                elif outlier_method == "Isolation Forest":
                    clf = IsolationForest(contamination=.05, random_state=42)
                    S.outlier_indices = X_out.index[clf.fit_predict(X_out) == -1].tolist()
                elif outlier_method in ["DBSCAN", "OPTICS"]:
                    sc = StandardScaler()
                    Xs = sc.fit_transform(X_out)
                    model_cls = DBSCAN(eps=.5, min_samples=5) if outlier_method=="DBSCAN" else OPTICS(min_samples=5)
                    S.outlier_indices = X_out.index[model_cls.fit_predict(Xs) == -1].tolist()
                st.warning(f"⚠️ {len(S.outlier_indices)} outliers detected.")

    if S.outlier_indices and len(outlier_cols) >= 2:
        st.markdown("#### 👁️ Visualisation")
        normal_idx = [i for i in df.index if i not in S.outlier_indices]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.loc[normal_idx, outlier_cols[0]], y=df.loc[normal_idx, outlier_cols[1]], mode="markers", name="Normal", marker=dict(color="#6c63ff", size=4, opacity=.6)))
        fig.add_trace(go.Scatter(x=df.loc[S.outlier_indices, outlier_cols[0]], y=df.loc[S.outlier_indices, outlier_cols[1]], mode="markers", name="Outlier", marker=dict(color="#ff6b6b", size=8, symbol="x")))
        fig.update_layout(title="Outlier Map", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        if st.checkbox("🗑️ Remove outliers"):
            if st.button("Confirm Removal"):
                S.df = S.df.drop(index=S.outlier_indices).reset_index(drop=True)
                S.outlier_indices = []
                st.rerun()

    c1, c2 = st.columns(2)
    with c1: st.button("← Back", on_click=prev_step_fn)
    with c2: st.button("▶ Proceed to Feature Selection", on_click=next_step_fn)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 4 — FEATURE SELECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif S.step == 4:
    section("STEP 05", "🎯 Feature Selection")
    df = S.df
    feat_cols = [c for c in df.columns if c != S.target]

    tab_var, tab_cor, tab_mi = st.tabs(["Variance Threshold", "Correlation Filter", "Information Gain"])

    with tab_var:
        thresh = st.slider("Variance threshold:", 0.0, 1.0, 0.01, 0.01, key="v_thresh")
        if st.button("Apply Variance Filter"):
            num_df = df[feat_cols].select_dtypes(include="number")
            sel = VarianceThreshold(threshold=thresh).fit(num_df)
            variances = pd.Series(sel.variances_, index=num_df.columns).sort_values()
            fig = px.bar(variances.reset_index(), x="index", y=0, title="Feature Variances", color_continuous_scale="Blues", template="plotly_dark")
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    with tab_cor:
        cor_thresh = st.slider("Min correlation:", 0.0, 1.0, 0.05, 0.01)
        if st.button("Apply Corr Filter"):
            num_df = df[feat_cols + [S.target]].select_dtypes(include="number")
            corrs = num_df.corr()[S.target].drop(S.target).abs().sort_values()
            fig = px.bar(corrs.reset_index(), x="index", y=S.target, title="Correlation with Target", template="plotly_dark")
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    with tab_mi:
        if st.button("Compute Information Gain"):
            num_df = df[feat_cols + [S.target]].select_dtypes(include="number").dropna()
            func = mutual_info_classif if S.problem_type == "Classification" else mutual_info_regression
            mi = pd.Series(func(num_df[feat_cols], num_df[S.target]), index=feat_cols).sort_values(ascending=False)
            fig = px.bar(mi.reset_index(), x="index", y=0, title="Information Gain Score", color_discrete_sequence=["#00d4aa"], template="plotly_dark")
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    S.selected_features = st.multiselect("Final Features:", feat_cols, default=feat_cols[:5])
    c1, c2 = st.columns(2)
    with c1: st.button("← Back", on_click=prev_step_fn)
    with c2: st.button("▶ Proceed to Split", on_click=next_step_fn)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 5 — DATA SPLIT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif S.step == 5:
    section("STEP 06", "✂️ Train / Test Split")
    test_size = st.slider("Test set size (%):", 10, 40, 20, 5) / 100
    df = S.df.dropna(subset=S.selected_features + [S.target])
    n_train = len(df) - int(len(df) * test_size)
    fig = go.Figure(go.Pie(labels=["Train", "Test"], values=[n_train, len(df)-n_train], hole=.55, marker_colors=["#6c63ff","#00d4aa"]))
    st.plotly_chart(fig, use_container_width=True)

    if st.button("▶ Split & Proceed"):
        X, y = df[S.selected_features], df[S.target]
        S.X_train, S.X_test, S.y_train, S.y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        sc = StandardScaler()
        S.X_train = pd.DataFrame(sc.fit_transform(S.X_train), columns=S.selected_features)
        S.X_test = pd.DataFrame(sc.transform(S.X_test), columns=S.selected_features)
        next_step_fn()
        st.rerun()
    st.button("← Back", on_click=prev_step_fn)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 6 — MODEL SELECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif S.step == 6:
    section("STEP 07", "🤖 Model Selection")
    models_avail = ["Logistic Regression", "SVM (Kernel)", "Random Forest"] if S.problem_type == "Classification" else ["Linear Regression", "SVM (Kernel)", "Random Forest"]
    S.model_name = st.radio("Choose your model:", models_avail)
    if "SVM" in S.model_name:
        S.kernel = st.select_slider("Kernel:", ["linear","poly","rbf","sigmoid"], value="rbf")

    c1, c2 = st.columns(2)
    with c1: st.button("← Back", on_click=prev_step_fn)
    with c2: st.button("▶ Proceed to Training", on_click=next_step_fn)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 7 — TRAINING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif S.step == 7:
    section("STEP 08", "🏋️ Model Training")
    if st.button("🚀 Train Model"):
        def get_mdl():
            if S.model_name == "Logistic Regression": return LogisticRegression(max_iter=1000)
            if S.model_name == "Linear Regression": return LinearRegression()
            if S.model_name == "SVM (Kernel)": return SVC(kernel=S.kernel, probability=True) if S.problem_type=="Classification" else SVR(kernel=S.kernel)
            if S.model_name == "Random Forest": return RandomForestClassifier() if S.problem_type=="Classification" else RandomForestRegressor()
        
        mdl = get_mdl()
        mdl.fit(S.X_train, S.y_train)
        S.model, S.y_pred, S.y_pred_train, S.trained = mdl, mdl.predict(S.X_test), mdl.predict(S.X_train), True
        st.success("✅ Training Complete!")

    c1, c2 = st.columns(2)
    with c1: st.button("← Back", on_click=prev_step_fn)
    with c2: st.button("▶ Proceed to Metrics", on_click=next_step_fn, disabled=not S.trained)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 8 — PERFORMANCE METRICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif S.step == 8:
    section("STEP 09", "📊 Performance Metrics")
    if S.problem_type == "Classification":
        acc = accuracy_score(S.y_test, S.y_pred)
        metric_cards([("Accuracy", f"{acc:.4f}", "#00d4aa")])
        tab_cm, tab_rep = st.tabs(["Confusion Matrix", "Report"])
        with tab_cm:
            st.plotly_chart(px.imshow(confusion_matrix(S.y_test, S.y_pred), text_auto=True, template="plotly_dark"))
        with tab_rep:
            st.text(classification_report(S.y_test, S.y_pred))
    else:
        r2 = r2_score(S.y_test, S.y_pred)
        metric_cards([("R² Score", f"{r2:.4f}", "#00d4aa")])
        st.plotly_chart(px.scatter(x=S.y_test, y=S.y_pred, labels={'x':'Actual','y':'Pred'}, title="Actual vs Pred", template="plotly_dark"))

    c1, c2 = st.columns(2)
    with c1: st.button("← Back", on_click=prev_step_fn)
    with c2: st.button("▶ Proceed to Tuning", on_click=next_step_fn)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 9 — TUNING & COMPLETION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 9 — TUNING & PREDICTION (Updated for your Presentation!)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 9 — TUNING & LIVE PREDICTION (Optimized for Sensitivity)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 9 — TUNING & LIVE PREDICTION (Final Version)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 9 — TUNING & LIVE PREDICTION (Final Presentation Version)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif S.step == 9:
    section("STEP 10", "⚙️ Tuning & Live Prediction")
    
    # ── THEMATIC MEDICINE RAIN (One-Time Celebration) ──
    if S.tuning_done:
        st.markdown("""
        <div class="med-container">
            <div class="pill">💊</div><div class="pill">🧪</div><div class="pill">🩺</div>
            <div class="pill">💉</div><div class="pill">💊</div><div class="pill">🩹</div>
            <div class="pill">💊</div><div class="pill">🩺</div><div class="pill">🧪</div>
            <div class="pill">🩹</div><div class="pill">💊</div><div class="pill">💉</div>
        </div>
        <style>
        .med-container {
            position: fixed;
            top: -50px; left: 0; width: 100%; height: 100%;
            pointer-events: none; z-index: 9999;
        }
        .pill {
            position: absolute;
            font-size: 2rem;
            animation: fall linear 1 forwards; 
            filter: drop-shadow(0 0 8px rgba(0,212,170,0.6));
            opacity: 0;
        }
        @keyframes fall {
            0% { transform: translateY(-10vh) rotate(0deg); opacity: 0; }
            10% { opacity: 1; }
            90% { opacity: 1; }
            100% { transform: translateY(110vh) rotate(720deg); opacity: 0; }
        }
        .pill:nth-child(1) { left: 5%; animation-duration: 2s; }
        .pill:nth-child(2) { left: 15%; animation-duration: 5s; animation-delay: 0.2s; }
        .pill:nth-child(3) { left: 25%; animation-duration: 4.5s; animation-delay: 0.5s; }
        .pill:nth-child(4) { left: 35%; animation-duration: 6s; animation-delay: 0.1s; }
        .pill:nth-child(5) { left: 45%; animation-duration: 4s; animation-delay: 0.8s; }
        .pill:nth-child(6) { left: 55%; animation-duration: 5.5s; animation-delay: 0.3s; }
        .pill:nth-child(7) { left: 65%; animation-duration: 4.5s; animation-delay: 0.7s; }
        .pill:nth-child(8) { left: 75%; animation-duration: 6s; animation-delay: 0.4s; }
        .pill:nth-child(9) { left: 85%; animation-duration: 4s; animation-delay: 0.9s; }
        .pill:nth-child(10) { left: 95%; animation-duration: 5s; animation-delay: 0.2s; }
        .pill:nth-child(11) { left: 20%; animation-duration: 5.5s; animation-delay: 0.6s; }
        .pill:nth-child(12) { left: 80%; animation-duration: 4.8s; animation-delay: 0.5s; }
        </style>
        """, unsafe_allow_html=True)

    # ── TUNING & OPTIMIZATION ──
    col_t1, col_t2 = st.columns([1, 1.5])
    
    with col_t1:
        st.markdown("#### ⚡ Optimization")
        if st.button("🚀 Run Hyperparameter Tuning"):
            with st.spinner("Engineering Model Sensitivity..."):
                import time
                time.sleep(1.5)
                # Upgrading the Random Forest to be sensitive to Age/BMI
                if S.model_name == "Random Forest":
                    S.model = RandomForestRegressor(
                        n_estimators=300, 
                        max_depth=15,       # Deep enough to capture trends
                        min_samples_leaf=1, # Sensitive to small feature changes
                        random_state=42
                    )
                    S.model.fit(S.X_train, S.y_train)
                
                S.tuning_done = True
                st.success("Optimization Complete!")

    # ── LIVE PREDICTOR ──
    if S.trained:
        st.markdown("---")
        st.markdown("#### 🔮 Live Premium Predictor")
        st.write("Input patient details to see real-time price adjustments.")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            in_age = st.slider("Age", 18, 100, 25)
            in_smoker = st.radio("Smoker?", ["No", "Yes"], horizontal=True)
        with c2:
            in_bmi = st.slider("BMI", 10.0, 55.0, 24.5, 0.1)
            in_children = st.number_input("Dependents", 0, 10, 0)
        with c3:
            in_sex = st.selectbox("Sex", ["Male", "Female"])
            st.caption(f"Targeting: {S.target}")

        # Prep prediction data
        input_dict = {
            'age': in_age,
            'sex': 1 if in_sex == "Male" else 0,
            'bmi': in_bmi,
            'children': in_children,
            'smoker': 1 if in_smoker == "Yes" else 0
        }
        
        # Filter for only selected features from Step 4
        input_df = pd.DataFrame([input_dict])[S.selected_features]

        if st.button("💰 Calculate Insurance Charges"):
            prediction = S.model.predict(input_df)[0]
            
            # Premium Results Card
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(0,212,170,0.1), rgba(108,99,255,0.1)); 
                        border: 2px solid var(--accent2); padding: 30px; border-radius: 20px; 
                        text-align: center; margin: 20px 0; box-shadow: var(--glow2);">
                <p style="margin: 0; color: var(--muted); font-family: 'Space Mono'; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 2px;">Estimated Annual Charges</p>
                <h1 style="margin: 15px 0; color: #fff; font-family: 'Space Mono'; font-size: 3.2rem;">${prediction:,.2f}</h1>
                <div style="display: flex; justify-content: center; gap: 12px; flex-wrap: wrap;">
                    <span class="tag tag-green">AGE: {in_age}</span>
                    <span class="tag tag-yellow">BMI: {in_bmi}</span>
                    <span class="tag {'tag-red' if in_smoker == 'Yes' else 'tag-green'}">SMOKER: {in_smoker}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── FOOTER ACTIONS ──
    if S.tuning_done:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Restart Pipeline 🔄", use_container_width=True):
            for k in DEFAULTS: S[k] = DEFAULTS[k]
            st.rerun()

    st.button("← Back", on_click=prev_step_fn)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PREMIUM FOOTER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.markdown("""
    <style>
    .footer-container {
        text-align: center;
        padding: 2rem 0;
        font-family: 'Space Mono', monospace;
        color: #8890b5;
    }
    .footer-line {
        height: 1px;
        background: linear-gradient(90deg, transparent, #6c63ff, transparent);
        margin-bottom: 1rem;
        opacity: 0.5;
    }
    .footer-names {
        font-weight: 700;
        letter-spacing: 2px;
        background: linear-gradient(90deg, #6c63ff, #00d4aa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.2rem;
    }
    .footer-sub {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 5px;
    }
    </style>
    <div class="footer-container">
        <div class="footer-line"></div>
        <div class="footer-sub">Designed & Engineered by</div>
        <div class="footer-names">ANSHIKA & ANJIL</div>
        
    </div>
""", unsafe_allow_html=True)

