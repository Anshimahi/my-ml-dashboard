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

def next_step_fn(): S.step += 1
def prev_step_fn(): S.step -= 1
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
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 0 — PROBLEM TYPE (Instant Update Version)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if S.step == 0:
    with st.container():
        section("STEP 01", "🧩 Define Your Problem")
        col1, col2 = st.columns([1.6, 1])
        
        with col1:
            st.markdown("#### What type of ML problem are you solving?")
            st.markdown('<span class="tag">Supervised</span> <span class="tag">Binary / Multi-class</span> <span class="tag-yellow">Regression</span>', unsafe_allow_html=True)
            st.markdown("")
            
            # Use 'key' to link the radio directly to session state for instant updates
            choice = st.radio(
                "Select problem type:",
                ["Classification", "Regression"],
                horizontal=True,
                index=0 if S.problem_type != "Regression" else 1,
                key="problem_selector" 
            )
            
            # Sync the choice to our shorthand S.problem_type immediately
            S.problem_type = choice

            if st.button("✅ Confirm & Proceed", key="btn_problem"):
                next_step_fn()
                st.rerun()

        with col2:
            # Now this block will re-render every time the radio button is clicked
            icon = "🏷️" if S.problem_type == "Classification" else "📈"
            color = "#6c63ff" if S.problem_type != "Regression" else "#00d4aa"
            
            st.markdown(f"""
            <div class="metric-card" style="margin-top:1rem; border: 2px solid {color}; background:rgba(0,0,0,.2); transition: all 0.3s ease;">
              <div style="font-size:2.5rem; margin-bottom:.5rem">{icon}</div>
              <div class="metric-val" style="color:{color}; font-size:1.2rem">{S.problem_type}</div>
              <div class="metric-lbl">Current Selection</div>
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
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 8 — PERFORMANCE METRICS (Advanced Version)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 8 — PERFORMANCE METRICS (With Feature Importance)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif S.step == 8:
    section("STEP 09", "📊 Performance Metrics & Model Logic")
    
    if S.trained:
        import numpy as np
        
        # ─── 1. CALCULATE METRICS BASED ON PROBLEM TYPE ───
        if S.problem_type == "Regression":
            from sklearn.metrics import r2_score, mean_squared_error
            r2 = r2_score(S.y_test, S.y_pred)
            mse = mean_squared_error(S.y_test, S.y_pred)
            rmse = np.sqrt(mse)
            
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.markdown(f'<div class="metric-card" style="border-left:5px solid var(--accent2)"><div class="metric-lbl">R² SCORE</div><div class="metric-val" style="color:var(--accent2)">{r2:.4f}</div></div>', unsafe_allow_html=True)
            with m_col2:
                st.markdown(f'<div class="metric-card" style="border-left:5px solid #ff4b4b"><div class="metric-lbl">MSE</div><div class="metric-val" style="color:#ff4b4b">{mse:,.0f}</div></div>', unsafe_allow_html=True)
            with m_col3:
                st.markdown(f'<div class="metric-card" style="border-left:5px solid #ff9f43"><div class="metric-lbl">RMSE</div><div class="metric-val" style="color:#ff9f43">${rmse:,.2f}</div></div>', unsafe_allow_html=True)

        else:
            from sklearn.metrics import accuracy_score, precision_score, recall_score
            acc = accuracy_score(S.y_test, S.y_pred)
            prec = precision_score(S.y_test, S.y_pred, average='weighted')
            rec = recall_score(S.y_test, S.y_pred, average='weighted')
            
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.markdown(f'<div class="metric-card" style="border-left:5px solid var(--accent1)"><div class="metric-lbl">ACCURACY</div><div class="metric-val" style="color:var(--accent1)">{acc:.2%}</div></div>', unsafe_allow_html=True)
            with m_col2:
                st.markdown(f'<div class="metric-card" style="border-left:5px solid var(--accent2)"><div class="metric-lbl">PRECISION</div><div class="metric-val" style="color:var(--accent2)">{prec:.2%}</div></div>', unsafe_allow_html=True)
            with m_col3:
                st.markdown(f'<div class="metric-card" style="border-left:5px solid var(--accent4)"><div class="metric-lbl">RECALL</div><div class="metric-val" style="color:var(--accent4)">{rec:.2%}</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        
        # ─── 2. GENERALIZED FEATURE IMPORTANCE ───
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("#### 🎯 Model Interpretability")
            
            feat_imp = None
            method_title = "Feature Importance"

            # Check for Tree-based (Random Forest)
            if hasattr(S.model, 'feature_importances_'):
                feat_imp = pd.Series(S.model.feature_importances_, index=S.selected_features)
                method_title = "Feature Importance (Tree Weights)"
            
            # Check for Linear-based (Regression/Logistic)
            elif hasattr(S.model, 'coef_'):
                # Handle multi-class coefficients by taking the mean of absolute values
                if len(S.model.coef_.shape) > 1:
                    weights = np.mean(np.abs(S.model.coef_), axis=0)
                else:
                    weights = np.abs(S.model.coef_)
                feat_imp = pd.Series(weights, index=S.selected_features)
                method_title = "Feature Importance (Absolute Coefficients)"

            if feat_imp is not None:
                feat_imp = feat_imp.sort_values(ascending=True)
                fig_imp = px.bar(feat_imp, orientation='h', color_continuous_scale="Viridis", template="plotly_dark", title=method_title)
                fig_imp.update_layout(showlegend=False, height=350, **PLOTLY_LAYOUT)
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("💡 Interpretability logic for this specific model (e.g., Non-linear SVM) requires advanced methods like SHAP or LIME.")

        with col_chart2:
            if S.problem_type == "Regression":
                st.markdown("#### 📈 Actual vs. Predicted")
                chart_data = pd.DataFrame({'Actual': S.y_test, 'Predicted': S.y_pred})
                st.scatter_chart(chart_data, x='Actual', y='Predicted', color="#00d4aa")
            else:
                st.markdown("#### 📝 Prediction Sample")
                sample_df = pd.DataFrame({'Actual': S.y_test, 'Predicted': S.y_pred}).head(10)
                st.dataframe(sample_df, use_container_width=True)

    # ─── FOOTER BUTTONS ───
    st.markdown("<br>", unsafe_allow_html=True)
    foot_left, foot_right = st.columns([1, 1])
    with foot_left:
        st.button("← Back", on_click=prev_step_fn, use_container_width=True)
    with foot_right:
        st.button("Proceed to Tuning ⚙️ →", on_click=next_step_fn, use_container_width=True)
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
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 9 — TUNING & LIVE PREDICTION (Final Interactive Version)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 9 — TUNING & LIVE PREDICTION (Fixed Prediction Display)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    
    # 2. DYNAMIC INPUT GENERATOR (This makes it work for ANY data)
elif S.step == 9:
    section("STEP 10", "⚙️ Tuning & Universal Prediction")
    
    # ── THEMATIC ANIMATION ──
    if S.fire_pills:
        st.markdown('<div class="med-container">' + ''.join([f'<div class="pill" style="left:{x}%; animation-duration:1s;">✨</div>' for x in range(5,95,10)]) + '</div>', unsafe_allow_html=True)
        S.fire_pills = False
    
    # ── 1. UNIVERSAL MODEL OPTIMIZATION ──
    if st.button(f"🚀 Optimize {S.model_name} for {S.target}", use_container_width=True):
        with st.spinner("Fine-tuning model architecture..."):
            if S.model_name == "Random Forest":
                if S.problem_type == "Regression":
                    S.model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
                else:
                    S.model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
            
            if S.X_train is not None:
                S.model.fit(S.X_train, S.y_train)
                S.tuning_done = True
                st.success(f"Model optimized for {S.problem_type} task!")

    st.markdown("---")
    
    # ── 2. DYNAMIC INPUT GENERATOR ──
    if S.trained:
        st.markdown(f"#### 🔮 Live Predictor: {S.target}")
        
        user_inputs = {}
        cols = st.columns(3)
        
        for i, col_name in enumerate(S.selected_features):
            with cols[i % 3]:
                if pd.api.types.is_numeric_dtype(S.raw_df[col_name]):
                    m_val = float(S.raw_df[col_name].min())
                    mx_val = float(S.raw_df[col_name].max())
                    avg_val = float(S.raw_df[col_name].mean())
                    user_inputs[col_name] = st.slider(f"{col_name}", m_val, mx_val, avg_val, key=f"input_{col_name}")
                else:
                    if col_name in S.encoders:
                        opts = list(S.encoders[col_name].classes_)
                        val = st.selectbox(f"{col_name}", opts, key=f"input_{col_name}")
                        user_inputs[col_name] = S.encoders[col_name].transform([val])[0]

        # ── 3. PREDICTION LOGIC ──
        if st.button(f"📊 Generate Prediction", use_container_width=True):
            input_df = pd.DataFrame([user_inputs])[S.selected_features]
            res = S.model.predict(input_df)[0]
            S.last_prediction = res
            S.fire_pills = True
            st.rerun()

        # ── 4. RESULTS DISPLAY ──
        if S.last_prediction is not None:
            if S.problem_type == "Classification":
                display_val = S.last_prediction
                if S.target in S.encoders:
                    display_val = S.encoders[S.target].inverse_transform([int(S.last_prediction)])[0]
                color = "var(--accent1)" 
            else:
                display_val = f"{S.last_prediction:,.2f}"
                color = "var(--accent2)"

            st.markdown(f"""
                <div class="metric-card" style="border:2px solid {color}; padding:30px;">
                    <div class="metric-lbl">PREDICTED {S.target.upper()}</div>
                    <div class="metric-val" style="font-size:3rem; color:#fff">{display_val}</div>
                </div>""", unsafe_allow_html=True)

    # ── 5. FOOTER (FIXED NAMES) ──
    st.markdown("<br>", unsafe_allow_html=True)
    fl, fr = st.columns([1,1])
    with fl: 
        # Using next_step / prev_step to match your helper function names
        st.button("← Back", on_click=prev_step, use_container_width=True)
    with fr: 
        if st.button("Restart Pipeline 🔄", use_container_width=True):
            S.last_prediction = None
            for k in DEFAULTS: S[k] = DEFAULTS[k]
            st.rerun()
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

