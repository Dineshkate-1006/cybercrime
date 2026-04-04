import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India Cybercrime Dashboard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

  html, body, [class*="css"] {
      font-family: 'Syne', sans-serif;
      background-color: #0a0e1a;
      color: #e8eaf0;
  }
  .stApp { background-color: #0a0e1a; }

  /* Sidebar */
  [data-testid="stSidebar"] {
      background: #0f1628;
      border-right: 1px solid #1e2d4a;
  }
  [data-testid="stSidebar"] * { color: #c8d0e7 !important; }

  /* Metric cards */
  .metric-card {
      background: linear-gradient(135deg, #111827 0%, #1a2540 100%);
      border: 1px solid #243456;
      border-radius: 12px;
      padding: 20px 24px;
      text-align: center;
  }
  .metric-value {
      font-family: 'Space Mono', monospace;
      font-size: 2.2rem;
      font-weight: 700;
      color: #60a5fa;
      line-height: 1.1;
  }
  .metric-label {
      font-size: 0.78rem;
      color: #8899bb;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      margin-top: 6px;
  }
  .metric-delta {
      font-family: 'Space Mono', monospace;
      font-size: 0.85rem;
      margin-top: 4px;
  }

  /* Section headers */
  .section-header {
      font-family: 'Syne', sans-serif;
      font-size: 1.1rem;
      font-weight: 800;
      color: #93c5fd;
      text-transform: uppercase;
      letter-spacing: 0.15em;
      border-bottom: 2px solid #1e3a5f;
      padding-bottom: 8px;
      margin: 24px 0 16px 0;
  }

  /* Title */
  .main-title {
      font-family: 'Syne', sans-serif;
      font-size: 2.6rem;
      font-weight: 800;
      background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      line-height: 1.2;
  }
  .subtitle {
      color: #64748b;
      font-size: 0.95rem;
      margin-top: 4px;
      font-family: 'Space Mono', monospace;
  }

  /* Hide streamlit branding */
  #MainMenu, footer { visibility: hidden; }
  header { visibility: hidden; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
      background: #0f1628;
      border-radius: 8px;
      gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
      color: #8899bb !important;
      font-family: 'Syne', sans-serif;
      font-weight: 600;
  }
  .stTabs [aria-selected="true"] {
      background: #1e3a5f !important;
      color: #93c5fd !important;
      border-radius: 6px;
  }

  div[data-testid="stMetric"] {
      background: #111827;
      border: 1px solid #243456;
      border-radius: 10px;
      padding: 16px;
  }
  div[data-testid="stMetric"] label { color: #8899bb !important; }
  div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #60a5fa !important; font-family: 'Space Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# ─── Load & Parse Data ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("CyberCrime_india.csv")

    # Aggregate rows: summary rows contain keywords
    summary_keywords = ["TOTAL", "Total", "ALL INDIA"]
    df_clean = df[~df["City"].str.contains("|".join(summary_keywords), na=False)].copy()

    # Classify: if city is a known state name keep as state; else city
    # Heuristic: cities appear in first ~35 rows; states appear after
    city_names = [
        "Agra","Allahabad","Amritsar","Asansol","Aurangabad","Bhopal",
        "Chandigarh City","Dhanbad","Durg-Bhilainagar","Faridabad","Gwalior",
        "Jabalpur","Jamshedpur","Jodhpur","Kannur","Kollam","Kota","Ludhiana",
        "Madurai","Malappuram","Meerut","Nasik","Raipur","Rajkot","Ranchi",
        "Srinagar","Thiruvananthapuram","Thrissur","Tiruchirapalli","Vadodara",
        "Varanasi","Vasai Virar","Vijayawada","Vishakhapatnam"
    ]
    ut_names = [
        "A & N Islands","Chandigarh","D & N Haveli","Daman & Diu",
        "D & N Haveli and Daman & Diu","Delhi","Lakshadweep","Puducherry",
        "Jammu & Kashmir","Ladakh"
    ]

    df_clean["Region_Type"] = df_clean["City"].apply(
        lambda x: "City" if x in city_names else ("UT" if x in ut_names else "State")
    )

    crime_cols = [c for c in df.columns if c not in ["City", "Total"]]
    return df_clean, crime_cols

df, crime_cols = load_data()

# Separate datasets
df_states = df[df["Region_Type"] == "State"].copy()
df_cities = df[df["Region_Type"] == "City"].copy()
df_uts    = df[df["Region_Type"] == "UT"].copy()

# ─── Plotly theme ──────────────────────────────────────────────────────────────
PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Syne, sans-serif", color="#c8d0e7"),
    colorway=["#60a5fa","#a78bfa","#f472b6","#34d399","#fb923c","#facc15","#38bdf8","#e879f9"],
    xaxis=dict(gridcolor="#1e2d4a", linecolor="#1e2d4a"),
    yaxis=dict(gridcolor="#1e2d4a", linecolor="#1e2d4a"),
)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔐 Filters")
    st.markdown("---")

    region_type = st.radio("Region Type", ["All", "States", "Cities", "Union Territories"], index=0)

    if region_type == "States":
        data_view = df_states
    elif region_type == "Cities":
        data_view = df_cities
    elif region_type == "Union Territories":
        data_view = df_uts
    else:
        data_view = df

    available_regions = sorted(data_view["City"].tolist())
    selected_regions = st.multiselect("Select Regions", available_regions, default=available_regions[:10])
    if selected_regions:
        data_filtered = data_view[data_view["City"].isin(selected_regions)]
    else:
        data_filtered = data_view

    st.markdown("---")
    selected_crimes = st.multiselect(
        "Crime Categories",
        crime_cols,
        default=crime_cols[:8]
    )
    if not selected_crimes:
        selected_crimes = crime_cols

    st.markdown("---")
    st.markdown("<div style='font-size:0.75rem; color:#4a5568;'>Data: NCRB India Cybercrime Statistics</div>", unsafe_allow_html=True)

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">India Cybercrime<br>Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">// National Crime Records Bureau · Motive-wise Analysis</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ─── KPI Row ───────────────────────────────────────────────────────────────────
total_crimes   = int(df["Total"].sum())
state_crimes   = int(df_states["Total"].sum())
city_crimes    = int(df_cities["Total"].sum())
top_crime_cat  = df[crime_cols].sum().idxmax()
top_crime_val  = int(df[crime_cols].sum().max())
top_state      = df_states.loc[df_states["Total"].idxmax(), "City"]
top_state_val  = int(df_states["Total"].max())

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Total Cases", f"{total_crimes:,}", help="Across all regions")
with c2:
    st.metric("State Cases", f"{state_crimes:,}")
with c3:
    st.metric("City Cases", f"{city_crimes:,}")
with c4:
    st.metric("Top Motive", top_crime_cat, delta=f"{top_crime_val:,} cases")
with c5:
    st.metric("Highest State", top_state, delta=f"{top_state_val:,} cases")

st.markdown("<br>", unsafe_allow_html=True)

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🗺️ Regional Breakdown", "🔍 Crime Categories", "📋 Raw Data"])

# ══════════════════════════════════════════════════════════════
# TAB 1: Overview
# ══════════════════════════════════════════════════════════════
with tab1:

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="section-header">Top 15 Regions by Total Cases</div>', unsafe_allow_html=True)
        top15 = data_filtered.nlargest(15, "Total")[["City", "Total"]].sort_values("Total")
        fig = px.bar(
            top15, x="Total", y="City", orientation="h",
            color="Total", color_continuous_scale=["#1e3a5f","#2563eb","#60a5fa","#a78bfa"],
            labels={"Total": "Total Cases", "City": ""},
        )
        fig.update_layout(**PLOTLY_THEME, showlegend=False, height=420,
                          margin=dict(l=0, r=10, t=10, b=10),
                          coloraxis_showscale=False)
        fig.update_traces(hovertemplate="<b>%{y}</b><br>Cases: %{x:,}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Crime Motive Composition</div>', unsafe_allow_html=True)
        motive_totals = data_filtered[crime_cols].sum().sort_values(ascending=False)
        motive_totals = motive_totals[motive_totals > 0]
        fig2 = px.pie(
            values=motive_totals.values,
            names=motive_totals.index,
            hole=0.55,
            color_discrete_sequence=["#60a5fa","#a78bfa","#f472b6","#34d399","#fb923c","#facc15","#38bdf8","#e879f9","#4ade80","#f87171"],
        )
        fig2.update_layout(**PLOTLY_THEME, height=420,
                           margin=dict(l=0, r=0, t=10, b=10),
                           legend=dict(font=dict(size=10)))
        fig2.update_traces(hovertemplate="<b>%{label}</b><br>%{value:,} cases (%{percent})<extra></extra>",
                           textposition="outside", textfont_size=10)
        st.plotly_chart(fig2, use_container_width=True)

    # Heatmap
    st.markdown('<div class="section-header">Crime Heatmap — Regions × Motives</div>', unsafe_allow_html=True)
    hm_data = data_filtered.set_index("City")[selected_crimes].fillna(0)
    # Show top 20 rows by total for readability
    hm_data = hm_data.loc[hm_data.sum(axis=1).nlargest(20).index]
    fig3 = px.imshow(
        hm_data,
        color_continuous_scale=["#0a0e1a","#1e3a5f","#2563eb","#60a5fa","#f472b6"],
        aspect="auto",
        labels=dict(color="Cases"),
    )
    fig3.update_layout(**PLOTLY_THEME, height=420, margin=dict(l=0, r=0, t=10, b=10))
    fig3.update_traces(hovertemplate="<b>%{y}</b> — %{x}<br>Cases: %{z:,}<extra></extra>")
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 2: Regional Breakdown
# ══════════════════════════════════════════════════════════════
with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">States — Total Cases</div>', unsafe_allow_html=True)
        states_sorted = df_states.sort_values("Total", ascending=False)
        fig_s = px.bar(
            states_sorted, x="City", y="Total",
            color="Total",
            color_continuous_scale=["#1e3a5f","#2563eb","#60a5fa"],
            labels={"City": "State", "Total": "Cases"},
        )
        fig_s.update_layout(**PLOTLY_THEME, height=380, showlegend=False,
                            coloraxis_showscale=False,
                            margin=dict(l=0, r=0, t=10, b=80),
                            xaxis_tickangle=-40)
        st.plotly_chart(fig_s, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Cities — Total Cases</div>', unsafe_allow_html=True)
        cities_sorted = df_cities.sort_values("Total", ascending=False)
        fig_c = px.bar(
            cities_sorted, x="City", y="Total",
            color="Total",
            color_continuous_scale=["#1e3a5f","#a78bfa","#f472b6"],
            labels={"City": "City", "Total": "Cases"},
        )
        fig_c.update_layout(**PLOTLY_THEME, height=380, showlegend=False,
                            coloraxis_showscale=False,
                            margin=dict(l=0, r=0, t=10, b=80),
                            xaxis_tickangle=-40)
        st.plotly_chart(fig_c, use_container_width=True)

    # Treemap
    st.markdown('<div class="section-header">Treemap — State-wise Crime Volume</div>', unsafe_allow_html=True)
    tm_data = df_states[df_states["Total"] > 0].copy()
    fig_tm = px.treemap(
        tm_data, path=["City"], values="Total",
        color="Total",
        color_continuous_scale=["#1e2d4a","#1d4ed8","#60a5fa","#f472b6"],
    )
    fig_tm.update_layout(**PLOTLY_THEME, height=400, margin=dict(l=0, r=0, t=10, b=10))
    fig_tm.update_traces(hovertemplate="<b>%{label}</b><br>Cases: %{value:,}<extra></extra>")
    st.plotly_chart(fig_tm, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 3: Crime Categories
# ══════════════════════════════════════════════════════════════
with tab3:

    col_x, col_y = st.columns([1, 1])

    with col_x:
        st.markdown('<div class="section-header">Motive Ranking (All India)</div>', unsafe_allow_html=True)
        crime_rank = df[crime_cols].sum().sort_values(ascending=True)
        crime_rank = crime_rank[crime_rank > 0]
        fig_cr = px.bar(
            x=crime_rank.values, y=crime_rank.index, orientation="h",
            color=crime_rank.values,
            color_continuous_scale=["#1e3a5f","#7c3aed","#f472b6"],
            labels={"x": "Total Cases", "y": "Motive"},
        )
        fig_cr.update_layout(**PLOTLY_THEME, height=400, showlegend=False,
                             coloraxis_showscale=False,
                             margin=dict(l=0, r=10, t=10, b=10))
        st.plotly_chart(fig_cr, use_container_width=True)

    with col_y:
        st.markdown('<div class="section-header">Top States per Crime Motive</div>', unsafe_allow_html=True)
        sel_crime = st.selectbox("Choose a crime motive", crime_cols, index=2)
        top_for_crime = df_states.nlargest(10, sel_crime)[["City", sel_crime]].sort_values(sel_crime)
        fig_tc = px.bar(
            top_for_crime, x=sel_crime, y="City", orientation="h",
            color=sel_crime,
            color_continuous_scale=["#1e3a5f","#0ea5e9","#34d399"],
            labels={sel_crime: "Cases", "City": ""},
        )
        fig_tc.update_layout(**PLOTLY_THEME, height=400, showlegend=False,
                             coloraxis_showscale=False,
                             margin=dict(l=0, r=10, t=10, b=10))
        st.plotly_chart(fig_tc, use_container_width=True)

    # Radar
    st.markdown('<div class="section-header">Crime Profile Radar — Selected Regions</div>', unsafe_allow_html=True)
    radar_regions = st.multiselect(
        "Pick regions to compare",
        sorted(df["City"].tolist()),
        default=["Karnataka","Maharashtra","Uttar Pradesh","Telangana","Assam"]
    )
    if radar_regions:
        radar_data = df[df["City"].isin(radar_regions)].set_index("City")[crime_cols]
        radar_data_norm = radar_data.div(radar_data.max(axis=0).replace(0, 1))  # normalise
        cats = crime_cols
        fig_rad = go.Figure()
        colors = ["#60a5fa","#a78bfa","#f472b6","#34d399","#fb923c"]
        for i, region in enumerate(radar_regions):
            if region in radar_data_norm.index:
                vals = radar_data_norm.loc[region].tolist()
                vals += [vals[0]]
                fig_rad.add_trace(go.Scatterpolar(
                    r=vals, theta=cats + [cats[0]],
                    fill="toself", name=region,
                    line_color=colors[i % len(colors)],
                    fillcolor=colors[i % len(colors)].replace(")", ", 0.1)").replace("rgb", "rgba") if "rgb" in colors[i % len(colors)] else colors[i % len(colors)] + "22",
                    opacity=0.85
                ))
        fig_rad.update_layout(
            **PLOTLY_THEME,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0,1], showticklabels=False, gridcolor="#1e2d4a"),
                angularaxis=dict(gridcolor="#1e2d4a", linecolor="#1e2d4a"),
            ),
            height=420, margin=dict(l=40, r=40, t=20, b=40)
        )
        st.plotly_chart(fig_rad, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 4: Raw Data
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Filtered Dataset</div>', unsafe_allow_html=True)
    show_cols = ["City", "Region_Type", "Total"] + selected_crimes
    st.dataframe(
        data_filtered[show_cols].style
            .background_gradient(subset=["Total"], cmap="Blues")
            .format("{:.0f}", subset=[c for c in show_cols if c not in ["City","Region_Type"]]),
        use_container_width=True, height=500
    )
    csv_out = data_filtered[show_cols].to_csv(index=False)
    st.download_button("⬇ Download CSV", csv_out, "filtered_cybercrime.csv", "text/csv")
