# =====================================================
# ECO 5012B — Streamlit Nowcasting App
# Germany GDP Sentiment-Based Nowcasting
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

# -----------------------------------------------------
# Page configuration
# -----------------------------------------------------
st.set_page_config(
    page_title="Germany GDP Nowcaster",
    page_icon="📈",
    layout="wide"
)

st.title("Germany GDP Nowcasting App")
st.markdown("Replication and extension of Ashwin et al. (2024) — Germany 2000-2023")

# -----------------------------------------------------
# Load data
# -----------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("germany_data.csv", index_col=0, parse_dates=True)
    df["gdp_lag"] = df["gdp_growth"].shift(1)
    df["bci_centered"] = df["bci"] - 100
    df = df.dropna()
    return df

df = load_data()

# -----------------------------------------------------
# Fit the OLS model to get baseline beta_S
# -----------------------------------------------------
@st.cache_data
def fit_model(data):
    y = data["gdp_growth"]
    X = data[["gdp_lag", "bci_centered"]]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

model = fit_model(df)
baseline_beta_0 = model.params["const"]
baseline_beta_1 = model.params["gdp_lag"]
baseline_beta_S = model.params["bci_centered"]

# -----------------------------------------------------
# Sidebar — User Controls
# -----------------------------------------------------
st.sidebar.header("Model Controls")

# Radio button — economic state
state = st.sidebar.radio(
    "Select Economic Regime:",
    ["Normal Times", "Supply Shock"]
)

# Slider — beta_S adjustment
beta_S_pct = st.sidebar.slider(
    "Adjust Sentiment Coefficient (β_S):",
    min_value=-50,
    max_value=50,
    value=0,
    step=5,
    format="%d%%"
)

# -----------------------------------------------------
# Apply state-dependent adjustment
# -----------------------------------------------------
# Supply shock amplifies sentiment effect (animal spirits dominate during crises)
if state == "Supply Shock":
    state_multiplier = 1.5
else:
    state_multiplier = 1.0

adjusted_beta_S = baseline_beta_S * (1 + beta_S_pct / 100) * state_multiplier

# -----------------------------------------------------
# Compute live nowcast for the most recent quarter
# -----------------------------------------------------
latest = df.iloc[-1]
nowcast = (
    baseline_beta_0
    + baseline_beta_1 * latest["gdp_lag"]
    + adjusted_beta_S * latest["bci_centered"]
)

# -----------------------------------------------------
# Display Metrics
# -----------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Live Nowcast (Latest Quarter)",
        value=f"{nowcast:.2f}%",
        delta=f"{nowcast - latest['gdp_growth']:.2f}% vs actual"
    )

with col2:
    st.metric(
        label="Adjusted β_S",
        value=f"{adjusted_beta_S:.3f}",
        delta=f"Baseline: {baseline_beta_S:.3f}"
    )

with col3:
    st.metric(
        label="Economic Regime",
        value=state
    )

# -----------------------------------------------------
# Compute fitted values for entire history
# -----------------------------------------------------
df["fitted"] = (
    baseline_beta_0
    + baseline_beta_1 * df["gdp_lag"]
    + adjusted_beta_S * df["bci_centered"]
)

# -----------------------------------------------------
# Plotly Chart — Actual vs Live Nowcast
# -----------------------------------------------------
plot_df = df.reset_index().melt(
    id_vars="DATE",
    value_vars=["gdp_growth", "fitted"],
    var_name="Series",
    value_name="GDP Growth (%)"
)

# Rename for cleaner legend
plot_df["Series"] = plot_df["Series"].replace({
    "gdp_growth": "Actual GDP Growth",
    "fitted": "Live Nowcast"
})

fig = px.line(
    plot_df,
    x="DATE",
    y="GDP Growth (%)",
    color="Series",
    title=f"Germany GDP Growth: Actual vs Live Nowcast | Regime: {state}",
    color_discrete_map={
        "Actual GDP Growth": "steelblue",
        "Live Nowcast": "darkorange"
    }
)

fig.add_hline(y=0, line_dash="dash", line_color="grey")

# Annotate crisis episodes
fig.add_vrect(x0="2008-09-01", x1="2009-06-30", fillcolor="red", opacity=0.15, line_width=0, annotation_text="GFC", annotation_position="top left")
fig.add_vrect(x0="2011-07-01", x1="2012-12-31", fillcolor="purple", opacity=0.15, line_width=0, annotation_text="Eurozone Crisis", annotation_position="top left")
fig
