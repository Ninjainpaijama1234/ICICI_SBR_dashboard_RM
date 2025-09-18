import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from statsmodels.api import OLS, add_constant
from math import log, sqrt, exp
import io

# ----------------------------
# Black-Scholes Option Pricing
# ----------------------------
def black_scholes(S, K, r, sigma, T, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    delta = norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type=="call" else -d2))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    rho = (K * T * np.exp(-r * T) *
           (norm.cdf(d2) if option_type=="call" else -norm.cdf(-d2)))

    return price, delta, gamma, theta, vega, rho

# ----------------------------
# VaR Computations
# ----------------------------
def historical_var(returns, conf=0.95):
    return np.percentile(returns, (1 - conf) * 100)

def parametric_var(returns, conf=0.95):
    mu, sigma = returns.mean(), returns.std()
    return mu - sigma * norm.ppf(conf)

def monte_carlo_var(S0, mu, sigma, T, n=10000, conf=0.95):
    sims = S0 * np.exp((mu - 0.5 * sigma**2) * T +
                       sigma * np.sqrt(T) * np.random.randn(n))
    rets = (sims - S0) / S0
    return np.percentile(rets, (1 - conf) * 100)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="ICICI Risk Dashboard", layout="wide")

st.title("📊 ICICI Bank Risk & Portfolio Dashboard")

# ---- File Upload ----
uploaded_file = st.file_uploader("Upload ICICI Dashboard Excel", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
else:
    df = pd.read_excel("icici dashboard data.xlsx")

df.columns = ["Date", "ICICI_Price", "ICICI_Return", "Nifty_Price", "Nifty_Return"]
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# ---- Filters ----
st.sidebar.header("Filters")
date_range = st.sidebar.date_input("Select Date Range",
                                   [df.index.min(), df.index.max()])
df = df.loc[date_range[0]:date_range[1]]

# Editable parameters
st.sidebar.subheader("Parameters")
notional = st.sidebar.number_input("Notional Value", 10000)
risk_free = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0) / 100
volatility = st.sidebar.slider("Volatility (%)", 0.0, 100.0, 30.0) / 100
time_horizon = st.sidebar.slider("Time Horizon (Years)", 0.1, 5.0, 1.0)

# ==========================
# 1. Performance Analysis
# ==========================
st.header("1️⃣ Performance Analysis")

df["ICICI_%Change"] = df["ICICI_Price"].pct_change()
df["Nifty_%Change"] = df["Nifty_Price"].pct_change()

# Reset index for Plotly charts
df_reset = df.reset_index()

fig1 = px.line(
    df_reset,
    x="Date",
    y=["ICICI_Price", "Nifty_Price"],
    title="ICICI vs Nifty Prices"
)
st.plotly_chart(fig1, use_container_width=True)

st.write("**Stats:**")
st.write(df[["ICICI_%Change", "Nifty_%Change"]].agg(["mean","var","std"]))

# ==========================
# 2. Risk-Return Analysis
# ==========================
st.header("2️⃣ Risk-Return Analysis")

# Sharpe & Sortino
rf = risk_free / 252
excess_returns = df["ICICI_%Change"] - rf
sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
downside = df["ICICI_%Change"][df["ICICI_%Change"] < 0]
sortino = np.sqrt(252) * excess_returns.mean() / downside.std()

# Beta & Alpha (fix alignment issue)
reg_df = df[["ICICI_%Change", "Nifty_%Change"]].dropna()
X = add_constant(reg_df["Nifty_%Change"])
y = reg_df["ICICI_%Change"]
model = OLS(y, X).fit()
alpha, beta = model.params

st.write(f"Sharpe Ratio: {sharpe:.3f}, Sortino Ratio: {sortino:.3f}")
st.write(f"Alpha: {alpha:.4f}, Beta: {beta:.3f}")

fig2 = px.scatter(
    reg_df, x="Nifty_%Change", y="ICICI_%Change",
    trendline="ols", title="Regression: ICICI vs Nifty"
)
st.plotly_chart(fig2, use_container_width=True)

# ==========================
# 3. Value at Risk
# ==========================
st.header("3️⃣ Value at Risk (VaR)")

conf = st.slider("Confidence Level", 0.90, 0.99, 0.95)

hist_var = historical_var(df["ICICI_%Change"].dropna(), conf)
param_var = parametric_var(df["ICICI_%Change"].dropna(), conf)
mc_var = monte_carlo_var(df["ICICI_Price"].iloc[-1],
                         df["ICICI_%Change"].mean(),
                         df["ICICI_%Change"].std(),
                         time_horizon, 10000, conf)

st.write(f"Historical VaR: {hist_var:.2%}")
st.write(f"Parametric VaR: {param_var:.2%}")
st.write(f"Monte Carlo VaR: {mc_var:.2%}")

# ==========================
# 4. Options & Greeks
# ==========================
st.header("4️⃣ Options & Greeks")

col1, col2 = st.columns(2)
with col1:
    S = st.number_input("Spot Price", value=100.0)
    K = st.number_input("Strike Price", value=100.0)
with col2:
    sigma = st.number_input("Volatility", value=0.2)
    T = st.number_input("Time to Maturity (Years)", value=1.0)

price, delta, gamma, theta, vega, rho = black_scholes(S, K, risk_free, sigma, T)

st.write(f"Option Price: {price:.2f}")
st.write(f"Delta: {delta:.3f}, Gamma: {gamma:.3f}, Theta: {theta:.3f}, Vega: {vega:.3f}, Rho: {rho:.3f}")

# ==========================
# 5. ALM Analysis
# ==========================
st.header("5️⃣ Asset Liability Management (ALM)")

alm_file = st.file_uploader("Upload ALM Maturity Pattern CSV", type=["csv"])
if alm_file:
    alm_df = pd.read_csv(alm_file)
    st.dataframe(alm_df)

    RSA = alm_df.loc[alm_df["Type"]=="Asset", "Amount"].sum()
    RSL = alm_df.loc[alm_df["Type"]=="Liability", "Amount"].sum()
    st.write(f"Rate Sensitive Assets: {RSA}, Rate Sensitive Liabilities: {RSL}")

# ==========================
# 6. Portfolio Simulation
# ==========================
st.header("6️⃣ Portfolio Simulation")

sims = np.random.normal(df["ICICI_%Change"].mean(),
                        df["ICICI_%Change"].std(),
                        10000)

fig3 = px.histogram(sims, nbins=50, title="Monte Carlo Portfolio Returns")
st.plotly_chart(fig3, use_container_width=True)

# ==========================
# 7. Download Results
# ==========================
st.header("7️⃣ Download Options")

output = io.BytesIO()
df.to_excel(output)
st.download_button("Download Processed Data", data=output.getvalue(),
                   file_name="processed_icici.xlsx")
