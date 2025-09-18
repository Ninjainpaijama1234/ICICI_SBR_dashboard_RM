import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.stats import norm
from statsmodels.api import OLS, add_constant

# ----------------------------
# Utils: safe stats
# ----------------------------
def safe_std(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    return np.nanstd(x, ddof=1)

def safe_mean(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    return np.nanmean(x)

def safe_div(a, b):
    if b is None or np.isnan(b) or b == 0:
        return np.nan
    return a / b

# ----------------------------
# Black-Scholes Option Pricing
# ----------------------------
def black_scholes(S, K, r, sigma, T, option_type="call"):
    try:
        S, K, r, sigma, T = map(float, [S, K, r, sigma, T])
        if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
            return (np.nan,)*6
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            rho =  K * T * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
            delta = -norm.cdf(-d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * (norm.cdf(d2) if option_type=="call" else norm.cdf(-d2)))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        return price, delta, gamma, theta, vega, rho
    except Exception:
        return (np.nan,)*6

# ----------------------------
# VaR Computations
# ----------------------------
def historical_var(returns, conf=0.95):
    r = pd.Series(returns).dropna().values
    if r.size == 0:
        return np.nan
    return np.percentile(r, (1 - conf) * 100)

def parametric_var(returns, conf=0.95):
    r = pd.Series(returns).dropna().values
    if r.size == 0:
        return np.nan
    mu, sigma = r.mean(), r.std(ddof=1)
    if np.isnan(mu) or np.isnan(sigma):
        return np.nan
    # lower-tail threshold at the chosen confidence
    return mu - sigma * norm.ppf(conf)

def monte_carlo_var(S0, mu, sigma, T, n=10000, conf=0.95):
    try:
        S0 = float(S0); mu = float(mu); sigma = float(sigma); T = float(T)
        if S0 <= 0 or sigma < 0 or T <= 0 or n <= 10:
            return np.nan
        sims = S0 * np.exp((mu - 0.5 * sigma**2) * T +
                           sigma * np.sqrt(T) * np.random.randn(int(n)))
        rets = (sims - S0) / S0
        if rets.size == 0:
            return np.nan
        return np.percentile(rets, (1 - conf) * 100)
    except Exception:
        return np.nan

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="ICICI Risk Dashboard", layout="wide")
st.title("📊 ICICI Bank Risk & Portfolio Dashboard")

# ---- File Upload ----
uploaded_file = st.file_uploader("Upload ICICI Dashboard Excel", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
else:
    # Fallback to local file name provided by you
    df = pd.read_excel("icici dashboard data.xlsx")

# Expecting 5 columns: Date, ICICI_Price, ICICI_Return, Nifty_Price, Nifty_Return
if df.shape[1] < 5:
    st.error("Input file must have 5 columns: Date, ICICI_Price, ICICI_Return, Nifty_Price, Nifty_Return")
    st.stop()

df.columns = ["Date", "ICICI_Price", "ICICI_Return", "Nifty_Price", "Nifty_Return"]
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
df = df.set_index("Date")

# ---- Sidebar Filters & Parameters ----
st.sidebar.header("Filters")
min_d, max_d = df.index.min(), df.index.max()
date_range = st.sidebar.date_input("Select Date Range", [min_d, max_d])
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df = df.loc[start_d:end_d]
else:
    st.warning("Invalid date range; using full data.")

if df.empty:
    st.warning("No data in the selected range. Adjust filters.")
    st.stop()

st.sidebar.subheader("Parameters")
notional = st.sidebar.number_input("Notional Value", min_value=0.0, value=10000.0, step=100.0)
risk_free = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0) / 100.0
volatility_ui = st.sidebar.slider("Volatility (%)", 0.0, 100.0, 30.0) / 100.0
time_horizon = st.sidebar.slider("Time Horizon (Years)", 0.1, 5.0, 1.0)

# 🔒 Global VaR confidence used everywhere
conf = st.sidebar.select_slider(
    "VaR Confidence Level",
    options=[0.90, 0.95, 0.99],
    value=0.95,
    format_func=lambda x: f"{int(x*100)}%"
)

# ==========================
# 1. Performance Analysis
# ==========================
st.header("1️⃣ Performance Analysis")

# If returns not provided, compute from price; otherwise prefer provided columns
if "ICICI_Return" in df and df["ICICI_Return"].notna().any():
    icici_ret = pd.to_numeric(df["ICICI_Return"], errors="coerce")
else:
    icici_ret = pd.to_numeric(df["ICICI_Price"], errors="coerce").pct_change()

if "Nifty_Return" in df and df["Nifty_Return"].notna().any():
    nifty_ret = pd.to_numeric(df["Nifty_Return"], errors="coerce")
else:
    nifty_ret = pd.to_numeric(df["Nifty_Price"], errors="coerce").pct_change()

df["ICICI_%Change"] = icici_ret
df["Nifty_%Change"] = nifty_ret

# Price chart
df_reset = df.reset_index()
fig1 = px.line(
    df_reset,
    x="Date",
    y=["ICICI_Price", "Nifty_Price"],
    title="ICICI vs Nifty Prices"
)
st.plotly_chart(fig1, use_container_width=True)

# Cumulative returns
cum_df = pd.DataFrame({
    "Date": df_reset["Date"],
    "ICICI_CumRet": (1 + df["ICICI_%Change"].fillna(0)).cumprod().values - 1,
    "Nifty_CumRet":  (1 + df["Nifty_%Change"].fillna(0)).cumprod().values - 1
})
fig_cum = px.line(cum_df, x="Date", y=["ICICI_CumRet", "Nifty_CumRet"], title="Cumulative Returns")
st.plotly_chart(fig_cum, use_container_width=True)

# Summary stats
stats_tbl = df[["ICICI_%Change", "Nifty_%Change"]].agg(["mean", "var", "std"])
st.write("**Return Stats (daily):**")
st.dataframe(stats_tbl)

# ==========================
# 2. Risk-Return Analysis
# ==========================
st.header("2️⃣ Risk-Return Analysis")

# Sharpe & Sortino (annualized)
rf_daily = risk_free / 252.0
excess = (df["ICICI_%Change"] - rf_daily).dropna().values
downside = (df["ICICI_%Change"][df["ICICI_%Change"] < 0] - rf_daily).dropna().values

excess_mean = safe_mean(excess)
excess_std = safe_std(excess)
down_std = safe_std(downside)

sharpe = np.sqrt(252.0) * safe_div(excess_mean, excess_std)
sortino = np.sqrt(252.0) * safe_div(excess_mean, down_std)

# Beta & Alpha
reg_df = df[["ICICI_%Change", "Nifty_%Change"]].dropna()
alpha = beta = np.nan
if not reg_df.empty and reg_df["Nifty_%Change"].nunique() > 1:
    X = add_constant(reg_df["Nifty_%Change"].values.astype(float))
    y = reg_df["ICICI_%Change"].values.astype(float)
    try:
        model = OLS(y, X).fit()
        params = model.params
        alpha = float(params[0])
        beta = float(params[1])
    except Exception as e:
        st.warning(f"Regression failed safely: {e}")

st.write(f"**Sharpe Ratio:** {sharpe:.3f} | **Sortino Ratio:** {sortino:.3f}")
st.write(f"**Alpha:** {alpha:.6f} | **Beta:** {beta:.4f}")

# Regression scatter
if not reg_df.empty:
    fig2 = px.scatter(reg_df.reset_index(), x="Nifty_%Change", y="ICICI_%Change",
                      trendline="ols", title="Regression: ICICI vs Nifty (daily returns)")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Insufficient data for regression plot.")

# ==========================
# 3. Value at Risk
# ==========================
st.header("3️⃣ Value at Risk (VaR)")

hist_var_val = historical_var(df["ICICI_%Change"], conf)
param_var_val = parametric_var(df["ICICI_%Change"], conf)
mc_var_val = monte_carlo_var(
    S0=df["ICICI_Price"].dropna().iloc[-1] if df["ICICI_Price"].dropna().size else np.nan,
    mu=safe_mean(df["ICICI_%Change"]),
    sigma=safe_std(df["ICICI_%Change"]),
    T=time_horizon,
    n=10000,
    conf=conf
)

st.write(f"**Confidence:** {int(conf*100)}%")
st.write(f"**Historical VaR (return):** {hist_var_val:.2%}")
st.write(f"**Parametric VaR (return):** {param_var_val:.2%}")
st.write(f"**Monte Carlo VaR (return over {time_horizon:.2f}y):** {mc_var_val:.2%}")

if notional and not np.isnan(notional):
    st.write(f"**Historical VaR (amt):** {notional * hist_var_val:,.2f}")
    st.write(f"**Parametric VaR (amt):** {notional * param_var_val:,.2f}")
    st.write(f"**Monte Carlo VaR (amt):** {notional * mc_var_val:,.2f}")

# ==========================
# 4. Options & Greeks
# ==========================
st.header("4️⃣ Options & Greeks")
col1, col2 = st.columns(2)
with col1:
    S = st.number_input("Spot Price", min_value=0.0, value=100.0)
    K = st.number_input("Strike Price", min_value=0.0, value=100.0)
with col2:
    sigma_ui = st.number_input("Volatility (σ, decimal)", min_value=0.0, value=0.2)
    T = st.number_input("Time to Maturity (Years)", min_value=0.0, value=1.0)
opt_type = st.selectbox("Option Type", ["call", "put"])

price, delta, gamma, theta, vega, rho = black_scholes(S, K, risk_free, sigma_ui, T, option_type=opt_type)
st.write(
    f"**Price:** {price:.4f} | **Delta:** {delta:.4f} | **Gamma:** {gamma:.6f} | "
    f"**Theta:** {theta:.4f} | **Vega:** {vega:.4f} | **Rho:** {rho:.4f}"
)

# ==========================
# 5. ALM Analysis (basic RSA/RSL)
# ==========================
st.header("5️⃣ Asset Liability Management (ALM)")
alm_file = st.file_uploader("Upload ALM Maturity Pattern CSV", type=["csv"])
if alm_file:
    alm_df = pd.read_csv(alm_file)
    st.dataframe(alm_df)
    if set(["Type", "Amount"]).issubset(alm_df.columns):
        RSA = pd.to_numeric(alm_df.loc[alm_df["Type"].str.lower()=="asset", "Amount"], errors="coerce").sum()
        RSL = pd.to_numeric(alm_df.loc[alm_df["Type"].str.lower()=="liability", "Amount"], errors="coerce").sum()
        st.write(f"**Rate Sensitive Assets (RSA):** {RSA:,.2f} | **Rate Sensitive Liabilities (RSL):** {RSL:,.2f}")
    else:
        st.info("ALM CSV must contain columns: Type, Amount (case-insensitive).")

# ==========================
# 6. Portfolio Simulation
# ==========================
st.header("6️⃣ Portfolio Simulation")
mu_sim = safe_mean(df["ICICI_%Change"])
sigma_sim = safe_std(df["ICICI_%Change"])
n_sims = st.slider("Number of simulations", 1000, 50000, 10000, step=1000)

if not np.isnan(mu_sim) and not np.isnan(sigma_sim) and sigma_sim >= 0:
    sims = np.random.normal(mu_sim, sigma_sim, n_sims)
    sim_df = pd.DataFrame({"Simulated Returns": sims})
    fig3 = px.histogram(sim_df, x="Simulated Returns", nbins=50, title="Monte Carlo Portfolio Returns (daily)")

    # 🔎 Overlay selected-confidence daily Parametric VaR as a vertical line
    if not np.isnan(param_var_val):
        fig3.add_vline(
            x=float(param_var_val),
            line_dash="dash",
            line_color="red",
            annotation_text=f"VaR {int(conf*100)}%",
            annotation_position="top right"
        )

    st.plotly_chart(fig3, use_container_width=True)
    st.write(f"**Probability of loss (daily):** {(sim_df['Simulated Returns'] < 0).mean():.2%}")
else:
    st.info("Insufficient data to simulate returns.")

# ==========================
# 7. Download Results
# ==========================
st.header("7️⃣ Download Options")
output = io.BytesIO()
df_export = df.copy()
df_export.reset_index().to_excel(output, index=False)
st.download_button(
    "Download Processed Data (Excel)",
    data=output.getvalue(),
    file_name="processed_icici.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
