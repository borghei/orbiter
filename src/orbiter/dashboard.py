"""Streamlit dashboard for Orbiter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Orbiter", page_icon="🪐", layout="wide")
st.title("Orbiter — Crypto Portfolio Optimizer")

# --- Sidebar ---
st.sidebar.header("Configuration")

DEFAULT_COINS = ["BTC", "ETH", "SOL", "AVAX", "BNB", "ADA", "DOT", "MATIC", "LINK", "UNI"]
symbols_input = st.sidebar.text_input(
    "Assets (comma-separated)",
    value=", ".join(DEFAULT_COINS[:5]),
)
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

strategy = st.sidebar.selectbox(
    "Strategy",
    ["max-sharpe", "min-vol", "min-cvar", "risk-parity", "hrp", "regime-aware", "factor-max-sharpe"],
    index=0,
)

days = st.sidebar.slider("Historical Days", 90, 730, 365)

cov_method = st.sidebar.selectbox(
    "Covariance Method",
    ["ledoit-wolf", "sample", "exponential"],
)

exchange = st.sidebar.selectbox("Exchange", ["binance", "bybit", "okx", "kraken"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Advanced")

run_backtest = st.sidebar.checkbox("Walk-Forward Backtest", value=False)
if run_backtest:
    train_days = st.sidebar.slider("Train Window (days)", 60, 365, 180)
    test_days = st.sidebar.slider("Test Window (days)", 7, 90, 30)

run_stress = st.sidebar.checkbox("Stress Testing", value=False)
if run_stress:
    stress_dist = st.sidebar.selectbox("MC Distribution", ["student-t", "normal"])
    stress_horizon = st.sidebar.slider("MC Horizon (days)", 7, 90, 30)
    corr_stress_factor = st.sidebar.slider("Correlation Stress Factor", 1.0, 3.0, 1.5, 0.1)

use_factors = st.sidebar.checkbox("Use Factor Model", value=(strategy == "factor-max-sharpe"))

run_button = st.sidebar.button("Optimize", type="primary", use_container_width=True)

if not run_button:
    st.info("Configure your portfolio in the sidebar and click **Optimize**.")
    st.stop()

if len(symbols) < 2:
    st.error("Please enter at least 2 assets.")
    st.stop()


# --- Data Loading ---
@st.cache_data(ttl=3600, show_spinner="Fetching price data...")
def load_data(symbols_tuple, exchange, days, timeframe="1d"):
    from orbiter.data import PriceLoader

    loader = PriceLoader(exchange=exchange)
    prices = loader.get_close_prices(list(symbols_tuple), timeframe=timeframe, days=days)
    returns = np.log(prices / prices.shift(1)).dropna()
    return prices, returns


try:
    prices, returns = load_data(tuple(symbols), exchange, days)
except Exception as e:
    st.error(f"Failed to fetch data: {e}")
    st.stop()

st.success(f"Loaded {len(returns)} days of data for {len(symbols)} assets")


# --- Factor Model ---
factor_model = None
if use_factors or strategy == "factor-max-sharpe":
    from orbiter.factors import CryptoFactorModel

    with st.spinner("Building factor model..."):
        factor_model = CryptoFactorModel(returns)
        factor_model.fit()


# --- Optimization ---
from orbiter.optimize import PortfolioOptimizer

with st.spinner(f"Optimizing ({strategy})..."):
    optimizer = PortfolioOptimizer(returns, cov_method=cov_method, factor_model=factor_model)
    result = optimizer.optimize(strategy)

# --- Layout ---
col1, col2 = st.columns([1, 1])

# Weights
with col1:
    st.subheader("Optimal Allocation")
    weights_df = pd.DataFrame({
        "Asset": result.weights.index,
        "Weight": result.weights.values,
    })
    weights_df = weights_df[weights_df["Weight"] > 0.001]
    fig_pie = px.pie(
        weights_df,
        values="Weight",
        names="Asset",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_pie.update_traces(textinfo="label+percent")
    st.plotly_chart(fig_pie, use_container_width=True)

# Metrics
with col2:
    st.subheader("Portfolio Metrics")
    m = result.metrics
    metric_cols = st.columns(4)
    metric_cols[0].metric("Sharpe", f"{m['sharpe_ratio']:.2f}")
    metric_cols[1].metric("Ann. Return", f"{m['annualized_return']:+.1%}")
    metric_cols[2].metric("Max Drawdown", f"{m['max_drawdown']:.1%}")
    metric_cols[3].metric("CVaR (95%)", f"{m['cvar_95']:.2%}")

    st.dataframe(
        pd.DataFrame([
            {"Metric": "Ann. Return", "Value": f"{m['annualized_return']:+.1%}"},
            {"Metric": "Ann. Volatility", "Value": f"{m['annualized_volatility']:.1%}"},
            {"Metric": "Sharpe Ratio", "Value": f"{m['sharpe_ratio']:.2f}"},
            {"Metric": "Sortino Ratio", "Value": f"{m['sortino_ratio']:.2f}"},
            {"Metric": "Max Drawdown", "Value": f"{m['max_drawdown']:.1%}"},
            {"Metric": "Calmar Ratio", "Value": f"{m['calmar_ratio']:.2f}"},
            {"Metric": "CVaR (95%)", "Value": f"{m['cvar_95']:.2%}"},
            {"Metric": "Omega Ratio", "Value": f"{m['omega_ratio']:.2f}"},
        ]).set_index("Metric"),
        use_container_width=True,
    )

# --- Efficient Frontier ---
st.subheader("Efficient Frontier")

with st.spinner("Computing efficient frontier..."):
    frontier = optimizer.efficient_frontier(n_points=40)

if not frontier.empty:
    fig_ef = px.scatter(
        frontier,
        x="volatility",
        y="return",
        color="sharpe",
        color_continuous_scale="Viridis",
        labels={"volatility": "Annualized Volatility", "return": "Annualized Return", "sharpe": "Sharpe"},
    )
    opt_ret = result.metrics["annualized_return"]
    opt_vol = result.metrics["annualized_volatility"]
    fig_ef.add_scatter(
        x=[opt_vol], y=[opt_ret],
        mode="markers",
        marker=dict(size=15, color="red", symbol="star"),
        name=f"Optimal ({strategy})",
    )
    for col in returns.columns:
        asset_ret = float(returns[col].mean() * 365)
        asset_vol = float(returns[col].std() * np.sqrt(365))
        fig_ef.add_scatter(
            x=[asset_vol], y=[asset_ret],
            mode="markers+text",
            marker=dict(size=10, color="orange"),
            text=[col],
            textposition="top center",
            name=col,
            showlegend=False,
        )
    fig_ef.update_layout(height=500)
    st.plotly_chart(fig_ef, use_container_width=True)

# --- Performance Charts ---
st.subheader("Performance")
col_perf1, col_perf2 = st.columns(2)

with col_perf1:
    port_returns = (returns * result.weights.values).sum(axis=1)
    equal_returns = returns.mean(axis=1)

    cum_port = (port_returns.cumsum().apply(np.exp) - 1) * 100
    cum_equal = (equal_returns.cumsum().apply(np.exp) - 1) * 100

    cum_df = pd.DataFrame({
        f"Optimized ({strategy})": cum_port,
        "Equal Weight": cum_equal,
    })
    st.line_chart(cum_df, y_label="Cumulative Return (%)")

with col_perf2:
    cum_wealth = port_returns.cumsum().apply(np.exp)
    running_max = cum_wealth.cummax()
    drawdown = ((cum_wealth - running_max) / running_max) * 100

    dd_df = pd.DataFrame({"Drawdown (%)": drawdown})
    st.area_chart(dd_df, y_label="Drawdown (%)", color=["#ff4b4b"])

# --- Correlation Heatmap ---
st.subheader("Correlation Matrix")
corr = returns.corr()
fig_corr = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    aspect="auto",
)
fig_corr.update_layout(height=400)
st.plotly_chart(fig_corr, use_container_width=True)

# --- Factor Model Results ---
if factor_model is not None and factor_model._exposures is not None:
    st.subheader("Factor Model")
    exp = factor_model._exposures

    fc1, fc2 = st.columns(2)
    with fc1:
        st.write("**Factor Loadings (Betas)**")
        fig_loadings = px.imshow(
            exp.loadings,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            aspect="auto",
        )
        st.plotly_chart(fig_loadings, use_container_width=True)

    with fc2:
        st.write("**Model R-squared per Asset**")
        r2_df = pd.DataFrame({"R²": exp.r_squared})
        st.bar_chart(r2_df)

# --- Stress Testing ---
if run_stress:
    st.subheader("Stress Testing")

    from orbiter.stress import correlation_stress, monte_carlo_stress

    weights = result.weights.values
    mu = returns.mean().values
    cov = returns.cov().values

    sc1, sc2 = st.columns(2)

    with sc1:
        with st.spinner("Running Monte Carlo..."):
            mc = monte_carlo_stress(
                weights, mu, cov,
                n_simulations=10000,
                horizon_days=stress_horizon,
                distribution=stress_dist,
            )

        st.write(f"**Monte Carlo ({stress_dist}, {stress_horizon}d)**")
        st.dataframe(pd.DataFrame([
            {"Metric": "VaR (95%)", "Value": f"{mc['var_95']:.2%}"},
            {"Metric": "CVaR (95%)", "Value": f"{mc['cvar_95']:.2%}"},
            {"Metric": "VaR (99%)", "Value": f"{mc['var_99']:.2%}"},
            {"Metric": "CVaR (99%)", "Value": f"{mc['cvar_99']:.2%}"},
            {"Metric": "Median Return", "Value": f"{mc['median_return']:+.2%}"},
            {"Metric": "Worst Case", "Value": f"{mc['worst_case']:.2%}"},
            {"Metric": "P(Loss)", "Value": f"{mc['prob_loss']:.1%}"},
        ]).set_index("Metric"), use_container_width=True)

    with sc2:
        from orbiter.covariance import get_covariance

        ann_cov = get_covariance(returns, method=cov_method)
        cs = correlation_stress(weights, ann_cov, stress_factor=corr_stress_factor)

        st.write(f"**Correlation Stress ({corr_stress_factor}x)**")
        st.metric("Original Volatility", f"{cs['original_volatility']:.1%}")
        st.metric("Stressed Volatility", f"{cs['stressed_volatility']:.1%}")
        st.metric("Vol Increase", f"{cs['increase_pct']:+.1f}%")

# --- Walk-Forward Backtest ---
if run_backtest:
    st.subheader("Walk-Forward Backtest")
    from orbiter.backtest import WalkForwardBacktest
    from orbiter.metrics import compute_metrics

    with st.spinner("Running walk-forward backtest..."):
        bt = WalkForwardBacktest(
            returns,
            train_days=train_days,
            test_days=test_days,
            strategy=strategy,
            cov_method=cov_method,
        )
        bt_result = bt.run()

    bt_col1, bt_col2 = st.columns(2)

    with bt_col1:
        st.write(f"**Rebalances:** {len(bt_result.rebalance_dates)}")
        bm = bt_result.metrics
        st.dataframe(
            pd.DataFrame([
                {"Metric": "Ann. Return", "Value": f"{bm['annualized_return']:+.1%}"},
                {"Metric": "Sharpe Ratio", "Value": f"{bm['sharpe_ratio']:.2f}"},
                {"Metric": "Max Drawdown", "Value": f"{bm['max_drawdown']:.1%}"},
                {"Metric": "CVaR (95%)", "Value": f"{bm['cvar_95']:.2%}"},
            ]).set_index("Metric"),
            use_container_width=True,
        )

    with bt_col2:
        st.write("**Weight History**")
        st.area_chart(bt_result.weights_history)
