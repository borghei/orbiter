"""Command-line interface for Orbiter."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

console = Console()

ALL_STRATEGIES = [
    "max-sharpe",
    "min-vol",
    "min-cvar",
    "risk-parity",
    "hrp",
    "regime-aware",
    "factor-max-sharpe",
    "black-litterman",
    "sentiment-regime",
    "yield-adjusted",
]


@click.group()
@click.version_option(package_name="orbiter")
def cli():
    """Orbiter — Lightweight crypto portfolio optimizer."""


@cli.command()
@click.argument("symbols", nargs=-1, required=True)
@click.option(
    "--strategy",
    type=click.Choice(ALL_STRATEGIES, case_sensitive=False),
    default="max-sharpe",
    help="Optimization strategy.",
)
@click.option("--days", default=365, help="Days of historical data.")
@click.option("--timeframe", default="1d", help="Candle timeframe.")
@click.option(
    "--cov-method",
    type=click.Choice(["sample", "ledoit-wolf", "exponential"]),
    default="ledoit-wolf",
    help="Covariance estimation method.",
)
@click.option("--exchange", default="binance", help="Exchange to fetch data from.")
@click.option("--use-factors", is_flag=True, help="Use factor model for expected returns.")
def optimize(symbols, strategy, days, timeframe, cov_method, exchange, use_factors):
    """Optimize a crypto portfolio.

    Example: orbiter optimize BTC ETH SOL AVAX --strategy max-sharpe
    """
    from orbiter.data import PriceLoader
    from orbiter.optimize import PortfolioOptimizer

    with console.status("[bold cyan]Fetching price data..."):
        loader = PriceLoader(exchange=exchange)
        returns = loader.get_returns(list(symbols), timeframe=timeframe, days=days)

    console.print(f"\n[dim]Loaded {len(returns)} days of data for {len(symbols)} assets[/dim]\n")

    factor_model = None
    if use_factors or strategy == "factor-max-sharpe":
        from orbiter.factors import CryptoFactorModel

        with console.status("[bold cyan]Building factor model..."):
            factor_model = CryptoFactorModel(returns)
            factor_model.fit()

    with console.status(f"[bold cyan]Optimizing ({strategy})..."):
        optimizer = PortfolioOptimizer(returns, cov_method=cov_method, factor_model=factor_model)
        result = optimizer.optimize(strategy)

    _print_result(result, strategy)


@cli.command()
@click.argument("symbols", nargs=-1, required=True)
@click.option(
    "--strategy",
    type=click.Choice(ALL_STRATEGIES, case_sensitive=False),
    default="max-sharpe",
)
@click.option("--days", default=365, help="Days of historical data.")
@click.option("--train-days", default=180, help="Training window size.")
@click.option("--test-days", default=30, help="Out-of-sample test window size.")
@click.option("--exchange", default="binance", help="Exchange for data.")
@click.option("--cov-method", default="ledoit-wolf")
@click.option("--maker-fee", default=0.001, help="Maker fee (default 0.1%).")
@click.option("--taker-fee", default=0.001, help="Taker fee (default 0.1%).")
def backtest(
    symbols, strategy, days, train_days, test_days, exchange, cov_method, maker_fee, taker_fee
):
    """Run walk-forward backtest.

    Example: orbiter backtest BTC ETH SOL --strategy hrp --train-days 90 --test-days 30
    """
    from orbiter.backtest import WalkForwardBacktest
    from orbiter.data import PriceLoader
    from orbiter.metrics import compute_metrics

    with console.status("[bold cyan]Fetching price data..."):
        loader = PriceLoader(exchange=exchange)
        returns = loader.get_returns(list(symbols), timeframe="1d", days=days)

    console.print(f"\n[dim]Loaded {len(returns)} days for {len(symbols)} assets[/dim]\n")

    with console.status(f"[bold cyan]Running walk-forward backtest ({strategy})..."):
        bt = WalkForwardBacktest(
            returns,
            train_days=train_days,
            test_days=test_days,
            strategy=strategy,
            cov_method=cov_method,
        )
        result = bt.run()

    # Compare vs equal-weight
    equal_returns = returns.mean(axis=1)
    eq_start = returns.index.get_loc(result.portfolio_returns.index[0])
    eq_end = returns.index.get_loc(result.portfolio_returns.index[-1]) + 1
    equal_oos = equal_returns.iloc[eq_start:eq_end]
    equal_metrics = compute_metrics(equal_oos)

    console.print(f"[bold]Walk-Forward Backtest: {strategy}[/bold]")
    console.print(
        f"[dim]Train: {train_days}d | Test: {test_days}d"
        f" | Rebalances: {len(result.rebalance_dates)}[/dim]\n"
    )

    table = Table(title="Out-of-Sample Performance")
    table.add_column("Metric", style="cyan")
    table.add_column(strategy, justify="right", style="green")
    table.add_column("Equal Weight", justify="right", style="yellow")

    metric_labels = {
        "annualized_return": ("Ann. Return", "{:+.1%}"),
        "annualized_volatility": ("Ann. Volatility", "{:.1%}"),
        "sharpe_ratio": ("Sharpe Ratio", "{:.2f}"),
        "sortino_ratio": ("Sortino Ratio", "{:.2f}"),
        "max_drawdown": ("Max Drawdown", "{:.1%}"),
        "calmar_ratio": ("Calmar Ratio", "{:.2f}"),
        "cvar_95": ("CVaR (95%)", "{:.2%}"),
    }

    for key, (label, fmt) in metric_labels.items():
        strat_val = result.metrics.get(key, 0)
        eq_val = equal_metrics.get(key, 0)
        table.add_row(label, fmt.format(strat_val), fmt.format(eq_val))

    console.print(table)

    # Latest weights
    if not result.weights_history.empty:
        console.print("\n[bold]Latest Allocation:[/bold]")
        latest = result.weights_history.iloc[-1]
        for asset, w in latest.items():
            if w > 0.001:
                bar = "█" * int(w * 40)
                console.print(f"  {asset:>6s}  {w:6.1%}  {bar}")


@cli.command()
@click.argument("symbols", nargs=-1, required=True)
@click.option(
    "--strategy", type=click.Choice(ALL_STRATEGIES[:5], case_sensitive=False), default="max-sharpe"
)
@click.option("--days", default=365, help="Days of historical data.")
@click.option("--exchange", default="binance")
@click.option("--simulations", default=10000, help="Number of Monte Carlo simulations.")
@click.option("--horizon", default=30, help="Stress test horizon in days.")
@click.option(
    "--distribution",
    type=click.Choice(["normal", "student-t"]),
    default="student-t",
)
@click.option("--corr-stress", default=1.5, help="Correlation stress factor (1.0 = no stress).")
def stress(symbols, strategy, days, exchange, simulations, horizon, distribution, corr_stress):
    """Run stress tests on a portfolio.

    Example: orbiter stress BTC ETH SOL AVAX --distribution student-t --horizon 30
    """
    from orbiter.data import PriceLoader
    from orbiter.optimize import PortfolioOptimizer
    from orbiter.stress import correlation_stress, monte_carlo_stress

    with console.status("[bold cyan]Fetching price data..."):
        loader = PriceLoader(exchange=exchange)
        returns = loader.get_returns(list(symbols), timeframe="1d", days=days)

    with console.status(f"[bold cyan]Optimizing ({strategy})..."):
        optimizer = PortfolioOptimizer(returns)
        result = optimizer.optimize(strategy)

    weights = result.weights.values
    mu = returns.mean().values
    cov = returns.cov().values

    console.print(f"\n[bold]Stress Test: {strategy}[/bold]")
    console.print(
        f"[dim]Horizon: {horizon}d | Simulations: {simulations}"
        f" | Distribution: {distribution}[/dim]\n"
    )

    # Monte Carlo
    with console.status("[bold cyan]Running Monte Carlo..."):
        mc = monte_carlo_stress(
            weights,
            mu,
            cov,
            n_simulations=simulations,
            horizon_days=horizon,
            distribution=distribution,
        )

    mc_table = Table(title=f"Monte Carlo ({distribution}, {horizon}d horizon)")
    mc_table.add_column("Metric", style="cyan")
    mc_table.add_column("Value", justify="right", style="bold")

    mc_table.add_row("VaR (95%)", f"{mc['var_95']:.2%}")
    mc_table.add_row("CVaR (95%)", f"{mc['cvar_95']:.2%}")
    mc_table.add_row("VaR (99%)", f"{mc['var_99']:.2%}")
    mc_table.add_row("CVaR (99%)", f"{mc['cvar_99']:.2%}")
    mc_table.add_row("Median Return", f"{mc['median_return']:+.2%}")
    mc_table.add_row("Worst Case", f"{mc['worst_case']:.2%}")
    mc_table.add_row("Best Case", f"{mc['best_case']:+.2%}")
    mc_table.add_row("P(Loss)", f"{mc['prob_loss']:.1%}")

    console.print(mc_table)

    # Correlation stress
    with console.status("[bold cyan]Running correlation stress..."):
        from orbiter.covariance import get_covariance

        ann_cov = get_covariance(returns, method="ledoit-wolf")
        cs = correlation_stress(weights, ann_cov, stress_factor=corr_stress)

    console.print()
    cs_table = Table(title=f"Correlation Stress (factor={corr_stress}x)")
    cs_table.add_column("Metric", style="cyan")
    cs_table.add_column("Value", justify="right", style="bold")
    cs_table.add_row("Original Vol", f"{cs['original_volatility']:.1%}")
    cs_table.add_row("Stressed Vol", f"{cs['stressed_volatility']:.1%}")
    cs_table.add_row("Vol Increase", f"{cs['increase_pct']:+.1f}%")
    console.print(cs_table)


@cli.command()
@click.option("--top", default=20, help="Number of top coins to show.")
@click.option("--min-mcap", default=1e9, help="Minimum market cap in USD.")
def discover(top, min_mcap):
    """Discover top cryptocurrencies by market cap.

    Example: orbiter discover --top 30
    """
    from orbiter.data_sources import CoinGeckoClient

    with console.status("[bold cyan]Fetching from CoinGecko..."):
        client = CoinGeckoClient()
        coins = client.top_coins(n=top, min_mcap_usd=min_mcap)

    table = Table(title=f"Top {len(coins)} Cryptocurrencies")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Symbol", style="cyan bold")
    table.add_column("Name")
    table.add_column("Price", justify="right", style="green")
    table.add_column("Market Cap", justify="right")
    table.add_column("24h Volume", justify="right")

    for _, row in coins.iterrows():
        table.add_row(
            str(int(row["rank"])) if row["rank"] else "-",
            row["symbol"],
            row["name"],
            f"${row['price']:,.2f}",
            f"${row['market_cap'] / 1e9:.1f}B",
            f"${row['volume_24h'] / 1e9:.1f}B",
        )

    console.print(table)
    symbols = " ".join(coins["symbol"].head(10).tolist())
    console.print(f"\n[dim]Try: orbiter optimize {symbols}[/dim]")


@cli.command()
@click.option("--port", default=8501, help="Port for the Streamlit dashboard.")
def dashboard(port):
    """Launch the interactive Streamlit dashboard."""
    import subprocess
    import sys

    try:
        import streamlit  # noqa: F401
    except ImportError:
        console.print(
            "[red]Streamlit not installed.[/red] "
            "Install with: [bold]pip install orbiter[dashboard][/bold]"
        )
        sys.exit(1)

    import os

    import orbiter.dashboard as dash_module

    dash_path = os.path.abspath(dash_module.__file__)
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", dash_path, "--server.port", str(port)],
    )


def _print_result(result, strategy):
    """Pretty-print optimization results."""
    strategy_names = {
        "max-sharpe": "Maximum Sharpe Ratio",
        "min-vol": "Minimum Volatility",
        "min-cvar": "Minimum CVaR",
        "risk-parity": "Risk Parity",
        "hrp": "Hierarchical Risk Parity",
        "regime-aware": "Regime-Aware (adaptive)",
        "factor-max-sharpe": "Factor Model Max Sharpe",
        "black-litterman": "Black-Litterman",
        "sentiment-regime": "Sentiment-Enhanced Regime",
        "yield-adjusted": "DeFi Yield-Adjusted",
    }

    display_name = strategy_names.get(strategy, result.strategy)
    console.print(f"[bold]Strategy: {display_name}[/bold]\n")

    # Weights table
    table = Table(title="Optimal Allocation")
    table.add_column("Asset", style="cyan", justify="right")
    table.add_column("Weight", justify="right", style="green")
    table.add_column("", justify="left")

    for asset, weight in result.weights.items():
        if weight > 0.001:
            bar = "█" * int(weight * 40)
            table.add_row(asset, f"{weight:.1%}", bar)

    console.print(table)

    # Metrics
    m = result.metrics
    console.print()
    metrics_table = Table(title="Portfolio Metrics")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", justify="right", style="bold")

    rows = [
        ("Ann. Return", f"{m['annualized_return']:+.1%}"),
        ("Ann. Volatility", f"{m['annualized_volatility']:.1%}"),
        ("Sharpe Ratio", f"{m['sharpe_ratio']:.2f}"),
        ("Sortino Ratio", f"{m['sortino_ratio']:.2f}"),
        ("Max Drawdown", f"{m['max_drawdown']:.1%}"),
        ("Calmar Ratio", f"{m['calmar_ratio']:.2f}"),
        ("CVaR (95%)", f"{m['cvar_95']:.2%}"),
        ("Omega Ratio", f"{m['omega_ratio']:.2f}"),
    ]
    for label, value in rows:
        metrics_table.add_row(label, value)

    console.print(metrics_table)
