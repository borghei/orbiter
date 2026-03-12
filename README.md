<p align="center">
  <img src="https://raw.githubusercontent.com/borghei/orbiter/main/assets/orbiter-banner.png" alt="Orbiter" width="100%">
</p>

<p align="center">
  <strong>Lightweight, open-source crypto portfolio optimizer.</strong><br>
  10 optimization strategies, 8 risk metrics, AI-powered views, DeFi yields, sentiment analysis — all from a single CLI or interactive dashboard.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT_+_Commons_Clause-yellow.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/tests-159%20passing-brightgreen.svg" alt="Tests">
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#who-is-this-for">Who Is This For</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#dashboard">Dashboard</a> •
  <a href="#strategies">Strategies</a> •
  <a href="#features">Features</a> •
  <a href="#python-api">Python API</a>
</p>

---

## Why Orbiter?

Most portfolio optimization tools are built for equities. Crypto is different — 24/7 markets, extreme volatility, fat-tailed returns, and correlations that spike during crashes. Orbiter is built specifically for this.

- **Fast** — Real optimization (not brute-force), results in seconds
- **Honest** — Walk-forward backtesting to avoid overfitting
- **Practical** — CLI for quick analysis, dashboard for exploration, Python API for integration
- **Comprehensive** — From basic mean-variance to regime detection and factor models

## Who Is This For

### Individual Investors & Traders

You hold multiple crypto assets and want to know the **optimal allocation** — not just gut feeling, but mathematically optimized weights based on real risk-return tradeoffs. Orbiter tells you exactly how much to put in each coin, shows you the efficient frontier, and validates the strategy with out-of-sample backtesting so you're not just curve-fitting to the past.

**What you get:**
- Run `orbiter optimize BTC ETH SOL AVAX` and get an allocation in seconds
- See how your portfolio would have performed with walk-forward backtesting
- Understand your actual risk exposure — max drawdown, CVaR, tail risk
- Compare 7 different strategies side-by-side to find what fits your risk tolerance

### Quantitative Researchers & Students

You're studying portfolio theory, crypto markets, or quantitative finance and need a clean, well-tested codebase to experiment with. Every module is independent, documented, and covered by tests.

**What you get:**
- Factor model with market, momentum, size, and liquidity factors for crypto
- Regime detection (HMM) to study bull/bear/sideways market dynamics
- Monte Carlo stress testing with fat-tailed (Student-t) distributions
- Efficient frontier computation, correlation analysis, covariance estimation
- All code is modular — import just the pieces you need into your own research

### Developers & Bot Builders

You're building a trading system, a portfolio dashboard, or an automated rebalancing bot. Orbiter gives you a clean Python API to plug into your stack.

**What you get:**
- `pip install orbiter-crypto` and import directly — no API keys needed for price data
- Programmatic access to all strategies, metrics, and stress tests
- Rebalancing simulation with transaction cost modeling (maker/taker fees, slippage)
- Calendar, threshold, or hybrid rebalancing triggers
- Works with any exchange supported by ccxt (Binance, Bybit, Kraken, OKX, etc.)

### Portfolio Managers & Analysts

You manage crypto allocations for clients or a fund and need institutional-grade analytics without paying for Bloomberg PORT.

**What you get:**
- Ledoit-Wolf shrinkage covariance — the same estimator used by institutional quant desks
- Risk parity and HRP — strategies used by Bridgewater and other systematic funds
- Correlation stress testing — see how your diversification breaks down during crashes
- Factor decomposition — understand what's driving your portfolio returns
- Interactive Streamlit dashboard for client presentations and exploration

## Installation

```bash
pip install orbiter-crypto
```

With the interactive dashboard:

```bash
pip install orbiter-crypto[dashboard]
```

Or install from source:

```bash
git clone https://github.com/borghei/orbiter.git
cd orbiter
pip install -e ".[dashboard]"
```

## Quick Start

### CLI

```bash
# Find the optimal allocation for a portfolio
orbiter optimize BTC ETH SOL AVAX --strategy max-sharpe

# Try different strategies
orbiter optimize BTC ETH SOL AVAX BNB --strategy hrp
orbiter optimize BTC ETH SOL --strategy risk-parity

# Regime-aware: auto-detects bull/bear/sideways and picks the best strategy
orbiter optimize BTC ETH SOL AVAX --strategy regime-aware

# Walk-forward backtest with out-of-sample validation
orbiter backtest BTC ETH SOL AVAX --strategy hrp --train-days 180 --test-days 30

# Stress test your portfolio (Monte Carlo + correlation stress)
orbiter stress BTC ETH SOL AVAX --distribution student-t --horizon 30

# Discover top coins by market cap
orbiter discover --top 30

# Launch the interactive dashboard
orbiter dashboard
```

### Example Output

```
Strategy: Hierarchical Risk Parity

            Optimal Allocation
┏━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Asset ┃ Weight ┃                       ┃
┡━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│   BTC │  53.3% │ █████████████████████ │
│   ETH │  17.0% │ ██████                │
│   SOL │  15.5% │ ██████                │
│  AVAX │  14.2% │ █████                 │
└───────┴────────┴───────────────────────┘

     Portfolio Metrics
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Metric          ┃  Value ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Ann. Return     │ -18.8% │
│ Ann. Volatility │  56.7% │
│ Sharpe Ratio    │  -0.33 │
│ Sortino Ratio   │  -0.46 │
│ Max Drawdown    │ -58.4% │
│ Calmar Ratio    │  -0.32 │
│ CVaR (95%)      │ -6.72% │
│ Omega Ratio     │   0.95 │
└─────────────────┴────────┘
```

## Dashboard

Launch with `orbiter dashboard` and open http://localhost:8501.

<p align="center">
  <img src="https://raw.githubusercontent.com/borghei/orbiter/main/assets/screenshots/allocation-metrics.png" alt="Dashboard — Allocation and Metrics" width="100%">
</p>

The dashboard provides:

- **Sidebar configuration** — pick assets, strategy, covariance method, exchange, lookback period
- **Allocation pie chart** — see how your portfolio is weighted
- **Metrics overview** — Sharpe, return, drawdown, CVaR at a glance
- **Efficient frontier** — interactive scatter plot showing the risk-return tradeoff

<p align="center">
  <img src="https://raw.githubusercontent.com/borghei/orbiter/main/assets/screenshots/efficient-frontier.png" alt="Efficient Frontier" width="100%">
</p>

- **Performance chart** — cumulative returns vs equal-weight benchmark
- **Drawdown chart** — visualize peak-to-trough declines
- **Correlation heatmap** — understand how your assets move together

Toggle advanced features in the sidebar:
- **Walk-Forward Backtest** — out-of-sample validation with weight history
- **Stress Testing** — Monte Carlo VaR/CVaR + correlation stress
- **Factor Model** — factor loadings heatmap and R-squared per asset

## Strategies

| Strategy | Description | Best For |
|---|---|---|
| `max-sharpe` | Maximize risk-adjusted return (Sharpe ratio) | Aggressive growth |
| `min-vol` | Minimize portfolio volatility | Capital preservation |
| `min-cvar` | Minimize tail risk (Conditional Value-at-Risk) | Downside protection |
| `risk-parity` | Equal risk contribution from each asset | Balanced exposure |
| `hrp` | Hierarchical Risk Parity — correlation-based, no return estimates | Robust diversification |
| `regime-aware` | HMM detects bull/bear/sideways, auto-switches strategy | Adaptive allocation |
| `factor-max-sharpe` | Max Sharpe using factor-model implied returns | Data-driven views |
| `black-litterman` | Black-Litterman model with investor views (manual or AI-generated) | Subjective + quantitative |
| `sentiment-regime` | HMM + Fear & Greed, funding rates, exchange flows | Early regime detection |
| `yield-adjusted` | Max Sharpe with DeFi staking/lending yields factored in | Yield-aware allocation |

## Features

### Risk Metrics

Every optimization returns 8 risk metrics:

| Metric | Description |
|---|---|
| Annualized Return | Geometric mean return, annualized to 365 days |
| Annualized Volatility | Standard deviation of returns, annualized |
| Sharpe Ratio | Return per unit of risk |
| Sortino Ratio | Return per unit of downside risk |
| Max Drawdown | Largest peak-to-trough decline |
| Calmar Ratio | Return divided by max drawdown |
| CVaR (95%) | Expected loss in the worst 5% of scenarios |
| Omega Ratio | Probability-weighted gains over losses |

### Covariance Estimation

| Method | Description |
|---|---|
| `ledoit-wolf` | Shrinkage estimator — default, most robust for small samples |
| `sample` | Standard sample covariance |
| `exponential` | Exponentially weighted — gives more weight to recent data |

### Walk-Forward Backtesting

Prevents overfitting by separating training and testing periods:

```bash
orbiter backtest BTC ETH SOL AVAX --strategy hrp --train-days 180 --test-days 30
```

1. Train on 180 days of data → find optimal weights
2. Test those weights on the next 30 days (out-of-sample)
3. Slide forward, repeat
4. Report aggregated out-of-sample performance vs equal-weight benchmark

### Stress Testing

```bash
orbiter stress BTC ETH SOL AVAX --distribution student-t --horizon 30 --corr-stress 2.0
```

- **Monte Carlo simulation** — 10,000 paths with normal or Student-t (fat-tailed) distributions
- **VaR & CVaR** — at 95% and 99% confidence levels
- **Correlation stress** — see how portfolio risk increases when correlations spike (as they do during crashes)

### Regime Detection

Hidden Markov Model identifies three market states:

| Regime | Characteristics | Strategy Used |
|---|---|---|
| Bull | Positive drift, moderate volatility | `max-sharpe` |
| Sideways | No clear direction, low volatility | `risk-parity` |
| Bear | Negative drift, high volatility | `min-vol` |

```bash
orbiter optimize BTC ETH SOL AVAX --strategy regime-aware
```

### Factor Model

Crypto-specific four-factor model:

- **Market** — overall market return (equal or cap-weighted)
- **Momentum** — long recent winners, short recent losers
- **Size** — small-cap minus large-cap returns
- **Liquidity** — high-volume minus low-volume returns

```bash
orbiter optimize BTC ETH SOL AVAX --strategy factor-max-sharpe --use-factors
```

### Black-Litterman with AI Views

Combine market equilibrium with your own views — or let AI generate them:

```python
from orbiter import BlackLitterman, View, get_ai
from orbiter.prompts import MARKET_VIEWS_SYSTEM, market_views_prompt
from orbiter.black_litterman import parse_ai_views

# Manual views
views = [
    View(asset="SOL", return_view=0.30, confidence=0.7),
    View(asset=("ETH", "BTC"), return_view=0.05, confidence=0.6),
]
bl = BlackLitterman(returns)
result = bl.optimize(views)

# Or generate views with AI (Claude, OpenAI, or Perplexity)
ai = get_ai("claude")  # reads ANTHROPIC_API_KEY from env
response = ai.generate(MARKET_VIEWS_SYSTEM, market_views_prompt(...))
views = parse_ai_views(response, assets=["BTC", "ETH", "SOL"])
result = bl.optimize(views)
```

Supports three AI providers with a single middleware:

| Provider | Model | Env Variable |
|---|---|---|
| Claude | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` |
| OpenAI | `gpt-4o` | `OPENAI_API_KEY` |
| Perplexity | `sonar-pro` | `PERPLEXITY_API_KEY` |

```bash
pip install orbiter-crypto[ai]  # installs anthropic + openai SDKs
```

### Sentiment-Enhanced Regime Detection

Goes beyond price-only HMM by fusing multiple sentiment signals:

- **Fear & Greed Index** — free API, updated daily
- **Funding rates** — crowded leverage detection from perpetual futures
- **Exchange net flows** — coins moving to exchanges = selling pressure

```python
from orbiter import SentimentCollector
from orbiter.regime import SentimentRegimeModel

collector = SentimentCollector(exchange="binance")
sentiment = collector.collect(["BTC", "ETH", "SOL"])

model = SentimentRegimeModel(n_regimes=3)
model.fit(market_returns, sentiment_features_df)
regime = model.current_regime(market_returns, sentiment_features_df)
```

### DeFi Yield-Adjusted Optimization

Staking ETH at 3.5% APY changes the efficient frontier. Orbiter accounts for this:

```python
from orbiter import YieldCollector
from orbiter.defi import adjust_expected_returns

yields = YieldCollector().collect(["ETH", "SOL", "ATOM"])
adjusted_mu = adjust_expected_returns(daily_returns.mean(), yields)
```

- Pulls live yields from DeFiLlama (7000+ protocols)
- Falls back to conservative manual estimates if API is down
- Includes yield reliability scoring (higher APY = more smart contract risk)

### Transaction Cost Modeling

Built-in fee and slippage models for realistic simulations:

- Maker/taker exchange fees
- Bid-ask spread costs
- Almgren-Chriss square-root market impact model

### Rebalancing Simulation

Three rebalancing triggers:

| Trigger | Description |
|---|---|
| `calendar` | Rebalance every N days |
| `threshold` | Rebalance when any weight drifts beyond a threshold |
| `hybrid` | Threshold check on a calendar schedule |

### Coin Discovery

```bash
orbiter discover --top 20 --min-mcap 1000000000
```

Fetches top coins from CoinGecko by market cap, with price and volume data.

## Python API

### Basic Optimization

```python
from orbiter import PriceLoader, PortfolioOptimizer

loader = PriceLoader(exchange="binance")
returns = loader.get_returns(["BTC", "ETH", "SOL", "AVAX"], days=365)

optimizer = PortfolioOptimizer(returns, cov_method="ledoit-wolf")
result = optimizer.optimize("max-sharpe")

print(result.weights)
print(result.metrics)
```

### Efficient Frontier

```python
frontier = optimizer.efficient_frontier(n_points=50)
# DataFrame with columns: return, volatility, sharpe, BTC, ETH, SOL, AVAX
```

### Walk-Forward Backtest

```python
from orbiter.backtest import WalkForwardBacktest

bt = WalkForwardBacktest(returns, train_days=180, test_days=30, strategy="hrp")
result = bt.run()

print(result.metrics)           # Out-of-sample performance
print(result.weights_history)   # Weights at each rebalance
```

### Stress Testing

```python
from orbiter.stress import monte_carlo_stress, correlation_stress

# Monte Carlo with fat tails
mc = monte_carlo_stress(
    weights=result.weights.values,
    mu=returns.mean().values,
    cov=returns.cov().values,
    distribution="student-t",
    df=5.0,
    horizon_days=30,
)
print(mc["cvar_95"], mc["var_99"])

# Correlation stress
cs = correlation_stress(weights, cov_matrix, stress_factor=2.0)
print(cs["stressed_volatility"])
```

### Factor Model

```python
from orbiter.factors import CryptoFactorModel

model = CryptoFactorModel(returns)
exposures = model.fit()

print(exposures.loadings)       # Factor betas per asset
print(exposures.r_squared)      # Model fit per asset
print(model.expected_returns()) # Factor-implied expected returns
```

### Regime Detection

```python
from orbiter.regime import RegimeModel

model = RegimeModel(n_regimes=3)
model.fit(returns["BTC"])

regime = model.current_regime(returns["BTC"])
print(regime)                   # Regime.BULL, Regime.SIDEWAYS, or Regime.BEAR
print(model.get_strategy(regime))
```

### Rebalancing Simulation

```python
from orbiter.rebalance import simulate_rebalancing, RebalanceConfig, RebalanceTrigger
from orbiter.costs import FeeSchedule

config = RebalanceConfig(
    trigger=RebalanceTrigger.THRESHOLD,
    drift_threshold=0.05,
    fee_schedule=FeeSchedule(maker=0.001, taker=0.001),
)
result = simulate_rebalancing(returns, target_weights, config)

print(result.total_turnover)
print(result.total_cost)
print(result.metrics)
```

## Data Sources

| Source | Data | API Key |
|---|---|---|
| Binance (via ccxt) | OHLCV price data | Not required |
| Any ccxt exchange | OHLCV price data | Not required |
| CoinGecko | Market caps, coin discovery | Not required (free tier) |
| Blockchain.com | BTC active addresses, NVT ratio | Not required |

Change the exchange:

```bash
orbiter optimize BTC ETH SOL --exchange kraken
orbiter optimize BTC ETH SOL --exchange bybit
```

## CLI Reference

| Command | Description |
|---|---|
| `orbiter optimize SYMBOLS...` | Optimize portfolio allocation |
| `orbiter backtest SYMBOLS...` | Walk-forward backtest |
| `orbiter stress SYMBOLS...` | Monte Carlo + correlation stress test |
| `orbiter discover` | List top coins by market cap |
| `orbiter dashboard` | Launch Streamlit dashboard |

Run `orbiter COMMAND --help` for full options.

## Architecture

```
src/orbiter/
├── data.py            # Price fetching via ccxt
├── data_sources.py    # CoinGecko, on-chain metrics
├── metrics.py         # Risk & performance metrics
├── covariance.py      # Covariance estimation (sample, Ledoit-Wolf, exponential)
├── optimize.py        # 10 optimization strategies + efficient frontier
├── backtest.py        # Walk-forward backtesting engine
├── costs.py           # Transaction cost & slippage modeling
├── rebalance.py       # Rebalancing simulation (calendar/threshold/hybrid)
├── regime.py          # HMM regime detection + sentiment-enhanced regime
├── stress.py          # Monte Carlo & correlation stress testing
├── factors.py         # Crypto factor model (market, momentum, size, liquidity)
├── black_litterman.py # Black-Litterman model with AI-generated views
├── ai.py              # AI middleware (Claude, OpenAI, Perplexity adapters)
├── prompts.py         # Prompt templates for AI-powered analysis
├── sentiment.py       # Fear & Greed, funding rates, exchange flows
├── defi.py            # DeFi yield data from DeFiLlama
├── cli.py             # Click CLI
└── dashboard.py       # Streamlit interactive dashboard
```

## License

**MIT + Commons Clause** — Free to use in open-source projects, personal use, education, and internal business workflows. You cannot sell this software or repackage it as a paid product. See [LICENSE](LICENSE) for full terms.

---

<p align="center">
  <a href="https://buymeacoffee.com/borghei"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me a Coffee"></a>
</p>

<p align="center">
  <a href="https://borghei.me">borghei.me</a>
</p>
