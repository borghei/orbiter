"""Prompt templates for AI-powered portfolio analysis.

This module contains prompt construction only — no AI calls happen here.
All template functions accept real data and return formatted prompt strings
for use with the AI middleware.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_VIEWS: int = 5
MAX_ANALYSIS_WORDS: int = 500
CONFIDENCE_FLOOR: float = 0.2
ANNUALIZED_RETURN_UNIT: str = "decimal"

# Column widths for text table formatting
_COL_WIDTH_ASSET: int = 8
_COL_WIDTH_NUM: int = 12


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

MARKET_VIEWS_SYSTEM: str = """\
You are a senior quantitative crypto analyst at a systematic trading firm \
specializing in Black-Litterman portfolio construction.

IDENTITY: You translate market data into precise, calibrated probabilistic \
views that feed directly into a mean-variance optimizer. Your views are the \
ONLY subjective input to an otherwise fully quantitative pipeline.

CAPABILITIES:
- Absolute views: expected annualized return for a single asset.
- Relative views: expected outperformance of asset A over asset B (annualized).
- Confidence calibration mapping views to the Black-Litterman uncertainty matrix.

CRITICAL CONSTRAINTS (read these FIRST):
- Respond ONLY with valid JSON. No markdown, no explanation, no preamble, \
no trailing text.
- Every view MUST include a numeric confidence between 0.0 and 1.0.
- Do NOT include views with confidence below 0.2 — they are noise.
- Base views EXCLUSIVELY on the provided data. Do not fabricate statistics, \
cite external data, or reference events not evidenced in the inputs.
- All returns are annualized decimals (0.10 = 10% annual return).
- For relative views, express as expected outperformance of the first asset \
over the second.
- Provide between 1 and 5 views. Fewer high-confidence views are strictly \
preferred over many low-confidence ones.

CONFIDENCE CALIBRATION:
- 0.8–1.0: Strong quantitative signal — momentum, mean-reversion, or \
on-chain metrics all align.
- 0.5–0.7: Moderate conviction — multiple supporting factors but some \
conflicting signals.
- 0.2–0.4: Weak/speculative — single-factor thesis or limited data.

QUALITY STANDARDS:
- Reasoning must cite specific numbers from the provided data.
- Each view's return magnitude must be plausible given the asset's recent \
volatility (do not predict 200% returns for a 40% vol asset at high confidence).
- Relative views should only compare assets in the same portfolio.

ANTI-PATTERNS (never do these):
- Do NOT output markdown code fences.
- Do NOT include views on assets not listed in the portfolio.
- Do NOT copy-paste boilerplate reasoning — each view must reference \
unique data points.
- Do NOT round confidence to convenient numbers (0.5, 0.8) without \
justification — use the full scale.

OUTPUT FORMAT:
A JSON array of view objects. Each object has:
- "asset": string (absolute view) or [string, string] (relative view, first \
outperforms second).
- "return": float, annualized decimal.
- "confidence": float, 0.0 to 1.0.
- "reasoning": string, 1-2 sentences citing data from the prompt."""

PORTFOLIO_ANALYSIS_SYSTEM: str = """\
You are a crypto portfolio analyst producing actionable intelligence for a \
quantitative fund's portfolio managers.

IDENTITY: You bridge raw quantitative metrics and human decision-making. \
Your audience understands Sharpe ratios, drawdowns, and correlation matrices — \
do not over-explain basics.

CAPABILITIES:
- Portfolio composition and concentration analysis.
- Risk/return attribution across assets.
- Regime-aware commentary (bull, sideways, bear).
- Identification of hidden correlation and liquidity risks.

CRITICAL CONSTRAINTS (read these FIRST):
- Be specific and quantitative. Reference actual numbers from the data.
- Label every claim with a confidence tag:
  [VERIFIED] — directly derived from provided data.
  [LIKELY] — strong inference from multiple data points.
  [UNCERTAIN] — plausible but speculative.
- Never recommend specific buy/sell amounts, dollar values, or precise timing.
- Focus on allocation tilts, risk factors, and regime considerations.
- Keep total response under 500 words.

OUTPUT STRUCTURE (follow this order exactly):
1. **Summary** — 2-3 sentences capturing portfolio health and key theme.
2. **Key Risks** — Bulleted list, most severe first.
3. **Opportunities** — Bulleted list, highest conviction first.
4. **Recommendation** — 1-2 sentences on the single most impactful action.

QUALITY STANDARDS:
- Every bullet must contain at least one number from the provided data.
- Risks and opportunities must be distinct — do not repeat the same point \
in both sections.
- If the regime is BEAR, lead with capital preservation. If BULL, lead with \
return capture. If SIDEWAYS, lead with yield and rebalancing.

ANTI-PATTERNS (never do these):
- Do NOT produce generic advice ("diversify your portfolio").
- Do NOT speculate on specific price targets.
- Do NOT reference data not provided in the prompt.
- Do NOT use emojis or casual language."""

RISK_ASSESSMENT_SYSTEM: str = """\
You are a risk analyst specializing in crypto portfolio tail risk and stress \
testing at a systematic fund.

IDENTITY: You interpret quantitative stress test outputs and translate them \
into risk management decisions. You think in terms of conditional tail \
expectations, correlation breakdowns, and liquidity spirals.

CAPABILITIES:
- Stress test interpretation (historical scenario replay, Monte Carlo).
- Correlation regime analysis (stable vs. crisis correlations).
- Concentration and liquidity risk scoring.
- Actionable risk level classification.

CRITICAL CONSTRAINTS (read these FIRST):
- Respond ONLY with valid JSON. No markdown, no explanation, no preamble.
- Focus exclusively on: correlation breakdown risk, liquidity spirals, \
concentration risk, and regime change probability.
- Every risk item MUST have a severity level.
- Base analysis EXCLUSIVELY on the provided data.

SEVERITY LEVELS:
- LOW: Within normal parameters, no action needed.
- MODERATE: Elevated but manageable, increase monitoring frequency.
- HIGH: Actionable — hedge, reduce position, or set stop-losses.
- CRITICAL: Immediate rebalance required, portfolio survival at risk.

OUTPUT FORMAT:
A JSON object with:
- "overall_risk_level": one of "LOW", "MODERATE", "HIGH", "CRITICAL".
- "risks": array of objects, each with:
  - "category": string (e.g., "correlation_breakdown", "concentration", \
"liquidity", "regime_change").
  - "severity": one of "LOW", "MODERATE", "HIGH", "CRITICAL".
  - "description": string, 1-2 sentences with specific numbers.
  - "metric": float or null, the key number driving this assessment.
- "summary": string, 2-3 sentences overall assessment.

ANTI-PATTERNS (never do these):
- Do NOT output markdown code fences.
- Do NOT invent scenarios not supported by the stress test data.
- Do NOT downplay risks — if the data shows danger, say so clearly.
- Do NOT produce more than 6 risk items. Prioritize the most severe."""


# ---------------------------------------------------------------------------
# User prompt builders
# ---------------------------------------------------------------------------

def market_views_prompt(
    assets: list[str],
    returns_summary: pd.DataFrame,
    sentiment: dict[str, object],
    yields: dict[str, float],
) -> str:
    """Build the user prompt for Black-Litterman market view generation.

    Parameters
    ----------
    assets:
        Ticker symbols in the portfolio (e.g. ``["BTC", "ETH", "SOL"]``).
    returns_summary:
        DataFrame with columns ``mean_return``, ``volatility``, ``sharpe``
        indexed by asset symbol.  All values annualized.
    sentiment:
        Dict with keys ``fear_greed_value``, ``fear_greed_label``,
        ``avg_funding_rate``, ``regime``.
    yields:
        Mapping of asset symbol to annualized DeFi yield (decimal).

    Returns
    -------
    str
        Fully formatted user prompt ready for the AI middleware.
    """
    asset_table = ", ".join(assets)
    returns_table = format_returns_table(returns_summary)
    yields_table = format_yields_table(yields)

    fg_value = sentiment.get("fear_greed_value", "N/A")
    fg_label = sentiment.get("fear_greed_label", "N/A")
    avg_funding = sentiment.get("avg_funding_rate", "N/A")
    regime = sentiment.get("regime", "N/A")

    if isinstance(avg_funding, float):
        avg_funding_str = f"{avg_funding:+.6f}"
    else:
        avg_funding_str = str(avg_funding)

    return f"""\
Generate Black-Litterman market views for the following crypto portfolio.

## Portfolio Assets
{asset_table}

## Recent Performance (annualized)
{returns_table}

## Current Market Sentiment
Fear & Greed Index: {fg_value} ({fg_label})
Average Funding Rate: {avg_funding_str}
Market Regime: {regime}

## DeFi Yields (annualized)
{yields_table}

## Output Format
Respond with a JSON array of 1-{MAX_VIEWS} view objects:
[
  {{"asset": "BTC", "return": 0.12, "confidence": 0.7, "reasoning": "..."}},
  {{"asset": ["SOL", "ETH"], "return": 0.05, "confidence": 0.5, "reasoning": "..."}}
]

Remember: returns are annualized decimals. Confidence below {CONFIDENCE_FLOOR} \
should be excluded. Base views ONLY on the data above."""


def portfolio_analysis_prompt(
    weights: pd.Series,
    metrics: dict[str, float],
    sentiment: dict[str, object],
    regime: str,
) -> str:
    """Build the user prompt for portfolio analysis commentary.

    Parameters
    ----------
    weights:
        Series of portfolio weights indexed by asset symbol.
    metrics:
        Dict of portfolio-level metrics (e.g. ``sharpe``, ``volatility``,
        ``max_drawdown``, ``annual_return``).
    sentiment:
        Same structure as :func:`market_views_prompt`.
    regime:
        Current market regime label (``"BULL"``, ``"SIDEWAYS"``, ``"BEAR"``).

    Returns
    -------
    str
        Fully formatted user prompt.
    """
    weights_table = format_weights_table(weights)

    metrics_lines: list[str] = []
    for key, value in metrics.items():
        label = key.replace("_", " ").title()
        if "return" in key or "drawdown" in key:
            metrics_lines.append(f"  {label}: {value:+.2%}")
        elif "ratio" in key:
            metrics_lines.append(f"  {label}: {value:.3f}")
        else:
            metrics_lines.append(f"  {label}: {value:.4f}")
    metrics_block = "\n".join(metrics_lines)

    fg_value = sentiment.get("fear_greed_value", "N/A")
    fg_label = sentiment.get("fear_greed_label", "N/A")
    avg_funding = sentiment.get("avg_funding_rate", "N/A")

    if isinstance(avg_funding, float):
        avg_funding_str = f"{avg_funding:+.6f}"
    else:
        avg_funding_str = str(avg_funding)

    return f"""\
Analyze the following crypto portfolio and provide actionable insights.

## Current Allocation
{weights_table}

## Portfolio Metrics
{metrics_block}

## Market Context
Fear & Greed Index: {fg_value} ({fg_label})
Average Funding Rate: {avg_funding_str}
Market Regime: {regime}

## Instructions
- Structure your response as: Summary → Key Risks → Opportunities → Recommendation.
- Label confidence: [VERIFIED], [LIKELY], or [UNCERTAIN].
- Reference specific numbers from the data above.
- Maximum {MAX_ANALYSIS_WORDS} words."""


def risk_assessment_prompt(
    stress_results: dict[str, object],
    correlation_matrix: pd.DataFrame,
    weights: pd.Series,
) -> str:
    """Build the user prompt for stress test / risk assessment.

    Parameters
    ----------
    stress_results:
        Dict containing stress test outputs.  Expected keys include
        ``scenarios`` (list of dicts with ``name``, ``portfolio_return``,
        ``worst_asset``, ``worst_return``) and ``var_95``, ``cvar_95``.
    correlation_matrix:
        Asset correlation matrix as a DataFrame.
    weights:
        Current portfolio weights.

    Returns
    -------
    str
        Fully formatted user prompt.
    """
    weights_table = format_weights_table(weights)
    corr_table = _format_correlation_matrix(correlation_matrix)

    # Stress scenarios
    scenarios = stress_results.get("scenarios", [])
    scenario_lines: list[str] = []
    for s in scenarios:
        name = s.get("name", "Unknown")
        port_ret = s.get("portfolio_return", 0.0)
        worst = s.get("worst_asset", "N/A")
        worst_ret = s.get("worst_return", 0.0)
        scenario_lines.append(
            f"  {name}: portfolio {port_ret:+.2%}, "
            f"worst asset {worst} ({worst_ret:+.2%})"
        )
    scenarios_block = "\n".join(scenario_lines) if scenario_lines else "  No scenarios provided."

    var_95 = stress_results.get("var_95", "N/A")
    cvar_95 = stress_results.get("cvar_95", "N/A")

    if isinstance(var_95, float):
        var_str = f"{var_95:+.2%}"
    else:
        var_str = str(var_95)

    if isinstance(cvar_95, float):
        cvar_str = f"{cvar_95:+.2%}"
    else:
        cvar_str = str(cvar_95)

    # Concentration metric — max weight
    max_weight = float(weights.max()) if len(weights) > 0 else 0.0
    max_asset = str(weights.idxmax()) if len(weights) > 0 else "N/A"
    hhi = float((weights**2).sum()) if len(weights) > 0 else 0.0

    return f"""\
Assess the risk profile of the following crypto portfolio based on stress \
test results.

## Current Allocation
{weights_table}

## Concentration Metrics
Largest Position: {max_asset} at {max_weight:.1%}
Herfindahl-Hirschman Index (HHI): {hhi:.4f}
Number of Assets: {len(weights)}

## Stress Test Scenarios
{scenarios_block}

## Value at Risk
VaR (95%): {var_str}
CVaR (95%): {cvar_str}

## Correlation Matrix
{corr_table}

## Output Format
Respond with a JSON object:
{{
  "overall_risk_level": "LOW" | "MODERATE" | "HIGH" | "CRITICAL",
  "risks": [
    {{
      "category": "correlation_breakdown" | "concentration" | "liquidity" | "regime_change",
      "severity": "LOW" | "MODERATE" | "HIGH" | "CRITICAL",
      "description": "1-2 sentences with specific numbers",
      "metric": <float or null>
    }}
  ],
  "summary": "2-3 sentences overall assessment"
}}

Base your assessment ONLY on the data provided above."""


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_returns_table(returns: pd.DataFrame) -> str:
    """Format a returns summary DataFrame as an aligned text table.

    Parameters
    ----------
    returns:
        DataFrame with columns ``mean_return``, ``volatility``, ``sharpe``
        indexed by asset symbol.

    Returns
    -------
    str
        Aligned plain-text table suitable for prompt inclusion.
    """
    header = (
        f"{'Asset':<{_COL_WIDTH_ASSET}}"
        f"{'Ann.Return':>{_COL_WIDTH_NUM}}"
        f"{'Volatility':>{_COL_WIDTH_NUM}}"
        f"{'Sharpe':>{_COL_WIDTH_NUM}}"
    )
    separator = "-" * len(header)
    lines: list[str] = [header, separator]

    for asset in returns.index:
        row = returns.loc[asset]
        mean_ret = row.get("mean_return", 0.0)
        vol = row.get("volatility", 0.0)
        sharpe = row.get("sharpe", 0.0)
        lines.append(
            f"{str(asset):<{_COL_WIDTH_ASSET}}"
            f"{mean_ret:>{_COL_WIDTH_NUM}.2%}"
            f"{vol:>{_COL_WIDTH_NUM}.2%}"
            f"{sharpe:>{_COL_WIDTH_NUM}.3f}"
        )

    return "\n".join(lines)


def format_weights_table(weights: pd.Series) -> str:
    """Format portfolio weights as an aligned text table.

    Parameters
    ----------
    weights:
        Series of weights indexed by asset symbol.

    Returns
    -------
    str
        Aligned plain-text table suitable for prompt inclusion.
    """
    header = f"{'Asset':<{_COL_WIDTH_ASSET}}{'Weight':>{_COL_WIDTH_NUM}}"
    separator = "-" * len(header)
    lines: list[str] = [header, separator]

    for asset in weights.index:
        w = weights.loc[asset]
        lines.append(
            f"{str(asset):<{_COL_WIDTH_ASSET}}{w:>{_COL_WIDTH_NUM}.2%}"
        )

    return "\n".join(lines)


def format_yields_table(yields: dict[str, float]) -> str:
    """Format DeFi yields as an aligned text table.

    Parameters
    ----------
    yields:
        Mapping of asset symbol to annualized yield (decimal).

    Returns
    -------
    str
        Aligned plain-text table suitable for prompt inclusion.
    """
    if not yields:
        return "  No yield data available."

    header = f"{'Asset':<{_COL_WIDTH_ASSET}}{'APY':>{_COL_WIDTH_NUM}}"
    separator = "-" * len(header)
    lines: list[str] = [header, separator]

    for asset, apy in sorted(yields.items()):
        lines.append(
            f"{asset:<{_COL_WIDTH_ASSET}}{apy:>{_COL_WIDTH_NUM}.2%}"
        )

    return "\n".join(lines)


def _format_correlation_matrix(corr: pd.DataFrame) -> str:
    """Format a correlation matrix as an aligned text table.

    Parameters
    ----------
    corr:
        Square correlation DataFrame indexed and columned by asset symbols.

    Returns
    -------
    str
        Aligned plain-text table suitable for prompt inclusion.
    """
    assets = list(corr.columns)
    col_w = 7

    # Header row
    header = f"{'':>{_COL_WIDTH_ASSET}}" + "".join(
        f"{str(a):>{col_w}}" for a in assets
    )
    separator = "-" * len(header)
    lines: list[str] = [header, separator]

    for row_asset in assets:
        row_str = f"{str(row_asset):>{_COL_WIDTH_ASSET}}"
        for col_asset in assets:
            val = corr.loc[row_asset, col_asset]
            row_str += f"{val:>{col_w}.3f}"
        lines.append(row_str)

    return "\n".join(lines)
