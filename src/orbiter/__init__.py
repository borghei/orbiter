"""Orbiter — Lightweight crypto portfolio optimizer."""

__version__ = "0.3.0"

# v0.3.0 — AI-powered features
from orbiter.ai import AIMiddleware as AIMiddleware
from orbiter.ai import get_ai as get_ai
from orbiter.black_litterman import BlackLitterman as BlackLitterman
from orbiter.black_litterman import View as View
from orbiter.data import PriceLoader as PriceLoader
from orbiter.defi import YieldCollector as YieldCollector
from orbiter.metrics import compute_metrics as compute_metrics
from orbiter.optimize import PortfolioOptimizer as PortfolioOptimizer
from orbiter.sentiment import SentimentCollector as SentimentCollector
