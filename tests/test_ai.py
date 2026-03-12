"""Tests for AI middleware."""

import json
from unittest.mock import patch

import pytest

from orbiter.ai import (
    _CODE_FENCE_RE,
    DEFAULT_MODELS,
    AIConfig,
    AIMiddleware,
    AIProvider,
    get_ai,
)


class TestAIConfig:
    def test_empty_key_raises(self):
        with pytest.raises(ValueError, match="API key required"):
            AIConfig(provider=AIProvider.OPENAI, api_key="")

    def test_default_model_claude(self):
        config = AIConfig(provider=AIProvider.CLAUDE, api_key="sk-test-key")
        assert config.model == DEFAULT_MODELS["claude"]

    def test_default_model_openai(self):
        config = AIConfig(provider=AIProvider.OPENAI, api_key="sk-test-key")
        assert config.model == DEFAULT_MODELS["openai"]

    def test_default_model_perplexity(self):
        config = AIConfig(provider=AIProvider.PERPLEXITY, api_key="pplx-test-key")
        assert config.model == DEFAULT_MODELS["perplexity"]

    def test_custom_model(self):
        config = AIConfig(
            provider=AIProvider.OPENAI, api_key="sk-test", model="gpt-4-turbo"
        )
        assert config.model == "gpt-4-turbo"


class TestAIMiddlewareFromEnv:
    @patch.dict("os.environ", {}, clear=True)
    def test_missing_env_var_raises(self):
        with pytest.raises(ValueError, match="not set"):
            AIMiddleware.from_env("openai")

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-123"})
    def test_with_env_var(self):
        mw = AIMiddleware.from_env("openai")
        assert mw.config.provider == AIProvider.OPENAI
        assert mw.config.api_key == "sk-test-123"


class TestGetAI:
    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_ai(provider="not-a-provider", api_key="key")

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"})
    def test_with_known_provider(self):
        mw = get_ai(provider="claude")
        assert mw.config.provider == AIProvider.CLAUDE

    def test_with_api_key(self):
        mw = get_ai(provider="openai", api_key="sk-direct-key")
        assert mw.config.api_key == "sk-direct-key"


class TestGenerateJSON:
    def test_parses_valid_json(self):
        config = AIConfig(provider=AIProvider.OPENAI, api_key="sk-test")
        mw = AIMiddleware(config)
        expected = {"views": [{"asset": "BTC", "return": 0.1}]}
        raw_json = json.dumps(expected)

        with patch.object(mw._adapter, "generate", return_value=raw_json):
            result = mw.generate_json("system", "user")
            assert result == expected

    def test_strips_code_fences(self):
        config = AIConfig(provider=AIProvider.OPENAI, api_key="sk-test")
        mw = AIMiddleware(config)
        expected = {"key": "value"}
        raw = f"```json\n{json.dumps(expected)}\n```"

        with patch.object(mw._adapter, "generate", return_value=raw):
            result = mw.generate_json("system", "user")
            assert result == expected

    def test_invalid_json_raises_runtime_error(self):
        config = AIConfig(provider=AIProvider.OPENAI, api_key="sk-test")
        mw = AIMiddleware(config)

        with patch.object(mw._adapter, "generate", return_value="not json"):
            with pytest.raises(RuntimeError, match="Failed to parse"):
                mw.generate_json("system", "user")


class TestCodeFenceRegex:
    def test_matches_json_fence(self):
        text = '```json\n{"key": "val"}\n```'
        match = _CODE_FENCE_RE.match(text)
        assert match is not None
        assert match.group(1).strip() == '{"key": "val"}'

    def test_matches_plain_fence(self):
        text = '```\n{"key": "val"}\n```'
        match = _CODE_FENCE_RE.match(text)
        assert match is not None
