"""AI middleware with provider adapters for Claude, OpenAI, and Perplexity."""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class AIProvider(Enum):
    CLAUDE = "claude"
    OPENAI = "openai"
    PERPLEXITY = "perplexity"


# Default models per provider
DEFAULT_MODELS: dict[str, str] = {
    AIProvider.CLAUDE.value: "claude-sonnet-4-20250514",
    AIProvider.OPENAI.value: "gpt-4o",
    AIProvider.PERPLEXITY.value: "sonar-pro",
}

# Environment variable names for API keys
API_KEY_ENV_VARS: dict[str, str] = {
    AIProvider.CLAUDE.value: "ANTHROPIC_API_KEY",
    AIProvider.OPENAI.value: "OPENAI_API_KEY",
    AIProvider.PERPLEXITY.value: "PERPLEXITY_API_KEY",
}

MAX_TOKENS = 4096
TEMPERATURE = 0.0


@dataclass
class AIConfig:
    provider: AIProvider
    api_key: str
    model: str = ""

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError(f"API key required for {self.provider.value}")
        if not self.model:
            self.model = DEFAULT_MODELS[self.provider.value]


class BaseAdapter(ABC):
    """Abstract base for AI provider adapters."""

    def __init__(self, config: AIConfig) -> None:
        self.config = config

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        ...


class ClaudeAdapter(BaseAdapter):

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required: pip install anthropic"
            )

        client = anthropic.Anthropic(api_key=self.config.api_key)
        response = client.messages.create(
            model=self.config.model,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text


class OpenAIAdapter(BaseAdapter):

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package required: pip install openai"
            )

        client = openai.OpenAI(api_key=self.config.api_key)
        response = client.chat.completions.create(
            model=self.config.model,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content


class PerplexityAdapter(BaseAdapter):

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package required: pip install openai"
            )

        client = openai.OpenAI(
            api_key=self.config.api_key,
            base_url="https://api.perplexity.ai",
        )
        response = client.chat.completions.create(
            model=self.config.model,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content


_ADAPTER_MAP: dict[AIProvider, type[BaseAdapter]] = {
    AIProvider.CLAUDE: ClaudeAdapter,
    AIProvider.OPENAI: OpenAIAdapter,
    AIProvider.PERPLEXITY: PerplexityAdapter,
}

# Regex to strip markdown code fences (```json ... ``` or ``` ... ```)
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", re.DOTALL)


class AIMiddleware:
    """Main interface for AI-powered text and JSON generation."""

    def __init__(self, config: AIConfig) -> None:
        adapter_cls = _ADAPTER_MAP.get(config.provider)
        if adapter_cls is None:
            raise ValueError(f"Unsupported provider: {config.provider.value}")
        self._adapter = adapter_cls(config)
        self.config = config

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self._adapter.generate(system_prompt, user_prompt)

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Generate a response and parse it as JSON.

        Strips markdown code fences if the model wraps the output.
        """
        raw = self.generate(system_prompt, user_prompt)
        text = raw.strip()

        # Strip code fences if present
        match = _CODE_FENCE_RE.match(text)
        if match:
            text = match.group(1).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Failed to parse AI response as JSON: {exc}\nRaw response:\n{raw}"
            ) from exc

    @classmethod
    def from_env(cls, provider: str = "openai") -> AIMiddleware:
        """Create middleware using API key from environment variables."""
        try:
            ai_provider = AIProvider(provider)
        except ValueError:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Choose from: {', '.join(p.value for p in AIProvider)}"
            )

        env_var = API_KEY_ENV_VARS[ai_provider.value]
        api_key = os.environ.get(env_var)
        if not api_key:
            raise ValueError(
                f"Environment variable {env_var} not set for provider '{provider}'"
            )

        config = AIConfig(provider=ai_provider, api_key=api_key)
        return cls(config)


def get_ai(
    provider: str = "openai", api_key: str | None = None
) -> AIMiddleware:
    """Convenience factory for AIMiddleware."""
    try:
        ai_provider = AIProvider(provider)
    except ValueError:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Choose from: {', '.join(p.value for p in AIProvider)}"
        )

    if api_key is None:
        return AIMiddleware.from_env(provider)

    config = AIConfig(provider=ai_provider, api_key=api_key)
    return AIMiddleware(config)
