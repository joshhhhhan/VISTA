"""Model providers behind the Claude Code transport.

Claude Code is the transport; the model it talks to is a separate choice. The
native provider is Anthropic through a subscription token. An
Anthropic-compatible provider is any server that implements ``/v1/messages``
and is selected purely through the environment variables Claude Code already
honors. Nothing here changes how
the transport is driven, only where its requests go.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClaudeCodeProvider:
    name: str
    credential_env: str
    endpoint_protocol: str
    default_model: str
    default_context_window: int
    default_base_url: str | None = None
    default_effort: str | None = None
    model_context_windows: tuple[tuple[str, int], ...] = ()
    # None leaves the transport's default unchanged.
    auto_compact_window: int | None = None
    max_output_tokens: int | None = None
    mcp_tool_timeout_ms: int | None = None

    def __post_init__(self) -> None:
        if self.endpoint_protocol not in {"anthropic-native", "anthropic-compatible"}:
            raise ValueError("Unknown Claude Code endpoint protocol")
        if not self.name or not self.credential_env or not self.default_model:
            raise ValueError("A provider needs a name, credential variable and model")

    @property
    def native(self) -> bool:
        return self.endpoint_protocol == "anthropic-native"

    def context_window(self, model: str) -> int:
        return dict(self.model_context_windows).get(model, self.default_context_window)

    def supports_model(self, model: str) -> bool:
        if not self.model_context_windows:
            return True
        return model in dict(self.model_context_windows)


ANTHROPIC_PROVIDER = ClaudeCodeProvider(
    name="anthropic",
    credential_env="CLAUDE_CODE_OAUTH_TOKEN",
    endpoint_protocol="anthropic-native",
    default_model="opus",
    default_context_window=1_000_000,
)

DEEPSEEK_PROVIDER = ClaudeCodeProvider(
    name="deepseek",
    credential_env="ANTHROPIC_AUTH_TOKEN",
    endpoint_protocol="anthropic-compatible",
    default_model="deepseek-flash[1m]",
    default_context_window=786_432,
    default_base_url="https://api.deepseek.com/anthropic",
    default_effort="max",
    auto_compact_window=786_432,
)

# Removed before applying provider settings. Model aliases are also cleared
# by their ANTHROPIC_DEFAULT_ prefix in provider_process_environment.
PROVIDER_ENV_NAMES = (
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL_NAME",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL_NAME",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL_NAME",
    "CLAUDE_CODE_SUBAGENT_MODEL",
    "CLAUDE_CODE_EFFORT_LEVEL",
    "CLAUDE_CODE_MAX_CONTEXT_TOKENS",
    "CLAUDE_CODE_AUTO_COMPACT_WINDOW",
    "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
    "MCP_TOOL_TIMEOUT",
    "ENABLE_TOOL_SEARCH",
)


def provider_environment(
    provider: ClaudeCodeProvider,
    *,
    credential: str,
    model: str,
    effort: str,
    base_url: str | None = None,
) -> dict[str, str]:
    """The environment variables that select this provider, and nothing else."""
    if not credential:
        raise ValueError(f"{provider.credential_env} must not be empty")
    if provider.native:
        return {"CLAUDE_CODE_OAUTH_TOKEN": credential}
    if not provider.supports_model(model):
        raise ValueError(f"Unsupported {provider.name} model: {model}")
    endpoint = (base_url or provider.default_base_url or "").rstrip("/")
    if not endpoint:
        raise ValueError(
            f"The {provider.name} provider requires an Anthropic-compatible base URL"
        )
    # Claude Code resolves its own aliases (opus/sonnet/haiku and the subagent
    # model) independently; every one is pinned so no request can escape to a
    # model the server does not serve.
    return {
        "ANTHROPIC_BASE_URL": endpoint,
        "ANTHROPIC_AUTH_TOKEN": credential,
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "CLAUDE_CODE_SUBAGENT_MODEL": model,
        "CLAUDE_CODE_EFFORT_LEVEL": effort,
    }
