"""Model matching and startup selection from ``core/model-resolver.ts``.

Only the two built-in providers have preferred model IDs. Custom providers use
their first configured model, rather than copying the upstream vendor table.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, replace
from typing import Sequence

from pi_agent_core.types import ThinkingLevel
from pi_ai.types import Model

from .defaults import DEFAULT_THINKING_LEVEL
from .model_runtime import ModelRuntime


DEFAULT_MODEL_PER_PROVIDER = {"anthropic": "claude-opus-4-8", "openai": "gpt-5.5"}
_THINKING_LEVELS = {"off", "minimal", "low", "medium", "high", "xhigh", "max"}


@dataclass(frozen=True)
class ScopedModel:
    model: Model
    thinking_level: ThinkingLevel | None = None


@dataclass(frozen=True)
class ModelScopeDiagnostic:
    code: str
    message: str
    pattern: str


@dataclass(frozen=True)
class ModelScopeResult:
    scoped_models: list[ScopedModel]
    diagnostics: list[ModelScopeDiagnostic]


@dataclass(frozen=True)
class ParsedModelResult:
    model: Model | None = None
    thinking_level: ThinkingLevel | None = None
    warning: str | None = None


@dataclass(frozen=True)
class CliModelResult(ParsedModelResult):
    error: str | None = None


@dataclass(frozen=True)
class InitialModelResult:
    model: Model | None
    thinking_level: ThinkingLevel = DEFAULT_THINKING_LEVEL
    fallback_message: str | None = None


def _same_model(left: Model, right: Model) -> bool:
    return left.provider == right.provider and left.id == right.id


def find_exact_model_reference_match(reference: str, models: Sequence[Model]) -> Model | None:
    normalized = reference.strip().lower()
    if not normalized:
        return None
    canonical = [model for model in models if f"{model.provider}/{model.id}".lower() == normalized]
    if canonical:
        return canonical[0] if len(canonical) == 1 else None
    bare = [model for model in models if model.id.lower() == normalized]
    return bare[0] if len(bare) == 1 else None


def _try_match_model(pattern: str, models: Sequence[Model]) -> Model | None:
    exact = find_exact_model_reference_match(pattern, models)
    if exact is not None:
        return exact
    lowered = pattern.lower()
    matches = [model for model in models if lowered in model.id.lower() or lowered in model.name.lower()]
    if not matches:
        return None
    aliases = [model for model in matches if model.id.endswith("-latest") or not re.search(r"-\d{8}$", model.id)]
    return sorted(aliases or matches, key=lambda model: model.id, reverse=True)[0]


def parse_model_pattern(
    pattern: str, models: Sequence[Model], *, allow_invalid_thinking_level_fallback: bool = True,
) -> ParsedModelResult:
    match = _try_match_model(pattern, models)
    if match is not None:
        return ParsedModelResult(model=match)
    if ":" not in pattern:
        return ParsedModelResult()
    prefix, suffix = pattern.rsplit(":", 1)
    if suffix in _THINKING_LEVELS:
        result = parse_model_pattern(prefix, models, allow_invalid_thinking_level_fallback=allow_invalid_thinking_level_fallback)
        if result.model is not None and result.warning is None:
            return replace(result, thinking_level=suffix)
        return result
    if not allow_invalid_thinking_level_fallback:
        return ParsedModelResult()
    result = parse_model_pattern(prefix, models)
    if result.model is not None:
        return replace(result, thinking_level=None, warning=f'Invalid thinking level "{suffix}" in pattern "{pattern}". Using default instead.')
    return result


def resolve_model_scope_from_models(patterns: Sequence[str], models: Sequence[Model]) -> ModelScopeResult:
    scoped: list[ScopedModel] = []
    diagnostics: list[ModelScopeDiagnostic] = []
    for pattern in patterns:
        if any(char in pattern for char in "*?["):
            glob, _, suffix = pattern.rpartition(":")
            level = suffix if suffix in _THINKING_LEVELS else None
            glob = glob if level else pattern
            matches = [
                model for model in models
                if fnmatch.fnmatchcase(f"{model.provider}/{model.id}".lower(), glob.lower())
                or fnmatch.fnmatchcase(model.id.lower(), glob.lower())
            ]
            results = [ScopedModel(model, level) for model in matches]
        else:
            parsed = parse_model_pattern(pattern, models)
            if parsed.warning:
                diagnostics.append(ModelScopeDiagnostic("invalid-thinking-level", parsed.warning, pattern))
            results = [ScopedModel(parsed.model, parsed.thinking_level)] if parsed.model else []
        if not results:
            diagnostics.append(ModelScopeDiagnostic("no-match", f'No models match pattern "{pattern}"', pattern))
        for item in results:
            if not any(_same_model(item.model, existing.model) for existing in scoped):
                scoped.append(item)
    return ModelScopeResult(scoped, diagnostics)


async def resolve_model_scope(patterns: Sequence[str], runtime: ModelRuntime) -> ModelScopeResult:
    return resolve_model_scope_from_models(patterns, await runtime.get_available())


def _preferred(models: Sequence[Model]) -> Model | None:
    for provider, model_id in DEFAULT_MODEL_PER_PROVIDER.items():
        found = next((model for model in models if model.provider == provider and model.id == model_id), None)
        if found is not None:
            return found
    return models[0] if models else None


async def resolve_cli_model(
    runtime: ModelRuntime, cli_model: str | None, cli_provider: str | None = None,
    cli_thinking: ThinkingLevel | None = None,
) -> CliModelResult:
    if not cli_model:
        return CliModelResult()
    models = list(runtime.get_models())
    if not models:
        return CliModelResult(error="No models available. Check your installation or add models to models.json.")
    provider_names = {model.provider.lower(): model.provider for model in models}
    provider = provider_names.get(cli_provider.lower()) if cli_provider else None
    if cli_provider and provider is None:
        return CliModelResult(error=f'Unknown provider "{cli_provider}". Use --list-models to see available providers/models.')
    pattern = cli_model
    inferred = False
    if provider is None and "/" in cli_model:
        prefix, rest = cli_model.split("/", 1)
        if prefix.lower() in provider_names:
            provider, pattern, inferred = provider_names[prefix.lower()], rest, True
    if provider is None:
        exact = [model for model in models if model.id.lower() == cli_model.lower() or f"{model.provider}/{model.id}".lower() == cli_model.lower()]
        if len(exact) == 1:
            return CliModelResult(model=exact[0])
        if len(exact) > 1:
            authenticated = [model for model in exact if await runtime.is_configured(model.provider)]
            if len(authenticated) == 1:
                return CliModelResult(model=authenticated[0])
            choices = ", ".join(sorted(f"{model.provider}/{model.id}" for model in exact))
            hint = "No matching provider is authenticated." if not authenticated else "More than one matching provider is authenticated."
            return CliModelResult(error=f'Model "{cli_model}" is ambiguous across providers: {choices}. {hint} Use --provider or provider/model.')
    if cli_provider and provider and cli_model.lower().startswith(f"{provider}/".lower()):
        pattern = cli_model[len(provider) + 1:]
    candidates = [model for model in models if model.provider == provider] if provider else models
    parsed = parse_model_pattern(pattern, candidates, allow_invalid_thinking_level_fallback=False)
    if parsed.model is not None:
        if inferred and not await runtime.is_configured(parsed.model.provider):
            raw = [model for model in models if model.id.lower() == cli_model.lower() and not _same_model(model, parsed.model)]
            authenticated = [model for model in raw if await runtime.is_configured(model.provider)]
            if len(authenticated) == 1:
                return CliModelResult(model=authenticated[0])
        return CliModelResult(parsed.model, parsed.thinking_level, parsed.warning)
    if inferred:
        fallback = parse_model_pattern(cli_model, models, allow_invalid_thinking_level_fallback=False)
        if fallback.model is not None:
            return CliModelResult(fallback.model, fallback.thinking_level, fallback.warning)
    if provider is not None:
        fallback_pattern = pattern
        fallback_thinking = None
        if cli_thinking is None and ":" in pattern:
            prefix, suffix = pattern.rsplit(":", 1)
            if suffix in _THINKING_LEVELS:
                fallback_pattern, fallback_thinking = prefix, suffix
        base = _preferred(candidates)
        if base is not None:
            fallback_model = replace(base, id=fallback_pattern, name=fallback_pattern,
                                     reasoning=True if (cli_thinking or fallback_thinking) not in (None, "off") else base.reasoning)
            return CliModelResult(fallback_model, fallback_thinking,
                                  f'Model "{fallback_pattern}" not found for provider "{provider}". Using custom model id.')
    display = f"{provider}/{pattern}" if provider else cli_model
    return CliModelResult(error=f'Model "{display}" not found. Use --list-models to see available models.')


async def find_initial_model(
    runtime: ModelRuntime, *, cli_provider: str | None = None, cli_model: str | None = None,
    scoped_models: Sequence[ScopedModel] = (), is_continuing: bool = False,
    default_provider: str | None = None, default_model_id: str | None = None,
    default_thinking_level: ThinkingLevel | None = None,
    model_thinking_levels: dict[str, ThinkingLevel] | None = None,
) -> InitialModelResult:
    if cli_provider and cli_model:
        resolved = await resolve_cli_model(runtime, cli_model, cli_provider)
        if resolved.error:
            raise ValueError(resolved.error)
        if resolved.model:
            return InitialModelResult(resolved.model)
    if scoped_models and not is_continuing:
        scoped = scoped_models[0]
        level = scoped.thinking_level or (model_thinking_levels or {}).get(f"{scoped.model.provider}/{scoped.model.id}") or default_thinking_level or DEFAULT_THINKING_LEVEL
        return InitialModelResult(scoped.model, level)
    if default_provider and default_model_id:
        model = runtime.get_model(default_provider, default_model_id)
        if model and await runtime.is_configured(default_provider):
            level = (model_thinking_levels or {}).get(f"{default_provider}/{default_model_id}") or default_thinking_level or DEFAULT_THINKING_LEVEL
            return InitialModelResult(model, level)
    return InitialModelResult(_preferred(runtime.get_available_snapshot()))


async def restore_model_from_session(
    provider: str, model_id: str, current: Model | None, runtime: ModelRuntime,
) -> tuple[Model | None, str | None]:
    restored = runtime.get_model(provider, model_id)
    if restored and await runtime.is_configured(provider):
        return restored, None
    reason = "model no longer exists" if restored is None else "no auth configured"
    fallback = current or _preferred(runtime.get_available_snapshot())
    message = f"Could not restore model {provider}/{model_id} ({reason})."
    if fallback:
        message += f" Using {fallback.provider}/{fallback.id}."
    return fallback, message


__all__ = [
    "DEFAULT_MODEL_PER_PROVIDER", "ScopedModel", "ModelScopeDiagnostic", "ModelScopeResult",
    "ParsedModelResult", "CliModelResult", "InitialModelResult",
    "find_exact_model_reference_match", "parse_model_pattern", "resolve_model_scope_from_models",
    "resolve_model_scope", "resolve_cli_model", "find_initial_model", "restore_model_from_session",
]
