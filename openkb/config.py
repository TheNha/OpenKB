from __future__ import annotations

import contextlib
import logging
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml

# Process-wide LLM runtime settings (extra_headers/extra_body/timeout/
# parallel_tool_calls + resolve_model_settings) live in openkb/llm_runtime.py
# (self-contained, no dependency on the rest of this module — split out to
# stay under the file-size limit, see docs/golden-principles.md#file-size).
# Re-exported here so existing ``from openkb.config import get_extra_headers``
# (etc.) call sites keep working unchanged.
from openkb.llm_runtime import (  # noqa: F401 (re-export for backward compat)
    get_extra_body,
    get_extra_headers,
    get_ingest_temperature,
    get_parallel_tool_calls,
    get_query_temperature,
    get_timeout,
    get_timeout_extra_args,
    resolve_model_settings,
    resolve_temperature,
    set_extra_body,
    set_extra_headers,
    set_ingest_temperature,
    set_parallel_tool_calls,
    set_query_temperature,
    set_timeout,
)
from openkb.locks import atomic_write_text, flock, funlock

logger = logging.getLogger(__name__)

# Default entity-type vocabulary. Overridable globally / per-KB via the optional
# ``entity_types:`` config key (see ``resolve_entity_types``).
DEFAULT_ENTITY_TYPES: tuple[str, ...] = (
    "person",
    "organization",
    "place",
    "product",
    "work",
    "event",
    "other",
)

DEFAULT_CONFIG: dict[str, Any] = {
    "model": "gpt-5.4",
    "language": "en",
    "pageindex_threshold": 20,
    # A GLOBAL_SCALAR_KEY like the three above, so the merged `effective` dict
    # always carries it (the layering/`sources` logic is type-agnostic). A
    # global/KB list overrides it wholesale; resolve_entity_types cleans the
    # effective value on read.
    "entity_types": list(DEFAULT_ENTITY_TYPES),
}

# Default ``~/.config/openkb``; override with env ``OPENKB_CONFIG_DIR`` to move
# global.yaml / the global .env / the default KB root out of $HOME entirely
# (e.g. into a project checkout). Read once at import — set it before the
# process starts (shell export, not mid-session).
_configured_config_dir = os.environ.get("OPENKB_CONFIG_DIR")
GLOBAL_CONFIG_DIR = (
    Path(_configured_config_dir).expanduser().resolve()
    if _configured_config_dir
    else Path.home() / ".config" / "openkb"
)
GLOBAL_CONFIG_PATH = GLOBAL_CONFIG_DIR / "global.yaml"
GLOBAL_CONFIG_LOCK_PATH = GLOBAL_CONFIG_DIR / "global.lock"

# Portable KB names for the REST API: letters (any script, incl. CJK), numbers,
# underscores, hyphens — but NO spaces, dots, or path separators. Lets clients
# address a knowledge base by a filesystem-safe short name (``kb``) instead of an
# absolute path, resolved via kb_aliases/known_kbs/OPENKB_KB_ROOT. `\w` is Unicode
# for str patterns in Python 3, so a CJK-named KB (e.g. 笔记) is a valid name.
KB_NAME_RE = re.compile(r"[\w-]+")


@contextlib.contextmanager
def _with_global_config_lock() -> Iterator[None]:
    GLOBAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    # Derive the lock path from GLOBAL_CONFIG_DIR (rather than the import-time
    # GLOBAL_CONFIG_LOCK_PATH constant) so it always co-locates with the dir we
    # just created — including when tests monkeypatch GLOBAL_CONFIG_DIR to a
    # throwaway path. Otherwise the lock would open the real ~/.config/openkb,
    # which fails on a fresh machine where that dir doesn't exist.
    lock_path = GLOBAL_CONFIG_DIR / "global.lock"
    with lock_path.open("a+", encoding="utf-8") as fh:
        flock(fh, exclusive=True)
        try:
            yield
        finally:
            funlock(fh)


def _atomic_yaml_dump(path: Path, config: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        yaml.safe_dump(config, allow_unicode=True, sort_keys=True),
    )


def _load_global_config_unlocked() -> dict[str, Any]:
    """Load global.yaml, returning ``{}`` when absent, empty, or malformed.

    The coercion of a non-mapping document (a bare list/scalar from a
    hand-edited global.yaml) to ``{}`` is sunk HERE — the single load choke
    point — so every reader (:func:`kb_root_dir`, :func:`registered_kbs`,
    :func:`resolve_kb_alias`, :func:`resolve_effective_config`, and the config
    PATCH in ``api_config``) can treat the result as a dict without repeating an
    ``isinstance`` guard, and a malformed file degrades to defaults on every
    command's hot path instead of raising ``AttributeError``.
    """
    if GLOBAL_CONFIG_PATH.exists():
        with GLOBAL_CONFIG_PATH.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if isinstance(data, dict):
            return data
    return {}


def resolve_entity_types(config: dict, *, warn: bool = True) -> list[str]:
    """Resolve the effective entity-type list from a loaded config dict.

    If ``config["entity_types"]`` is a non-empty list, each string item is
    cleaned (lowercased, trimmed, restricted to ``[a-z0-9 _-]`` so a stray
    brace/punctuation can't leak into a prompt template or frontmatter value);
    non-string items (YAML nulls, numbers) are skipped. The cleaned list is
    de-duped (order preserving) and ``"other"`` is always appended when missing
    (it is the coercion fallback). Otherwise — key absent, not a list, empty,
    or fully malformed — :data:`DEFAULT_ENTITY_TYPES` is returned, so behavior
    is byte-identical to the default.

    ``warn`` logs a warning when ``entity_types`` is present-but-malformed. It is
    ``True`` for the compile path (a real config problem the author should see)
    and passed ``False`` by the config-READ endpoints, which run on every GET and
    must not spam the log for a malformed value.
    """
    raw = config.get("entity_types")
    if raw is None:
        return list(DEFAULT_ENTITY_TYPES)
    if not isinstance(raw, list):
        if warn:
            logger.warning(
                "config: 'entity_types' must be a list of strings, got %s — "
                "falling back to the default entity types.",
                type(raw).__name__,
            )
        return list(DEFAULT_ENTITY_TYPES)
    cleaned: list[str] = []
    for x in raw:
        if not isinstance(x, str):
            continue  # skip YAML nulls/numbers (str(None) would become "none")
        s = re.sub(r"[^a-z0-9 _-]+", "", x.strip().lower()).strip()
        if s and s not in cleaned:
            cleaned.append(s)
    if not cleaned:
        if warn:
            logger.warning(
                "config: 'entity_types' was present but yielded no usable values — "
                "falling back to the default entity types.",
            )
        return list(DEFAULT_ENTITY_TYPES)
    if "other" not in cleaned:
        cleaned.append("other")
    return cleaned


def resolve_extra_headers(config: dict) -> dict[str, str]:
    """Resolve the optional ``extra_headers:`` config key into a str→str dict.

    Some LiteLLM providers need extra HTTP headers on every request (e.g.
    GitHub Copilot's ``Editor-Version`` IDE-auth headers). Users opt in via
    an ``extra_headers:`` mapping in config.yaml; the result is forwarded to
    LiteLLM's ``extra_headers`` parameter on all LLM calls.

    Values are stringified (YAML may parse version-like values as numbers).
    Entries with a non-string/empty key or a non-scalar value are skipped.
    A non-mapping ``extra_headers`` is ignored entirely. Warnings are logged
    only when the key was present but malformed.
    """
    raw = config.get("extra_headers")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        logger.warning(
            "config: 'extra_headers' must be a mapping of header name to "
            "value, got %s — ignoring it.",
            type(raw).__name__,
        )
        return {}
    headers: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.strip():
            logger.warning(
                "config: skipping 'extra_headers' entry with non-string or empty key: %r",
                key,
            )
            continue
        if value is None or not isinstance(value, (str, int, float, bool)):
            logger.warning(
                "config: skipping 'extra_headers' entry %r with non-scalar value: %r",
                key,
                value,
            )
            continue
        headers[key.strip()] = str(value)
    return headers


def resolve_extra_body(config: dict) -> dict[str, Any]:
    """Resolve the optional ``extra_body:`` mapping, forwarded verbatim as
    LiteLLM's ``extra_body`` on every LLM call (e.g. vLLM/SGLang's
    ``chat_template_kwargs: {enable_thinking: false}`` for Qwen3). Values may
    be nested JSON, unlike ``extra_headers``. Non-mapping is ignored, warned.
    """
    raw = config.get("extra_body")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        logger.warning(
            "config: 'extra_body' must be a mapping, got %s — ignoring it.",
            type(raw).__name__,
        )
        return {}
    return dict(raw)


def resolve_parallel_tool_calls(config: dict) -> tuple[bool | None, bool]:
    """Resolve the optional ``parallel_tool_calls:`` key to ``(value, was_explicit)``.

    Absent → ``(None, False)`` so each agent applies its own default (see
    ``resolve_model_settings``). ``true``/``false`` → that bool; explicit
    ``null`` → ``None`` (omit — the Amazon Bedrock #175 escape hatch); both with
    ``was_explicit=True`` so they override every agent uniformly. An invalid
    value degrades to omit (never breaks a provider), with a warning.
    """
    if "parallel_tool_calls" not in config:
        return None, False
    value = config["parallel_tool_calls"]
    if value is None:
        return None, True
    if isinstance(value, bool):
        return value, True
    logger.warning(
        "config: 'parallel_tool_calls' must be true, false, or null, got %r "
        "— omitting the setting.",
        value,
    )
    return None, True


def resolve_timeout(config: dict) -> float | None:
    """Resolve the optional ``timeout:`` key to a finite positive number of seconds.

    Returns ``None`` (use LiteLLM's default) when absent or invalid; rejects
    bools and ``nan``/``inf``, warning when present but unusable.
    """
    raw = config.get("timeout")
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        logger.warning(
            "config: 'timeout' must be a positive number of seconds, got %s — ignoring it.",
            type(raw).__name__,
        )
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "config: 'timeout' must be a positive number of seconds, got %r — ignoring it.",
            raw,
        )
        return None
    if not math.isfinite(value) or value <= 0:
        logger.warning(
            "config: 'timeout' must be a finite positive number of seconds, got %s — ignoring it.",
            value,
        )
        return None
    return value


def resolve_concurrency(config: dict) -> int | None:
    """Resolve the optional ``concurrency:`` key — the cap on concurrent LLM
    calls OpenKB makes during ingest (PageIndex indexing and concept/entity
    compilation alike; they never run at the same time for one document, so
    one setting covers both).

    Returns ``None`` when absent, explicitly ``null``, or invalid (rejecting
    bools and non-positive values with a warning when present but unusable —
    an explicit ``null``/absent key is the normal "unset" case and stays
    silent, matching ``resolve_timeout``). Callers apply their own default (or
    omit the setting entirely) when this returns ``None``.
    """
    value = config.get("concurrency")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        logger.warning(
            "config: 'concurrency' must be a positive integer, got %r — ignoring it.",
            value,
        )
        return None
    return value


def resolve_effective_concurrency(config: dict) -> int | None:
    """``resolve_concurrency`` plus a global.yaml fallback for KBs that don't
    set their own — same rationale as extra_body/temperature: ``concurrency``
    isn't in ``GLOBAL_SCALAR_KEYS``, so ``resolve_effective_config`` doesn't
    layer it in from global.yaml on its own. ``or`` is safe here (unlike
    temperature): ``resolve_concurrency`` never returns 0, only a positive
    int or ``None``.
    """
    return resolve_concurrency(config) or resolve_concurrency(load_global_config())


def resolve_litellm_settings(config: dict) -> dict[str, Any]:
    """Resolve the optional ``litellm:`` mapping of LiteLLM module settings.

    Values are forwarded verbatim (the user owns them); only the container shape
    is enforced — returns ``{}`` if absent or not a mapping, and drops non-string
    keys. ``cli._apply_litellm_settings`` applies them.
    """
    raw = config.get("litellm")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        logger.warning(
            "config: 'litellm' must be a mapping of LiteLLM settings, got %s — ignoring it.",
            type(raw).__name__,
        )
        return {}
    settings: dict[str, Any] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            logger.warning("config: skipping 'litellm' entry with non-string key %r.", key)
            continue
        settings[key] = value
    return settings


# Process-wide extra headers for LLM requests, resolved from the active KB's
# config by the CLI entry points (cli._setup_llm_key). LLM call sites read it
# via get_extra_headers() so the value doesn't have to be threaded through
# every compile/agent call chain — mirroring how the API key is applied
# globally via litellm.api_key / provider env vars.


# Shared by cli._setup_llm_key and resolve_credential_bundle so the CLI and
# REST paths apply identical litellm.* override semantics.
def resolve_per_request_overrides(
    config: dict[str, Any],
) -> tuple[dict[str, str], float | None, dict[str, Any], dict[str, Any]]:
    """Resolve extra_headers / timeout / extra_body / litellm_settings.

    ``litellm.extra_headers`` / ``litellm.timeout`` / ``litellm.extra_body``
    override their top-level counterparts (legacy precedence) and are popped
    from ``litellm_settings`` so they aren't also applied as process-wide
    litellm module settings. ``ingest_temperature`` / ``query_temperature``
    are resolved separately by callers, like ``parallel_tool_calls``.
    """
    extra_headers = resolve_extra_headers(config)
    timeout = resolve_timeout(config)
    extra_body = resolve_extra_body(config)
    litellm_settings = resolve_litellm_settings(config)
    if "extra_headers" in litellm_settings:
        extra_headers = resolve_extra_headers(
            {"extra_headers": litellm_settings.pop("extra_headers")}
        )
    if "timeout" in litellm_settings:
        timeout = resolve_timeout({"timeout": litellm_settings.pop("timeout")})
    if "extra_body" in litellm_settings:
        extra_body = resolve_extra_body({"extra_body": litellm_settings.pop("extra_body")})
    return extra_headers, timeout, extra_body, litellm_settings


@dataclass(frozen=True)
class LlmCredentialBundle:
    """Immutable per-request LLM credential + config bundle.

    Resolved once from a KB's ``.env`` and ``config.yaml`` via
    :func:`resolve_credential_bundle`, then threaded through to LLM call sites
    so concurrent requests on a shared event-loop thread never see each other's
    key/headers/timeout (unlike the process-wide globals in
    :func:`set_extra_headers` / :func:`set_timeout`).
    """

    api_key: str | None = None
    base_url: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    timeout: float | None = None
    ingest_temperature: float | None = None
    query_temperature: float | None = None
    parallel_tool_calls: bool | None = None
    parallel_tool_calls_explicit: bool = False


def resolve_credential_bundle(kb_dir: Path) -> LlmCredentialBundle:
    """Build an :class:`LlmCredentialBundle` from a KB's config, purely.

    Resolves ``LLM_API_KEY`` / ``OPENAI_API_BASE`` with per-KB precedence — the
    KB's own ``.env`` first, then the process environment, then the global
    ``~/.config/openkb/.env`` — and reads the KB's ``config.yaml`` for
    ``extra_headers`` / ``extra_body`` / ``timeout`` / ``litellm`` (``extra_body``
    additionally falls back to global.yaml's own ``extra_body`` when the KB
    doesn't set one). This honors the same sources as ``cli._setup_llm_key``
    (so a server configured via a process env var or the global ``.env`` keeps
    working) but is side-effect-free: ``os.environ`` is never written, so
    concurrent requests for different KBs cannot see each other's key.
    """
    kb_values: dict[str, str | None] = {}
    kb_env = kb_dir / ".env"
    if kb_env.exists():
        from dotenv import dotenv_values

        kb_values = dict(dotenv_values(str(kb_env)))

    global_values: dict[str, str | None] = {}
    global_env = GLOBAL_CONFIG_DIR / ".env"
    if global_env.exists():
        from dotenv import dotenv_values

        global_values = dict(dotenv_values(str(global_env)))

    def _resolve_env(key: str) -> str | None:
        # KB-local .env wins, then the process environment, then the global
        # .env; empty values are treated as unset so they fall through.
        return kb_values.get(key) or os.environ.get(key) or global_values.get(key) or None

    api_key = _resolve_env("LLM_API_KEY")
    base_url = _resolve_env("OPENAI_API_BASE")

    # resolve_effective_config handles a missing KB config.yaml internally, so
    # no existence check is needed here (matches cli._setup_llm_key's use of
    # the same call). extra_headers/timeout/extra_body/parallel_tool_calls are
    # NOT in GLOBAL_SCALAR_KEYS, so this only pulls in the KB's own config.yaml
    # for them (verbatim, same as the old load_config(config_path) call).
    config = resolve_effective_config(kb_dir)[0]
    # Shared resolver so CLI and REST apply identical litellm.* override
    # semantics. litellm module-level settings (drop_params etc.) are
    # process globals and intentionally not carried per-request: the REST
    # server runs multiple KBs in one process, so they cannot be isolated.
    extra_headers, timeout, extra_body, _ = resolve_per_request_overrides(config)
    if not extra_body:
        # extra_body has no typed REST config-management surface (unlike
        # model/language/etc — see GLOBAL_SCALAR_KEYS), so it doesn't layer
        # through resolve_effective_config. Fall back to global.yaml's own
        # extra_body directly: a KB-set value always wins, letting one global
        # extra_body (e.g. Qwen3's enable_thinking toggle) apply to every KB
        # that doesn't override it.
        extra_body = resolve_extra_body(load_global_config())

    def _resolve_temp(key: str) -> float | None:
        # Same KB-wins-over-global fallback as extra_body above. Explicit
        # `is None` (not `or`) since 0 is a valid, falsy temperature.
        value = resolve_temperature(config, key)
        return value if value is not None else resolve_temperature(load_global_config(), key)

    ingest_temperature = _resolve_temp("ingest_temperature")
    query_temperature = _resolve_temp("query_temperature")
    parallel_tool_calls, parallel_tool_calls_explicit = resolve_parallel_tool_calls(config)

    return LlmCredentialBundle(
        api_key=api_key,
        base_url=base_url,
        extra_headers=extra_headers,
        extra_body=extra_body,
        timeout=timeout,
        ingest_temperature=ingest_temperature,
        query_temperature=query_temperature,
        parallel_tool_calls=parallel_tool_calls,
        parallel_tool_calls_explicit=parallel_tool_calls_explicit,
    )


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config from config_path, merged with DEFAULT_CONFIG.

    If the file does not exist, returns a copy of the defaults.
    """
    config = dict(DEFAULT_CONFIG)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        config.update(data)
    return config


def save_config(config_path: Path, config: dict) -> None:
    """Persist config dict to YAML, creating parent directories as needed."""
    _atomic_yaml_dump(config_path, config)


def load_global_config() -> dict[str, Any]:
    """Load the global config from ~/.config/openkb/global.yaml."""
    return _load_global_config_unlocked()


# Whitelisted scalar keys that global.yaml may provide as shared defaults for
# every KB. Non-scalar global keys (known_kbs, kb_aliases, default_kb) are NOT
# config and must never leak into a KB's effective runtime config.
# Config keys that layer global.yaml -> KB config.yaml with per-key `sources`
# tracking (a KB explicit null = inherit). Mostly scalars; `entity_types` is the
# one list-valued member — the layering rule (a non-null value wins over the
# layer below) is type-agnostic, so a KB list overrides the global list wholesale.
#
# `extra_body` is deliberately NOT here: it isn't part of the typed
# config-management REST API (GlobalConfigResponse / _KbConfigWritable), so
# adding it would trip test_global_scalar_keys_match_api_writable_keys without
# a matching PATCH/GET surface. It still gets a KB-wins-over-global fallback —
# see resolve_credential_bundle / cli._setup_llm_key, which read global.yaml's
# extra_body directly when the KB doesn't set one.
GLOBAL_SCALAR_KEYS: tuple[str, ...] = (
    "model",
    "language",
    "pageindex_threshold",
    "entity_types",
)


def resolve_effective_config(kb_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Layer DEFAULT_CONFIG -> global.yaml (whitelisted scalars) -> KB config.yaml.

    Returns ``(effective, sources)``. ``effective`` is the fully merged config
    dict — a drop-in superset of :func:`load_config`'s result: it carries every
    KB key plus the global-defaulted scalars. ``sources`` maps each key in
    :data:`GLOBAL_SCALAR_KEYS` to the layer that supplied its effective value:
    ``'kb'`` (KB config.yaml set it), ``'global'`` (global.yaml set it and the
    KB did not), or ``'default'`` (neither did, so DEFAULT_CONFIG stands).

    Precedence is the standard one: a KB-set scalar wins over global, which wins
    over the default. A scalar written as explicit ``null`` in the KB config
    means "inherit" (matches the RFC 7386 revert semantics of the config PATCH)
    and does NOT clobber the layer below. Non-scalar keys come only from the KB
    file and are copied verbatim (including a meaningful ``null`` such as
    ``parallel_tool_calls: null``).
    """
    effective: dict[str, Any] = dict(DEFAULT_CONFIG)
    sources: dict[str, str] = {key: "default" for key in GLOBAL_SCALAR_KEYS}

    # A malformed global.yaml (bare list/scalar) is coerced to {} inside the
    # loader (_load_global_config_unlocked), so no guard is needed here.
    global_config = load_global_config()
    for key in GLOBAL_SCALAR_KEYS:
        value = global_config.get(key)
        if value is not None:
            effective[key] = value
            sources[key] = "global"

    kb_path = kb_dir / ".openkb" / "config.yaml"
    if kb_path.exists():
        with kb_path.open("r", encoding="utf-8") as fh:
            kb_config = yaml.safe_load(fh) or {}
        # Same defensive coercion for a malformed KB config.yaml (list/scalar):
        # .items() below would otherwise raise and break this single KB.
        if not isinstance(kb_config, dict):
            kb_config = {}
        for key, value in kb_config.items():
            # A GLOBAL_SCALAR_KEYS value explicitly nulled — or an empty
            # entity_types list, since an empty vocabulary is meaningless —
            # means "inherit": don't clobber the global/default layer, and keep
            # `sources` reporting the layer that actually supplied the value. The
            # gate is on GLOBAL_SCALAR_KEYS so non-scalar nulls (parallel_tool_calls) stay.
            if key in GLOBAL_SCALAR_KEYS and (value is None or value == []):
                continue
            effective[key] = value
            if key in GLOBAL_SCALAR_KEYS:
                sources[key] = "kb"

    return effective, sources


def save_global_config(config: dict[str, Any]) -> None:
    """Save the global config to ~/.config/openkb/global.yaml."""
    with _with_global_config_lock():
        _atomic_yaml_dump(GLOBAL_CONFIG_PATH, config)


def register_kb(kb_path: Path) -> None:
    """Register a KB path in the global config's known_kbs list."""
    with _with_global_config_lock():
        gc = _load_global_config_unlocked()
        known = gc.get("known_kbs", [])
        resolved = str(kb_path.resolve())
        if resolved not in known:
            known.append(resolved)
            gc["known_kbs"] = known
        gc["default_kb"] = resolved
        _atomic_yaml_dump(GLOBAL_CONFIG_PATH, gc)


def kb_root_dir() -> Path:
    """Return the root directory for API-addressable KB names.

    Precedence: env ``OPENKB_KB_ROOT`` > global.yaml ``kb_root`` > the default
    ``<GLOBAL_CONFIG_DIR>/kbs``. The env var stays the top override (a deployer
    pins it per-process); a ``kb_root`` string in global.yaml lets the UI move
    the root persistently. Each candidate is ``expanduser().resolve()``d so
    ``~`` and relative segments normalize. A malformed global.yaml (coerced to
    ``{}`` by the loader) or a blank/non-string ``kb_root`` degrades to the
    default.
    """
    configured_root = os.environ.get("OPENKB_KB_ROOT")
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    global_root = _load_global_config_unlocked().get("kb_root")
    if isinstance(global_root, str) and global_root.strip():
        return Path(global_root).expanduser().resolve()
    return (GLOBAL_CONFIG_DIR / "kbs").resolve()


def _is_kb_dir(kb_dir: Path) -> bool:
    """Whether ``kb_dir`` is a KB directory (has both ``.openkb`` and ``wiki``).

    Mirrors ``openkb.api_helpers._is_kb_dir``; duplicated here (rather than
    imported) so this low-level config module — used by the CLI — does not pull
    in the FastAPI web stack, and to avoid an import cycle (``api_helpers``
    imports this module). Used to recognize ``kb_root_dir()`` children so a
    root-level KB is never shadowed by a same-named off-root registry entry.
    """
    return (kb_dir / ".openkb").is_dir() and (kb_dir / "wiki").is_dir()


def _root_child_names() -> set[str]:
    """Directory names of ``kb_root_dir()`` children that look like a KB.

    These names are reserved in the unified name→path registry: a root-level KB
    wins a basename tie over a bare ``known_kbs`` entry, so an off-root KB with a
    colliding basename can neither shadow it (:func:`resolve_kb_alias`) nor
    duplicate its card (:func:`registered_kbs` / ``_list_knowledge_bases``).
    """
    root = kb_root_dir()
    if not root.is_dir():
        return set()
    return {child.name for child in root.iterdir() if child.is_dir() and _is_kb_dir(child)}


def resolve_init_kb_dir(kb_name: str, path: str | None = None) -> Path:
    """Resolve the target directory for a newly-initialized KB.

    Default: ``kb_root_dir() / kb_name``. When ``path`` is given it must be an
    ABSOLUTE directory (a relative path raises ``ValueError`` — the REST init
    endpoint maps that to a 400); it is used verbatim after ``expanduser()`` +
    ``resolve()``. Any absolute location is allowed, matching the CLI's
    cwd-based init (OpenKB is a local-first tool). The caller still registers
    the alias (``register_kb_alias``) so name→path resolution keeps working for
    a custom-path KB.
    """
    if path is not None:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            raise ValueError("KB path must be an absolute directory")
        return candidate.resolve()
    return (kb_root_dir() / kb_name).resolve()


def validate_kb_name(name: str) -> str:
    """Validate and return a portable KB name."""
    normalized = name.strip()
    if not normalized or not KB_NAME_RE.fullmatch(normalized):
        raise ValueError(
            "KB name may contain only letters, numbers, underscores, and hyphens "
            "(no spaces, dots, or path separators)"
        )
    return normalized


def register_kb_alias(name: str, kb_path: Path) -> None:
    """Register a portable KB name alias for a KB path.

    Held under the global-config lock so concurrent API writes (uvicorn
    threadpool) can't lose an alias to a read-modify-write race.
    """
    alias = validate_kb_name(name)
    resolved = str(kb_path.resolve())
    with _with_global_config_lock():
        gc = _load_global_config_unlocked()
        known = gc.get("known_kbs", [])
        if resolved not in known:
            known.append(resolved)
            gc["known_kbs"] = known
        aliases = gc.get("kb_aliases", {})
        aliases[alias] = resolved
        gc["kb_aliases"] = aliases
        gc["default_kb"] = resolved
        _atomic_yaml_dump(GLOBAL_CONFIG_PATH, gc)


def resolve_kb_alias(name: str) -> Path:
    """Resolve a portable KB name to its path, using the unified name→path map.

    Resolution is kept in lock-step with :func:`registered_kbs` (and hence the
    ``/api/v1/kbs`` list) so the two can never disagree. Order:

    1. an explicit ``kb_aliases`` entry (user-set name→path) wins outright;
    2. else ``<kb_root>/<name>`` when it is a KB dir — a root-level KB is never
       shadowed by a same-named off-root ``known_kbs`` entry;
    3. else a :func:`registered_kbs` name match — an off-root registered KB is
       addressable by its directory basename (e.g. ``~/Documents/笔记`` as
       ``笔记``);
    4. else the ``<kb_root>/<name>`` fallback, so a freshly-created KB is
       addressable before it is registered.

    The old standalone ``known_kbs`` basename loop (which ran before the root
    check, could shadow a root child, and picked an arbitrary first match on a
    basename collision) is gone: resolution now goes through the same
    collision-resolved registry as discovery.
    """
    alias = validate_kb_name(name)
    with _with_global_config_lock():
        gc = _load_global_config_unlocked()
    aliases = gc.get("kb_aliases", {})
    if isinstance(aliases, dict):
        raw = aliases.get(alias)
        if isinstance(raw, str):
            return Path(raw).expanduser().resolve()
    root_candidate = (kb_root_dir() / alias).resolve()
    if _is_kb_dir(root_candidate):
        return root_candidate
    for reg_name, reg_path in registered_kbs():
        if reg_name == alias:
            return reg_path
    return root_candidate


def registered_kbs() -> list[tuple[str, Path]]:
    """Enumerate registered KBs as UNIQUE ``(name, resolved_path)`` pairs.

    The single name→path registry view shared by API discovery
    (``_list_knowledge_bases``) and name resolution (:func:`resolve_kb_alias`),
    so the KB list and resolution can never disagree. Names are deduped by name
    AND resolved path:

    * ``kb_aliases`` (explicit name→path) come first and keep their name;
    * a bare ``known_kbs`` path is surfaced by its directory basename ONLY when
      that name is not already claimed by an alias or by a ``kb_root_dir()``
      child (a root child wins the basename tie). A colliding entry is SKIPPED
      with a warning — never silently truncated to a wrong path or emitted as a
      duplicate name.

    Root children are NOT emitted here (``_list_knowledge_bases`` surfaces them
    from the filesystem and :func:`resolve_kb_alias` prefers them before this
    registry); they participate only as reserved names for the tie-break above.
    A malformed global.yaml (coerced to ``{}`` by the loader) or a non-string
    entry is tolerated rather than raising.
    """
    global_config = _load_global_config_unlocked()
    seen_paths: set[str] = set()
    # Root-child names are reserved up front so a colliding bare known_kbs entry
    # is hidden rather than shadowing/duplicating the same-named root-level KB.
    taken_names: set[str] = set(_root_child_names())
    result: list[tuple[str, Path]] = []
    aliases = global_config.get("kb_aliases", {})
    if isinstance(aliases, dict):
        for name, raw in aliases.items():
            if not isinstance(name, str) or not isinstance(raw, str):
                continue
            resolved = Path(raw).expanduser().resolve()
            if name in taken_names or str(resolved) in seen_paths:
                continue
            taken_names.add(name)
            seen_paths.add(str(resolved))
            result.append((name, resolved))
    known = global_config.get("known_kbs", [])
    if isinstance(known, list):
        for raw in known:
            if not isinstance(raw, str):
                continue
            resolved = Path(raw).expanduser().resolve()
            if str(resolved) in seen_paths:
                continue
            name = resolved.name
            if name in taken_names:
                logger.warning(
                    "config: registered KB %s is hidden — its name %r is already "
                    "claimed by an alias or a root-level KB.",
                    resolved,
                    name,
                )
                continue
            taken_names.add(name)
            seen_paths.add(str(resolved))
            result.append((name, resolved))
    return result
