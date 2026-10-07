"""Local LLM backend — OpenAI or Gemini, selected by LLM_PROVIDER.

Gemini resilience model:

  * Every call goes to ``GEMINI_MODEL`` first, then to each entry of
    ``GEMINI_FALLBACK_MODELS`` in order.
  * For each model, transient errors (HTTP 429/500/502/503/504 and network
    timeouts / connection failures) are retried up to ``GEMINI_MAX_RETRIES``
    times with exponential backoff (2s, 4s, 8s, ... capped).
  * A 404 for a model (retired / not available to this account) is not
    retried; we move straight to the next fallback model.
  * Permanent errors (invalid API key, permission denied, malformed request,
    unsupported request, ...) are raised immediately without retry or
    fallback — they would fail identically on every model.
"""
import time
from typing import Callable, List, Optional

from ..config import (
    GEMINI_API_KEY,
    GEMINI_FALLBACK_MODELS,
    GEMINI_MAX_RETRIES,
    GEMINI_MODEL,
    LLM_PROVIDER,
    OPENAI_MODEL,
    require_gemini_key,
    require_openai_key,
)

# HTTP status codes considered temporary: worth retrying the same model.
GEMINI_TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
# Status codes that mean "this model is not available" — skip to the next model.
GEMINI_MODEL_UNAVAILABLE_STATUS_CODES = frozenset({404})

GEMINI_RETRY_BASE_DELAY_SECONDS = 2.0
GEMINI_RETRY_MAX_DELAY_SECONDS = 60.0
# Hard cap on a single Gemini HTTP request so a wedged call fails fast enough
# to be retried / failed over instead of hanging the pipeline.
GEMINI_REQUEST_TIMEOUT_SECONDS = 300

_gemini_client = None


def _log(msg: str) -> None:
    print(f"[gemini] {msg}", flush=True)


class GeminiAllModelsFailedError(RuntimeError):
    """Raised when the primary and every fallback Gemini model failed.

    The pipeline attaches ``transcript_path`` so the final message can tell the
    user their transcript is safe and a rerun will not retranscribe.
    """

    def __init__(
        self,
        primary_model: str,
        fallback_models: List[str],
        last_error: str,
        transcript_path: Optional[str] = None,
    ):
        self.primary_model = primary_model
        self.fallback_models = list(fallback_models)
        self.last_error = last_error
        self.transcript_path = transcript_path
        super().__init__(last_error)

    def __str__(self) -> str:
        fallbacks = ", ".join(self.fallback_models) if self.fallback_models else "(none configured)"
        lines = [
            "[gemini] All configured models failed.",
            "",
            "Primary model:",
            self.primary_model,
            "",
            "Fallback models:",
            fallbacks,
            "",
            "Last error:",
            self.last_error,
        ]
        if self.transcript_path:
            lines += [
                "",
                "The transcript has been preserved in:",
                self.transcript_path,
                "",
                "You can safely retry without retranscribing the video.",
            ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

def call_openai_llm(prompt: str) -> str:
    """OpenAI Chat Completions backend used by --mode local."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "openai is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    client = OpenAI(api_key=require_openai_key())
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

def _import_genai():
    try:
        from google import genai  # type: ignore
        from google.genai import errors  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "google-genai is required for LLM_PROVIDER=gemini. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e
    return genai, errors


def _get_gemini_client():
    """Create the Gemini client once per process (API-key auth)."""
    global _gemini_client
    if _gemini_client is None:
        genai, _ = _import_genai()
        _gemini_client = genai.Client(
            api_key=require_gemini_key(),
            http_options={"timeout": GEMINI_REQUEST_TIMEOUT_SECONDS * 1000},  # ms
        )
    return _gemini_client


def gemini_model_chain() -> List[str]:
    """Primary model followed by fallbacks, de-duplicated, order preserved."""
    chain: List[str] = []
    for name in [GEMINI_MODEL, *GEMINI_FALLBACK_MODELS]:
        name = (name or "").strip()
        if name and name not in chain:
            chain.append(name)
    return chain


def _error_code(exc: BaseException) -> Optional[int]:
    code = getattr(exc, "code", None)
    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


def _is_transport_error(exc: BaseException) -> bool:
    """Network-level failures (timeouts, connection resets) are transient."""
    try:
        import httpx  # type: ignore
    except ImportError:  # pragma: no cover - httpx ships with google-genai
        return False
    return isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError))


def _is_transient_error(exc: BaseException) -> bool:
    if _is_transport_error(exc):
        return True
    return _error_code(exc) in GEMINI_TRANSIENT_STATUS_CODES


def _is_model_unavailable_error(exc: BaseException) -> bool:
    return _error_code(exc) in GEMINI_MODEL_UNAVAILABLE_STATUS_CODES


def _describe_error(exc: BaseException) -> str:
    code = _error_code(exc)
    status = getattr(exc, "status", None)
    message = getattr(exc, "message", None)
    if code is not None:
        head = f"{code} {status}" if status else str(code)
        return f"{head}: {message}" if message else head
    return f"{type(exc).__name__}: {exc}"


def _backoff_delay(attempt: int) -> float:
    """attempt 1 → 2s, attempt 2 → 4s, attempt 3 → 8s, ... capped."""
    return min(GEMINI_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)), GEMINI_RETRY_MAX_DELAY_SECONDS)


def _extract_text(response) -> str:
    """Pull the text out of a GenerateContentResponse, or "" if there is none.

    An empty string makes the highlight layer treat the reply as malformed and
    re-prompt, which is the right behavior for truncated / blocked output.
    """
    try:
        text = response.text
    except Exception:  # SDK raises on unexpected part layouts
        text = None
    if text:
        return text

    # Surface *why* the model returned nothing so the user can act on it.
    reason = None
    try:
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            reason = getattr(candidates[0], "finish_reason", None)
        feedback = getattr(response, "prompt_feedback", None)
        block = getattr(feedback, "block_reason", None) if feedback else None
        if block:
            reason = f"blocked: {block}"
    except Exception:
        pass
    _log(f"empty response from model (finish_reason={reason or 'unknown'})")
    return ""


def _gemini_generate_once(model: str, prompt: str) -> str:
    """Single structured-JSON request. No tools => AFC explicitly disabled."""
    client = _get_gemini_client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
            "max_output_tokens": 8192,
            # This app never declares tools/functions, so automatic function
            # calling is pointless. Disabling it also keeps the SDK off the AFC
            # code path (and silences its "use Chat.send_message" warning).
            "automatic_function_calling": {"disable": True},
        },
    )
    return _extract_text(response)


def call_gemini_llm(
    prompt: str,
    *,
    _sleep: Optional[Callable[[float], None]] = None,
) -> str:
    """Gemini backend used by --mode local when LLM_PROVIDER=gemini.

    Tries the primary model, then each fallback. Transient errors are retried
    with exponential backoff; permanent errors are raised immediately.
    """
    _sleep = _sleep or time.sleep
    _, errors = _import_genai()
    models = gemini_model_chain()
    if not models:
        raise RuntimeError("[gemini] GEMINI_MODEL is not configured.")

    max_attempts = max(1, GEMINI_MAX_RETRIES)
    last_error = "unknown"

    for model_index, model in enumerate(models):
        if model_index > 0:
            _log(f"switching to fallback model={model}")

        for attempt in range(1, max_attempts + 1):
            _log(f"model={model} attempt={attempt}/{max_attempts}")
            try:
                return _gemini_generate_once(model, prompt)
            except errors.APIError as exc:
                last_error = _describe_error(exc)
                if _is_model_unavailable_error(exc):
                    _log(f"model not available ({last_error}); not retrying this model")
                    break  # next fallback model
                if not _is_transient_error(exc):
                    # Auth / permission / bad request: identical on every model.
                    _log(f"permanent error ({last_error}); not retrying")
                    raise
            except Exception as exc:  # network-level failures
                if not _is_transport_error(exc):
                    raise
                last_error = _describe_error(exc)

            if attempt < max_attempts:
                delay = _backoff_delay(attempt)
                _log(f"temporary error {last_error.split(':', 1)[0]}, retrying in {delay:g}s...")
                _sleep(delay)
            else:
                _log(f"temporary error {last_error.split(':', 1)[0]}; giving up on model={model}")

    raise GeminiAllModelsFailedError(
        primary_model=models[0],
        fallback_models=models[1:],
        last_error=last_error,
    )


def validate_gemini_config() -> None:
    """Fail fast on missing config and print a safe summary (never the key)."""
    if not GEMINI_API_KEY:
        require_gemini_key()  # raises the canonical "[gemini] GEMINI_API_KEY is not configured." error
    models = gemini_model_chain()
    if not models:
        raise RuntimeError("[gemini] GEMINI_MODEL is not configured.")
    _import_genai()  # surface a missing google-genai install before downloading anything
    _log("API key loaded: yes")
    _log(f"primary model: {models[0]}")
    _log(f"fallback models: {', '.join(models[1:]) if len(models) > 1 else '(none)'}")
    _log(f"max retries per model: {max(1, GEMINI_MAX_RETRIES)}")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _provider() -> str:
    return (LLM_PROVIDER or "openai").strip().lower()


def validate_local_llm_config() -> None:
    """Check the configured provider's credentials before any heavy work runs."""
    provider = _provider()
    if provider == "openai":
        require_openai_key()
        return
    if provider == "gemini":
        validate_gemini_config()
        return
    raise RuntimeError(
        f"Unknown LLM_PROVIDER={provider!r}. Use 'openai' or 'gemini'."
    )


def call_local_llm(prompt: str) -> str:
    """Dispatch to the configured local LLM provider."""
    provider = _provider()
    if provider == "openai":
        return call_openai_llm(prompt)
    if provider == "gemini":
        return call_gemini_llm(prompt)
    raise RuntimeError(
        f"Unknown LLM_PROVIDER={provider!r}. Use 'openai' or 'gemini'."
    )
