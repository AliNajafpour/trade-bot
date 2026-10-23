"""Callable LLM providers. Codex remains the key-free default; APIs are explicit opt-in.

Compatible API format: https://developers.openai.com/api/docs/guides/structured-outputs
Credentials are sent only in the Authorization header, never in persisted receipts.
"""
from datetime import datetime, timezone
import base64
import hashlib
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

from . import llm

def _adapter_hash():
    return hashlib.sha256(Path(__file__).read_bytes() + Path(llm.__file__).read_bytes()).hexdigest()


class CodexProvider:
    """Use the existing Codex login, tool-free invocation, validation and cache unchanged."""

    @property
    def cache_identity(self):
        return {"provider": "codex", "requested_model": None, "adapter_sha256": _adapter_hash()}

    def __call__(self, packet, prompt, schema, output_root, timeout=120, model=None, images=None, reasoning_effort=None):
        kwargs = {"images": images} if images is not None else {}
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        return llm.run_one(packet, prompt, schema, output_root, timeout=timeout, model=model, **kwargs)


class OpenAICompatibleProvider:
    """Non-streaming Chat Completions with strict JSON Schema and receipt-backed caching."""

    def __init__(self, api_key, base_url, model):
        if not isinstance(api_key, str) or not api_key.strip() or any(ch in api_key for ch in "\r\n"):
            raise ValueError("LLM_API_KEY must contain a nonempty API credential")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("LLM_MODEL must name a model supported by the configured API")
        try:
            url = urlsplit(base_url)
            valid = (url.scheme in {"http", "https"} and url.hostname and not url.username
                     and not url.password and not url.query and not url.fragment
                     and (url.scheme == "https" or url.hostname in {"localhost", "127.0.0.1", "::1"}))
            url.port
        except (ValueError, TypeError, AttributeError):
            valid = False
        if not valid:
            raise ValueError("LLM_BASE_URL requires HTTPS or loopback HTTP, with no credentials, query or fragment")
        if api_key in base_url or api_key in model:
            raise ValueError("API configuration must not embed credential material")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    @property
    def cache_identity(self):
        return {"provider": "openai_compatible", "base_url": self.base_url,
                "requested_model": self.model, "adapter_sha256": _adapter_hash()}

    def __call__(self, packet, prompt, schema, output_root, timeout=120, model=None, images=None, reasoning_effort=None):
        # Lazy import keeps the existing Codex path usable without an API dependency.
        import httpx

        if type(packet) is not dict or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A packet object and nonempty prompt required")
        if type(timeout) not in (int, float) or not 0 < timeout <= 120:
            raise ValueError("Timeout must be within 120 seconds")
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise ValueError("Model must be a nonempty string")
        llm.validate_reasoning_effort(reasoning_effort)
        requested_model = self.model if model is None else model
        images = llm.image_inputs(images)
        content = llm._json(packet)
        if images:
            content = [{"type": "text", "text": content}, *[
                {"type": "image_url", "image_url": {"url": "data:" + item["media_type"] + ";base64," + base64.b64encode(item["data"]).decode("ascii")}}
                for item in images]]
        request = {"model": requested_model,
                   "messages": [{"role": "system", "content": llm.PREFIX + prompt},
                                {"role": "user", "content": content}],
                   "response_format": {"type": "json_schema", "json_schema": {
                       "name": "wallex_response", "strict": True, "schema": schema}}}
        if reasoning_effort is not None:
            request["reasoning_effort"] = reasoning_effort
        request_text = llm._json(request)
        if self._api_key in request_text or any(self._api_key.encode() in item["data"] for item in images):
            raise ValueError("Inference inputs must not contain API credential material")
        identity = {**self.cache_identity, "requested_model": requested_model, "request": request,
                    "images": llm.image_manifest(images)}
        key = hashlib.sha256(llm._json(identity).encode()).hexdigest()
        output = Path(output_root).resolve() / key
        receipt_path = output / "receipt.json"
        if receipt_path.exists():
            receipt = llm._load(receipt_path.read_text(encoding="utf-8"))
            files = receipt.get("files_sha256", {})
            if (receipt.get("cache_key") != key or not all(llm.cache_filename(name) for name in files)
                    or not {"packet.json", "prompt.txt", "schema.json", "request.json"} <= set(files)):
                raise ValueError("Cached inference receipt failed integrity check")
            if images and not {"images.json", *(item["name"] for item in images)} <= set(files):
                raise ValueError("Cached inference images missing")
            for name, digest in files.items():
                if hashlib.sha256((output / name).read_bytes()).hexdigest() != digest:
                    raise ValueError("Cached inference receipt failed integrity check")
            if receipt["status"] == "ok":
                if not {"response.json", "final.json"} <= set(files):
                    raise ValueError("Cached inference response missing")
                llm.validate_value(receipt["response"], schema)
                if receipt["response"] != llm._load((output / "final.json").read_text(encoding="utf-8")):
                    raise ValueError("Cached inference response mismatch")
            if self._api_key in llm._json(receipt):
                raise ValueError("Cached inference receipt contains credential material")
            return dict(receipt, cache_hit=True, api_calls=0, cli_calls=0)
        output.mkdir(parents=True, exist_ok=False)
        llm.save_images(output, images)

        def save(name, raw):
            (output / name).write_text(raw, encoding="utf-8")

        save("packet.json", llm._json(packet))
        save("prompt.txt", llm.PREFIX + prompt)
        save("schema.json", llm._json(schema))
        save("request.json", request_text)
        receipt = {"status": "error", "provider": "openai_compatible", "cache_key": key,
                   "receipt_path": str(receipt_path), "started_utc": datetime.now(timezone.utc).isoformat(),
                   "requested_model": requested_model, "model_selection": "explicit" if model else "provider_config",
                   "requested_reasoning_effort": reasoning_effort,
                   "base_url": self.base_url, "actual_model": None, "model_identity_verified": False,
                   "observed_models": [], "adapter_sha256": identity["adapter_sha256"], "images": llm.image_manifest(images),
                   "tool_calls": 0, "api_calls": 0, "cli_calls": 0, "cache_hit": False}
        started = time.monotonic()
        try:
            receipt["api_calls"] = 1
            save("receipt.json", llm._json(receipt))
            with httpx.Client(timeout=timeout, follow_redirects=False) as client:
                response = client.post(self.base_url + "/chat/completions", json=request,
                                       headers={"Authorization": "Bearer " + self._api_key})
            receipt["http_status"] = response.status_code
            if response.status_code != 200:
                raise ValueError("LLM API HTTP " + str(response.status_code))
            envelope = llm._load(response.text)
            if self._api_key in llm._json(envelope):
                raise ValueError("LLM API response contained credential material")
            choices = envelope["choices"]
            if type(choices) is not list or len(choices) != 1:
                raise ValueError("LLM API response requires one choice")
            choice = choices[0]
            message = choice["message"]
            if choice.get("finish_reason") != "stop" or message.get("refusal"):
                raise ValueError("LLM API response was refused or incomplete")
            if message.get("tool_calls") or message.get("function_call"):
                raise ValueError("Tool activity invalidates inference")
            if message.get("role") != "assistant" or not isinstance(message.get("content"), str):
                raise ValueError("LLM API response requires assistant JSON text")
            result = llm._load(message["content"])
            if self._api_key in llm._json(result):
                raise ValueError("LLM API response contained credential material")
            llm.validate_value(result, schema)
            save("response.json", response.text)
            save("final.json", llm._json(result))
            actual = envelope.get("model")
            actual = actual if isinstance(actual, str) and actual.strip() else None
            receipt.update(status="ok", response=result, actual_model=actual,
                           observed_models=[actual] if actual else [], usage=envelope.get("usage"))
        except httpx.TimeoutException:
            receipt["error"] = "LLM API request timed out"
        except httpx.HTTPError:
            receipt["error"] = "LLM API transport failed"
        except (ValueError, KeyError, TypeError, IndexError, AttributeError):
            # Provider bodies and exception text can echo credentials or request headers.
            receipt["error"] = ("LLM API HTTP " + str(receipt["http_status"]) if receipt.get("http_status") != 200
                                and "http_status" in receipt else "LLM API returned invalid or incomplete structured output")
        receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
        receipt["ended_utc"] = datetime.now(timezone.utc).isoformat()
        receipt["files_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in output.iterdir() if path.is_file() and llm.cache_filename(path.name)}
        save("receipt.json", llm._json(receipt))
        return receipt


def provider_from_env(environ=None):
    """Only LLM_PROVIDER=openai_compatible enables API use; a key alone never does."""
    env = os.environ if environ is None else environ
    name = env.get("LLM_PROVIDER", "codex").strip().lower()
    if name == "codex":
        return CodexProvider()
    if name == "openai_compatible":
        return OpenAICompatibleProvider(env.get("LLM_API_KEY"), env.get("LLM_BASE_URL", "https://api.openai.com/v1"),
                                        env.get("LLM_MODEL"))
    raise ValueError("LLM_PROVIDER must be codex or openai_compatible")


def run_one(packet, prompt, schema, output_root, timeout=120, model=None, images=None, reasoning_effort=None):
    kwargs = {"images": images} if images is not None else {}
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    return provider_from_env()(packet, prompt, schema, output_root, timeout=timeout, model=model, **kwargs)
