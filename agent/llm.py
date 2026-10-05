import json
import os
import random
import time
from dataclasses import dataclass, field


class LLMError(Exception):
    """
    Model call failed in a way retrying won't fix (or retries ran out).
    """


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str

@dataclass
class Usage:
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0   
    reasoning_tokens: int = 0


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall]
    usage: Usage
    finish_reason: str
    latency_ms: float
    retries: list[dict] = field(default_factory=list)

    def as_message(self) -> dict:
        msg = {
            "role": "assistant", 
            "content": self.content
        }
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id, 
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments}
                }
                for tc in self.tool_calls
            ]
        return msg


class AzureLLM:
    def __init__(self, cfg):
        from openai import AzureOpenAI

        self.cfg = cfg
        # Missing credentials are reported on the first call, not here, so the
        # run still goes through the loop and ends with valid JSON and a trace.
        missing = [v for v in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY") if not os.getenv(v)]
        self.setup_error = f"missing environment variables: {', '.join(missing)} (see .env.example)" \
            if missing else None
        if missing:
            return
        self.client = AzureOpenAI(
            azure_endpoint=base_endpoint(os.environ["AZURE_OPENAI_ENDPOINT"]),
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
            max_retries=0,  # we retry ourselves, see complete()
        )

    def complete(self, messages, tools, tool_choice="auto", deadline: float | None = None,
                 max_output: int | None = None) -> LLMResponse:
        import openai

        if self.setup_error:
            raise LLMError(self.setup_error)
        retries = []
        attempt = 0
        while True:
            remaining = (deadline - time.monotonic()) if deadline else self.cfg.request_timeout_s
            if remaining <= 1:
                raise LLMError("no wall-clock time left for a model call")
            started = time.monotonic()
            try:
                resp = self.client.chat.completions.create(
                    model=self.cfg.deployment,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    parallel_tool_calls=True,
                    max_completion_tokens=max_output or self.cfg.max_completion_tokens,
                    reasoning_effort=self.cfg.reasoning_effort,
                    timeout=min(self.cfg.request_timeout_s, remaining),
                )
            except (openai.RateLimitError, openai.APIConnectionError,
                    openai.APITimeoutError, openai.InternalServerError) as e:
                attempt += 1
                wait = _retry_after(e) or min(2 ** attempt, 20) + random.random()
                retries.append({"attempt": attempt, "error": type(e).__name__, "wait_s": round(wait, 1)})
                left = (deadline - time.monotonic()) if deadline else float("inf")
                if attempt > self.cfg.max_retries or wait > left - 2:
                    raise LLMError(f"{type(e).__name__} after {attempt} attempt(s): {e}") from None
                time.sleep(wait)
                continue
            except openai.APIError as e:
                # 400s (bad request, content filter, auth) won't get better on retry.
                raise LLMError(f"{type(e).__name__}: {e}") from None

            choice = resp.choices[0]
            u = resp.usage
            usage = Usage(
                input_tokens=u.prompt_tokens,
                cached_tokens=(u.prompt_tokens_details.cached_tokens or 0) if u.prompt_tokens_details else 0,
                output_tokens=u.completion_tokens,
                reasoning_tokens=(u.completion_tokens_details.reasoning_tokens or 0)
                if u.completion_tokens_details else 0,
            )
            calls = [ToolCall(tc.id, tc.function.name, tc.function.arguments)
                     for tc in (choice.message.tool_calls or []) if tc.type == "function"]
            return LLMResponse(
                content=choice.message.content,
                tool_calls=calls,
                usage=usage,
                finish_reason=choice.finish_reason,
                latency_ms=round((time.monotonic() - started) * 1000, 1),
                retries=retries,
            )


def base_endpoint(url: str) -> str:
    from urllib.parse import urlsplit
    parts = urlsplit(url.strip())
    return f"{parts.scheme}://{parts.netloc}/"


def _retry_after(e) -> float | None:
    try:
        return float(e.response.headers.get("retry-after"))
    except Exception:
        return None


def to_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str, separators=(",", ":"))
