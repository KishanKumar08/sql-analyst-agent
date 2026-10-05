import json
import time

from agent.llm import LLMError, LLMResponse, ToolCall, Usage


def call(name, **args):
    return ("call", name, args)


def raw_call(name, raw: str):
    return ("call", name, raw)


class FakeLLM:
    """
    Each script step is one model response: a list of calls, a string
    (text with no tool call), an Exception (raised), or a callable that gets
    (messages, tool_choice) and returns one of those.
    """

    def __init__(self, script, usage=(1000, 100), latency_s=0.0, repeat_last=False):
        self.script = list(script)
        self.usage = usage
        self.latency_s = latency_s
        self.repeat_last = repeat_last
        self.requests = []
        self._n = 0

    def complete(self, messages, tools, tool_choice="auto", deadline=None, max_output=None):
        self.requests.append({"messages": [dict(m) for m in messages], "tool_choice": tool_choice,
                              "max_output": max_output})
        if self._n < len(self.script):
            step = self.script[self._n]
        elif self.repeat_last and self.script:
            step = self.script[-1]
        else:
            raise LLMError("fake script exhausted")
        self._n += 1
        if callable(step):
            step = step(messages, tool_choice)
        if isinstance(step, Exception):
            raise step
        if self.latency_s:
            time.sleep(self.latency_s)
        calls, content = [], None
        if isinstance(step, str):
            content = step
        else:
            for i, (_, name, args) in enumerate(step):
                raw = args if isinstance(args, str) else json.dumps(args)
                calls.append(ToolCall(f"call_{self._n}_{i}", name, raw))
        return LLMResponse(content=content, tool_calls=calls,
                           # Like the real API, output can't exceed max_output.
                           usage=Usage(self.usage[0], 0, min(self.usage[1], max_output or 10**9), 0),
                           finish_reason="tool_calls" if calls else "stop",
                           latency_ms=self.latency_s * 1000)
