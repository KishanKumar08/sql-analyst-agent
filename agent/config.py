import os
from dataclasses import dataclass, fields


@dataclass
class Config:
    # --- model -----------------------------------------------------------
    deployment: str = "gpt-5-mini"
    reasoning_effort: str = "medium"       # minimal | low | medium | high
    max_completion_tokens: int = 4000      # per call; includes reasoning tokens
    request_timeout_s: float = 60.0
    max_retries: int = 4                   # on 429 / 5xx / network errors

    # USD per 1M tokens. Defaults are gpt-5-mini list prices (checked Oct 2026).
    price_input: float = 0.25
    price_cached_input: float = 0.025
    price_output: float = 2.00

    # --- run budgets ------------------------------------------------------
    max_turns: int = 15                    # one turn = one model call
    max_tokens: int = 200_000              # input + output, summed over the run
    max_cost_usd: float = 0.10
    max_wall_s: float = 180.0
    # Fraction of each budget held back so a final "submit what you have"
    # call can still run after exploration stops.
    finalize_reserve: float = 0.15

    # --- database limits --------------------------------------------------
    query_timeout_s: float = 5.0
    max_rows: int = 200                    # rows fetched from SQLite per query
    max_cell_chars: int = 200

    # --- context management -----------------------------------------------
    rows_to_model: int = 50                # rows of a result the model actually sees
    max_tool_result_chars: int = 6000
    compact_after_tokens: int = 16_000     # prompt size that triggers compaction
    keep_recent_tool_results: int = 4      # newest results never compacted
    schema_in_prompt: bool = True
    max_schema_chars: int = 12_000
    schema_sample_rows: int = 2            # example rows per table in the preloaded schema

    # --- output ------------------------------------------------------------
    trace_dir: str = "traces"

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        cfg = cls()
        if os.getenv("AZURE_OPENAI_DEPLOYMENT"):
            cfg.deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
        for f in fields(cls):
            raw = os.getenv(f"AGENT_{f.name.upper()}")
            if raw is not None:
                if f.type is bool:
                    value = raw.strip().lower() in ("1", "true", "yes", "on")
                elif f.type in (int, float):
                    value = f.type(raw)
                else:
                    value = raw
                setattr(cfg, f.name, value)
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)
        return cfg

    def cost(self, input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
        fresh = max(input_tokens - cached_tokens, 0)
        return (fresh * self.price_input
                + cached_tokens * self.price_cached_input
                + output_tokens * self.price_output) / 1_000_000
