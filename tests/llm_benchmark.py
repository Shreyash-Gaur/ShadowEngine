#!/usr/bin/env python3
"""
LLM Inference Benchmarker
Measures: TTFT, ITL, TPOT, E2E Latency, TPS (system & per-user), Goodput

Usage:
    python tests/llm_bench.py
    python tests/llm_bench.py --runs 5
    python tests/llm_bench.py --runs 10 --quiet
"""

import os
import requests
import time
import json
import statistics
from dotenv import load_dotenv

load_dotenv()

API_URL     = os.getenv("API_URL")
_u, _p = os.getenv("AUTH_USER"), os.getenv("AUTH_PASS")
AUTH    = (_u, _p) if _u and _p else None
MODEL       = os.getenv("MODEL")
MAX_TOKENS  = int(os.getenv("MAX_TOKENS", "8192"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# SLA thresholds — override via .env
SLA_TTFT_MS = float(os.getenv("SLA_TTFT_MS", "3000"))   # 3 seconds
SLA_TPOT_MS = float(os.getenv("SLA_TPOT_MS", "20"))    # 20 ms per output token
SLA_E2E_MS  = float(os.getenv("SLA_E2E_MS",  "50000"))  # 50 seconds end-to-end

PROMPT = os.getenv(
    "BENCH_PROMPT",
    "Write a highly detailed, 4-paragraph story about a cyberpunk detective. Be descriptive."
)


# ─────────────────────────────────────────────
#  Core streaming benchmark — single request
# ─────────────────────────────────────────────

def run_single_request(verbose: bool = True) -> dict:
    payload = {
        "model":       MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": PROMPT},
        ],
        "stream":         True,
        "max_tokens":     MAX_TOKENS,
        "temperature":    TEMPERATURE,
        "stream_options": {"include_usage": True},  # request real BPE token count in final chunk
    }

    request_sent_at  = time.perf_counter()
    response = requests.post(API_URL, json=payload, auth=AUTH, stream=True, timeout=120)
    response.raise_for_status()

    first_token_at   = None
    prev_token_at    = None
    inter_token_gaps = []  
    token_count      = 0    
    is_thinking      = False
    full_text        = []

    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if not decoded.startswith("data: "):
            continue
        data_str = decoded[6:]
        if data_str == "[DONE]":
            break

        try:
            data = json.loads(data_str)

            usage = data.get("usage")
            if usage:
                token_count = usage.get("completion_tokens", token_count)

            choices = data.get("choices")
            if not choices:
                continue

            delta = choices[0].get("delta", {})

            reasoning_text = delta.get("reasoning_content", "")
            content_text = delta.get("content", "")

            if reasoning_text:
                if not is_thinking:
                    if verbose: print("\n[THINKING START]\n\033[90m", end="")
                    is_thinking = True
                if verbose: print(reasoning_text, end="", flush=True)

                now = time.perf_counter()
                if first_token_at is None:
                    first_token_at = now
                if prev_token_at is not None:
                    inter_token_gaps.append((now - prev_token_at) * 1000)
                prev_token_at = now
                token_count += 1
                full_text.append(reasoning_text)
    
            if content_text:
                if is_thinking:
                    if verbose: print("\033[0m\n\n[THINKING END]\n")
                    is_thinking = False
                if verbose: print(content_text, end="", flush=True)

                now = time.perf_counter()
                if first_token_at is None:
                    first_token_at = now
                if prev_token_at is not None:
                    inter_token_gaps.append((now - prev_token_at) * 1000)
                prev_token_at = now
                token_count += 1
                full_text.append(content_text)

            # ── FIX: override chunk-count with real BPE token count ──────────
            # vLLM attaches usage on the final chunk when stream_options is set.
            # This is the only accurate token count regardless of SSE batching.
            usage = data.get("usage")
            if usage and usage.get("completion_tokens"):
                token_count = usage["completion_tokens"]
            # ────────────────────────────────────────────────────────────────

        except json.JSONDecodeError:
            continue

    last_token_at = time.perf_counter()

    if verbose and is_thinking:
        print("\033[0m")

    if first_token_at is None or token_count == 0:
        raise RuntimeError("No tokens received from the model.")

    # ── Derived metrics ───────────────────────────────────────────────────
    ttft_ms  = (first_token_at - request_sent_at) * 1000   # request → first token
    e2e_ms   = (last_token_at  - request_sent_at) * 1000   # request → last token
    gen_ms   = (last_token_at  - first_token_at)  * 1000   # first token → last token

    # TPOT = generation_window / token_count
    tpot_ms  = gen_ms / token_count if token_count > 1 else 0.0

    # Per-user TPS = tokens / E2E seconds
    tps_user = token_count / (e2e_ms / 1000)

    # ITL percentile stats
    sorted_itl = sorted(inter_token_gaps)

    def _pct(lst, p):
        return lst[max(0, int(p * len(lst)) - 1)] if lst else 0.0

    return {
        "ttft_ms":     round(ttft_ms,  2),
        "e2e_ms":      round(e2e_ms,   2),
        "gen_ms":      round(gen_ms,   2),
        "tpot_ms":     round(tpot_ms,  2),
        "tps_user":    round(tps_user, 2),
        "token_count": token_count,
        "itl_avg_ms":  round(statistics.mean(inter_token_gaps)     if inter_token_gaps else 0.0, 2),
        "itl_p50_ms":  round(_pct(sorted_itl, 0.50), 2),
        "itl_p90_ms":  round(_pct(sorted_itl, 0.90), 2),
        "itl_p99_ms":  round(_pct(sorted_itl, 0.99), 2),
        "sla_ttft":    ttft_ms <= SLA_TTFT_MS,
        "sla_tpot":    tpot_ms <= SLA_TPOT_MS,
        "sla_e2e":     e2e_ms  <= SLA_E2E_MS,
        "output_text": "".join(full_text),
    }


# ─────────────────────────────────────────────
#  Multi-run benchmark + aggregate stats
# ─────────────────────────────────────────────

def run_benchmark(n_runs: int = 3, verbose_first: bool = True) -> None:
    print(f"\n{'='*60}")
    print(f"  LLM Inference Benchmarker")
    print(f"  Model  : {MODEL}")
    print(f"  URL    : {API_URL}")
    print(f"  Runs   : {n_runs}")
    print(f"{'='*60}\n")
    print(f"  Sending: {PROMPT[:80]}...\n")

    # FIX 3 (before the run loop): warmup request to flush cold-start overhead
    print("  Warming up...")
    try:
        run_single_request(verbose=False)
    except Exception:
        pass
    print()

    results   = []
    start_all = time.perf_counter()

    for i in range(1, n_runs + 1):
        print(f"─── Run {i}/{n_runs} {'─'*40}")
        try:
            r = run_single_request(verbose=(i == 1 and verbose_first))
            results.append(r)
            print(f"\n  TTFT       : {r['ttft_ms']:.1f} ms  {'✓' if r['sla_ttft'] else '✗ SLA miss'}")
            print(f"  E2E        : {r['e2e_ms']:.1f} ms  {'✓' if r['sla_e2e'] else '✗ SLA miss'}")
            print(f"  TPOT       : {r['tpot_ms']:.1f} ms/tok  {'✓' if r['sla_tpot'] else '✗ SLA miss'}")
            print(f"  TPS (user) : {r['tps_user']:.2f} tok/s")
            print(f"  Tokens     : {r['token_count']}")
            print(f"  ITL        : avg {r['itl_avg_ms']:.1f} ms | p50 {r['itl_p50_ms']:.1f} | p90 {r['itl_p90_ms']:.1f} | p99 {r['itl_p99_ms']:.1f}")
        except Exception as e:
            print(f"  [ERROR] {e}")
        print()

    if not results:
        print("No successful runs.")
        return

    wall_time_s = time.perf_counter() - start_all

    # ── Aggregate ─────────────────────────────────────────────────────────
    def pct(vals, p):
        s = sorted(vals)
        return s[max(0, int(p * len(s)) - 1)]

    ttfts  = [r["ttft_ms"]     for r in results]
    tpots  = [r["tpot_ms"]     for r in results]
    e2es   = [r["e2e_ms"]      for r in results]
    tpss   = [r["tps_user"]    for r in results]
    tokens = [r["token_count"] for r in results]

    total_tokens = sum(tokens)
    system_tps   = total_tokens / wall_time_s

    passing     = sum(1 for r in results if r["sla_ttft"] and r["sla_tpot"] and r["sla_e2e"])
    goodput_pct = (passing / len(results)) * 100
    goodput_tps = system_tps * (goodput_pct / 100)

    print(f"\n{'='*60}")
    print(f"  AGGREGATE  ({len(results)}/{n_runs} successful runs)")
    print(f"{'='*60}")

    print(f"\n  TTFT (ms)")
    print(f"    avg {statistics.mean(ttfts):>8.1f}  |  p50 {pct(ttfts,.50):>8.1f}  |  p90 {pct(ttfts,.90):>8.1f}  |  p99 {pct(ttfts,.99):>8.1f}")
    print(f"    min {min(ttfts):>8.1f}  |  max {max(ttfts):>8.1f}")

    print(f"\n  E2E Latency (ms)")
    print(f"    avg {statistics.mean(e2es):>8.1f}  |  p50 {pct(e2es,.50):>8.1f}  |  p90 {pct(e2es,.90):>8.1f}  |  p99 {pct(e2es,.99):>8.1f}")

    print(f"\n  TPOT (ms/token)")
    print(f"    avg {statistics.mean(tpots):>8.1f}  |  p50 {pct(tpots,.50):>8.1f}  |  p90 {pct(tpots,.90):>8.1f}  |  p99 {pct(tpots,.99):>8.1f}")

    print(f"\n  Per-user TPS (tok/s)")
    print(f"    avg {statistics.mean(tpss):>8.2f}  |  p50 {pct(tpss,.50):>8.2f}  |  min {min(tpss):>8.2f}  |  max {max(tpss):>8.2f}")

    print(f"\n  System TPS   : {system_tps:.2f} tok/s")

    print(f"\n  Goodput")
    print(f"    {passing}/{len(results)} runs passed all SLAs")
    print(f"    Goodput TPS  : {goodput_tps:.2f} tok/s")
    print(f"    Goodput %    : {goodput_pct:.1f}%")

    print(f"\n  SLA Thresholds")
    print(f"    TTFT ≤ {SLA_TTFT_MS:.0f} ms  |  TPOT ≤ {SLA_TPOT_MS:.0f} ms/tok  |  E2E ≤ {SLA_E2E_MS:.0f} ms")

    print(f"\n  Formulas")
    print(f"    E2E        = TTFT + generation_time")
    print(f"    TPOT       = generation_time / token_count")
    print(f"    TPS (user) = token_count / E2E_seconds")
    print(f"    TPS (sys)  = total_tokens / wall_clock_seconds")
    print(f"    Goodput    = TPS(sys) × (SLA_passing / total_runs)")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLM Inference Benchmarker")
    parser.add_argument("--runs",  type=int, default=3,
                        help="Number of benchmark runs (default: 3)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress streamed output on first run")
    args = parser.parse_args()
    run_benchmark(n_runs=args.runs, verbose_first=not args.quiet)