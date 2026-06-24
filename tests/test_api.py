#!/usr/bin/env python3
"""
ShadowEngine — vLLM API Health Check & Throughput Benchmark
============================================================
Tests the running endpoint with concurrent requests and reports throughput metrics.

Usage:
    python tests/test_api.py --endpoint https://abc.ngrok-free.app --requests 20
    python tests/test_api.py --endpoint http://localhost:8000 --concurrent 8
    python tests/test_api.py --endpoint http://localhost:8000 --requests 5 --no-benchmark
"""

import os
import argparse
import sys
import time
import statistics
import json
import requests as req
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

AUTH_USER        = os.getenv("AUTH_USER")
AUTH_PASS        = os.getenv("AUTH_PASS")
AUTH             = (AUTH_USER, AUTH_PASS) if AUTH_USER and AUTH_PASS else None
MODEL            = os.getenv("MODEL")
MAX_TOKENS       = int(os.getenv("MAX_TOKENS", "4096"))
TEMPERATURE      = float(os.getenv("TEMPERATURE", "0.7"))
DEFAULT_API_BASE = os.getenv("API_URL_BASE", "http://127.0.0.1:8000")

PROMPT_TEXT = os.getenv(
    "BENCH_PROMPT",
    "Write a poem on Qwen architecture of around 10 lines."
)

DEFAULT_MSGS = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": PROMPT_TEXT},
]


# ─────────────────────────────────────────────
#  Result dataclass
# ─────────────────────────────────────────────

@dataclass
class RequestResult:
    url:         str
    index:       int
    status_code: int   = 0
    latency_ms:  float = 0.0
    ttft_ms:     float = 0.0
    token_count: int   = 0
    error:       str   = ""


# ─────────────────────────────────────────────
#  Single request — streaming so TTFT is real
# ─────────────────────────────────────────────

def make_chat_request(url: str, index: int, msgs: Optional[list] = None) -> RequestResult:
    """
    Streaming request so we can capture:
      - TTFT  : time from send → first token chunk
      - E2E   : time from send → last token chunk
      - tokens: counted from actual stream chunks, not word-split
    """

    result = RequestResult(url=url, index=index)
    msgs   = msgs or DEFAULT_MSGS

    payload = {
        "model":          MODEL,
        "messages":       msgs,
        "temperature":    TEMPERATURE,
        "top_p":          0.9,
        "max_tokens":     MAX_TOKENS,
        "stream":         True,
        "stream_options": {"include_usage": True},
    }

    request_sent_at = time.perf_counter()
    try:
        resp = req.post(
            f"{url}/v1/chat/completions",
            auth=AUTH,
            json=payload,
            stream=True,
            timeout=1200,
        )
        result.status_code = resp.status_code
        resp.raise_for_status()

        first_token_at = None
        token_count    = 0
        last_data      = None  # tracks last parsed chunk so usage is accessible after loop

        for line in resp.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if not decoded.startswith("data: "):
                continue
            data_str = decoded[6:]
            if data_str == "[DONE]":
                break
            try:
                data      = json.loads(data_str)
                last_data = data                       # keep reference to last parsed chunk
                delta     = data["choices"][0]["delta"]
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

            text = delta.get("content") or delta.get("reasoning_content") or ""
            if not text:
                continue

            now = time.perf_counter()
            if first_token_at is None:
                first_token_at = now
            token_count += 1

        end_time = time.perf_counter()

        if last_data:
            usage = last_data.get("usage")
            if usage and usage.get("completion_tokens"):
                token_count = usage["completion_tokens"]

        if first_token_at is not None:
            result.ttft_ms    = (first_token_at - request_sent_at) * 1000
        result.latency_ms     = (end_time        - request_sent_at) * 1000
        result.token_count    = max(token_count, 1)

    except Exception as e:
        result.latency_ms = (time.perf_counter() - request_sent_at) * 1000
        result.error      = str(e)

    return result


# ─────────────────────────────────────────────
#  Health check
# ─────────────────────────────────────────────

def health_check(url: str) -> bool:
    try:
        resp = req.get(f"{url}/health", auth=AUTH, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"  Health check failed: {e}")
        return False


# ─────────────────────────────────────────────
#  Benchmark runner
# ─────────────────────────────────────────────

def run_benchmark(url: str, n_requests: int = 10, concurrent: int = 4) -> None:
    print(f"\n{'='*60}")
    print(f"  ShadowEngine — Throughput Benchmark")
    print(f"{'='*60}")
    print(f"  Endpoint   : {url}/v1/chat/completions")
    print(f"  Model      : {MODEL}")
    print(f"  Total reqs : {n_requests}")
    print(f"  Concurrent : {concurrent}")
    print(f"  Prompt     : {PROMPT_TEXT[:80]}...")
    print(f"{'='*60}\n")

    results: List[RequestResult] = []
    overall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = {
            executor.submit(make_chat_request, url, i + 1): i
            for i in range(n_requests)
        }
        for fut in as_completed(futures):
            results.append(fut.result())

    total_time_s = time.perf_counter() - overall_start

    successful = [r for r in results if r.status_code == 200 and not r.error]
    failed     = [r for r in results if r.error]

    if not successful:
        print("  ❌ All requests failed!")
        for r in failed:
            print(f"     Req #{r.index}: {r.error}")
        return

    latencies = [r.latency_ms  for r in successful]
    ttfts     = [r.ttft_ms     for r in successful if r.ttft_ms > 0]
    tokens    = [r.token_count for r in successful]

    def pct(vals, p):
        s = sorted(vals)
        return s[max(0, int(p * len(s)) - 1)]

    total_tokens = sum(tokens)
    rps          = len(successful) / total_time_s
    system_tps   = total_tokens    / total_time_s

    print(f"  Results : {len(successful)} successful, {len(failed)} failed\n")

    print(f"  E2E Latency (ms)")
    print(f"    avg  {statistics.mean(latencies):>8.1f}  |  min  {min(latencies):>8.1f}  |  max  {max(latencies):>8.1f}")
    print(f"    p50  {pct(latencies, .50):>8.1f}  |  p90  {pct(latencies, .90):>8.1f}  |  p99  {pct(latencies, .99):>8.1f}")
    print()

    if ttfts:
        print(f"  TTFT (ms)")
        print(f"    avg  {statistics.mean(ttfts):>8.1f}  |  min  {min(ttfts):>8.1f}  |  max  {max(ttfts):>8.1f}")
        print(f"    p50  {pct(ttfts, .50):>8.1f}  |  p90  {pct(ttfts, .90):>8.1f}  |  p99  {pct(ttfts, .99):>8.1f}")
        print()

    print(f"  Throughput")
    print(f"    RPS (requests/sec) : {rps:>8.3f}")
    print(f"    System TPS (tok/s) : {system_tps:>8.2f}")
    print(f"    Avg tokens / req   : {total_tokens / len(successful):>8.1f}")
    print(f"    Total tokens       : {total_tokens:>8d}")
    print(f"    Wall time          : {total_time_s:>8.2f}s")
    print()

    sorted_results = sorted(successful, key=lambda r: r.index)
    show = sorted_results[-min(5, len(sorted_results)):]
    print(f"  {'Last' if len(successful) > 5 else 'All'} {len(show)} requests:")
    for r in show:
        flag = "✅" if not r.error else f"❌ {r.error}"
        print(f"    Req #{r.index:>2}: E2E {r.latency_ms:7.1f}ms | TTFT {r.ttft_ms:6.1f}ms | {r.token_count:4d} tok | {flag}")
    print()


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ShadowEngine API benchmark")
    parser.add_argument("--endpoint",     default=DEFAULT_API_BASE, help="Base URL of the server")
    parser.add_argument("--requests", "-n", type=int, default=10,   help="Number of requests (default: 10)")
    parser.add_argument("--concurrent", "-c", type=int, default=4,  help="Concurrent connections (default: 4)")
    parser.add_argument("--no-benchmark", action="store_true",      help="Skip benchmark, health check only")
    args = parser.parse_args()

    base_url = args.endpoint.rstrip("/")

    print("Checking server health...")
    if not health_check(base_url):
        print(f"  ❌ Server at {base_url} is not reachable!")
        print(f"  Make sure vllm_server.py is running.")
        sys.exit(1)
    print("  ✅ Server is healthy!\n")

    if not args.no_benchmark:
        run_benchmark(
            base_url,
            n_requests=args.requests,
            concurrent=args.concurrent,
        )


if __name__ == "__main__":
    main()