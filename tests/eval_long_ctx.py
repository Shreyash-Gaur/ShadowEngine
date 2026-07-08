#!/usr/bin/env python3
"""
ShadowEngine — Long Context Evaluation Suite
=============================================
Tests: NIAH (basic), Multi-Needle NIAH, NoLiMa (paraphrased), RULER-lite, BABILong-lite
Target: llama.cpp or vLLM server via local proxy (proxy_stream.py / proxy_buffer.py)

Usage:
    python eval_long_ctx.py --ctx 32768 192000 --suite all
    python eval_long_ctx.py --ctx 65536 --suite niah ruler --save-results
    python eval_long_ctx.py --ctx 192000 --suite all --checkpoint ./ckpt.json --save-results
    python eval_long_ctx.py --ctx 65536 --suite all --no-wiki          # skip Wikipedia, use static filler
    python eval_long_ctx.py --ctx 65536 --suite all --wiki-sentences 40000

Filler corpus:
    Haystack filler is streamed from the wikimedia/wikipedia HF dataset
    (streaming=True — never downloads the full ~20GB dump) and cached to
    wiki_sentence_pool.json next to this script, so the network cost is paid
    once per session. Requires `pip install datasets --break-system-packages`.
    Falls back automatically to a small static corpus (_FILLER_CORPUS_FALLBACK)
    if `datasets` isn't installed or the stream can't be reached — use
    --no-wiki to force that path deliberately (e.g. for a fast offline
    smoke-test of the harness itself).

.env (same as proxy):
    REMOTE_HOST=http://127.0.0.1:8000     # point at local proxy port
    MODEL_ALIAS=Qwen3.6-35B-A3B-Uncensored  # must match llama-server --alias
    EVAL_TIMEOUT=3600        # seconds — matches the proxy's new ceiling; raise further for 192K
    EVAL_RETRIES=2           # transient-error retries per call, not counted as a real fail
    EVAL_RETRY_WAIT=5        # seconds between retries

IMPORTANT — disable reasoning server-side before running this suite.
`--reasoning on` is set at llama-server boot (see deploy_llama.py) and is not
reliably overridable per-request on recent llama.cpp builds. This script
still sends chat_template_kwargs={"enable_thinking": False} as a best-effort,
but the real fix is: set reasoning.enabled = False in deploy_llama.py and
redeploy before running this. preflight_check() below will warn you if it
looks like reasoning is still on.
"""

from __future__ import annotations
import argparse, json, os, random, re, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL    = os.getenv("API_URL_BASE", "http://127.0.0.1:8000")
MODEL       = os.getenv("MODEL", "Qwen3.6-35B-A3B-Uncensored")
API_KEY     = os.getenv("API_KEY", "")
AUTH_USER   = os.getenv("AUTH_USER", "")
AUTH_PASS   = os.getenv("AUTH_PASS", "")
BASIC_AUTH  = (AUTH_USER, AUTH_PASS) if AUTH_USER and AUTH_PASS else None
TIMEOUT     = int(os.getenv("EVAL_TIMEOUT", "10800"))
MAX_TOKENS  = int(os.getenv("EVAL_MAX_TOKENS", "256"))
RETRIES     = int(os.getenv("EVAL_RETRIES", "0"))
RETRY_WAIT  = int(os.getenv("EVAL_RETRY_WAIT", "5"))

HEADERS = {"Content-Type": "application/json"}
if API_KEY:
    HEADERS["Authorization"] = f"Bearer {API_KEY}"


# ─── Utilities ────────────────────────────────────────────────────────────────

def _chat(messages: List[Dict], max_tokens: int = MAX_TOKENS, temperature: float = 0.0) -> Dict[str, Any]:
    """
    Returns a dict, not a bare string, so callers can see finish_reason and
    reasoning_content — that's what tells you whether a "failure" was actually
    a retrieval miss, or just the response getting cut off mid-thought.

    Streams (`"stream": True`) rather than waiting for one buffered response.
    With stream=False, llama.cpp/vLLM send ZERO bytes until generation is
    fully done — on a long-context call that's many minutes of a completely
    silent connection, which is exactly what gets killed by an idle-timeout
    on any hop you don't control (ngrok's free-tier edge, in particular).
    Streaming keeps real bytes flowing the whole time instead.
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        # Best-effort. If the server was booted with --reasoning on, this is
        # likely ignored — see the module docstring.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers=HEADERS, json=payload, timeout=TIMEOUT, auth=BASIC_AUTH,
        stream=True,
    )
    resp.raise_for_status()
    # Force UTF-8 for the decode step below. Without this, `requests` guesses the
    # encoding from the Content-Type header (get_encoding_from_headers), and
    # `text/event-stream` responses essentially never declare a charset — so the
    # guess falls back to ISO-8859-1 per the old HTTP/1.1 default. Any multi-byte
    # UTF-8 character (e.g. "°", 0xC2 0xB0) then gets decoded byte-by-byte as
    # Latin-1, producing mojibake like "Â°" instead of "°". This silently breaks
    # exact-match/keyword scoring on any needle containing non-ASCII text.
    resp.encoding = "utf-8"

    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    finish_reason = ""

    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data:"):
            continue  # blank keep-alive lines between SSE events
        data = raw_line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta", {})
        if delta.get("content"):
            content_parts.append(delta["content"])
        if delta.get("reasoning_content"):
            reasoning_parts.append(delta["reasoning_content"])
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

    content = "".join(content_parts).strip()
    reasoning = "".join(reasoning_parts).strip()
    # If the model's answer ended up in reasoning_content instead of content
    # (happens on some builds when thinking can't be fully suppressed),
    # fall back to it rather than silently scoring an empty answer as a miss.
    answer = content if content else reasoning
    return {"answer": answer, "content": content, "reasoning": reasoning, "finish_reason": finish_reason}


def _chat_with_retry(messages: List[Dict], max_tokens: int = MAX_TOKENS, temperature: float = 0.0) -> Dict[str, Any]:
    """Retries transient errors (timeouts, dropped tunnel, etc.) so a network
    blip doesn't get permanently checkpointed as a quality failure."""
    last_err: Optional[Exception] = None
    for attempt in range(RETRIES + 1):
        try:
            return _chat(messages, max_tokens=max_tokens, temperature=temperature)
        except Exception as e:
            last_err = e
            if attempt < RETRIES:
                time.sleep(RETRY_WAIT)
    assert last_err is not None
    raise last_err


def count_tokens(text: str) -> Optional[int]:
    """
    Real token count via the server's /tokenize endpoint. None if not exposed.
    llama.cpp's server expects {"content": ...} with no model field; vLLM's
    expects {"model": ..., "prompt": ...}. Sending all three keys at once
    satisfies either schema without needing to know which backend is live.
    """
    try:
        r = requests.post(
            f"{BASE_URL}/tokenize", headers=HEADERS,
            json={"content": text, "model": MODEL, "prompt": text},
            timeout=30, auth=BASIC_AUTH,
        )
        r.raise_for_status()
        return len(r.json().get("tokens", []))
    except Exception:
        return None


_FILLER_CORPUS_FALLBACK = [
    # Biology
    "The human genome contains approximately three billion base pairs distributed across 23 chromosome pairs.",
    "Ribosomes translate messenger RNA sequences into chains of amino acids that fold into proteins.",
    "ATP synthase harnesses a proton gradient across the mitochondrial membrane to produce adenosine triphosphate.",
    "The cerebellum coordinates voluntary movement, fine motor control, and spatial orientation.",
    "CRISPR-Cas9 acts as molecular scissors that allow targeted editing of specific DNA sequences.",
    "Apoptosis is the process of programmed cell death that removes damaged or unwanted cells during development.",
    "Monoclonal antibodies are engineered proteins that bind to specific antigens on target cells.",
    "The blood-brain barrier selectively restricts which substances can pass from the bloodstream into the brain.",
    "Herd immunity is reached when enough individuals in a population are immune to prevent widespread transmission.",
    "Telomeres are repetitive DNA sequences at chromosome ends that shorten with each cell division.",
    # Physics
    "Entropy in an isolated system tends to increase over time according to the second law of thermodynamics.",
    "The Pauli exclusion principle states that no two fermions can occupy the same quantum state simultaneously.",
    "Superconductivity occurs when certain materials conduct electricity with zero resistance below a critical temperature.",
    "The Heisenberg uncertainty principle establishes fundamental limits on the precision of position and momentum measurements.",
    "General relativity describes gravity as the curvature of spacetime caused by mass and energy.",
    "Quantum entanglement links two particles such that measuring one instantly affects the state of the other.",
    "The photoelectric effect demonstrated that light behaves as discrete packets of energy called photons.",
    "Black holes form when matter collapses to such density that not even light can escape the gravitational field.",
    # Chemistry
    "Carbon forms more compounds than any other element due to its ability to bond in tetrahedral arrangements.",
    "Le Chatelier's principle states that a system at equilibrium shifts to counteract any imposed change.",
    "Electronegativity measures an atom's tendency to attract shared electrons in a chemical bond.",
    "Catalysts lower the activation energy of a reaction without being consumed in the process.",
    "The Arrhenius equation relates reaction rate to temperature and activation energy.",
    "Isotopes of an element have the same number of protons but different numbers of neutrons.",
    # History
    "The Treaty of Westphalia in 1648 established the principle of state sovereignty in international relations.",
    "The Black Death reduced Europe's population by an estimated one-third during the fourteenth century.",
    "The Silk Road connected China to the Mediterranean world and facilitated the exchange of goods and ideas.",
    "Johannes Gutenberg's printing press, invented around 1440, revolutionised the spread of written knowledge.",
    "The Industrial Revolution began in Britain in the late eighteenth century and transformed manufacturing processes.",
    "The Congress of Vienna in 1815 redrew the map of Europe after the Napoleonic Wars.",
    "The abolition of the transatlantic slave trade by Britain in 1807 marked a turning point in global history.",
    # Geography
    "The Amazon River discharges more freshwater into the Atlantic Ocean than any other river on Earth.",
    "Lake Baikal in Siberia holds approximately twenty percent of the world's unfrozen surface freshwater.",
    "The Mariana Trench reaches a depth of nearly eleven kilometres at its deepest point, the Challenger Deep.",
    "The Sahara Desert spans approximately 9.2 million square kilometres across northern Africa.",
    "Antarctica contains around seventy percent of the world's fresh water locked in its ice sheets.",
    "The Himalayan mountain range was formed by the collision of the Indian and Eurasian tectonic plates.",
    "The Nile River flows northward through eleven countries before emptying into the Mediterranean Sea.",
    # Economics
    "Supply and demand curves intersect at the equilibrium price where quantity supplied equals quantity demanded.",
    "Gross domestic product measures the total monetary value of goods and services produced within a country.",
    "Inflation erodes the purchasing power of currency by increasing the general price level over time.",
    "The Gini coefficient is a statistical measure of income distribution and economic inequality.",
    "Central banks adjust interest rates to influence borrowing, spending, and the overall pace of economic activity.",
    "Comparative advantage explains why nations benefit from specialising in goods they produce most efficiently.",
    "The Phillips curve historically suggested an inverse relationship between inflation and unemployment.",
    # Technology
    "Transistors are the fundamental switching components underlying all modern digital logic circuits.",
    "The internet relies on the TCP/IP protocol suite to route data packets across interconnected networks.",
    "Public-key cryptography allows two parties to communicate securely without sharing a secret key in advance.",
    "Solid-state drives store data using flash memory cells and have no moving mechanical parts.",
    "Machine learning models identify patterns in training data and generalise to make predictions on new inputs.",
    "Fibre-optic cables transmit data as pulses of light and offer far greater bandwidth than copper wire.",
    "The von Neumann architecture separates memory from processing units and underlies most modern computers.",
    # Mathematics
    "Euler's identity elegantly relates the five most important constants in mathematics in a single equation.",
    "The Riemann hypothesis, still unproven, concerns the location of non-trivial zeros of the zeta function.",
    "A Fourier transform decomposes a time-domain signal into its underlying frequency components.",
    "Gödel's incompleteness theorems show that any consistent formal system contains true but unprovable statements.",
    "Bayes' theorem provides a framework for updating the probability of a hypothesis in light of new evidence.",
    "The travelling salesman problem asks for the shortest route visiting a set of cities exactly once.",
    "Prime numbers have no divisors other than one and themselves and are infinite in number.",
    # Linguistics
    "The Sapir-Whorf hypothesis proposes that the language one speaks influences how one perceives the world.",
    "Phonemes are the minimal units of sound that distinguish one word from another in a given language.",
    "Creole languages develop from simplified contact languages into fully expressive natural languages over generations.",
    "The Indo-European family encompasses most languages of Europe, Iran, and the northern Indian subcontinent.",
    "Syntax defines the hierarchical rules by which words and phrases are combined into grammatical sentences.",
    "The Great Vowel Shift was a major change in English pronunciation that occurred between 1400 and 1700.",
    # Astronomy
    "Pulsars are rotating neutron stars that emit highly regular beams of electromagnetic radiation.",
    "The cosmic microwave background radiation is the thermal afterglow of the early universe.",
    "Dark matter neither emits nor absorbs light but exerts measurable gravitational effects on visible matter.",
    "Stellar nucleosynthesis forges heavier elements from hydrogen and helium through nuclear fusion inside stars.",
    "A parsec equals approximately 3.26 light-years and is the standard unit for measuring interstellar distances.",
    "The Hubble constant describes the rate at which the universe is currently expanding.",
    # Philosophy
    "Kant's categorical imperative holds that one should act only according to maxims that could be universal laws.",
    "Hume argued that the mind perceives constant conjunction, not direct causation, between events.",
    "Popper's falsifiability criterion holds that a claim is scientific only if it can in principle be disproved.",
    "Aristotle distinguished syllogistic deduction from inductive generalisation as separate forms of reasoning.",
    "The problem of induction raises the question of how repeated observations can justify universal conclusions.",
    # Environmental Science
    "The greenhouse effect occurs when atmospheric gases absorb and re-emit infrared radiation from the Earth's surface.",
    "Ocean acidification results from the dissolution of excess carbon dioxide into seawater.",
    "Biodiversity hotspots are regions with an exceptional concentration of endemic species under threat.",
    "The nitrogen cycle describes the continuous transformation of nitrogen between the atmosphere, soil, and organisms.",
    "Permafrost stores vast quantities of organic carbon that may be released as a greenhouse gas if it thaws.",
]


# ─── Wikipedia-backed filler pool ────────────────────────────────────────────
#
# _FILLER_CORPUS_FALLBACK above is only used if `datasets` isn't installed or
# the Wikipedia stream can't be reached (e.g. Kaggle's internet toggle is
# off). With ~80 short, topically disconnected sentences repeated on a
# shuffle-cycle, it under-stresses q4_0 KV quantization: real prose has a
# much messier token distribution (rare tokens, varied sentence length and
# structure, topical drift across paragraphs) than a small, fixed set of
# declarative facts cycled forever. It also makes a synthetic needle stand
# out by being the only sentence in the whole context about a specific
# person/company/measurement — an anomaly a model can spot by genre alone
# rather than by actually retrieving it.

WIKI_DATASET_NAME   = "wikimedia/wikipedia"
WIKI_DATASET_CONFIG = "20231101.en"
WIKI_POOL_CACHE     = Path(__file__).parent / "wiki_sentence_pool.json"

_WIKI_POOL: Optional[List[str]] = None   # lazy singleton, set by _get_filler_pool()
_WIKI_POOL_DISABLED = False              # set by --no-wiki


def _build_wiki_sentence_pool(n_sentences: int, cache_path: Path) -> Optional[List[str]]:
    """
    Streams the English Wikipedia snapshot via HF `datasets` (streaming=True,
    so this never downloads the full ~20GB dataset — it stops the moment it
    has enough sentences) and caches the result to disk so a checkpoint-resume
    or a second run in the same Kaggle session doesn't re-stream over the
    network. Returns None (caller falls back to the static corpus) if
    `datasets` isn't installed or the stream can't be reached.
    """
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                pool = json.load(f)
            if len(pool) >= n_sentences:
                print(f"[wiki-pool] Loaded {len(pool)} cached sentences from {cache_path}\n")
                return pool
        except Exception:
            pass  # corrupt/partial cache file — rebuild below

    try:
        from datasets import load_dataset
    except ImportError:
        print("[!] `datasets` not installed — falling back to the small static filler corpus.\n"
              "    Run: pip install datasets --break-system-packages\n")
        return None

    print(f"[wiki-pool] Streaming {WIKI_DATASET_NAME} ({WIKI_DATASET_CONFIG}) to build a "
          f"{n_sentences}-sentence pool (one-time cost this session, then cached to "
          f"{cache_path})...")
    try:
        stream = load_dataset(WIKI_DATASET_NAME, WIKI_DATASET_CONFIG, split="train", streaming=True)
        pool: List[str] = []
        for article in stream:
            for s in article["text"].split(". "):
                s = s.strip()
                # Skip short fragments and leftover wiki markup/list noise.
                if len(s) > 40 and not s.startswith(("*", "|", "==", "#")):
                    pool.append(s)
            if len(pool) >= n_sentences:
                break
    except Exception as e:
        print(f"[!] Wikipedia stream failed ({e}) — falling back to the small static filler corpus.\n")
        return None

    try:
        with open(cache_path, "w") as f:
            json.dump(pool, f)
    except Exception as e:
        print(f"[!] Could not write cache to {cache_path} ({e}) — continuing without cache.\n")
    print(f"[wiki-pool] Built {len(pool)} sentences.\n")
    return pool


def _get_filler_pool(n_sentences: int = 20000) -> List[str]:
    """Lazy singleton so the (potentially slow, network-bound) pool build
    happens once per process, not once per _pad_to_tokens() call."""
    global _WIKI_POOL
    if _WIKI_POOL is not None:
        return _WIKI_POOL
    if _WIKI_POOL_DISABLED:
        _WIKI_POOL = _FILLER_CORPUS_FALLBACK
        return _WIKI_POOL
    pool = _build_wiki_sentence_pool(n_sentences, WIKI_POOL_CACHE)
    _WIKI_POOL = pool if pool else _FILLER_CORPUS_FALLBACK
    return _WIKI_POOL


def _pad_to_tokens(target_tokens: int) -> str:
    """
    Generate filler prose sized to roughly target_tokens. Starts from a fast
    1.3-words/token estimate, then refines against the server's real
    tokenizer (a handful of /tokenize calls, not hundreds) so the context
    length you report is the context length you actually sent.
    """
    def make_words(n_words: int) -> List[str]:
        pool = _get_filler_pool()[:]
        random.shuffle(pool)
        words: List[str] = []
        while len(words) < n_words:
            for sentence in pool:
                words.extend(sentence.split())
                if len(words) >= n_words:
                    break
            random.shuffle(pool)  # re-shuffle on each full pass for variety
        return words[:n_words]

    words = make_words(max(0, int(target_tokens / 1.3)))

    for _ in range(3):
        n = count_tokens(" ".join(words))
        if n is None:
            break  # /tokenize not exposed by the proxy — stick with the estimate
        if n >= target_tokens:
            break
        ratio = target_tokens / max(n, 1)
        words = make_words(int(len(words) * ratio) + 20)

    return " ".join(words)


def _inject_needle(filler: str, needle: str, position: float = 0.5) -> str:
    sentences = filler.split(". ")
    idx = max(0, int(len(sentences) * position))
    sentences.insert(idx, needle)
    return ". ".join(sentences)


def _kw_pattern(keyword: str) -> "re.Pattern":
    """
    Case-insensitive containment check that respects token boundaries
    without using bare `\\b`, and treats "one more digit" and "a letter"
    as different kinds of boundary for a numeric keyword.

    Round 1 was plain substring `in` matching — passed "40" inside "1840"
    and "billion" inside "billionaire". Real false positives.

    Round 2 switched to `(?<!\\w)...(?!\\w)` (lookaround instead of `\\b`,
    since `\\b` never fires between two non-word characters — e.g. a space
    then a leading "-" — which silently failed to match "-40" preceded by
    whitespace). That part was correct and is unchanged below.

    But `(?!\\w)` blocks ANY trailing word character equally: one more
    digit ("41500") or a trailing letter ("415M") look identical to it. For
    a numeric keyword those are not the same thing — "$415M" is standard
    financial notation for exactly 415, not a different number — and this
    eval's own needles are phrased that way ("$120M", "$85M", "$210M").
    `(?!\\w)` can't tell "more of the same number" apart from "a unit
    suffix," so it rejected a model answer that was actually correct
    (BABILong's aggregation case: the model answered "= $415M" and this
    pattern still scored it a miss).

    Fix: when the keyword starts/ends in a digit, only a further DIGIT
    immediately adjacent counts as "still the same token" — a letter does
    not. A leading/trailing digit is still always blocked either way, so
    "415" still correctly rejects "41500" and "1415".

    Trade-off, stated plainly rather than hidden: for an opaque alphanumeric
    ID that ends in a digit (e.g. "DELTA-7749"), this would in principle
    also accept a directly-glued trailing letter as a match
    ("DELTA-7749X"). NIAH's system prompt forces bare single-token answers,
    so this hasn't been observed and isn't expected to be — but it's a real
    trade-off, not a free lunch.
    """
    kw = keyword.strip()
    lookbehind = r"(?<!\d)" if kw[:1].isdigit() else r"(?<!\w)"
    lookahead  = r"(?!\d)" if kw[-1:].isdigit() else r"(?!\w)"
    return re.compile(lookbehind + re.escape(kw) + lookahead, re.IGNORECASE)


def _score_exact(answer: str, expected: str) -> bool:
    return bool(_kw_pattern(expected).search(answer))


def _score_partial(answer: str, keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    found = sum(1 for kw in keywords if _kw_pattern(kw).search(answer))
    return found / len(keywords)


def _haystack_collisions(text: str, keywords: List[str]) -> List[str]:
    """Keywords that already appear in `text` by chance, using the same
    word-boundary-aware match as scoring. Used to catch a real text corpus
    incidentally containing an answer before the needle is even injected."""
    return [kw for kw in keywords if _kw_pattern(kw).search(text)]


def _build_filler_avoiding(target_tokens: int, avoid_keywords: List[str], max_attempts: int = 3) -> str:
    """
    _pad_to_tokens(), but resamples if the haystack already contains one of
    the answer keywords by chance. This barely mattered with the old
    10-sentence static corpus; it matters a lot with a large real-text pool,
    where a common word like "billion" or a short numeric string can
    plausibly turn up on its own within 100K+ tokens of real prose — letting
    a model "pass" a retrieval test without ever needing the needle.
    """
    filler = ""
    collisions: List[str] = []
    for attempt in range(max_attempts):
        filler = _pad_to_tokens(target_tokens)
        collisions = _haystack_collisions(filler, avoid_keywords)
        if not collisions:
            return filler
        print(f"    [!] haystack collision on {collisions} (attempt {attempt + 1}/{max_attempts}) "
              f"— resampling filler...")
    print(f"    [!] WARNING: proceeding with a haystack that still contains {collisions} "
          f"after {max_attempts} attempts — a PASS on this case may be partly attributable to "
          f"incidental co-occurrence rather than genuine retrieval.")
    return filler


def resolve_model_name(requested: str) -> str:
    """
    deploy_llama.py serves the alias 'Qwen3.6-35B-A3B-Uncensored'; deploy_vllm.py
    serves 'shieldstar/Qwen3.6-35B-A3B-int4-AutoRound-EC' and 'Qwen3.6-35B-A3B'
    (no '-Uncensored'). If MODEL doesn't match what the server actually reports,
    every single request below would 404 on "model not found" — burning the
    whole session before you'd notice. Check first, auto-correct if there's
    exactly one model loaded.
    """
    try:
        r = requests.get(f"{BASE_URL}/v1/models", headers=HEADERS, timeout=10, auth=BASIC_AUTH)
        r.raise_for_status()
        available = [m["id"] for m in r.json().get("data", [])]
    except Exception as e:
        print(f"[!] Could not query {BASE_URL}/v1/models ({e}) — proceeding with '{requested}' as given.\n")
        return requested

    if requested in available:
        return requested
    if available:
        print(f"[!] '{requested}' isn't being served here. Available model name(s): {available}. "
              f"Using '{available[0]}' for this run.\n")
        return available[0]
    print(f"[!] /v1/models returned no models — proceeding with '{requested}' as given.\n")
    return requested


# ─── Preflight ────────────────────────────────────────────────────────────────

def preflight_check() -> None:
    """One cheap call before committing a 12-hour session to the full suite."""
    try:
        result = _chat([
            {"role": "system", "content": "Answer with ONLY the single word: ready"},
            {"role": "user", "content": "Are you ready? Reply with one word."},
        ], max_tokens=64)
    except Exception as e:
        print(f"[!] Preflight call failed: {e}\n    Check the server/proxy is actually reachable before running the full suite.\n")
        return

    if not result["content"].strip() and result["reasoning"]:
        print(f"[!] WARNING: `content` came back empty but `reasoning_content` has "
              f"{len(result['reasoning'])} chars on a TRIVIAL one-word prompt. The server is almost "
              f"certainly still running with --reasoning on. Every call below will spend time thinking "
              f"instead of just answering, and short max_tokens will truncate before reaching an answer.\n"
              f"    Fix: set reasoning.enabled = False in deploy_llama.py and redeploy, then re-run this.\n")
    elif result["finish_reason"] == "length" and not result["content"].strip():
        print("[!] WARNING: hit max_tokens with no answer on a trivial prompt — reasoning is very "
              "likely still forced on server-side.\n")
    else:
        print(f"[✓] Preflight OK — sample answer: {result['content'][:60]!r}\n")


# ─── Result Tracker with Checkpointing ───────────────────────────────────────

class ResultTracker:
    def __init__(self, checkpoint_path: Optional[str] = None):
        self.results: List[Dict] = []
        self.checkpoint_path = checkpoint_path
        if checkpoint_path:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)  # add this line
            if Path(checkpoint_path).exists():
                with open(checkpoint_path) as f:
                    self.results = json.load(f)
                print(f"[checkpt] Resumed: {len(self.results)} prior results loaded")

    def add(self, suite: str, ctx_len: int, test_name: str, passed: bool,
            score: float, latency_s: float, details: str = "", is_error: bool = False):
        entry = {
            "suite": suite, "ctx_len": ctx_len, "test": test_name,
            "passed": passed, "score": round(score, 3),
            "latency_s": round(latency_s, 2), "details": details,
            "is_error": is_error,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        self.results.append(entry)
        status = "ERROR" if is_error else ("✅ PASS" if passed else "❌ FAIL")
        print(f"  {status} | {suite}/{test_name} @ {ctx_len//1000}K | "
              f"score={score:.2f} | {latency_s:.1f}s")
        if details:
            print(f"         → {details[:550]}")
        if self.checkpoint_path:
            with open(self.checkpoint_path, "w") as f:
                json.dump(self.results, f, indent=2)

    def already_done(self, suite: str, ctx_len: int, test_name: str) -> bool:
        # Errors (timeouts, dropped tunnel, etc.) don't count as "done" — only
        # a real pass/fail does. Otherwise a transient blip gets permanently
        # checkpointed and silently skipped on every future resume.
        return any(
            r["suite"] == suite and r["ctx_len"] == ctx_len and r["test"] == test_name
            and not r.get("is_error", False)
            for r in self.results
        )

    def summary(self) -> Dict:
        scored = [r for r in self.results if not r.get("is_error", False)]
        total = len(scored)
        passed = sum(1 for r in scored if r["passed"])
        avg_score = sum(r["score"] for r in scored) / total if total else 0
        by_suite: Dict[str, Dict] = {}
        for r in scored:
            s = r["suite"]
            if s not in by_suite:
                by_suite[s] = {"total": 0, "passed": 0, "scores": []}
            by_suite[s]["total"] += 1
            by_suite[s]["passed"] += int(r["passed"])
            by_suite[s]["scores"].append(r["score"])
        n_errors = len(self.results) - total
        return {"total": total, "passed": passed, "errors": n_errors,
                "avg_score": round(avg_score, 3), "by_suite": by_suite}

    def save_report(self, path: str):
        report = {"summary": self.summary(), "results": self.results}
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[✓] Full report saved → {path}")


# ─── Suite 1: Basic NIAH ─────────────────────────────────────────────────────

NIAH_NEEDLES = [
    ("The secret code is DELTA-7749.", "DELTA-7749"),
    ("The launch password is AURORA-PRIME.", "AURORA-PRIME"),
    ("The hidden phrase is CRIMSON-FALCON.", "CRIMSON-FALCON"),
]

def run_niah(tracker: ResultTracker, ctx_len: int):
    suite = "niah"
    for depth_name, position in [("start", 0.05), ("middle", 0.50), ("end", 0.95)]:
        for needle_text, expected in NIAH_NEEDLES[:1]:
            test_name = f"depth_{depth_name}"
            if tracker.already_done(suite, ctx_len, test_name):
                print(f"  [skip] {suite}/{test_name} @ {ctx_len//1000}K"); continue
            filler = _build_filler_avoiding(ctx_len - 200, [expected])
            context = _inject_needle(filler, needle_text, position)
            messages = [
                {"role": "system", "content": "Answer with ONLY the exact value asked for. No explanation."},
                {"role": "user",   "content": f"{context}\n\nWhat is the secret code mentioned in the text?"},
            ]
            t0 = time.time()
            try:
                result = _chat_with_retry(messages)
                latency = time.time() - t0
                answer = result["answer"]
                passed = _score_exact(answer, expected)
                tracker.add(suite, ctx_len, test_name, passed, float(passed), latency,
                            f"expected={expected!r} | got={answer[:80]!r} | finish={result['finish_reason']}")
            except Exception as e:
                tracker.add(suite, ctx_len, test_name, False, 0.0, time.time() - t0,
                            f"ERROR: {e}", is_error=True)


# ─── Suite 2: Multi-Needle ───────────────────────────────────────────────────

MULTI_NEEDLES = [
    ("The Alpha key value is 1847.", "1847"),
    ("The Beta key value is 5293.",  "5293"),
    ("The Gamma key value is 7761.", "7761"),
    ("The Delta key value is 3042.", "3042"),
]

def run_multi_needle(tracker: ResultTracker, ctx_len: int):
    suite = "multi_needle"
    for n_needles in [2, 4]:
        test_name = f"n{n_needles}_needles"
        if tracker.already_done(suite, ctx_len, test_name):
            print(f"  [skip] {suite}/{test_name} @ {ctx_len//1000}K"); continue
        needles = MULTI_NEEDLES[:n_needles]
        expected_values = [v for _, v in needles]
        filler = _build_filler_avoiding(ctx_len - 300 * n_needles, expected_values)
        for i, (needle_text, _) in enumerate(needles):
            filler = _inject_needle(filler, needle_text, (i + 1) / (n_needles + 1))
        query = ("List ALL key values mentioned in the text (Alpha, Beta, Gamma, Delta). "
                 "Return only the numbers, one per line.")
        messages = [
            {"role": "system", "content": "Extract and list ONLY the exact values. No explanation."},
            {"role": "user",   "content": f"{filler}\n\n{query}"},
        ]
        t0 = time.time()
        try:
            result = _chat_with_retry(messages, max_tokens=128)
            latency = time.time() - t0
            answer = result["answer"]
            score = _score_partial(answer, expected_values)
            tracker.add(suite, ctx_len, test_name, score >= 0.75, score, latency,
                        f"expected={expected_values} | score={score:.2f} | got={answer[:200]!r} | "
                        f"finish={result['finish_reason']}")
        except Exception as e:
            tracker.add(suite, ctx_len, test_name, False, 0.0, time.time() - t0,
                        f"ERROR: {e}", is_error=True)


# ─── Suite 3: NoLiMa (paraphrased needle) ────────────────────────────────────

NOLIMA_CASES = [
    {
        "needle":   "The annual revenue of HelixCorp in 2024 was $4.7 billion.",
        "query":    "How much money did HelixCorp earn across all of last year?",
        "keywords": ["4.7", "billion"],
        "pos":      0.1,   # early-context retrieval
    },
    {
        "needle":   "Dr. Elara Voss discovered the compound RX-9 inhibits tau protein aggregation.",
        "query":    "Which researcher found that RX-9 prevents the clumping of tau proteins?",
        "keywords": ["elara", "voss"],
        "pos":      0.5,   # mid-context retrieval
    },
    {
        "needle":   "The minimum operating temperature of the Titan-III reactor is -40°C.",
        "query":    "What is the coldest temperature at which the Titan-III system can function?",
        "keywords": ["-40"],   # "40" was a substring of "-40"; dropped to avoid wrong-sign passes
        "pos":      0.9,   # late-context retrieval
    },
]

def run_nolima(tracker: ResultTracker, ctx_len: int):
    suite = "nolima"
    for i, case in enumerate(NOLIMA_CASES):
        test_name = f"case_{i+1}"
        if tracker.already_done(suite, ctx_len, test_name):
            print(f"  [skip] {suite}/{test_name} @ {ctx_len//1000}K"); continue
        filler  = _build_filler_avoiding(ctx_len - 200, case["keywords"])
        context = _inject_needle(filler, case["needle"], position=case["pos"])
        messages = [
            {"role": "system", "content": "Answer precisely based only on the provided text. Be brief."},
            {"role": "user",   "content": f"{context}\n\n{case['query']}"},
        ]
        t0 = time.time()
        try:
            result = _chat_with_retry(messages)
            latency = time.time() - t0
            answer = result["answer"]
            score = _score_partial(answer, case["keywords"])
            tracker.add(suite, ctx_len, test_name, score >= 1.0, score, latency,
                        f"keywords={case['keywords']} | got={answer[:200]!r} | finish={result['finish_reason']}")
        except Exception as e:
            tracker.add(suite, ctx_len, test_name, False, 0.0, time.time() - t0,
                        f"ERROR: {e}", is_error=True)


# ─── Suite 4: RULER-lite (2-hop reasoning) ───────────────────────────────────

RULER_CASES = [
    {
        "fact_a":   "Project Orion is led by scientist Marcus Webb.",
        "fact_b":   "Marcus Webb graduated from the University of Cape Town.",
        "query":    "Which university did the leader of Project Orion graduate from?",
        "keywords": ["cape town"], "pos_a": 0.2, "pos_b": 0.8,
    },
    {
        "fact_a":   "The Helios satellite operates at an altitude of 35,786 km.",
        "fact_b":   "Objects at 35,786 km altitude are in geostationary orbit.",
        "query":    "What type of orbit does the Helios satellite use?",
        "keywords": ["geostationary"], "pos_a": 0.3, "pos_b": 0.7,
    },
    {
        "fact_a":   "Compound Z-40 was synthesised by Dr. Amara Singh in 2019.",
        "fact_b":   "Dr. Amara Singh is a researcher at the Nairobi Institute of Chemistry.",
        "query":    "At which institution was compound Z-40 synthesised?",
        "keywords": ["nairobi"], "pos_a": 0.15, "pos_b": 0.85,
    },
]

def run_ruler(tracker: ResultTracker, ctx_len: int):
    suite = "ruler"
    for i, case in enumerate(RULER_CASES):
        test_name = f"hop2_case_{i+1}"
        if tracker.already_done(suite, ctx_len, test_name):
            print(f"  [skip] {suite}/{test_name} @ {ctx_len//1000}K"); continue
        filler = _build_filler_avoiding(ctx_len - 400, case["keywords"])
        filler = _inject_needle(filler, case["fact_a"], case["pos_a"])
        filler = _inject_needle(filler, case["fact_b"], case["pos_b"])
        messages = [
            {"role": "system", "content": "Reason carefully over the text and answer in one sentence."},
            {"role": "user",   "content": f"{filler}\n\nQuestion: {case['query']}"},
        ]
        t0 = time.time()
        try:
            result = _chat_with_retry(messages)
            latency = time.time() - t0
            answer = result["answer"]
            score = _score_partial(answer, case["keywords"])
            tracker.add(suite, ctx_len, test_name, score >= 1.0, score, latency,
                        f"keywords={case['keywords']} | got={answer[:200]!r} | finish={result['finish_reason']}")
        except Exception as e:
            tracker.add(suite, ctx_len, test_name, False, 0.0, time.time() - t0,
                        f"ERROR: {e}", is_error=True)


# ─── Suite 5: BABILong-lite (multi-doc chain) ────────────────────────────────

def run_babilong(tracker: ResultTracker, ctx_len: int):
    suite = "babilong"
    cases = [
        {
            "test_name": "entity_chain",
            "facts":     [
                "Document 14: The Zephyr protocol was created by organisation NOVA.",
                "Document 27: NOVA is headquartered in the city of Reykjavik.",
                "Document 51: Reykjavik is the capital of Iceland.",
            ],
            "query":    "In which country was the Zephyr protocol created?",
            "keywords": ["iceland"],
            "positions": [0.1, 0.4, 0.75],
        },
        {
            "test_name": "aggregation",
            "facts":     [
                "Report A: Division North achieved a profit of $120M.",
                "Report B: Division South achieved a profit of $85M.",
                "Report C: Division East achieved a profit of $210M.",
            ],
            "query":    "What is the total combined profit of all three divisions?",
            "keywords": ["415"],   # "415m"/"$415" were redundant substrings; "415" matches all valid formats
            "positions": [0.2, 0.5, 0.8],
        },
    ]
    for case in cases:
        test_name = case["test_name"]
        if tracker.already_done(suite, ctx_len, test_name):
            print(f"  [skip] {suite}/{test_name} @ {ctx_len//1000}K"); continue
        filler = _build_filler_avoiding(ctx_len - 600, case["keywords"])
        for fact, pos in zip(case["facts"], case["positions"]):
            filler = _inject_needle(filler, fact, pos)
        messages = [
            {"role": "system", "content": "Answer based on all documents in the text. Be concise."},
            {"role": "user",   "content": f"{filler}\n\nQuestion: {case['query']}"},
        ]
        t0 = time.time()
        try:
            result = _chat_with_retry(messages, max_tokens=512)
            latency = time.time() - t0
            answer = result["answer"]
            score = _score_partial(answer, case["keywords"])
            tracker.add(suite, ctx_len, test_name, score >= 1.0, score, latency,
                        f"keywords={case['keywords']} | got={answer[:500]!r} | finish={result['finish_reason']}")
        except Exception as e:
            tracker.add(suite, ctx_len, test_name, False, 0.0, time.time() - t0,
                        f"ERROR: {e}", is_error=True)


# ─── Entrypoint ───────────────────────────────────────────────────────────────

SUITE_MAP = {
    "niah":         run_niah,
    "multi_needle": run_multi_needle,
    "nolima":       run_nolima,
    "ruler":        run_ruler,
    "babilong":     run_babilong,
}

def main():
    global BASE_URL, MODEL
    parser = argparse.ArgumentParser(description="ShadowEngine Long-Context Eval Suite")
    parser.add_argument("--ctx", nargs="+", type=int,
                        default=[32768, 65536, 131072, 192000],
                        help="Context lengths (tokens).")
    parser.add_argument("--suite", nargs="+", default=["all"],
                        choices=list(SUITE_MAP.keys()) + ["all"])
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="JSON checkpoint file — resume from partial run.")
    parser.add_argument("--base-url", type=str, default=BASE_URL)
    parser.add_argument("--model",    type=str, default=MODEL)
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip the reasoning-mode sanity check (not recommended).")
    parser.add_argument("--wiki-sentences", type=int, default=20000,
                        help="Size of the Wikipedia-derived filler pool (default: 20000).")
    parser.add_argument("--no-wiki", action="store_true",
                        help="Skip Wikipedia streaming; use the small static filler corpus instead.")
    args = parser.parse_args()

    global _WIKI_POOL_DISABLED
    _WIKI_POOL_DISABLED = args.no_wiki

    BASE_URL = args.base_url
    MODEL    = args.model
    suites   = list(SUITE_MAP.keys()) if "all" in args.suite else args.suite

    print("=" * 62)
    print("  ShadowEngine — Long Context Evaluation Suite")
    print("=" * 62)
    print(f"  API        : {BASE_URL}")
    print(f"  Model      : {MODEL}")
    print(f"  Ctx lengths: {[f'{c//1000}K' for c in args.ctx]}")
    print(f"  Suites     : {suites}")
    print(f"  Checkpoint : {args.checkpoint or 'disabled'}")
    print("=" * 62 + "\n")

    try:
        r = requests.get(f"{BASE_URL}/health", timeout=10, auth=BASIC_AUTH)
        print(f"[✓] /health → {r.status_code}\n" if r.status_code == 200
              else f"[!] /health → {r.status_code} (server may not be ready)\n")
    except Exception as e:
        print(f"[!] Cannot reach {BASE_URL}: {e}")
        print("    Start proxy_stream.py or proxy_buffer.py first.\n")

    MODEL = resolve_model_name(MODEL)
    if not args.skip_preflight:
        preflight_check()

    print("[wiki-pool] Preparing filler corpus (one-time cost this session)...")
    pool = _get_filler_pool(n_sentences=args.wiki_sentences)
    using_fallback = pool is _FILLER_CORPUS_FALLBACK
    print(f"[wiki-pool] Ready — {len(pool)} sentences "
          f"({'static fallback corpus' if using_fallback else 'Wikipedia-derived'}).\n")

    tracker = ResultTracker(checkpoint_path=args.checkpoint)

    for ctx_len in sorted(args.ctx):
        print(f"\n{'─'*50}")
        print(f"  Context: {ctx_len:,} tokens ({ctx_len//1000}K)")
        print(f"{'─'*50}")
        for suite_name in suites:
            print(f"\n  [{suite_name.upper()}]")
            SUITE_MAP[suite_name](tracker, ctx_len)

    summary = tracker.summary()
    print("\n" + "=" * 62)
    print("  RESULTS")
    print("=" * 62)
    print(f"  Total : {summary['total']}  |  Passed: {summary['passed']}  |  "
          f"Errors: {summary['errors']}  |  Avg score: {summary['avg_score']:.3f}")
    print()
    for s_name, s in summary["by_suite"].items():
        avg = sum(s["scores"]) / len(s["scores"]) if s["scores"] else 0
        print(f"  {s_name:<15} {s['passed']}/{s['total']} passed  avg={avg:.2f}")
    print("=" * 62)

    if args.save_results:
        results_dir = Path(__file__).parent / "eval_results"
        results_dir.mkdir(exist_ok=True)
        ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = results_dir / f"report_{ts}.json"
        tracker.save_report(str(path))


if __name__ == "__main__":
    main()