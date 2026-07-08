# deploy_vllm.py
"""
ShadowEngine — Kaggle deploy script (vLLM Edition)
=========================================================================
Features: vLLM API server, ngrok tunnel, GPU health check, ntfy.sh remote control, and local file logging.
Flow:
    1. install_deps()        -> apt + pip installs on the Kaggle kernel
    2. kill_switch_loop()    -> background thread listening on ntfy.sh for SHUTDOWN_GPU
    3. start_http_server()   -> launches vLLM's OpenAI-compatible API server (subprocess)
    4. start_ngrok_tunnel() or start_cloudflare_tunnel()  -> exposes that server publicly via pyngrok or clouflare
    5. keep-alive loop until killed (ntfy) or the kernel times out

Then on YOUR machine, point local_proxy.py's REMOTE_HOST at the printed
ngrok URL and hit http://127.0.0.1:8000/v1/... as if it were a local server.

--------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import re
import queue
from pathlib import Path
from typing import Any, Dict, Optional
import warnings
warnings.filterwarnings("ignore")
import requests

# =============================================================================
# SECTION 0 — SECRETS (Kaggle-only, private kernel; no .env on Kaggle)
# =============================================================================

def _get_secret(name: str, fallback: str = "") -> str:
    """Prefer Kaggle Secrets (Add-ons -> Secrets) over a hardcoded fallback."""
    try:
        from kaggle_secrets import UserSecretsClient 
        val = UserSecretsClient().get_secret(name)
        return val if val else fallback
    except Exception:
        return fallback

NGROK_TOKEN = _get_secret("NGROK_TOKEN", fallback="YOUR_NGROK_TOKEN")
HF_TOKEN = _get_secret("HF_TOKEN", fallback="YOUR_HF_TOKEN")
NTFY_CHANNEL = _get_secret("NTFY_CHANNEL", fallback="YOUR_NTFY_CHANNEL")  
BASIC_AUTH_USER = _get_secret("BASIC_AUTH_USER", fallback="YOUR_AUTH_USERNAME")
BASIC_AUTH_PASS = _get_secret("BASIC_AUTH_PASS", fallback="YOUR_AUTH_PASSWORD")

# =============================================================================
# SECTION 1 — INLINE CONFIG (was config.yaml; no sibling files on Kaggle)
# =============================================================================
 
CONFIG: Dict[str, Any] = {
    "model": {
        "name": "shieldstar/Qwen3.6-35B-A3B-int4-AutoRound-EC",
        "dtype": "float16",             # Forced FP16 for T4 compatibility
        "quantization": None,           # Native INT4 math
        "max_model_len": 262144,        # 256k context window
        "gpu_memory_utilization": 0.95, # High utilization to maximize KV Cache
        "trust_remote_code": True,
        "tokenizer_mode": "auto",
        "tensor_parallel_size": 2,      # Splitting across both T4s
        "max_num_seqs": 6,              # Dedicated single-stream processing
        # DOES NOT SUPPORT FP8 math If you force vLLM to use FP8 KV cache,
        # it will emulate the decompression in software, causing inference to crawl.

        # "kv_cache_dtype": "fp8",         # Reduces KV memory footprint for 256K attempts 
        # "calculate_kv_scales": True,     # Better than fixed 1.0 scales for FP8 KV
        # "reasoning_parser": "qwen3",     # It processes the thinking block internally but only returns the final, clean answer to your API request.
    },
    "serve": {
        "host": "0.0.0.0",
        "port": 8000,
        "api_keys": [], 
    },
    "ngrok": {
        "enabled": True,
        "region": "",           # us, eu, ap, etc. — blank = auto
        "subdomain": "",        # paid plan only
        "basic_auth": True,     # True = secured, False = open
    },
    "performance": {
        "enforce_eager": False,                  # Crucial to prevent MoE Graph OOM
        "enable_prefix_caching": False,
        "disable_custom_all_reduce": True,
        "language_model_only": True,            # Drops unused vision projector logic
        "prefetch_safetensors": True,
    },
    "monitoring": {
        "log_dir": "./logs",
    },
}
 
 
# =============================================================================
# SECTION 2 — Notifications (ntfy.sh)
# =============================================================================
 
def send_log(message: str, priority: int = 3) -> None:
    """Pushes deployment updates to your remote terminal."""
    try:
        requests.post(
            f"https://ntfy.sh/{NTFY_CHANNEL}",
            data=f"[KAGGLOG]: {message}".encode("utf-8"),
            headers={"Priority": str(priority)},
            timeout=10,
        )
    except Exception:
        pass
 
 
# =============================================================================
# SECTION 3 — Logging
# =============================================================================
 
def setup_logging(log_dir: str, verbose: bool = False) -> None:
    """Configures global logging: INFO to console, DEBUG to file."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] — %(message)s", datefmt="%H:%M:%S")
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    
    fh = logging.FileHandler(os.path.join(log_dir, "server.log"), mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    logging.getLogger("pyngrok").setLevel(logging.WARNING)
 
 
# =============================================================================
# SECTION 4 — Dependency installation (Kaggle environment)
# =============================================================================
 
def detect_gpu() -> Dict[str, Any]:
    info: Dict[str, Any] = {"count": 0, "names": [], "total_vram_gb": 0.0}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    info["count"] += 1
                    info["names"].append(parts[0])
                    try:
                        info["total_vram_gb"] += int(parts[1].split()[0]) / 1024
                    except ValueError:
                        pass
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return info

def install_deps() -> None:
    logging.info("[*] Installing dependencies (vLLM, pyngrok)...")
    send_log("Phase 1: Installing vLLM dependencies...", priority=3)

    subprocess.run(
        "apt-get update && apt-get install -y zstd",
        shell=True, check=False,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    logging.info("[*] zstd installed")
    # vLLM FlashInfer libcuda.so symlink fix for Kaggle T4s
    _NVIDIA_LIB   = "/usr/local/nvidia/lib64"
    _CUDA_COMPAT  = "/usr/local/cuda-12.8/compat"
    _TMP_STUBS    = "/tmp/flashinfer_cuda_stubs"
    os.makedirs(_TMP_STUBS, exist_ok=True)
    _cuda_src = next(
        (p for p in [
            f"{_NVIDIA_LIB}/libcuda.so",
            f"{_NVIDIA_LIB}/libcuda.so.1",
            f"{_CUDA_COMPAT}/libcuda.so",
            f"{_CUDA_COMPAT}/libcuda.so.1",
        ] if os.path.exists(p)),
        None,
    )

    if _cuda_src:
        _dst = f"{_TMP_STUBS}/libcuda.so"
        if not os.path.lexists(_dst):
            os.symlink(_cuda_src, _dst)
        for _evar in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
            _prev = os.environ.get(_evar, "")
            os.environ[_evar] = f"{_NVIDIA_LIB}:{_TMP_STUBS}:{_prev}".rstrip(":")
        logging.info(f"[*] libcuda.so fix: {_cuda_src} → {_dst} | LIBRARY_PATH updated")
    else:
        logging.warning("[*] libcuda.so fix: WARNING — libcuda.so not found in known locations")
 
    cmd = [
        sys.executable, "-m", "pip", "install", "--quiet", "--upgrade",
        "opentelemetry-api>=1.36.0,<1.39.0", "opentelemetry-sdk>=1.36.0,<1.39.0",
        "starlette>=0.49.1,<1.0.0", "vllm", "pyngrok", "requests",
    ]

    result = subprocess.run(cmd, check=False, capture_output=True, text=True)

    if result.returncode != 0:
        logging.error(f"[!] pip install failed: {result.stderr[-2000:]}")
        sys.exit(1)
    else:
        logging.info("[*] Python packages installed successfully.")
    
    subprocess.run(
        "wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb && dpkg -i cloudflared-linux-amd64.deb",
        shell=True, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    logging.info("[*] cloudflared installed")

# =============================================================================
# SECTION 5 — vLLM server + ngrok tunnel
# =============================================================================
 
class VLLMServer:
    """Launch vLLM's OpenAI-compatible API server as a subprocess and tunnel it."""
 
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self._server_process: Optional[subprocess.Popen] = None
        self._stop_event = threading.Event()
 
    def start_http_server(self) -> None:
        """Launch vLLM's OpenAI-compatible server as a subprocess."""
        model_cfg = self.cfg["model"]
        serve_cfg = self.cfg["serve"]
        perf_cfg = self.cfg["performance"]

        tp_size = model_cfg.get("tensor_parallel_size", 1)  # use every visible GPU (e.g. Kaggle's 2x T4)

        logging.info(f"Preparing to launch vLLM for model: {model_cfg['name']} (TP: {tp_size})")
        send_log(f"Phase 2: Booting vLLM Engine for {model_cfg['name']}...", priority=4)

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_cfg["name"],
            "--served-model-name", model_cfg["name"], "Qwen3.6-35B-A3B",
            "--host", serve_cfg["host"],
            "--port", str(serve_cfg["port"]),
            "--dtype", model_cfg.get("dtype", "float16"),
            "--max-model-len", str(model_cfg.get("max_model_len", 262144)),
            "--gpu-memory-utilization", str(model_cfg.get("gpu_memory_utilization", 0.95)),
            "--tokenizer-mode", model_cfg.get("tokenizer_mode", "auto"),
            "--tensor-parallel-size", str(tp_size),
            "--attention-backend", "TRITON_ATTN",
        ]

        # Apply Model Config Options
        if model_cfg.get("max_num_seqs"):
            cmd += ["--max-num-seqs", str(model_cfg["max_num_seqs"])]
        if model_cfg.get("trust_remote_code"):
            cmd.append("--trust-remote-code")
        if model_cfg.get("quantization"):
            cmd += ["--quantization", model_cfg["quantization"]]
        if model_cfg.get("reasoning_parser"):
            cmd += ["--reasoning-parser", model_cfg["reasoning_parser"]]

        # Apply Serving Config Options
        if serve_cfg.get("api_keys"):
            cmd += ["--api-key", *serve_cfg["api_keys"]]

        # Apply Performance Config Options
        if perf_cfg.get("enable_prefix_caching"):
            cmd.append("--enable-prefix-caching")
        if perf_cfg.get("enforce_eager"):
            cmd.append("--enforce-eager")
        if perf_cfg.get("disable_custom_all_reduce") and tp_size > 1:
            cmd.append("--disable-custom-all-reduce")
        if perf_cfg.get("language_model_only"):
            cmd.append("--language-model-only")
        if perf_cfg.get("prefetch_safetensors", True):
            cmd += ["--safetensors-load-strategy", "prefetch"]
        
        cmd += ["--generation-config", "vllm"]  # ignore HF generation_config.json
 
        logging.info(f"Executing: {' '.join(cmd)}")
        
        sub_env = os.environ.copy()
    
        sub_env["VLLM_LOGGING_LEVEL"] = "ERROR"
        sub_env["VLLM_CONFIGURE_LOGGING"] = "0"
        sub_env["TRANSFORMERS_VERBOSITY"] = "error"
        sub_env["PYTHONWARNINGS"] = "ignore"
        sub_env["NCCL_P2P_DISABLE"] = "1"
        sub_env["NCCL_IB_DISABLE"] = "1"
        sub_env["OMP_NUM_THREADS"] = "1"
        sub_env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        # sub_env["VLLM_DISABLE_DEEP_GEMM"] = "1"

        if HF_TOKEN:
            sub_env["HF_TOKEN"] = HF_TOKEN
            sub_env["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN  
                                                            
        cuda_stubs = "/usr/local/cuda/lib64/stubs"          # LIBRARY_PATH → compile-time linker search path for FlashInfer's JIT kernels.
        nvidia_lib = "/usr/local/nvidia/lib64"              # The real libcuda.so on Kaggle T4 is in /usr/local/nvidia/lib64; we also
        extra = f"{nvidia_lib}:{cuda_stubs}"                # include the stubs dir (which now has a symlink to it) for belt+suspenders.
        sub_env["LIBRARY_PATH"] = f"{extra}:{sub_env.get('LIBRARY_PATH', '')}"
        sub_env["LD_LIBRARY_PATH"] = f"{extra}:{sub_env.get('LD_LIBRARY_PATH', '')}"

        self._server_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=sub_env,
        )
 
        # Stream subprocess output into our own logger in the background.
        def _pump_output() -> None:
            if not self._server_process or not self._server_process.stdout:
                return
            for line in self._server_process.stdout:
                logging.info(f"[vllm] {line.rstrip()}")
 
        threading.Thread(target=_pump_output, daemon=True).start()
 
        # Large models can take many minutes to load — poll /health generously.
        base = f"http://localhost:{serve_cfg['port']}"
        deadline = time.time() + 1800  # 30 min ceiling
        logging.info("Waiting for HTTP API to become healthy (timeout: 30 min)…")
        last_ping_log = 0.0
 
        while time.time() < deadline:
            if self._server_process.poll() is not None:
                raise RuntimeError(
                    f"vLLM server process exited early with code {self._server_process.returncode}"
                )
            try:
                r = requests.get(f"{base}/health", timeout=3)
                if r.status_code == 200:
                    logging.info("HTTP API is HEALTHY at %s", base)
                    return
            except requests.RequestException:
                pass

            now = time.time()
            if now - last_ping_log > 60:
                logging.info("...still waiting for model load (this is normal for 30B-class models)")
                last_ping_log = now
            time.sleep(3)
 
        raise RuntimeError("vLLM API server did not become healthy within 30 minutes")

    # ---- Cloudflare tunnel -------------------------------------------------

    def start_cloudflare_tunnel(self) -> str:
        local_port = self.cfg["serve"]["port"]
        logging.info("Starting Cloudflare tunnel on port %s...", local_port)

        self._cloudflare_process = subprocess.Popen(
            ["cloudflared", "tunnel", "--url", f"http://localhost:{local_port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        line_queue: queue.Queue = queue.Queue()

        def _reader() -> None:
            for line in self._cloudflare_process.stdout:
                line_queue.put(line)
            line_queue.put(None)  # sentinel — process stdout closed

        threading.Thread(target=_reader, daemon=True).start()

        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                line = line_queue.get(timeout=1)  # unblocks every 1s to recheck deadline
                if line is None:                  # process died, stdout closed
                    break
                logging.debug("[cloudflare] %s", line.rstrip())
                match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
                if match:
                    return match.group(0)
            except queue.Empty:
                if self._cloudflare_process.poll() is not None:
                    break  # process crashed silently
                continue    # no line yet, loop back and recheck deadline

        raise RuntimeError("Cloudflare tunnel did not produce a URL within 60s")
 
    # ---- ngrok tunnel (via pyngrok's Python API) ---------------------------
 
    def start_ngrok_tunnel(self, authtoken: str) -> str:
        local_port = self.cfg["serve"]["port"]
 
        from pyngrok import ngrok, conf
        ngrok.set_auth_token(authtoken)
        
        if self.cfg["ngrok"].get("region"): conf.get_default().region = self.cfg["ngrok"]["region"]
        
        connect_kwargs: Dict[str, Any] = {"proto": "http", "host_header": "localhost"}
        if self.cfg["ngrok"].get("subdomain"): connect_kwargs["subdomain"] = self.cfg["ngrok"]["subdomain"]
        if self.cfg["ngrok"].get("basic_auth"):
            connect_kwargs["basic_auth"] = [f"{BASIC_AUTH_USER}:{BASIC_AUTH_PASS}"]
 
        tunnel = ngrok.connect(local_port, **connect_kwargs)
        return tunnel.public_url
 
    def shutdown(self, signum=None, frame=None) -> None:
        logging.info("Shutdown initiated…")
        self._stop_event.set()
        if hasattr(self, "_cloudflare_process") and self._cloudflare_process.poll() is None:
            self._cloudflare_process.terminate()
        if self._server_process and self._server_process.poll() is None:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=10)
            except Exception:
                self._server_process.kill()
        try:
            from pyngrok import ngrok
            ngrok.kill()
        except Exception:
            pass


# =============================================================================
# SECTION 6 — Kill-switch listener (ntfy.sh)
# =============================================================================
 
_SERVER: Optional[VLLMServer] = None  # set in main() so the kill switch can reach it
 
 
def kill_switch_loop() -> None:
    send_log("Kill-switch listener online. Send 'SHUTDOWN_vLLM' to terminate GPU.", priority=2)
    logging.info(f"Kill-switch listener online. Send 'SHUTDOWN_vLLM' to ntfy.sh/{NTFY_CHANNEL} to terminate.")
    url = f"https://ntfy.sh/{NTFY_CHANNEL}/raw"

    while True:
        try:
            resp = requests.get(url, stream=True, timeout=(10, None))
            for line in resp.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8", errors="ignore")
                if "SHUTDOWN_vLLM" in decoded:
                    send_log("Kill switch activated! Shutting down GPU session...", priority=5)
                    logging.warning("Kill switch activated remotely! Shutting down GPU session...")
                    if _SERVER is not None:
                        _SERVER.shutdown()
                    os._exit(0)
        except Exception as e:
            send_log(f"Kill-switch listener dropped ({e}) — reconnecting in 5s", priority=2)
            logging.warning(f"Kill-switch listener dropped ({e}) — reconnecting in 5s")
            time.sleep(5)
 
 
# =============================================================================
# SECTION 7 — Main
# =============================================================================
 
def main() -> None:
    global _SERVER
 
    # 1. Parse Args
    parser = argparse.ArgumentParser(description="ShadowEngine — vLLM Deploy")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--no-ngrok", action="store_true", help="Use Cloudflare tunnel instead of ngrok")
    args = parser.parse_args()
 
    # 2. Setup Logging
    setup_logging(CONFIG["monitoring"]["log_dir"], verbose=args.verbose)
    
    print("\n" + "=" * 60)
    print(" ShadowEngine — Kaggle vLLM Deploy (Optimized)")
    print("=" * 60 + "\n")
 
    send_log("🚀 Kaggle Session Started. Initiating ShadowEngine boot sequence...", priority=4)
 
    # 3. Hardware Health Check
    gpu_info = detect_gpu()
    if gpu_info["count"] == 0:
        err_msg = "CRITICAL: No GPUs detected! Ensure Kaggle environment is set to GPU T4x2."
        logging.error(err_msg)
        send_log(err_msg, priority=5)
        sys.exit(1)
    else:
        logging.info(f"Hardware Verified: {gpu_info['count']}x {gpu_info['names'][0]} detected ({gpu_info['total_vram_gb']:.1f} GB Total VRAM)")
 
    # 4. Install & Configure
    install_deps()
    cfg = CONFIG
    if args.no_ngrok: cfg["ngrok"]["enabled"] = False
 
    server = VLLMServer(cfg)
    _SERVER = server
 
    # 5. Start Kill-Switch Listener
    threading.Thread(target=kill_switch_loop, daemon=True).start()
 
    def _handler(signum, frame): server.shutdown(signum, frame)
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)
 
    try:
        server.start_http_server()
        if cfg["ngrok"]["enabled"]:
            public_url = server.start_ngrok_tunnel(authtoken=NGROK_TOKEN)
        else:
            logging.info("ngrok disabled — falling back to Cloudflare tunnel")
            public_url = server.start_cloudflare_tunnel()
        
        print("\n" + "=" * 60)
        print(" ✅ ShadowEngine (vLLM) is LIVE!")
        print(f"  Local API : http://localhost:{cfg['serve']['port']}")
        print(f"  Public URL: {public_url}")
        print("=" * 60 + "\n")
        
        send_log(f"✅ ShadowEngine is LIVE!\nPublic URL: {public_url}", priority=5)
 
        while not server._stop_event.is_set():
            if server._server_process and server._server_process.poll() is not None:
                logging.error(f"vLLM server died unexpectedly (code {server._server_process.returncode}).")
                send_log("❌ Fatal Error: vLLM process died unexpectedly. Session ending.", priority=5)
                break
            server._stop_event.wait(2)
 
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.exception("Fatal error — shutting down")
        send_log("❌ ShadowEngine crashed during startup. Check Kaggle logs.", priority=5)
        sys.exit(1)
    finally:
        server.shutdown()
 
if __name__ == "__main__":
    main()