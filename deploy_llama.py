# deploy_llama.py
"""
ShadowEngine — Kaggle deploy script (llama.cpp Edition)
===================================================================
Features: llama-server, ngrok tunnel, GPU health check, and ntfy.sh kill-switch.
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

import requests

# =============================================================================
# SECTION 0 — SECRETS (Kaggle-only, private kernel)
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
# SECTION 1 — CONFIG
# =============================================================================
 
CONFIG: Dict[str, Any] = {
    "model": {
        # Model weights ~19GB + Q4_0 KV cache @128K ~5.5GB = ~25.5GB total
        "hf_repo": "HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive",
        "hf_file": "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf",
        "alias": "Qwen3.6-35B-A3B-Uncensored",
        "ctx_size": 262144,         # 256K context
        "fit": "on",
        "n_gpu_layers": 999,        # Offload all layers
        "split_mode": "layer",      # Better for 2x T4 over PCIe
        "tensor_split": [1, 1],     # Equal split across both GPUs
        "main_gpu": 0,
        "flash_attn": True,
        "cache_type_k": "q4_0",     # Required to fit 128K
        "cache_type_v": "q4_0",     # Required to fit 128K
    },
    "mtp": {
        "enabled": False,           # speculative decoding — needs MTP layers in model
        "spec_type": "draft-mtp",
        "spec_draft_n_max": 3,
        "spec_draft_p_min": 0.75,
    },
    "reasoning": {
        "enabled": True,            # thinking mode — just a token budget, always works
        "budget": 65536,
    },
    "serve": {
        "host": "0.0.0.0",
        "port": 8000,
        "parallel": 1,              
        "batch_size": 512,
        "ubatch_size": 512,
        "threads": 10,
    },
    "ngrok": {
        "enabled": True,
        "basic_auth": True,
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
    
    # 1. Set the global base level to DEBUG so the file can catch everything
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] — %(message)s", datefmt="%H:%M:%S")
    # 2. Console Handler (Only show INFO unless verbose is True)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    # 3. File Handler (Always save DEBUG logs to the file)
    fh = logging.FileHandler(os.path.join(log_dir, "server.log"), mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    # 4. Silence the massive pyngrok download bar
    logging.getLogger("pyngrok").setLevel(logging.WARNING)
 
# =============================================================================
# SECTION 4 — System & Dependencies (llama.cpp)
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
    """Install pyngrok, huggingface_hub, and compile llama.cpp server binary."""
    logging.info("[*] Installing dependencies and compiling llama-server...")
    send_log("Phase 1: Downloading tools & compiling llama-server...", priority=3)
 
    subprocess.run(
        "apt-get update && apt-get install -y unzip curl build-essential cmake git",
        shell=True, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    logging.info("[*] Core tools installed")

    logging.info("[*] Compiling llama.cpp from source (Optimized for Tesla T4)...")
 
    script = """
    # 1. Point the C++ compiler and linker to Kaggle's hidden CUDA drivers
    export CUDACXX=/usr/local/cuda/bin/nvcc
    export CMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/local/nvidia/lib64:$CMAKE_LIBRARY_PATH
    export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/local/nvidia/lib64:$LIBRARY_PATH
    export LDFLAGS="-L/usr/local/cuda/lib64/stubs -L/usr/local/nvidia/lib64"
    
    # 2. Clone the official repository
    git clone --branch b9780 --depth 1 https://github.com/ggml-org/llama.cpp.git /tmp/llama.cpp
    cd /tmp/llama.cpp
    
    # 3. Build with CUDA enabled specifically for Compute Capability 75 (Tesla T4) and Disabled UI
    # The > /dev/null silences the verbose configuration logs you mentioned
    cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES="75" \
          -DGGML_CCACHE=OFF -DLLAMA_BUILD_UI=OFF > /dev/null
    cmake --build build --config Release -j $(nproc) --target llama-server
    
    # 4. Move executable AND shared libraries to system paths before cleanup
    cp build/bin/llama-server /usr/local/bin/
    
    # Copy all generated .so files (preserving symlinks if they exist)
    cp -P build/bin/*.so* /usr/local/lib/ 2>/dev/null || true
    cp -P build/src/*.so* /usr/local/lib/ 2>/dev/null || true
    
    # Register the new libraries with the OS
    ldconfig 2>/dev/null || true
    
    # 5. Clean up source files to free disk space
    cd /kaggle/working
    rm -rf /tmp/llama.cpp
    
    chmod +x /usr/local/bin/llama-server
    """
    subprocess.run(script, shell=True, check=True)
    logging.info("[*] llama-server compiled and installed to /usr/local/bin/llama-server!")
    subprocess.run("pip install --quiet huggingface_hub pyngrok requests", shell=True, check=False)
    logging.info("[*] Python packages installed")
    
    subprocess.run(
    "wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb && dpkg -i cloudflared-linux-amd64.deb",
    shell=True, check=False,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,)
    logging.info("[*] cloudflared installed")

    logging.info("[*] Compilation and installation complete!")
 
# =============================================================================
# SECTION 5 — llama-server + ngrok tunnel
# =============================================================================
 
class LlamaServer:
    """Launch llama.cpp's OpenAI-compatible API server and tunnel it."""
 
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self._server_process: Optional[subprocess.Popen] = None
        self._stop_event = threading.Event()
 
    def start_http_server(self) -> None:
        """Launch the llama-server subprocess."""
        from huggingface_hub import hf_hub_download
        model_cfg = self.cfg["model"]
        serve_cfg = self.cfg["serve"]
        mtp_cfg = self.cfg["mtp"]
        reasoning_cfg = self.cfg.get("reasoning", {})
 
        logging.info(f"Downloading {model_cfg['hf_file']} from Hugging Face...")
        send_log(f"Phase 2: Downloading model {model_cfg['hf_file']}...", priority=3)
        
        model_path = hf_hub_download(
            repo_id=model_cfg["hf_repo"],
            filename=model_cfg["hf_file"],
            local_dir="/tmp/models", # <-- FIX: Bypasses Kaggle's 20GB working directory quota
            token=HF_TOKEN
        )
 
        # Construct the llama-server launch command based on the Reddit spec
        cmd = [
            "llama-server",
            "-lv", "1",                   # Uses the verbosity flag (-lv 1) to hide all the INFO and WAR
            "--model", model_path,
            "--alias", model_cfg["alias"],
            "--ctx-size", str(model_cfg["ctx_size"]),
            "--parallel", str(serve_cfg["parallel"]),
            "--split-mode", model_cfg["split_mode"],
            "--host", serve_cfg["host"],
            "--port", str(serve_cfg["port"]),
            "--threads", str(serve_cfg["threads"]),
            "--n-gpu-layers", str(model_cfg["n_gpu_layers"]),
            "--cache-type-k", model_cfg["cache_type_k"],
            "--cache-type-v", model_cfg["cache_type_v"],
            *(["--fit", model_cfg["fit"]] if model_cfg.get("fit") else []),
        ]
        cmd.extend([
            "--tensor-split",
            ",".join(map(str, model_cfg["tensor_split"]))
            ])
        cmd.extend([
            "--main-gpu",
            str(model_cfg["main_gpu"])
            ])
        cmd.extend([
            "--batch-size",
            str(serve_cfg["batch_size"]),
            "--ubatch-size",
            str(serve_cfg["ubatch_size"]),
            ])
        cmd.append("--metrics")
        if model_cfg.get("flash_attn"):
            cmd.extend(["--flash-attn", "on"])
        if mtp_cfg.get("enabled"):
            cmd.extend([
                "--spec-type", mtp_cfg["spec_type"],
                "--spec-draft-n-max", str(mtp_cfg["spec_draft_n_max"]),
                "--spec-draft-p-min", str(mtp_cfg["spec_draft_p_min"]),
            ])
        if reasoning_cfg.get("enabled"):
            cmd.extend([
                "--reasoning", "on",
                "--reasoning-budget", str(reasoning_cfg["budget"]),
            ])
        
        sub_env = os.environ.copy()
        sub_env["LD_LIBRARY_PATH"] = f"/usr/local/lib:/usr/local/nvidia/lib64:{sub_env.get('LD_LIBRARY_PATH', '')}"

        send_log("Phase 3: Booting llama-server. Loading weights into VRAM...", priority=4)

        self._server_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=sub_env,
        )
 
        def _pump_output() -> None:
            if not self._server_process or not self._server_process.stdout:
                return
            for line in self._server_process.stdout:
                logging.info(f"[llama] {line.rstrip()}")
 
        threading.Thread(target=_pump_output, daemon=True).start()
 
        base = f"http://localhost:{serve_cfg['port']}"
        deadline = time.time() + 600
        logging.info("Waiting for HTTP API to become healthy…")
 
        while time.time() < deadline:
            if self._server_process.poll() is not None:
                raise RuntimeError(
                    f"llama-server process exited early with code {self._server_process.returncode}"
                )
            try:
                r = requests.get(f"{base}/health", timeout=3)
                if r.status_code == 200:
                    logging.info("HTTP API is HEALTHY.")
                    return
            except requests.RequestException:
                pass
            time.sleep(3)
 
        raise RuntimeError("llama-server API did not become healthy within 10 minutes")

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

    def start_ngrok_tunnel(self, authtoken: str) -> str:
        local_port = self.cfg["serve"]["port"]
 
        from pyngrok import ngrok
        
        ngrok.set_auth_token(authtoken)
        connect_kwargs: Dict[str, Any] = {"proto": "http", "host_header": "localhost"}
        
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
 
_SERVER: Optional[LlamaServer] = None
 
def kill_switch_loop() -> None:
    send_log("Kill-switch listener online. Send 'SHUTDOWN_llama' to terminate GPU.", priority=2)
    logging.info(f"Kill-switch listener online. Send 'SHUTDOWN_llama' to ntfy.sh/{NTFY_CHANNEL} to terminate.")
    url = f"https://ntfy.sh/{NTFY_CHANNEL}/raw"
    while True:
        try:
            resp = requests.get(url, stream=True, timeout=(5, None))
            for line in resp.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8", errors="ignore")
                if "SHUTDOWN_llama" in decoded:
                    send_log("Kill switch activated! Shutting down GPU session...", priority=5)
                    logging.warning("Kill switch activated remotely! Shutting down GPU session...")
                    if _SERVER is not None:
                        _SERVER.shutdown()
                    os._exit(0)
        except Exception:
            time.sleep(5)
 
# =============================================================================
# SECTION 7 — Main
# =============================================================================
 
def main() -> None:
    global _SERVER
 
    # 1. Parse Arguments FIRST
    parser = argparse.ArgumentParser(description="ShadowEngine — llama-server + ngrok tunnel (Kaggle)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--no-ngrok", action="store_true", help="Use Cloudflare tunnel instead of ngrok")
    args = parser.parse_args()

    # 2. Setup Logging SECOND (so it catches everything that follows)
    setup_logging(CONFIG["monitoring"]["log_dir"], verbose=args.verbose)
 
    print("=" * 60)
    print(" ShadowEngine — Kaggle Llama.cpp Deploy")
    print("=" * 60)
 
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
 
    logging.info("ShadowEngine — starting up…")
 
    server = LlamaServer(cfg)
    _SERVER = server
    
    # 5. Start Kill-Switch Listener
    threading.Thread(target=kill_switch_loop, daemon=True).start()

    # Handle local interruption (Ctrl+C)
    def _handler(signum, frame): server.shutdown(signum, frame)
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)
 
    try:
        send_log("Starting llama-server...", priority=4)
        server.start_http_server()
 
        if cfg["ngrok"]["enabled"]:
            public_url = server.start_ngrok_tunnel(authtoken=NGROK_TOKEN)
        else:
            logging.info("ngrok disabled — falling back to Cloudflare tunnel")
            public_url = server.start_cloudflare_tunnel()

        send_log(f"✅ ShadowEngine LIVE. URL: {public_url}", priority=5)
 
        print("\n" + "=" * 60)
        print("  ShadowEngine is running!")
        print(f"  Local API : http://localhost:{cfg['serve']['port']}")
        print(f"  Public URL: {public_url}")
        print("=" * 60)
 
        while not server._stop_event.is_set():
            if server._server_process and server._server_process.poll() is not None:
                logging.error("llama-server process died unexpectedly (code %s)", server._server_process.returncode)
                send_log("⚠️ llama-server process died unexpectedly. Session ending.", priority=5)
                break
            server._stop_event.wait(2)
 
    except KeyboardInterrupt:
        pass
    except Exception:
        logging.exception("Fatal error — shutting down")
        send_log("❌ ShadowEngine crashed during startup. Check Kaggle logs.", priority=5)
        sys.exit(1)
    finally:
        server.shutdown()
 
if __name__ == "__main__":
    main()