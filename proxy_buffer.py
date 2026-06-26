"""
ShadowEngine — Local API Proxy (Buffered)

Run this on your local machine to route local API requests to your Kaggle GPU.
This version buffers the entire response in memory (stream=False) and calculates
the exact Content-Length before replying. Ideal for strict HTTP clients, 
legacy tools, or batch JSON data extraction.

Supports:
- Tunnels: ngrok (ngrok-free.app) OR Cloudflare (trycloudflare.com)
- Auth: Optional Basic Auth (automatically disabled if missing from .env)
- Models: vLLM, Llama.cpp, or Ollama endpoints

Example .env:
  REMOTE_HOST=https://your-url.trycloudflare.com
  LOCAL_PORT=8000
  AUTH_USER=
  AUTH_PASS=
"""

import os
import base64
import requests
from flask import Flask, request, Response
from dotenv import load_dotenv

load_dotenv()

REMOTE_HOST = os.getenv("REMOTE_HOST") or os.getenv("VLLM_REMOTE_HOST")
LOCAL_PORT = int(os.getenv("LOCAL_VLLM_PORT", "8000"))
AUTH_USER = os.getenv("AUTH_USER")
AUTH_PASS = os.getenv("AUTH_PASS")

app = Flask(__name__)

@app.route("/", defaults={"path": ""}, methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
@app.route("/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
def universal_proxy(path):
    if not REMOTE_HOST:
        return "Error: REMOTE_HOST variable not found in your local .env file.", 500

    target_url = f"{REMOTE_HOST.rstrip('/')}/{path}"

    # 1. Duplicate incoming headers
    headers = {key: value for (key, value) in request.headers if key.lower() != "host"}

    # 2. Inject tunnel bypass header (Harmless if using Cloudflare)
    headers["ngrok-skip-browser-warning"] = "true"

    # 3. Inject optional Basic Auth
    if AUTH_USER and AUTH_PASS:
        auth_string = f"{AUTH_USER}:{AUTH_PASS}"
        b64_auth = base64.b64encode(auth_string.encode()).decode()
        headers["Authorization"] = f"Basic {b64_auth}"

    try:
        remote_response = requests.request(
            method=request.method,
            url=target_url,
            headers=headers,
            data=request.get_data(),
            cookies=request.cookies,
            allow_redirects=False,
            stream=False,           # <--- BUFFERED IN MEMORY
            timeout=600,
        )

        # Block headers that interfere with dynamic chunked transfer streaming
        excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
        response_headers = {
            name: value for name, value in remote_response.headers.items()
            if name.lower() not in excluded_headers
        }

        # Stream chunks from remote_response directly into Flask's generator
        return Response(
            remote_response.content,
            status=remote_response.status_code,
            headers=response_headers
        )

    except requests.exceptions.RequestException as e:
        return f"Proxy Error: Unable to reach remote host. Details: {str(e)}", 502

if __name__ == "__main__":
    print("📡 ShadowEngine vLLM Local Proxy Online!")
    print(f"🔗 Intercepting: http://127.0.0.1:{LOCAL_PORT} -> Forwarding to: {REMOTE_HOST}")
    print("REMOTE_HOST =", REMOTE_HOST)
    print("AUTH_USER =", AUTH_USER)
    print("AUTH_PASS set =", bool(AUTH_PASS))
    app.run(host="127.0.0.1", port=LOCAL_PORT, debug=False, threaded=True)