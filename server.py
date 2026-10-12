"""Start/stop/query the local llama-server (llama.cpp) process."""
import subprocess, time
import requests
import config

URL = f"http://{config.HOST}:{config.PORT}"


def check_files():
    """Return a list of human-readable problems (empty = OK)."""
    problems = []
    if not config.LLAMA_SERVER.exists():
        problems.append(f"llama-server not built: {config.LLAMA_SERVER}\n  -> run: bash scripts/setup.sh")
    for f in (config.MODEL_FILE, config.MMPROJ_FILE):
        if not (config.MODEL_DIR / f).exists():
            problems.append(f"model file missing: {config.MODEL_DIR / f}\n  -> run: bash scripts/setup.sh")
    return problems


def server_cmd():
    return [str(config.LLAMA_SERVER),
            "-m", str(config.MODEL_DIR / config.MODEL_FILE),
            "--mmproj", str(config.MODEL_DIR / config.MMPROJ_FILE),
            "-ngl", str(config.GPU_LAYERS), "-c", str(config.CTX_SIZE),
            "--host", config.HOST, "--port", str(config.PORT)]


def is_up():
    try:
        return requests.get(URL + "/health", timeout=1).status_code == 200
    except requests.RequestException:
        return False


def start(log_path, on_status=print, timeout=240):
    """Start the server if not already running. Returns the Popen (or None if external)."""
    if is_up():
        on_status("Using model server already running on " + URL)
        return None
    problems = check_files()
    if problems:
        raise RuntimeError("\n".join(problems))
    log = open(log_path, "w")
    proc = subprocess.Popen(server_cmd(), stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            tail = open(log_path).read()[-1500:]
            hint = ""
            if "out of memory" in tail.lower() or "cudaMalloc" in tail:
                hint = ("\nLooks like out of memory: close other apps (browser), "
                        "or lower CTX_SIZE in config.py, then retry.")
            raise RuntimeError(f"Model server exited (code {proc.returncode}). Log tail:\n{tail}{hint}")
        if is_up():
            on_status(f"Model loaded in {time.time() - t0:.0f}s")
            return proc
        time.sleep(1)
    proc.terminate()
    raise RuntimeError(f"Model server did not become ready in {timeout}s. See {log_path}")
