#!/usr/bin/env bash
# One-shot, re-runnable setup. Needs sudo ONLY for apt (CUDA toolkit + cmake).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
say() { printf '\n\033[1;32m== %s ==\033[0m\n' "$*"; }
die() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

say "1/6 Verify Jetson / CUDA"
grep -qi jetson /proc/device-tree/model 2>/dev/null || die "Not a Jetson."
tr -d '\0' < /proc/device-tree/model; echo
head -1 /etc/nv_tegra_release
nvidia-smi >/dev/null 2>&1 || die "nvidia-smi failed: NVIDIA driver not working."
nvidia-smi | grep "CUDA Version"
free -h | sed -n 1,2p
[ "$(awk '/MemTotal/{print int($2/1024)}' /proc/meminfo)" -ge 6500 ] || echo "WARNING: <6.5GB RAM; the model may not fit."

say "2/6 System packages (cuda-toolkit, cmake, build tools)"
CUDA_PKG=$(apt-cache search --names-only '^cuda-toolkit-[0-9]+-[0-9]+$' | awk '{print $1}' | sort -V | tail -1)
[ -n "$CUDA_PKG" ] || die "No cuda-toolkit-X-Y apt package found (is the NVIDIA Jetson apt repo enabled?)."
need=()
[ -x /usr/local/cuda/bin/nvcc ] || need+=("$CUDA_PKG")
command -v cmake >/dev/null || need+=(cmake)
command -v git   >/dev/null || need+=(git)
command -v curl  >/dev/null || need+=(curl)
dpkg -s python3-venv python3-tk >/dev/null 2>&1 || need+=(python3-venv python3-tk)
if [ ${#need[@]} -gt 0 ]; then
  echo "Installing: ${need[*]}"; sudo apt-get update && sudo apt-get install -y "${need[@]}"
fi
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version | tail -2

say "3/6 Python venv (opencv, pillow, requests only)"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt

say "4/6 Build llama.cpp with CUDA (Orin = sm_87; takes ~15-30 min once)"
if [ ! -x third_party/llama.cpp/build/bin/llama-server ]; then
  [ -d third_party/llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp third_party/llama.cpp
  cd third_party/llama.cpp
  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
  cmake --build build --config Release -j4 --target llama-server
  cd "$ROOT"
else echo "already built"; fi

say "5/6 Download model (Qwen2.5-VL-3B Q4_K_M, ~2.6 GB total)"
mkdir -p models
for f in $(.venv/bin/python -c "import config;print(config.MODEL_FILE, config.MMPROJ_FILE)"); do
  [ -s "models/$f" ] || curl -L --fail -C - -o "models/$f" \
    "https://huggingface.co/$(.venv/bin/python -c 'import config;print(config.HF_REPO)')/resolve/main/$f"
done
ls -lh models

say "6/6 Tests: camera + model load"
.venv/bin/python scripts/test_camera.py || echo "(camera test failed - fix before launching the app)"
.venv/bin/python - <<'PY'
import server, time, subprocess
proc = server.start("server.log")
print("MODEL LOAD OK" if proc is not None or server.is_up() else "MODEL LOAD FAILED")
if proc: proc.terminate(); proc.wait()
PY

say "Done. Launch with:  bash scripts/run.sh"
