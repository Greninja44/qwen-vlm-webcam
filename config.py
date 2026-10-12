"""Central settings. Edit here, no other file needs to change."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LLAMA_DIR = ROOT / "third_party" / "llama.cpp"
LLAMA_SERVER = LLAMA_DIR / "build" / "bin" / "llama-server"
MODEL_DIR = ROOT / "models"

# Qwen2.5-VL-3B-Instruct, 4-bit (GGUF) + vision projector
HF_REPO = "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF"
MODEL_FILE = "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
MMPROJ_FILE = "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf"
QUANT = "Q4_K_M (4-bit) + Q8_0 vision projector"

# llama-server (bound to localhost only: nothing leaves the machine)
HOST, PORT = "127.0.0.1", 8080
CTX_SIZE = 3072          # prompt + image tokens + answer
GPU_LAYERS = 99          # offload everything to the GPU
MAX_TOKENS = 300         # max answer length

# Camera
CAMERA_INDEX = None      # None = auto-detect (tries 0..3)
CAM_WIDTH, CAM_HEIGHT, CAM_FPS = 640, 480, 30

# Image sent to the model: longest side in pixels, JPEG quality
INFER_MAX_SIDE = 640
JPEG_QUALITY = 85

DEFAULT_QUESTION = "Describe everything you can see in this image."

# Navigation mode (structured JSON answers)
NAV_MAX_SIDE = 448       # smaller image = fewer tokens = faster
NAV_MAX_TOKENS = 160
