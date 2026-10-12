# qwen-vlm-webcam

Point a webcam at something, press a key, and ask a vision-language model about it. Everything runs
on the device: [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) at 4-bit,
served by [llama.cpp](https://github.com/ggml-org/llama.cpp) on the Jetson's GPU. No cloud APIs, and no
frame ever leaves the machine.

It started as a way to check how well a small VLM understands what a webcam sees, so it is deliberately
small: a live preview, a capture key, a question box, and an answer. There is also a structured JSON
mode for when another program needs to consume the answer.

```
webcam ──▶ capture (on demand) ──▶ resize + JPEG ──▶ llama-server (Qwen2.5-VL-3B Q4_K_M, CUDA) ──▶ answer
```

## Features

- Live preview that runs independently of the model. Frames are only sent when you ask.
- Ask anything about the captured frame, with a menu of example questions.
- **Navigate mode** returns schema-constrained JSON (obstacles, a clear/blocked flag, a suggested action).
  Any parse or inference failure falls back to `stop`.
- A headless loop (`nav_loop.py`) that prints one JSON line per decision and logs frames to disk.
- Inference time printed after every answer; GPU, CUDA, model and quantization printed at startup.
- Readable errors for a missing camera, camera permissions, CUDA problems, out-of-memory and model load failures.

## Requirements

Developed and tested on a Jetson Orin Nano Developer Kit (Super), 8 GB, JetPack 7 (L4T R39.2.1,
CUDA 13.2), Python 3.12, with a USB webcam. It should work on other Jetsons with a CUDA toolkit and
enough memory, but that hasn't been tried.

- About 3 GB of free RAM for the model, and about 2.6 GB of disk for the weights. Close your browser first.
- Internet is needed once, for setup. After that it runs offline.
- `sudo` is used only to `apt install` the CUDA toolkit and cmake.

## Quick start

```bash
git clone <this-repo> qwen-vlm-webcam
cd qwen-vlm-webcam
bash scripts/setup.sh     # one-time; asks for your sudo password; the llama.cpp build takes 15-30 min
bash scripts/run.sh
```

`setup.sh` is safe to re-run. It checks Jetson and CUDA, installs the toolkit, builds llama.cpp for
the Orin GPU (sm_87), downloads the model, then tests the camera and the model load.

## Using it

| Key | Action |
|---|---|
| **C** | Capture the current frame (shown in the right-hand panel) |
| **A** or **Enter** | Ask the question in the box about the captured frame |
| **N** | Navigate mode: the box becomes the goal, e.g. `the door` |
| **Q** / **Esc** | Quit |

C, A, N and Q type letters while the cursor is in the question box. Click outside it to use them as
shortcuts. Enter always asks.

The first model load takes 10-60 seconds. On the test board, loading took about 7 seconds (warm),
open-ended answers about 9 seconds, and navigate decisions about 3 seconds.

### Headless navigation

```bash
.venv/bin/python nav_loop.py --once                  # one decision
.venv/bin/python nav_loop.py --goal "the door"       # continuous
```

Each line looks like:

```json
{"path_clear": true, "obstacles": [], "goal_visible": false, "action": "stop", "reason": "The door is not visible.", "latency_s": 2.9}
```

Frames and decisions are saved to `nav_logs/` (git-ignored) so you can review mistakes later.

> **Not a safety system.** The model's left/right and near/far judgements are unreliable, and a decision
> takes seconds. Don't drive motors from it without a depth or range sensor and a hard stop underneath.

## Configuration

Edit `config.py`:

| Setting | Effect |
|---|---|
| `INFER_MAX_SIDE` | Longest image side sent to the model. Lower is faster and uses less memory. |
| `NAV_MAX_SIDE`, `NAV_MAX_TOKENS` | Same, for navigate mode. |
| `CTX_SIZE` | Context length. Lower it if you run out of memory. |
| `MAX_TOKENS` | Longest answer. |
| `CAMERA_INDEX` | `None` auto-detects. Set a number to force `/dev/videoN`. |

## How it works

| File | Role |
|---|---|
| `app.py` | Tk interface |
| `camera.py` | Webcam reader thread (V4L2, MJPG) |
| `server.py` | Starts and monitors the local `llama-server`, bound to `127.0.0.1` |
| `vlm.py` | Image resize/encode and the model request |
| `nav.py` | Navigation prompt, JSON schema and fail-safe parsing |
| `nav_loop.py` | Headless navigation loop |
| `scripts/` | `setup.sh`, `run.sh`, `start_server.sh`, `test_camera.py` |

PyTorch, vLLM and TensorRT-LLM were considered and skipped: they need multi-gigabyte stacks, and
llama.cpp with GGUF weights was the smallest thing that ran this model on the board.

## Troubleshooting

- **Out of memory:** close other apps, then lower `INFER_MAX_SIDE` or `CTX_SIZE`.
- **Camera permission denied:** `sudo usermod -aG video $USER`, then log out and back in.
- **No window over SSH:** run from the desktop session, or `export DISPLAY=:0`.
- **Model won't start:** check `server.log` in the project folder.
- **Camera check:** `.venv/bin/python scripts/test_camera.py` saves `camera_test.jpg`.

## Credits

Model: Qwen2.5-VL by the Qwen team (Alibaba Cloud), GGUF conversion by `ggml-org`. Inference: llama.cpp.
Check each project's license for the weights and code you download.
