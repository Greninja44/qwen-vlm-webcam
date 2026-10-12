"""Webcam -> Capture -> Qwen2.5-VL-3B (local llama.cpp) -> Answer.

Keys:  C = capture   N = navigate (JSON action; question box = goal)   Ctrl+Enter / Enter in question box = ask   Q or Esc = quit
"""
import subprocess, threading, time, tkinter as tk
from tkinter import scrolledtext

import cv2, requests
from PIL import Image, ImageTk

import config, server, vlm, nav, json
from camera import Camera

EXAMPLES = [
    config.DEFAULT_QUESTION,
    "What objects do you see?",
    "Describe this scene.",
    "What is directly in front of the camera?",
    "Are there any obstacles?",
    "What text can you read?",
    "What is the person doing?",
    "Describe the scene from the perspective of a robot.",
]


def gpu_info():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        cuda = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5).stdout
        cuda = cuda.split("CUDA Version:")[1].split()[0] if "CUDA Version:" in cuda else "?"
        return (out or "Orin (nvgpu)"), cuda
    except Exception:
        return "unavailable (nvidia-smi failed - CUDA may not work)", "?"


class App:
    def __init__(self, root):
        self.root, self.cam, self.proc = root, None, None
        self.captured, self.busy, self.ready = None, False, False
        root.title("Qwen2.5-VL-3B webcam test")
        root.protocol("WM_DELETE_WINDOW", self.quit)

        top = tk.Frame(root); top.pack(padx=6, pady=6)
        self.live = tk.Label(top, text="starting camera...", width=64, height=24, bg="black", fg="white")
        self.live.grid(row=0, column=0)
        self.snap = tk.Label(top, text="(no capture yet)", width=64, height=24, bg="#222", fg="white")
        self.snap.grid(row=0, column=1, padx=6)
        tk.Label(top, text="Live").grid(row=1, column=0)
        tk.Label(top, text="Captured (this is what the model sees)").grid(row=1, column=1)

        qf = tk.Frame(root); qf.pack(fill="x", padx=6)
        self.q = tk.StringVar(value=config.DEFAULT_QUESTION)
        e = tk.Entry(qf, textvariable=self.q, font=("TkDefaultFont", 12)); e.pack(side="left", fill="x", expand=True)
        e.bind("<Return>", lambda _e: self.ask())
        tk.Button(qf, text="Capture (C)", command=self.capture).pack(side="left", padx=3)
        self.ask_btn = tk.Button(qf, text="Ask (A)", command=self.ask, state="disabled")
        self.ask_btn.pack(side="left")
        self.nav_btn = tk.Button(qf, text="Navigate (N)", command=self.navigate, state="disabled")
        self.nav_btn.pack(side="left", padx=3)
        ex = tk.OptionMenu(qf, tk.StringVar(value="Examples"), *EXAMPLES, command=self.q.set)
        ex.pack(side="left", padx=3)

        self.out = scrolledtext.ScrolledText(root, height=9, wrap="word", font=("TkDefaultFont", 12))
        self.out.pack(fill="both", expand=True, padx=6, pady=6)
        self.status = tk.Label(root, anchor="w", text="Starting..."); self.status.pack(fill="x")

        # C / A / Q only when not typing in the entry box
        for k, fn in (("c", self.capture), ("a", self.ask), ("n", self.navigate), ("q", self.quit)):
            root.bind(f"<KeyPress-{k}>", lambda ev, fn=fn: fn() if ev.widget.winfo_class() != "Entry" else None)
        root.bind("<Escape>", lambda _e: self.quit())
        root.bind("<Control-Return>", lambda _e: self.ask())

        self.log("Starting camera and model (first load takes ~30-60 s)...")
        threading.Thread(target=self.startup, daemon=True).start()
        self.refresh()

    # ---- helpers (thread-safe UI updates) ----
    def log(self, text):
        self.root.after(0, lambda: (self.out.insert("end", text + "\n"), self.out.see("end")))

    def set_status(self, text):
        self.root.after(0, lambda: self.status.config(text=text))

    def startup(self):
        try:
            self.cam = Camera()
        except Exception as e:
            self.log(f"CAMERA ERROR: {e}")
        gpu, cuda = gpu_info()
        self.log(f"GPU: {gpu}\nCUDA (driver): {cuda}\nModel: Qwen2.5-VL-3B-Instruct\n"
                 f"Quantization: {config.QUANT}\nImage resolution sent: max {config.INFER_MAX_SIDE}px side\n"
                 f"Backend: llama.cpp (llama-server, CUDA, localhost only)")
        try:
            self.proc = server.start("server.log", on_status=self.log)
            self.ready = True
            self.log("Ready. Press C to capture, then ask a question.")
            self.root.after(0, lambda: (self.ask_btn.config(state="normal"), self.nav_btn.config(state="normal")))
        except Exception as e:
            self.log(f"MODEL ERROR: {e}")

    def refresh(self):
        f = self.cam.latest() if self.cam else None
        if f is not None:
            self.show(self.live, f)
            self.set_status(f"camera /dev/video{self.cam.index}  {self.cam.fps:.0f} fps"
                            + ("   | model ready" if self.ready else "   | model loading..."))
        self.root.after(33, self.refresh)

    def show(self, label, bgr):
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        img.thumbnail((480, 360))
        ph = ImageTk.PhotoImage(img)
        label.config(image=ph, text="", width=ph.width(), height=ph.height())
        label.image = ph

    # ---- actions ----
    def capture(self):
        f = self.cam.latest() if self.cam else None
        if f is None:
            self.log("Cannot capture: no camera frame available.")
            return
        self.captured = f
        self.show(self.snap, f)
        self.log(f"Captured frame {f.shape[1]}x{f.shape[0]}.")

    def navigate(self):
        self.ask(nav_mode=True)

    def ask(self, nav_mode=False):
        if self.busy:
            return
        if not self.ready:
            self.log("Model not ready yet.")
            return
        if self.captured is None:
            self.capture()
            if self.captured is None:
                return
        question = self.q.get().strip() or config.DEFAULT_QUESTION
        self.busy = True
        self.ask_btn.config(state="disabled"); self.nav_btn.config(state="disabled")
        self.log(f"\n{'NAV goal' if nav_mode else 'Q'}: {question if nav_mode else question}\n(thinking...)")
        threading.Thread(target=self.infer, args=(self.captured.copy(), question, nav_mode), daemon=True).start()

    def infer(self, frame, question, nav_mode=False):
        try:
            b64, (w, h) = vlm.encode_image(frame, config.NAV_MAX_SIDE if nav_mode else None)
        except Exception as e:
            self.log(f"INVALID IMAGE: {e}")
            return self.done()
        t0 = time.time()
        try:
            if nav_mode:
                d, _ = nav.navigate(b64, "" if question == config.DEFAULT_QUESTION else question)
                obs = "; ".join(f"{o['what']} ({o['position']}, {o['distance']})" for o in d["obstacles"]) or "none"
                ans = (f"ACTION: {d['action'].upper()}   path_clear={d['path_clear']}   goal_visible={d['goal_visible']}\n"
                       f"obstacles: {obs}\nreason: {d['reason']}\n{json.dumps(d)}")
            else:
                ans = vlm.ask_model(b64, question)
            dt = time.time() - t0
            self.log(f"A: {ans}\nInference: {dt:.1f} seconds  (image {w}x{h}, {len(b64) // 1024} KB)")
        except requests.ConnectionError:
            self.log("INFERENCE ERROR: model server is not reachable (it may have crashed - see server.log, restart the app).")
        except Exception as e:
            msg = str(e)
            if "memory" in msg.lower():
                msg += "\nOut of memory: lower INFER_MAX_SIDE or CTX_SIZE in config.py."
            self.log(f"INFERENCE ERROR: {msg}")
        self.done()

    def done(self):
        self.busy = False
        st = "normal" if self.ready else "disabled"
        self.root.after(0, lambda: (self.ask_btn.config(state=st), self.nav_btn.config(state=st)))

    def quit(self):
        if self.cam:
            self.cam.close()
        if self.proc:
            self.proc.terminate()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
