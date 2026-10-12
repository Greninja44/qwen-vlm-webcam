"""Headless navigation loop: prints one JSON line per decision, logs frames to disk.

  .venv/bin/python nav_loop.py --goal "the door" --every 0   # back-to-back
  .venv/bin/python nav_loop.py --once
"""
import argparse, json, pathlib, time
import cv2
import config, server, vlm, nav
from camera import Camera

ap = argparse.ArgumentParser()
ap.add_argument("--goal", default="")
ap.add_argument("--every", type=float, default=0.0, help="min seconds between decisions")
ap.add_argument("--once", action="store_true")
ap.add_argument("--log", default="nav_logs", help="folder for frames + decisions ('' to disable)")
a = ap.parse_args()

cam = Camera()
proc = server.start("server.log")
logdir = pathlib.Path(a.log) if a.log else None
if logdir:
    logdir.mkdir(exist_ok=True)
time.sleep(1)
try:
    while True:
        t0 = time.time()
        frame = cam.latest()
        if frame is None:
            print(json.dumps({"action": "stop", "reason": "no camera frame"}), flush=True)
            time.sleep(0.5)
            continue
        try:
            b64, _ = vlm.encode_image(frame, config.NAV_MAX_SIDE)
            d, _raw = nav.navigate(b64, a.goal)
        except Exception as e:
            d = {"action": "stop", "reason": f"inference error: {e}"}
        d["latency_s"] = round(time.time() - t0, 2)
        print(json.dumps(d), flush=True)
        if logdir:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(str(logdir / f"{stamp}.jpg"), frame)
            (logdir / f"{stamp}.json").write_text(json.dumps(d, indent=1))
        if a.once:
            break
        time.sleep(max(0, a.every - (time.time() - t0)))
except KeyboardInterrupt:
    pass
finally:
    cam.close()
    if proc:
        proc.terminate()
