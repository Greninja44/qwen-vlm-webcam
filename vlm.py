"""Image encoding + calls to the local model server (no GUI dependencies)."""
import base64
import cv2, requests
import config, server


def encode_image(frame_bgr, max_side=None):
    """Resize + JPEG-compress a frame, return (base64 str, (w, h))."""
    max_side = max_side or config.INFER_MAX_SIDE
    h, w = frame_bgr.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        frame_bgr = cv2.resize(frame_bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
    if not ok:
        raise ValueError("could not encode image")
    h, w = frame_bgr.shape[:2]
    return base64.b64encode(buf).decode(), (w, h)


def ask_model(b64, question, schema=None, max_tokens=None, system=None):
    """Ask the VLM. With `schema` (JSON schema) the output is grammar-constrained to valid JSON."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
        {"type": "text", "text": question}]})
    body = {"max_tokens": max_tokens or config.MAX_TOKENS, "temperature": 0.1, "messages": msgs}
    if schema:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "nav", "schema": schema}}
    r = requests.post(server.URL + "/v1/chat/completions", timeout=300, json=body)
    if r.status_code != 200:
        raise RuntimeError(f"server returned {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"].strip()
