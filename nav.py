"""Structured navigation query: image (+ optional goal) -> validated JSON."""
import json
import vlm, config

SYSTEM = ("You are the vision module of a ground robot. The camera looks straight ahead. "
          "Judge only what is visible. Be conservative: if unsure, say the path is not clear.")

SCHEMA = {
    "type": "object",
    "properties": {
        "path_clear": {"type": "boolean"},
        "obstacles": {"type": "array", "maxItems": 5, "items": {
            "type": "object",
            "properties": {
                "what": {"type": "string"},
                "position": {"enum": ["left", "center", "right"]},
                "distance": {"enum": ["near", "mid", "far"]}},
            "required": ["what", "position", "distance"]}},
        "goal_visible": {"type": "boolean"},
        "action": {"enum": ["forward", "left", "right", "stop"]},
        "reason": {"type": "string"}},
    "required": ["path_clear", "obstacles", "goal_visible", "action", "reason"],
}


def navigate(b64, goal=""):
    """Return (parsed dict, raw text). near = within ~1 m, mid = 1-3 m, far = beyond."""
    q = ("Decide the next move. near = closer than about 1 m, mid = 1-3 m, far = beyond. "
         "action must be stop if any obstacle is near and center. Keep reason under 20 words. ")
    q += f"Goal: {goal}. Set goal_visible true only if the goal is clearly visible." if goal else \
         "There is no goal; set goal_visible false and just pick the safest action."
    raw = vlm.ask_model(b64, q, schema=SCHEMA, max_tokens=config.NAV_MAX_TOKENS, system=SYSTEM)
    try:
        return json.loads(raw), raw
    except json.JSONDecodeError:
        return {"path_clear": False, "obstacles": [], "goal_visible": False,
                "action": "stop", "reason": "unparseable model output"}, raw  # fail safe
