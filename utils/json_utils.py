"""
Simple JSON helpers used by the pipeline.
"""

import json
from pathlib import Path

def load_json(path):
    path = Path(path)
    with open(path, "r") as f:
        return json.load(f)

def find_dicts_with_key(obj, key):
    out = []
    if isinstance(obj, dict):
        if key in obj:
            out.append(obj)
        for v in obj.values():
            out += find_dicts_with_key(v, key)
    elif isinstance(obj, list):
        for it in obj:
            out += find_dicts_with_key(it, key)
    return out

