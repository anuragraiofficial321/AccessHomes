#!/usr/bin/env python3
import json

def load_json(path):
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
