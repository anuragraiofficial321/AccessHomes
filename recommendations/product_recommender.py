#!/usr/bin/env python3
"""
product_recommender_new_openai.py

Single-file pipeline (static paths). Uses only the NEW OpenAI Python client (openai>=1.0.0).

Flow:
 - Extract frames from VIDEO_PATH (1fps or gap_seconds)
 - For each group of frames (frames_per_query): prepare up to `frames_per_query` frames + up to 2 floor-plan images,
   embed them as data: URIs and send prompt + images to the OpenAI chat completions API.
 - Parse JSON result {selected_id, selected_name, reason, confidence}
 - If OpenAI call fails for a group, fall back to MockModelClient for that group
 - Save per-group JSON to OUT_DIR/recommendations/

Edit CONFIG at the top (VIDEO_PATH, PRODUCTS_PATH, OPENAI_API_KEY, etc.).
"""
import base64
import csv
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# new OpenAI client (>=1.0.0)
try:
    import openai
except Exception:
    openai = None

# -----------------------
# CONFIG (edit these)
# -----------------------
VIDEO_PATH = Path("/home/anuragrai/Desktop/AccessHomes/input_data/video.mp4")
PRODUCTS_PATH = Path("/home/anuragrai/Desktop/AccessHomes/recommendations/Ageing In Place products list v7 Heastabit 1.xlsx")
OUT_DIR = Path("/home/anuragrai/Desktop/AccessHomes/recommendations_output")
PERSONA = "Owner is an elderly person with mobility issues and a preference for minimalist design."
USE_OPENAI = True
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
FRAMES_LIMIT = None     
FRAMES_PER_QUERY = 5    
MAX_WIDTH = 512
WAIT_BETWEEN_CALLS = 0.5
MODEL_NAME = "gpt-4o"
SYSTEM_PROMPT_PATH = Path("/home/anuragrai/Desktop/AccessHomes/recommendations/prompt.txt")
# -----------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# -----------------------
# Product loading & parsing
# -----------------------
def parse_weight(value: str) -> Optional[float]:
    if not value:
        return None
    v = value.strip()
    m = re.search(r"([0-9]*\.?[0-9]+)", v.replace(",", "."))
    if m:
        try:
            return float(m.group(1))
        except:
            return None
    return None

def parse_size(value: str) -> Optional[Tuple[Optional[float], Optional[float], Optional[float]]]:
    if not value:
        return None
    s = value.strip().lower().replace("cm", "").replace("mm", "")
    s = s.replace("×", "x").replace("X", "x").replace(",", ".")
    parts = re.split(r"[x×\*]|[,;]", s)
    nums = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        m = re.search(r"([0-9]*\.?[0-9]+)", p)
        if m:
            try:
                nums.append(float(m.group(1)))
            except:
                pass
    if not nums:
        m_all = re.findall(r"([0-9]*\.?[0-9]+)", s)
        nums = [float(x) for x in m_all] if m_all else []
    if not nums:
        return None
    if len(nums) == 1:
        return (nums[0], None, None)
    if len(nums) == 2:
        return (nums[0], nums[1], None)
    return (nums[0], nums[1], nums[2])

def load_products_text(path: Path) -> List[Dict[str,Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Products file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            products_raw = data
        else:
            products_raw = list(data)
    except Exception:
        lines = [l for l in text.splitlines() if l.strip()]
        delimiter = "\t" if ("\t" in lines[0]) else ","
        reader = csv.reader(lines, delimiter=delimiter)
        rows = list(reader)
        header = [h.strip().lower() for h in rows[0]]
        products_raw = []
        for r in rows[1:]:
            row = {header[i] if i < len(header) else f"col_{i}": r[i].strip() if i < len(r) else "" for i in range(len(header))}
            products_raw.append(row)

    products = []
    for p in products_raw:
        def get_field(keys, default=""):
            for k in keys:
                if isinstance(p, dict) and k in p:
                    return str(p[k]).strip()
            return default

        name = get_field(["product","name","product name","item"], "")
        if not name:
            if isinstance(p, dict):
                vals = list(p.values())
                name = str(vals[0]).strip() if vals else ""
            else:
                name = str(p).strip()
        type_ = get_field(["type","category","product type"], "")
        supplier = get_field(["supplier","brand"], "")
        size_raw = get_field(["size","dimensions","size  "], "")
        weight_raw = get_field(["weight","weight kg","weight (kg)","weight kg "], "")
        tags = get_field(["tags"], "")
        weight_kg = parse_weight(weight_raw)
        size_parsed = parse_size(size_raw)
        prod = {
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, name + supplier))[:12],
            "name": name,
            "type": type_,
            "supplier": supplier,
            "size_raw": size_raw,
            "size_parsed": {
                "w_cm": size_parsed[0] if size_parsed else None,
                "d_cm": size_parsed[1] if size_parsed else None,
                "h_cm": size_parsed[2] if size_parsed else None,
            } if size_parsed else None,
            "weight_kg": weight_kg,
            "tags": tags,
            "raw_row": p
        }
        products.append(prod)
    return products

import pandas as pd

def load_products(path: Path) -> List[Dict[str, Any]]:
    """
    Load products from Excel (.xlsx/.xls) or fallback to JSON/CSV/TSV.
    """
    if not path.exists():
        raise FileNotFoundError(f"Products file not found: {path}")

    # --- Excel Support ---
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]
        rows = df.to_dict(orient="records")
    else:
        # fallback: JSON / CSV / TXT
        text = path.read_text(encoding="utf-8").strip()
        try:
            data = json.loads(text)
            rows = data if isinstance(data, list) else list(data)
        except Exception:
            lines = [l for l in text.splitlines() if l.strip()]
            delimiter = "\t" if ("\t" in lines[0]) else ","
            reader = csv.reader(lines, delimiter=delimiter)
            header = [h.strip().lower() for h in next(reader)]
            rows = [{header[i]: r[i].strip() if i < len(r) else "" for i in range(len(header))} for r in reader]

    products = []
    for p in rows:
        def get_field(keys, default=""):
            for k in keys:
                if isinstance(p, dict) and k in p:
                    return str(p[k]).strip()
            return default

        name = get_field(["product","name","product name","item"], "")
        type_ = get_field(["type","category","product type"], "")
        supplier = get_field(["supplier","brand"], "")
        size_raw = get_field(["size","dimensions"], "")
        weight_raw = get_field(["weight","weight kg","weight (kg)"], "")

        weight_kg = parse_weight(weight_raw)
        size_parsed = parse_size(size_raw)

        prod = {
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, name + supplier))[:12],
            "name": name,
            "type": type_,
            "supplier": supplier,
            "size_raw": size_raw,
            "size_parsed": {
                "w_cm": size_parsed[0] if size_parsed else None,
                "d_cm": size_parsed[1] if size_parsed else None,
                "h_cm": size_parsed[2] if size_parsed else None,
            } if size_parsed else None,
            "weight_kg": weight_kg,
            "tags": get_field(["tags"], ""),
            "raw_row": p
        }
        products.append(prod)

    return products

# -----------------------
# Frame extraction (1 fps)
# -----------------------
def extract_frames_at_1fps(video_path: Path, out_dir: Path, limit: Optional[int] = None) -> List[Path]:
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps else 0.0
    duration_int = int(np.floor(duration))
    saved = []
    idx = 0
    for sec in range(0, duration_int + 1):
        if limit is not None and idx >= limit:
            break
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if not ret:
            continue
        fname = out_dir / f"frame_{idx:04d}.jpg"
        cv2.imwrite(str(fname), frame)
        saved.append(fname)
        idx += 1
    cap.release()
    return saved

# -----------------------
# Frame extraction (custom gap in seconds)
# -----------------------
def extract_frames_with_gap(video_path: Path, out_dir: Path, gap_seconds: int = 1, limit: Optional[int] = None) -> List[Path]:
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps else 0.0
    duration_int = int(np.floor(duration))
    saved = []
    idx = 0
    for sec in range(0, duration_int + 1, gap_seconds):
        if limit is not None and idx >= limit:
            break
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if not ret:
            continue
        fname = out_dir / f"frame_{idx:04d}.jpg"
        cv2.imwrite(str(fname), frame)
        saved.append(fname)
        idx += 1
    cap.release()
    return saved

# -----------------------
# Image helpers
# -----------------------
def load_and_resize_image_bytes(path: Path, max_width: int = 1024) -> bytes:
    img = Image.open(str(path)).convert("RGB")
    w, h = img.size
    if w > max_width:
        new_h = int(h * (max_width / w))
        img = img.resize((max_width, new_h), Image.LANCZOS)
    from io import BytesIO
    bio = BytesIO()
    img.save(bio, format="PNG", optimize=True)
    return bio.getvalue()

def image_bytes_to_data_uri(img_bytes: bytes, mime: str = "image/png") -> str:
    b64 = base64.b64encode(img_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"

# -----------------------
# Prompt builder (persona-only)
# -----------------------
PROMPT_TEMPLATE = (
    "You are an expert interior product recommender.\n\n"
    "Product catalog (JSON array):\n{product_list}\n\n"
    "User persona: \"{persona}\"\n\n"
    "Task:\n"
    "Analyze the attached image(s) of an interior space.\n"
    "1) From the catalog pick exactly one product (use selected_id and selected_name) that best fits the image and persona.\n"
    "2) Explain in 1–2 sentences why this product is the best choice for this specific user (consider accessibility, hygiene, security, or comfort).\n"
    "3) Return a confidence score between 0 and 1.\n\n"
    "Output ONLY valid JSON with keys:selected_id, selected_name, reason, confidence, placement.\n"
)

def build_prompt_text(product_list: List[Dict[str,Any]], persona: str) -> str:
    slim_products = [
        {"id": p["id"], "name": p["name"], "type": p["type"], "supplier": p["supplier"],
         "size_raw": p.get("size_raw"), "weight_kg": p.get("weight_kg"), "tags": p.get("tags")}
        for p in product_list  # limit catalog to first 30 for token control
    ]
    short_catalog = json.dumps(slim_products, ensure_ascii=False)
    return PROMPT_TEMPLATE.format(product_list=short_catalog, persona=persona)

# -----------------------
# Load system prompt (optional)
# -----------------------
system_prompt = ""
if SYSTEM_PROMPT_PATH.exists():
    try:
        system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    except Exception:
        system_prompt = ""

# -----------------------
# OpenAI wrapper (NEW API only)
# -----------------------
class OpenAIModelClient:
    """
    Uses new openai.OpenAI client (openai>=1.0.0) exclusively.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o", max_images: int = 6):
        if openai is None:
            raise RuntimeError("openai package not installed. Please `pip install openai` (>=1.0.0).")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.model = model
        self.max_images = max_images  # maximum images to embed per request (safety)
        try:
            # newer openai client supports OpenAI(api_key=...)
            self.client = openai.OpenAI(api_key=self.api_key)
        except TypeError:
            os.environ.setdefault("OPENAI_API_KEY", self.api_key)
            self.client = openai.OpenAI()

    def query(
        self,
        image_bytes: Optional[bytes],
        product_list: List[Dict[str, Any]],
        persona: str,
        temperature: float = 0.5,
        max_tokens: int = 300,
        timeout: int = 60,
        image_bytes_list: Optional[List[bytes]] = None,
        image_paths: Optional[List[Union[str, Path]]] = None
    ) -> Dict[str, Any]:
        """
        Query the model with prompt + images embedded as data: URIs.
        Accepts multiple images via image_bytes_list or image_paths or single image_bytes.
        """
        from urllib.request import urlopen
        from pathlib import Path as _Path
        from io import BytesIO

        imgs_bytes: List[bytes] = []
        if image_bytes_list:
            for b in image_bytes_list:
                if b:
                    imgs_bytes.append(b)

        # Fill from image_paths if we have capacity
        if len(imgs_bytes) < self.max_images and image_paths:
            for p in image_paths:
                if len(imgs_bytes) >= self.max_images:
                    break
                if not p:
                    continue
                try:
                    ps = str(p)
                    if ps.lower().startswith("http://") or ps.lower().startswith("https://"):
                        try:
                            with urlopen(ps) as resp:
                                imgs_bytes.append(resp.read())
                        except Exception as e:
                            print(f"Warning: failed to fetch remote image {ps}: {e}")
                            continue
                    else:
                        pp = _Path(ps)
                        if pp.exists():
                            try:
                                imgs_bytes.append(pp.read_bytes())
                            except Exception as e:
                                print(f"Warning: failed to read local image {pp}: {e}")
                                continue
                        else:
                            print(f"Warning: image_path not found: {pp}")
                except Exception as e:
                    print(f"Warning: error processing image_path {p}: {e}")
                    continue

        if not imgs_bytes and image_bytes:
            imgs_bytes.append(image_bytes)

        if not imgs_bytes:
            raise RuntimeError("No images provided to query(). Provide image_bytes, image_bytes_list, or image_paths.")

        # Limit to configured maximum images
        imgs_bytes = imgs_bytes[: self.max_images]

        # Convert images to data URIs
        data_uris: List[str] = []
        for i, b in enumerate(imgs_bytes):
            try:
                data_uris.append(image_bytes_to_data_uri(b, mime="image/png"))
            except Exception as e:
                raise RuntimeError(f"Failed to convert image #{i} to data URI: {e}")

        # Build messages
        prompt_text = build_prompt_text(product_list, persona)
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        content_array: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        for uri in data_uris:
            content_array.append({"type": "image_url", "image_url": {"url": uri}})

        messages.append({"role": "user", "content": content_array})

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )
            try:
                content = resp.choices[0].message.content
            except Exception:
                content = str(resp)
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")

        json_text = extract_first_json(content)
        parsed = None
        if json_text:
            try:
                parsed = json.loads(json_text)
            except Exception:
                try:
                    parsed = json.loads(json_text.replace("'", '"'))
                except Exception:
                    parsed = None
        return {"raw": content, "parsed": parsed}

# -----------------------
# JSON extractor helper
# -----------------------
def extract_first_json(text: str) -> Optional[str]:
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None

# -----------------------
# Smart Mock client (persona-only)
# -----------------------
def score_product_against_persona(prod: Dict[str,Any], persona: str) -> float:
    score = 0.0
    name = (prod.get("name","") or "").lower()
    type_ = (prod.get("type","") or "").lower()
    tags = (prod.get("tags","") or "").lower()
    for token in re.findall(r"\w+", (persona).lower()):
        if token in type_ or token in name or token in tags:
            score += 1.5
    if "minimalist" in (persona or "").lower():
        sp = prod.get("size_parsed") or {}
        w = sp.get("w_cm") if sp else None
        d = sp.get("d_cm") if sp else None
        if w and w < 60:
            score += 1.0
        if d and d < 60:
            score += 0.5
    weight = prod.get("weight_kg")
    if weight and weight < 5:
        score += 0.5
    return score

class MockModelClient:
    def query(self, image_bytes: Optional[bytes], product_list: List[Dict[str,Any]], persona: str, image_bytes_list: Optional[List[bytes]] = None, **kwargs):
        """
        Mock selection ignores images content, scores products against persona.
        image_bytes_list kept for compatibility.
        """
        scored = []
        for p in product_list:
            s = score_product_against_persona(p, persona)
            scored.append((s, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = scored[0][1] if scored else {"id":"none","name":"none"}
        parsed = {
            "selected_id": selected.get("id", str(uuid.uuid4())),
            "selected_name": selected.get("name", "Unknown"),
            "reason": f"Mock selection: matched persona/keyword with type '{selected.get('type')}' and size {selected.get('size_raw')}.",
            "confidence": round(min(0.95, 0.3 + scored[0][0] * 0.25), 2) if scored else 0.5
        }
        return {"raw": None, "parsed": parsed}

# -----------------------
# Orchestration (static)
# -----------------------
def process_video_and_recommend_static(
    video_path: Path,
    products_path: Path,
    out_dir: Path,
    persona: str,
    use_openai_flag: bool,
    api_key: Optional[str] = None,
    frames_limit: Optional[int] = None,
    max_width: int = 1024,
    floor_plan_image_paths: Optional[List[Path]] = None,
    gap_seconds: int = 40,
    frames_per_query: int = 1
):
    """
    Extract frames and for each group of `frames_per_query` frames send a single model query:
    images order: [frame1_bytes, frame2_bytes, ..., floorplan1_bytes, floorplan2_bytes] truncated to client's max_images.
    """
    from io import BytesIO

    frames_dir = out_dir / "frames"
    rec_dir = out_dir / "recommendations"
    ensure_dir(frames_dir)
    ensure_dir(rec_dir)

    # Normalize floor plan inputs
    floor_plan_image_paths = floor_plan_image_paths or []
    valid_floor_plans = [p for p in floor_plan_image_paths if p and p.exists()][:2]
    if valid_floor_plans:
        print(f"Using floor plan images: {[str(p) for p in valid_floor_plans]}")
    else:
        valid_floor_plans = []

    product_list = load_products(products_path)
    print(f"Loaded {len(product_list)} products (showing first 5):")
    for p in product_list[:5]:
        print(f" - {p['name']} | Type={p.get('type')} | Supplier={p.get('supplier')} | Size={p.get('size_raw')} | Weight={p.get('weight_kg')}")

    if frames_limit is None:
        print("FRAMES_LIMIT is None -> will process all frames from the video (subject to gap_seconds).")
    else:
        print(f"FRAMES_LIMIT set to {frames_limit} -> will process up to {frames_limit} frames.")

    print("Extracting frames...")
    frames = extract_frames_with_gap(video_path, frames_dir, gap_seconds=gap_seconds, limit=frames_limit)
    print(f"Extracted {len(frames)} frames to {frames_dir}")

    # Preload floor-plan PIL images and their bytes
    preloaded_floor_imgs: List[Image.Image] = []
    preloaded_floor_bytes: List[bytes] = []
    for p in valid_floor_plans:
        try:
            im = Image.open(str(p)).convert("RGB")
            preloaded_floor_imgs.append(im)
        except Exception as e:
            print(f"Warning: failed to open floor plan image {p}: {e}")

    # helper to convert PIL image to PNG bytes with resizing to max_each
    def pil_to_png_bytes(img: Image.Image, max_width_each: int) -> bytes:
        w, h = img.size
        if w > max_width_each:
            new_h = int(h * (max_width_each / w))
            img_r = img.resize((max_width_each, new_h), Image.LANCZOS)
        else:
            img_r = img.copy()
        buf = BytesIO()
        img_r.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    if preloaded_floor_imgs:
        max_each = max(200, int(max_width / 3))
        for im in preloaded_floor_imgs[:2]:
            try:
                b = pil_to_png_bytes(im, max_each)
                preloaded_floor_bytes.append(b)
            except Exception as e:
                print(f"Warning: failed to convert floorplan to bytes: {e}")

    # initialize OpenAI client (new API) or mock
    client = None
    if use_openai_flag:
        try:
            client = OpenAIModelClient(api_key=api_key or OPENAI_API_KEY, model=MODEL_NAME)
            print("OpenAI client (new API) initialized.")
        except Exception as e:
            print("OpenAI init failed:", e)
            print("Falling back to MockModelClient for all groups.")
            client = MockModelClient()
    else:
        client = MockModelClient()
        print("Using MockModelClient (no OpenAI).")

    # Iterate groups of frames
    total_frames = len(frames)
    if frames_per_query <= 0:
        frames_per_query = 1

    for i in tqdm(range(0, total_frames, frames_per_query), desc="Frame groups"):
        group = frames[i:i + frames_per_query]
        try:
            # prepare bytes for each frame in group (resize each to reasonable width)
            frame_bytes_list: List[bytes] = []
            frame_imgs_for_composite: List[Image.Image] = []
            for fp in group:
                img = Image.open(str(fp)).convert("RGB")
                fw, fh = img.size
                if fw > max_width:
                    new_h = int(fh * (max_width / fw))
                    img = img.resize((max_width, new_h), Image.LANCZOS)
                frame_imgs_for_composite.append(img)
                try:
                    buf = BytesIO()
                    img.save(buf, format="PNG", optimize=True)
                    frame_bytes_list.append(buf.getvalue())
                except Exception:
                    frame_bytes_list.append(pil_to_png_bytes(img, max_width))

            # build combined image_bytes_list: frames first, then floorplans
            image_bytes_list: List[bytes] = []
            for fb in frame_bytes_list:
                image_bytes_list.append(fb)
            for fb in preloaded_floor_bytes:
                image_bytes_list.append(fb)
            # ensure we don't exceed client's max image allowance
            if isinstance(client, OpenAIModelClient):
                image_bytes_list = image_bytes_list[:client.max_images]
            else:
                # default guard
                image_bytes_list = image_bytes_list[:6]

            # Build composite (for records): horizontal join of all those images (frames + floorplans)
            try:
                imgs_for_composite = frame_imgs_for_composite + preloaded_floor_imgs[:2]
                n = max(1, len(imgs_for_composite))
                target_each = max(200, int(max_width / n))
                resized = []
                widths = []
                heights = []
                for im in imgs_for_composite:
                    w, h = im.size
                    if w > target_each:
                        new_h = int(h * (target_each / w))
                        im_r = im.resize((target_each, new_h), Image.LANCZOS)
                    else:
                        im_r = im.copy()
                    resized.append(im_r)
                    widths.append(im_r.size[0])
                    heights.append(im_r.size[1])
                total_w = sum(widths)
                max_h = max(heights) if heights else 0
                composite = Image.new("RGB", (total_w, max_h), (255, 255, 255))
                x = 0
                for im_r in resized:
                    y = (max_h - im_r.size[1]) // 2
                    composite.paste(im_r, (x, y))
                    x += im_r.size[0]
                # name composite by group index and frame range
                start_idx = i
                end_idx = i + len(group) - 1
                composite_fname = frames_dir / f"composite_group_{start_idx:04d}_{end_idx:04d}.png"
                try:
                    composite.save(str(composite_fname), format="PNG", optimize=True)
                except Exception:
                    composite.save(str(composite_fname), format="PNG")
            except Exception as e:
                composite_fname = frames_dir / f"composite_group_{i:04d}.png"
                print(f"Warning: failed to create composite for group starting at {i}: {e}")

            # Query model with multiple images (frame group + floorplans)
            try:
                resp = client.query(
                    image_bytes=None,
                    product_list=product_list,
                    persona=persona,
                    image_bytes_list=image_bytes_list
                )
            except Exception as e:
                print("Model query failed for this group (attempting mock fallback):", e)
                # mock fallback: pass product_list and persona (mock ignores images)
                resp = MockModelClient().query(None, product_list, persona, image_bytes_list=image_bytes_list)

            raw = resp.get("raw")
            parsed = resp.get("parsed")

            # fallback parse from raw text
            if parsed is None and raw:
                try:
                    json_text = extract_first_json(raw if isinstance(raw, str) else str(raw))
                    if json_text:
                        parsed = json.loads(json_text)
                        print("Recovered parsed JSON using extract_first_json()")
                except Exception:
                    parsed = None

            out = {
                "frame_group": [str(fp) for fp in group],
                "composite_image": str(composite_fname) if 'composite_fname' in locals() else None,
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "persona": persona,
                "product_count": len(product_list),
                "model_raw": raw,
                "recommendation": parsed
            }

            print(f"Group {i}-{i+len(group)-1} result:", parsed)
            fname = rec_dir / f"recommend_group_{start_idx:04d}_{end_idx:04d}.json"
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)

            time.sleep(WAIT_BETWEEN_CALLS)
        except KeyboardInterrupt:
            print("Interrupted by user.")
            break
        except Exception as e:
            print("Failed group starting at", i, ":", e)
            continue

    print("All done. Recommendations in:", rec_dir)

# -----------------------
# Run (no CLI) - uses static CONFIG above
# -----------------------
def main():
    if not VIDEO_PATH.exists():
        print("Video not found:", VIDEO_PATH)
        sys.exit(1)
    if not PRODUCTS_PATH.exists():
        print("Products file not found:", PRODUCTS_PATH)
        sys.exit(1)

    api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if USE_OPENAI and not api_key:
        print("Warning: USE_OPENAI=True but OPENAI_API_KEY not set. Will attempt to init and likely fail; mock fallback will be used.")

    # --- USER: set your two floor-plan image paths here (can be PNG/JPG) ---
    FP_IMAGE_PATH_1 = Path("/home/anuragrai/Desktop/AccessHomes/input_data/floor_plan.png")
    FP_IMAGE_PATH_2 = Path("/home/anuragrai/Desktop/AccessHomes/output_data/input_sample_output_3/floor_plan_output.png")
    # -------------------------------------------------------

    # Build list of Path objects (only include those that exist)
    fp_images: List[Path] = []
    for p in (FP_IMAGE_PATH_1, FP_IMAGE_PATH_2):
        if p and p.exists():
            fp_images.append(p)
        else:
            print(f"Warning: floor-plan image not found, skipping: {p}")

    process_video_and_recommend_static(
        video_path=VIDEO_PATH,
        products_path=PRODUCTS_PATH,
        out_dir=OUT_DIR,
        persona=PERSONA,
        use_openai_flag=USE_OPENAI,
        api_key=api_key,
        frames_limit=FRAMES_LIMIT,
        max_width=MAX_WIDTH,
        floor_plan_image_paths=fp_images,
        gap_seconds=10,
        frames_per_query=FRAMES_PER_QUERY
    )

if __name__ == "__main__":
    main()
