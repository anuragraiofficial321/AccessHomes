"""
multimodal_recommender.py
Single-file pipeline:
- Frame extraction (1 fps)
- Prompt build (persona + requirements)
- Send multimodal request to OpenAI 4o (example)
- Receive recommendation + placement coordinates
- Generate a transparent PNG icon for recommended object
- Validate/adjust placement to avoid overlap (simple heuristic)
- Place icon on floor plan and save output
- Expose Flask endpoint /recommend

Usage (example):
  export OPENAI_API_KEY="sk-..."
  python multimodal_recommender.py

Then POST to http://127.0.0.1:5000/recommend with multipart form:
- file: video file (.mp4) OR frame image (.jpg/.png)
- floorplan: floor plan image (jpg/png) used to check overlap & to paste icon
- persona: e.g., "minimalist"
- requirements: e.g., "needs charging point nearby, child-safe"

The endpoint returns JSON describing saved files and coordinates.
"""

import os
import io
import cv2
import uuid
import json
import base64
import tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, jsonify, send_from_directory
import openai
from typing import Tuple, Dict, List

# ---------------------------
# Configuration / constants
# ---------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY environment variable.")
openai.api_key = OPENAI_API_KEY

OUTPUT_DIR = "output_data"
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
ICONS_DIR = os.path.join(OUTPUT_DIR, "icons")
PLACED_DIR = os.path.join(OUTPUT_DIR, "placed")
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(ICONS_DIR, exist_ok=True)
os.makedirs(PLACED_DIR, exist_ok=True)

# Font for icon generation (PIL will use default if not found)
try:
    DEFAULT_FONT = ImageFont.truetype("arial.ttf", 20)
except Exception:
    DEFAULT_FONT = ImageFont.load_default()

# ---------------------------
# Utility functions
# ---------------------------

def extract_frames(video_path: str, out_dir: str = FRAMES_DIR, fps: int = 1) -> List[str]:
    """
    Extract 1 frame per second from a video and save as JPEG files.
    Returns list of saved frame paths.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = total_frames / video_fps if video_fps else 0
    saved_paths = []

    # Extract frame every `fps` seconds => sample_interval_frames frames
    sample_interval_frames = int(round(video_fps / fps))
    idx = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_interval_frames == 0:
            fname = os.path.join(out_dir, f"frame_{saved_count+1:04d}.jpg")
            cv2.imwrite(fname, frame)
            saved_paths.append(fname)
            saved_count += 1
        idx += 1

    cap.release()
    return saved_paths

def image_to_data_url(path: str) -> str:
    """
    Convert image file to a base64 data URL (png).
    """
    with open(path, "rb") as f:
        b = f.read()
    b64 = base64.b64encode(b).decode("utf-8")
    # guess mime-type by extension
    ext = os.path.splitext(path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"

def build_prompt(persona: str, requirements: str, example_instruction: str = None) -> str:
    """
    Build a clear product-recommendation prompt to send to the multimodal model.
    """
    parts = [
        "You are an intelligent interior design assistant. Look at the image provided and recommend one additional object (e.g., furniture, lamp, rug) that best improves the room usability and aesthetics.",
        f"User persona: {persona}.",
        f"User requirements: {requirements}.",
        "Return a JSON object EXACTLY in the following schema (without extra text):\n"
        '{\n  "recommendation": "NAME of object (short, e.g., coffee table)",\n'
        '  "reasoning": "short text explaining why",\n'
        '  "suggested_size_cm": {"w": number, "h": number},\n'
        '  "placement": {"x": number, "y": number},\n'
        '  "placement_units": "pixels"  // placement coordinates are in pixels relative to the input image\n'
        "}\n",
    ]
    if example_instruction:
        parts.append("Example instruction: " + example_instruction)
    return "\n".join(parts)

def call_multimodal_model(image_path: str, prompt_text: str, model_name: str = "gpt-4o") -> Dict:
    """
    Send the image + prompt to a multimodal model and parse JSON response.
    NOTE: This code uses a demonstration pattern of embedding the image as a data URL inside the message.
    You may need to adapt this to your OpenAI client version that supports images (e.g., `client.responses.create`
    or `chat.completions` with attachments). Replace the request block below as appropriate.
    """

    # Convert image to data URL and include in the message
    data_url = image_to_data_url(image_path)
    user_message = f"<image>{data_url}</image>\n\n{prompt_text}"

    # -------------------------
    # ### OPENAI CALL — ADAPT AS NEEDED ###
    # This is an illustrative example using ChatCompletion. Some OpenAI clients require different
    # argument names for multimodal inputs or a `files` parameter. Replace this block if your
    # SDK expects another format (for instance, the Responses API supports `input` with multimodal).
    # -------------------------
    try:
        resp = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=512,
            temperature=0.2,
        )
    except Exception as e:
        # Provide helpful debugging info
        raise RuntimeError("OpenAI API call failed: " + str(e))

    # extract text
    text = ""
    try:
        text = resp["choices"][0]["message"]["content"]
    except Exception:
        text = str(resp)

    # Model is instructed to return pure JSON — attempt to parse:
    parsed = None
    # Attempt to find first JSON object in the response
    import re
    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(1))
        except Exception:
            # fallback: try to cleanup common issues (trailing commas)
            try:
                cleaned = re.sub(r",\s*}", "}", m.group(1))
                cleaned = re.sub(r",\s*\]", "]", cleaned)
                parsed = json.loads(cleaned)
            except Exception:
                parsed = {"raw_text": text}
    else:
        parsed = {"raw_text": text}

    return parsed

def generate_transparent_icon(name: str, out_path: str, size: Tuple[int,int] = (128,128)) -> str:
    """
    Generate a simple transparent PNG icon with the name text centered.
    For production you would fetch or generate a real icon/PNG from an icon library or image model.
    """
    w, h = size
    img = Image.new("RGBA", (w,h), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    # draw rounded rectangle as background (low opacity)
    rect_margin = 8
    draw.rounded_rectangle([rect_margin, rect_margin, w-rect_margin, h-rect_margin], radius=12, fill=(255,255,255,200))
    # draw text
    font = DEFAULT_FONT
    text = name[:16]
    tw, th = draw.textsize(text, font=font)
    draw.text(((w-tw)/2, (h-th)/2), text, fill=(10,10,10,255), font=font)
    img.save(out_path)
    return out_path

def detect_obstacles_on_floorplan(floorplan_path: str) -> List[Tuple[int,int,int,int]]:
    """
    Heuristic detection: treat non-white pixels as obstacles (furniture/walls).
    Returns list of bounding boxes (x, y, w, h).
    If you have structured JSON of furniture coordinates, replace this.
    """
    pil = Image.open(floorplan_path).convert("RGB")
    arr = np.array(pil)
    # heuristics: consider pixel not very bright as obstacle
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)  # white -> background
    # find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        # ignore tiny specks
        if w*h < 100:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))
    return boxes

def bbox_overlap(b1, b2) -> bool:
    """
    Check overlap between two boxes (x,y,w,h).
    """
    x1,y1,w1,h1 = b1
    x2,y2,w2,h2 = b2
    left = max(x1, x2)
    right = min(x1+w1, x2+w2)
    top = max(y1, y2)
    bottom = min(y1+h1, y2+h2)
    return (right > left) and (bottom > top)

def validate_and_adjust_placement(suggested: Dict, floorplan_path: str, icon_size=(128,128), max_attempts=50) -> Dict:
    """
    Ensure the suggested placement doesn't overlap obstacles. If it does, try to nudge slightly
    around in a spiral until a valid free spot is found (pixels).
    Input suggested placement uses 'x','y' in pixels (origin top-left).
    Returns dict: {"x": int, "y": int, "w": int, "h": int}
    """
    # get obstacles bounding boxes
    obstacles = detect_obstacles_on_floorplan(floorplan_path)
    img = Image.open(floorplan_path).convert("RGBA")
    W, H = img.size
    w_icon, h_icon = icon_size

    sx = int(suggested.get("x", W//2))
    sy = int(suggested.get("y", H//2))

    # center the icon around given point (so x,y in center)
    def centered_box(cx, cy):
        x = int(cx - w_icon/2)
        y = int(cy - h_icon/2)
        return (x,y,w_icon,h_icon)

    # clamp
    def clamp_box(box):
        x,y,w,h = box
        x = max(0, min(W-w, x))
        y = max(0, min(H-h, y))
        return (x,y,w,h)

    candidate = clamp_box(centered_box(sx, sy))
    if not any(bbox_overlap(candidate, obs) for obs in obstacles):
        # fine
        return {"x": candidate[0], "y": candidate[1], "w": w_icon, "h": h_icon}

    # Spiral search
    step = 20
    attempts = 0
    radius = step
    while attempts < max_attempts:
        for dx in range(-radius, radius+1, step):
            for dy in range(-radius, radius+1, step):
                cx = sx + dx
                cy = sy + dy
                candidate = clamp_box(centered_box(cx, cy))
                if not any(bbox_overlap(candidate, obs) for obs in obstacles):
                    return {"x": candidate[0], "y": candidate[1], "w": w_icon, "h": h_icon}
                attempts += 1
                if attempts >= max_attempts:
                    break
            if attempts >= max_attempts:
                break
        radius += step
    # if we fail to find a non-overlapping spot, return the clamped original candidate (best-effort)
    return {"x": candidate[0], "y": candidate[1], "w": w_icon, "h": h_icon}

def place_icon_on_floorplan(floorplan_path: str, icon_path: str, placement: Dict, out_path: str) -> str:
    """
    Paste icon onto floorplan at placement (x, y) top-left and save.
    Returns path to saved image.
    """
    floor = Image.open(floorplan_path).convert("RGBA")
    icon = Image.open(icon_path).convert("RGBA")
    x = int(placement["x"])
    y = int(placement["y"])
    floor.paste(icon, (x,y), icon)
    floor.save(out_path)
    return out_path

# ---------------------------
# Flask API
# ---------------------------
app = Flask(__name__)

@app.route("/recommend", methods=["POST"])
def recommend_endpoint():
    """
    Accepts multipart:
    - file: video.mp4 OR image frame jpg/png
    - floorplan: floorplan image (jpg/png)
    - persona: string
    - requirements: string
    Returns JSON with:
    - recommendation
    - reasoning
    - suggested_size_cm
    - final placement coordinates (x,y,w,h)
    - paths to generated icon & placed image
    """

    # Validate inputs
    if 'file' not in request.files:
        return jsonify({"error": "Missing 'file' (video or image)."}), 400
    if 'floorplan' not in request.files:
        return jsonify({"error": "Missing 'floorplan' image."}), 400

    file = request.files['file']
    floorplan = request.files['floorplan']
    persona = request.form.get("persona", "general")
    requirements = request.form.get("requirements", "no special requirements")

    # Save uploaded files to temp files
    uid = uuid.uuid4().hex[:8]
    temp_dir = os.path.join(FRAMES_DIR, f"upload_{uid}")
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    file.save(file_path)

    floorplan_path = os.path.join(temp_dir, f"floorplan_{uid}.png")
    floorplan.save(floorplan_path)

    # If file is video (heuristic by extension), extract frames and choose first frame
    ext = os.path.splitext(file.filename)[1].lower()
    frame_path = None
    if ext in [".mp4", ".mov", ".avi", ".mkv"]:
        frames = extract_frames(file_path, out_dir=temp_dir, fps=1)
        if not frames:
            return jsonify({"error": "No frames extracted from video."}), 500
        frame_path = frames[0]
    else:
        # treat as image
        frame_path = file_path

    # Build prompt
    prompt = build_prompt(persona, requirements)

    # Call model
    try:
        model_resp = call_multimodal_model(frame_path, prompt)
    except Exception as e:
        return jsonify({"error": "Model call failed", "details": str(e)}), 500

    # parse model result
    # expected keys: recommendation, reasoning, suggested_size_cm, placement
    recommendation = model_resp.get("recommendation", "unknown_object")
    reasoning = model_resp.get("reasoning", "")
    suggested_size = model_resp.get("suggested_size_cm", {"w": 50, "h": 50})
    suggested_placement = model_resp.get("placement", {"x": 100, "y": 100})

    # generate icon
    icon_name = recommendation.replace(" ", "_") + "_" + uid + ".png"
    icon_path = os.path.join(ICONS_DIR, icon_name)
    generate_transparent_icon(recommendation, icon_path, size=(128,128))

    # validate/adjust placement against floorplan
    final_placement = validate_and_adjust_placement(suggested_placement, floorplan_path, icon_size=(128,128), max_attempts=400)

    # paste onto floorplan
    placed_fname = f"placed_{recommendation.replace(' ','_')}_{uid}.png"
    placed_path = os.path.join(PLACED_DIR, placed_fname)
    place_icon_on_floorplan(floorplan_path, icon_path, final_placement, placed_path)

    response = {
        "recommendation": recommendation,
        "reasoning": reasoning,
        "suggested_size_cm": suggested_size,
        "final_placement": final_placement,  # x,y,w,h (pixels)
        "icon_path": icon_path,
        "placed_floorplan_path": placed_path,
        "model_raw_response": model_resp if isinstance(model_resp, dict) else {"raw_text": str(model_resp)}
    }
    return jsonify(response)

@app.route("/files/<path:filename>", methods=["GET"])
def serve_file(filename):
    # simple endpoint to fetch saved files under output_data
    full = os.path.join(OUTPUT_DIR, filename)
    dirname = os.path.dirname(full)
    basename = os.path.basename(full)
    if not os.path.exists(full):
        return jsonify({"error": "file not found"}), 404
    return send_from_directory(dirname, basename)

# ---------------------------
# CLI helper (process video + local flow)
# ---------------------------
def demo_local_run(video_or_image_path: str, floorplan_path: str, persona: str, requirements: str):
    """
    Convenience helper to run the full pipeline locally without Flask.
    """
    print("1) Extracting frames (if input is video)...")
    ext = os.path.splitext(video_or_image_path)[1].lower()
    temp_dir = tempfile.mkdtemp(prefix="mm_demo_")
    if ext in [".mp4", ".mov", ".avi", ".mkv"]:
        frames = extract_frames(video_or_image_path, out_dir=temp_dir, fps=1)
        frame_path = frames[0]
    else:
        # copy image
        frame_path = os.path.join(temp_dir, os.path.basename(video_or_image_path))
        Image.open(video_or_image_path).save(frame_path)

    print("2) Building prompt...")
    prompt = build_prompt(persona, requirements)

    print("3) Calling multimodal model (OpenAI)...")
    model_resp = call_multimodal_model(frame_path, prompt)
    print("Model response (parsed):", model_resp)

    recommendation = model_resp.get("recommendation", "unknown")
    suggested_placement = model_resp.get("placement", {"x":100,"y":100})
    icon_path = os.path.join(ICONS_DIR, f"{recommendation}_{uuid.uuid4().hex[:6]}.png")
    generate_transparent_icon(recommendation, icon_path, size=(128,128))
    final_placement = validate_and_adjust_placement(suggested_placement, floorplan_path, icon_size=(128,128))
    placed_path = os.path.join(PLACED_DIR, f"placed_{recommendation}_{uuid.uuid4().hex[:6]}.png")
    place_icon_on_floorplan(floorplan_path, icon_path, final_placement, placed_path)

    print("Output saved:", {"icon": icon_path, "placed_floorplan": placed_path, "placement": final_placement})
    return {"icon": icon_path, "placed_floorplan": placed_path, "placement": final_placement, "model_resp": model_resp}

# ---------------------------
# Main: run Flask
# ---------------------------
if __name__ == "__main__":
    print("Multimodal recommender service starting...")
    print("Ensure OPENAI_API_KEY environment variable is set.")
    print("API endpoint: POST /recommend (file, floorplan, persona, requirements)")
    print("\nQuick ASCII flow diagram:\n")
    print("""
    [Input video/image] ---> Extract 1fps frames ---> Choose frame
                   |
                   v
               [Build prompt]  (persona + requirements)
                   |
                   v
      [Multimodal model (image + prompt) -> JSON]
                   |
                   v
    [Generate icon PNG] ---> [Validate placement w/ floorplan obstacles] ---> [Paste icon on floorplan]
                   |
                   v
               [Return JSON + file paths]
    """)
    app.run(debug=True, port=5000)
