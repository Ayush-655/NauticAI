# =====================================================================
# OIL SPILL API  -  paste this whole file into ONE new Colab cell
# (Runtime must be T4 GPU, and Google Drive must hold your .pth file)
# =====================================================================

# ---- 1. Install the web-server libraries (quiet = less text) ----
!pip install -q fastapi uvicorn python-multipart nest_asyncio

# ---- 2. Imports ----
import io, base64, threading, subprocess, re, time, os          # standard tools
import numpy as np                                                # number grids (images are grids of numbers)
import cv2                                                        # image processing (blur, contours, colours)
import torch                                                      # the AI framework
import torch.nn as nn                                             # building blocks for the network
from torchvision import models, transforms                        # EfficientNet + image preparation
from PIL import Image                                             # opens uploaded images

# ---- 3. Settings you may need to edit ----
MODEL_PATH = "/content/drive/MyDrive/oil_spill_efficientnet_b0.pth"  # where your trained model lives
TILE = 400            # the model was trained on ~400x400 chips, so we cut images into 400x400 windows
STRIDE = 200          # slide the window 200 px each step (50% overlap = smoother result)
MAX_SIDE = 4096       # shrink gigantic uploads so the free GPU doesn't run out of memory
OIL_INDEX = 1         # class order was ["no_oil","oil"], so "oil" is index 1

# ---- 4. Connect Google Drive (skip if already mounted) ----
from google.colab import drive                                    # Colab's Drive helper
if not os.path.exists("/content/drive/MyDrive"):                  # only mount if not already mounted
    drive.mount("/content/drive")                                 # asks you for permission once

# ---- 5. Rebuild the exact network you trained, then load the saved numbers into it ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # GPU if available
model = models.efficientnet_b0(weights=None)                      # empty EfficientNet-B0 (weights come from your file)
model.classifier = nn.Sequential(                                 # SAME head as Cell 7 of your training notebook
    nn.Linear(1280, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2))
ckpt = torch.load(MODEL_PATH, map_location=device)                # read the .pth file
model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)  # fill in learned numbers
model = model.to(device).eval()                                   # eval() = switch off dropout, we're predicting now
print("Model loaded on", device)

# Same preparation steps used in training (resize to 224, normalise like ImageNet)
prep = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

@torch.no_grad()                                                  # no gradients needed = faster, less memory
def predict_tiles(tiles):
    """Takes a list of PIL tiles, returns a list of oil probabilities (0..1)."""
    probs = []
    for i in range(0, len(tiles), 32):                            # process 32 tiles at a time
        batch = torch.stack([prep(t) for t in tiles[i:i+32]]).to(device)   # stack into one tensor on the GPU
        p = torch.softmax(model(batch), dim=1)[:, OIL_INDEX]      # scores -> probability of "oil"
        probs += p.cpu().tolist()                                 # bring results back to normal Python
    return probs

# ---- 6. The analysis pipeline (no AI-specific code below this line) ----
def to_b64(arr_bgr):
    ok, buf = cv2.imencode(".png", arr_bgr)                       # turn the image grid into PNG bytes
    return "data:image/png;base64," + base64.b64encode(buf).decode()   # wrap as a string the browser can show

def analyze_image(pil_img, threshold=0.5, predict_fn=predict_tiles):
    img = pil_img.convert("L")                                    # grayscale (SAR is one channel)
    w, h = img.size
    if max(w, h) > MAX_SIDE:                                      # shrink very large uploads
        s = MAX_SIDE / max(w, h); img = img.resize((int(w*s), int(h*s))); w, h = img.size
    gray = np.array(img)                                          # PIL -> numpy grid, shape (h, w)

    # 6a. Cut into overlapping tiles (if image is smaller than a tile, use the whole image as one tile)
    tw, th = min(TILE, w), min(TILE, h)
    xs = list(range(0, max(w - tw, 0) + 1, STRIDE)); ys = list(range(0, max(h - th, 0) + 1, STRIDE))
    if xs[-1] != w - tw: xs.append(w - tw)                        # make sure the right edge is covered
    if ys[-1] != h - th: ys.append(h - th)                        # make sure the bottom edge is covered
    boxes = [(x, y) for y in ys for x in xs]
    valid = gray > 5                                              # pixels that are real data (0 = black border/no data)
    tiles, keep = [], []
    for (x, y) in boxes:
        if valid[y:y+th, x:x+tw].mean() < 0.3: continue           # skip tiles that are mostly empty border
        t = gray[y:y+th, x:x+tw].astype(np.float32)               # same per-tile contrast stretch (2%-98%) your Kochi scene pipeline used
        lo, hi = np.percentile(t, (2, 98))
        t = np.clip((t - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)
        tiles.append(Image.fromarray(t).convert("RGB")); keep.append((x, y))

    # 6b. Ask the model about every tile, paint each tile's probability onto the pixels it covers, then average
    probs = predict_fn(tiles) if tiles else []
    acc = np.zeros((h, w), np.float32); cnt = np.zeros((h, w), np.float32)
    for (x, y), p in zip(keep, probs):
        acc[y:y+th, x:x+tw] += p; cnt[y:y+th, x:x+tw] += 1
    prob_map = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0).astype(np.float32)
    prob_map = cv2.GaussianBlur(prob_map, (0, 0), max(TILE // 12, 3))   # soften hard tile edges

    # 6c. Refine: oil = "model suspects this area" AND "pixel is dark" (slicks are dark in SAR)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)                      # blur away speckle noise before thresholding
    vals = blur[valid]
    dark_cut = cv2.threshold(vals.reshape(-1, 1).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] if vals.size else 0
    dark = (blur < dark_cut) & valid                              # Otsu picks the dark/bright split automatically
    mask = ((prob_map >= threshold) & dark).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)              # remove tiny specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)             # fill small holes
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    min_area = 0.0005 * h * w                                     # ignore blobs smaller than 0.05% of the image
    final = np.zeros_like(mask); regions = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < min_area: continue
        final[labels == i] = 1
        regions.append({
            "id": len(regions) + 1, "area_px": a,
            "area_pct": round(100 * a / (h * w), 2),
            "bbox": [int(stats[i, 0]), int(stats[i, 1]), int(stats[i, 2]), int(stats[i, 3])],
            "centroid": [round(float(cents[i][0]) / w, 4), round(float(cents[i][1]) / h, 4)],   # 0..1 position in image
            "confidence": round(float(prob_map[labels == i].mean()), 3)})

    # 6d. Draw the pretty visuals
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)                 # grayscale -> 3 channels so we can add colour
    glow = cv2.GaussianBlur(final.astype(np.float32), (0, 0), 12)[..., None]   # soft halo around the spill
    core = cv2.GaussianBlur(final.astype(np.float32), (0, 0), 2)[..., None]
    overlay = base.astype(np.float32)
    overlay = overlay * (1 - 0.45 * glow) + np.array([40, 140, 255], np.float32) * 0.45 * glow    # orange glow (BGR)
    overlay = overlay * (1 - 0.35 * core) + np.array([30, 40, 255], np.float32) * 0.35 * core     # red core
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    cnts, _ = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (255, 255, 0), max(2, w // 500))   # cyan outline
    for r in regions:                                             # label each region with its number
        cx, cy = int(r["centroid"][0] * w), int(r["centroid"][1] * h)
        cv2.putText(overlay, str(r["id"]), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, max(0.8, w / 1200), (255, 255, 255), 2, cv2.LINE_AA)
    heat = cv2.applyColorMap((np.clip(prob_map, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)   # blue=safe, red=oil
    heat = cv2.addWeighted(base, 0.4, heat, 0.6, 0)
    mask_img = cv2.cvtColor(final * 255, cv2.COLOR_GRAY2BGR)

    def small(a):                                                 # keep response size reasonable (max 1400 px wide)
        return cv2.resize(a, (1400, int(h * 1400 / w))) if w > 1400 else a
    return {
        "oil_detected": len(regions) > 0,
        "max_probability": round(float(prob_map.max()) if tiles else 0.0, 3),
        "coverage_pct": round(100 * float(final.sum()) / (h * w), 2),
        "regions": regions, "image_size": [w, h], "tiles_scanned": len(tiles), "threshold": threshold,
        "original": to_b64(small(base)), "overlay": to_b64(small(overlay)),
        "heatmap": to_b64(small(heat)), "mask": to_b64(small(mask_img))}

# ---- 7. Web server ----
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, nest_asyncio
nest_asyncio.apply()                                              # lets the server run inside Colab's notebook

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])   # lets your website call us

@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}                # quick "are you alive?" check

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), threshold: float = Form(0.5)):
    data = await file.read()                                      # raw bytes of the uploaded picture
    img = Image.open(io.BytesIO(data))                            # open them as an image
    return analyze_image(img, threshold)                          # run the pipeline, return JSON

threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning"), daemon=True).start()
time.sleep(3)                                                     # give the server a moment to start

# ---- 8. Public HTTPS link (free Cloudflare tunnel, same trick you used before) ----
if not os.path.exists("/content/cloudflared"):
    !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /content/cloudflared
    !chmod +x /content/cloudflared
proc = subprocess.Popen(["/content/cloudflared", "tunnel", "--url", "http://localhost:8000"],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:                                          # read the tunnel's log until it prints the link
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
    if m:
        print("\n==== YOUR API URL (give this to the frontend) ====\n" + m.group(0) + "\n"); break
