# NauticEye: AI Oil Spill Detection from SAR Satellite Images

Upload a Sentinel-1 SAR image and NauticEye tells you whether it contains an oil slick, shows **where** (highlighted overlay, probability map, mask) and **how sure** the model is.

Built for an MLH hackathon. Open source under the MIT license.

## How it works
1. **Model:** EfficientNet-B0 (ImageNet pre-trained, transfer learning) with a custom 1280→128→2 head, trained on Sentinel-1 SAR chips (oil vs no-oil). Test set: ~94% accuracy, ~94% oil recall.
2. **Localization:** large images are cut into overlapping 400×400 tiles. Each tile gets an oil probability. Tile probabilities are merged into a map and combined with a dark-pixel (low-backscatter) threshold to outline the slick.
3. **Backend:** FastAPI server (`backend/colab_backend.py`) runs on a Google Colab T4 GPU and is exposed via a Cloudflare quick tunnel.
4. **Frontend:** a single static page (`index.html`) deployed on Netlify. Upload an image, see the verdict, before/after slider, probability map and region table.

> Note: the model classifies tiles, so the outline is an estimate (model + dark-pixel refinement), not a pixel-level segmentation model. Dark look-alikes (low wind, algae) can cause false alarms; results are meant for analyst review.

## Run it yourself
1. Train or obtain the model file `oil_spill_efficientnet_b0.pth` (see `notebooks/nauticeye.ipynb`) and put it in your Google Drive (`MyDrive`).
2. Open a new Google Colab notebook, set **Runtime → T4 GPU**, run `!pip install -q -U sympy fastapi uvicorn python-multipart nest_asyncio`, then **Runtime → Restart session**.
3. Paste `backend/colab_backend.py` into a cell (delete its first `!pip` line), run it, and copy the printed `https://....trycloudflare.com` URL.
4. Open `index.html` (locally or hosted) and paste that URL into the API URL box.

The tunnel URL changes each time Colab restarts.

## Repository layout
- `index.html`: frontend
- `backend/colab_backend.py`: inference API
- `notebooks/nauticeye.ipynb`: training and Kochi scene analysis
- `LICENSE`: MIT

## Data and credits
- Training data: [Sentinel-1 SAR Oil Spill Detection Dataset](https://www.kaggle.com/datasets/harikrishnacs/sentinel-1-sar-oil-spill-detection-dataset) (CC BY-SA 4.0), derived from CSIRO researchers' work. Credit them if you reuse it.
- Scene demo: Copernicus Sentinel-1 data (ESA), downloaded via ASF.
- Model: EfficientNet-B0 (Tan & Le, 2019) via torchvision.

## Team
Add your team names here.
