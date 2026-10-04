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
1. Train or obtain the model file `oi detection/notebooks/nauticeye(4).ipynb` (see `notebooks/nauticeye.ipynb`) and put it in your Google Drive (`MyDrive`).
2. Open a new Google Colab notebook, set **Runtime → T4 GPU**, run `!pip install -q -U sympy fastapi uvicorn python-multipart nest_asyncio`, then **Runtime → Restart session**.
3. Paste `backend/colab_backend.py` into a cell (delete its first `!pip` line), run it, and copy the printed `https://....trycloudflare.com` URL.
4. Open `ai detection/analyze.html` (locally or hosted) and paste that URL into the API URL box.

The tunnel URL changes each time Colab restarts.


## ⚠️ Important: the API link must be recreated

NauticEye's AI model runs on a Google Colab server, and the website reaches it through a temporary public link (the **API URL**). **This link expires whenever the Colab server stops or restarts, so it has to be recreated before the live demo will work.**

**If the demo shows "Could not analyze", do this:**
1. Open the Colab notebook and run the backend cell (`backend/colab_backend.py`) on a T4 GPU.
2. Wait until it prints a new link ending in `trycloudflare.com` (or your ngrok domain).
3. Open the website (`ai detection/analyze.html`), paste the new link into the **API URL** box at the top, and click **Detect oil**.


## AI Model API Setup

1. Download **`oil_spill_efficientnet_b0(1).pth`** from the project's **Models** section and upload it to your Google Drive.
2. Open **`Untitled1.ipynb`** in Google Colab and connect/mount your Google Drive.
3. Locate the uploaded `.pth` file and update the notebook with its correct **Google Drive path**.
4. Skip/delete the **pip installation/download section** and keep the model and API code unchanged.
5. Run the **main model/API block** and wait for the AI model to load successfully.
6. Copy the **newly generated AI API key** provided by the notebook.
7. Open **`nautieyes.netlify.app`**, paste the API key into the required configuration, and save it.
8. The NauticEye frontend can now communicate with the trained **oil-spill detection model** through the API.

No API keys are stored in this repository. If the server is offline, see the demo video: **[add your video link]**.
## Repository layout
   - `ai detection/analyze.html`: frontend nauticeyes.netlify.app
   - `backend/colab_backend.py`: inference API
   - `ai detection/notebooks/`: training and Kochi scene analysis
   - `LICENSE`: MIT


## Data and credits
- Training data: [Sentinel-1 SAR Oil Spill Detection Dataset](https://www.kaggle.com/datasets/harikrishnacs/sentinel-1-sar-oil-spill-detection-dataset) (CC BY-SA 4.0), derived from CSIRO researchers' work. Credit them if you reuse it.
- Scene demo: Copernicus Sentinel-1 data (ESA), downloaded via ASF.
- Model: EfficientNet-B0 (Tan & Le, 2019) via torchvision.

## Team RawOnions
Aryan Shukla, 
Anshuman Sengar, 
Harshit Chaturvedi, 
Paridhi Vishwakarma

