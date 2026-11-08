"""
Builds a standalone static site in `docs/` for GitHub Pages deployment.
Pre-computes real PyTorch (ViT-B/16, Swin-T, ConvNeXt-Tiny, ResNet-50) investigations,
Grad-CAM maps, DeepLabV3 masks, counterfactuals, curves, and feature probes into JSON
files in `docs/data/`, and bundles an automatic static/serverless fallback into `docs/app.js`.
"""
import json
import shutil
from pathlib import Path
from PIL import Image

from backend.app import (
    PRESET_SAMPLES,
    list_samples,
    api_investigate,
    api_analyze_cross_model,
    api_analyze_curves,
    api_analyze_probes,
    InvestigateRequest,
    CrossModelRequest,
    CurvesAnalysisRequest,
    ProbesAnalysisRequest,
)
from backend.investigator import SAMPLES_DIR, pil_to_base64

DOCS_DIR = Path("docs")
DATA_DIR = DOCS_DIR / "data"
DOCS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

print("1. Exporting /api/samples metadata...")
samples_meta = list_samples()
(DATA_DIR / "samples.json").write_text(json.dumps(samples_meta))

# Also export raw 256x256 foreground alpha masks for client-side canvas compositing on GitHub Pages
from backend.app import investigator

masks_dict = {}
raw_bgs = {}
for bg_id, bg_info in investigator.backgrounds.items():
    raw_bgs[bg_id] = pil_to_base64(bg_info["image"].resize((384, 384)), quality=82)

models_to_export = ["vit_b_16", "swin_t", "convnext_tiny", "resnet50"]

for s in PRESET_SAMPLES:
    sid = s["id"]
    print(f"2. Pre-computing PyTorch telemetry for sample: {sid}...")
    img = Image.open(SAMPLES_DIR / s["filename"]).convert("RGB").resize((512, 512))
    fg_alpha, _, _ = investigator.segment_foreground(img)
    # Downsample mask to 64x64 for lightweight client-side habitat swapping
    mask_64 = (
        Image.fromarray((fg_alpha * 255).astype("uint8"))
        .resize((64, 64), Image.Resampling.BILINEAR)
    )
    masks_dict[sid] = pil_to_base64(mask_64, fmt="PNG")

    for mid in models_to_export:
        inv = api_investigate(InvestigateRequest(sample_id=sid, grid_size=8, model_id=mid))
        # Compress base64 images slightly to keep static bundle fast
        (DATA_DIR / f"inv_{sid}_{mid}.json").write_text(json.dumps(inv))

    # Cross-model, Curves, and Probes for each sample
    cm = api_analyze_cross_model(CrossModelRequest(sample_id=sid, model_ids=models_to_export))
    (DATA_DIR / f"cm_{sid}.json").write_text(json.dumps(cm))

    curves = api_analyze_curves(CurvesAnalysisRequest(sample_id=sid, grid_size=8, model_id="vit_b_16"))
    (DATA_DIR / f"curves_{sid}.json").write_text(json.dumps(curves))

    probes = api_analyze_probes(ProbesAnalysisRequest(sample_id=sid, model_id="vit_b_16"))
    (DATA_DIR / f"probes_{sid}.json").write_text(json.dumps(probes))

(DATA_DIR / "client_assets.json").write_text(
    json.dumps({"masks": masks_dict, "backgrounds": raw_bgs})
)

# Copy index.html (with relative ./app.js path) and app.js into docs/
html_text = Path("frontend/index.html").read_text().replace('src="/static/app.js"', 'src="./app.js"')
(DOCS_DIR / "index.html").write_text(html_text)
shutil.copy2("frontend/app.js", DOCS_DIR / "app.js")
(DOCS_DIR / ".nojekyll").write_text("")

print("Done precomputing static GitHub Pages bundle in docs/!")
