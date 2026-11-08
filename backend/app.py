import io
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFilter
from pydantic import BaseModel

from backend.investigator import (
    ASSETS_DIR,
    SAMPLES_DIR,
    ModelInvestigator,
    base64_to_pil,
    fetch_image_from_url,
    pil_to_base64,
)

app = FastAPI(
    title="ShortcutLab — AI Model Shortcut & Spurious Correlation Investigator",
    description=(
        "Interactive ML interpretability suite using PyTorch Vision Transformers & ConvNets, "
        "Grad-CAM, Semantic Segmentation, Patch Occlusion, Counterfactual Synthesis, "
        "Deletion/Insertion Curves, and Feature Bias Probes."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global investigator singleton
investigator = ModelInvestigator()

PRESET_SAMPLES: List[Dict] = [
    {
        "id": "wolf_snow",
        "filename": "wolf_snow.jpg",
        "title": "Wolf in Snowy Blizzard",
        "subtitle": "Classic Snow Shortcut Case",
        "badge": "Shortcut Exposed",
        "badge_color": "rose",
        "custom": False,
        "description": (
            "A wolf standing in heavy snow. Removing the snowy background drops 'timber wolf' "
            "confidence by 38.6%, and swapping to a desert environment flips the prediction to 'coyote'."
        ),
    },
    {
        "id": "wolf_clear",
        "filename": "wolf_clear.jpg",
        "title": "Wolf (High-Contrast Coat)",
        "subtitle": "Moderate Context Coupling",
        "badge": "Moderate Coupling",
        "badge_color": "amber",
        "custom": False,
        "description": (
            "High-contrast Eurasian wolf in snow. Foreground anatomy anchors the prediction, "
            "though removing the snow still reduces confidence by 16.2%."
        ),
    },
    {
        "id": "husky_snow",
        "filename": "husky_snow.jpg",
        "title": "Siberian Husky in Snow",
        "subtitle": "Robust Foreground Features",
        "badge": "Robust Subject",
        "badge_color": "emerald",
        "custom": False,
        "description": (
            "Siberian Husky / Eskimo dog in snow. Facial markings strongly anchor the model "
            "even when transplanted across forest, desert, or urban scenes."
        ),
    },
    {
        "id": "waterbird_lake",
        "filename": "waterbird_lake.jpg",
        "title": "Mallard Drake on Water",
        "subtitle": "Waterbird Habitat Test",
        "badge": "Habitat Prior",
        "badge_color": "sky",
        "custom": False,
        "description": (
            "Tests the classic Waterbirds benchmark hypothesis: even when the bird is completely "
            "removed, the lake water ripples alone elicit a 47.0% 'drake' prediction!"
        ),
    },
    {
        "id": "camel_desert",
        "filename": "camel_desert.jpg",
        "title": "Camel in Arid Outback",
        "subtitle": "Desert Context Invariance",
        "badge": "Robust Subject",
        "badge_color": "emerald",
        "custom": False,
        "description": (
            "Arabian camel in arid terrain. Tests whether transplanting a desert animal into "
            "an Arctic snowfield or city street breaks the Vision Transformer."
        ),
    },
]

CUSTOM_SAMPLES_FILE = ASSETS_DIR / "custom_samples.json"


def _load_custom_samples() -> List[Dict]:
    if not CUSTOM_SAMPLES_FILE.exists():
        return []
    try:
        items = json.loads(CUSTOM_SAMPLES_FILE.read_text())
        return [s for s in items if (SAMPLES_DIR / s["filename"]).exists()]
    except Exception:
        return []


def _save_custom_samples(items: List[Dict]):
    CUSTOM_SAMPLES_FILE.write_text(json.dumps(items, indent=2))


def get_all_samples() -> List[Dict]:
    return PRESET_SAMPLES + _load_custom_samples()


class InvestigateRequest(BaseModel):
    sample_id: Optional[str] = None
    image_base64: Optional[str] = None
    grid_size: int = 8
    model_id: str = "vit_b_16"
    target_idx: Optional[int] = None


class PerturbRequest(BaseModel):
    sample_id: Optional[str] = None
    image_base64: Optional[str] = None
    occluded_cells: List[Tuple[int, int]] = []
    grid_size: int = 8
    fill_mode: str = "grey"
    bg_swap_id: Optional[str] = "original"
    target_idx: Optional[int] = None
    model_id: str = "vit_b_16"


class CrossModelRequest(BaseModel):
    sample_id: Optional[str] = None
    image_base64: Optional[str] = None
    model_ids: Optional[List[str]] = None
    target_idx: Optional[int] = None


class CurvesAnalysisRequest(BaseModel):
    sample_id: Optional[str] = None
    image_base64: Optional[str] = None
    grid_size: int = 8
    model_id: str = "vit_b_16"
    target_idx: Optional[int] = None


class ProbesAnalysisRequest(BaseModel):
    sample_id: Optional[str] = None
    image_base64: Optional[str] = None
    model_id: str = "vit_b_16"
    target_idx: Optional[int] = None


class AddSampleRequest(BaseModel):
    title: str
    subtitle: str = "Custom Experimental Image"
    badge: str = "Custom Case"
    badge_color: str = "sky"
    description: str = "User-added image for shortcut and counterfactual analysis."
    image_base64: Optional[str] = None
    image_url: Optional[str] = None


class AddBackgroundRequest(BaseModel):
    label: str
    description: str = "Custom counterfactual environment"
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    procedural_type: Optional[str] = None  # 'studio_white' | 'dark_void' | 'checkerboard' | 'autumn_foliage'


class LoadModelRequest(BaseModel):
    model_id: str


def _resolve_image(sample_id: Optional[str], image_base64: Optional[str]) -> Image.Image:
    if image_base64:
        return base64_to_pil(image_base64)
    if sample_id:
        all_samples = get_all_samples()
        spec = next((s for s in all_samples if s["id"] == sample_id), None)
        if not spec:
            raise HTTPException(status_code=404, detail=f"Unknown sample_id: {sample_id}")
        p = SAMPLES_DIR / spec["filename"]
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Sample file missing: {p.name}")
        return Image.open(p).convert("RGB")
    raise HTTPException(status_code=400, detail="Provide either sample_id or image_base64")


def _generate_procedural_background(proc_type: str) -> Image.Image:
    w, h = 512, 512
    rng = np.random.default_rng(123)
    if proc_type == "studio_white":
        # High-albedo studio backdrop with subtle vignette
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt(((x - 256) / 256.0) ** 2 + ((y - 256) / 256.0) ** 2)
        val = np.clip(248 - dist * 28, 200, 252).astype(np.uint8)
        arr = np.stack([val, val, val], axis=-1)
        return Image.fromarray(arr)
    elif proc_type == "dark_void":
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt(((x - 256) / 256.0) ** 2 + ((y - 256) / 256.0) ** 2)
        val = np.clip(32 - dist * 18, 10, 40).astype(np.uint8)
        arr = np.stack([val, val, val + 5], axis=-1)
        return Image.fromarray(arr)
    elif proc_type == "checkerboard":
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        tile = 32
        for r in range(h // tile):
            for c in range(w // tile):
                color = 215 if (r + c) % 2 == 0 else 65
                arr[r * tile : (r + 1) * tile, c * tile : (c + 1) * tile] = color
        return Image.fromarray(arr)
    else:
        # Autumn amber woodland texture
        base = np.zeros((h, w, 3), dtype=np.float32)
        base[..., 0] = 195 + rng.normal(0, 28, (h, w))
        base[..., 1] = 105 + rng.normal(0, 24, (h, w))
        base[..., 2] = 35 + rng.normal(0, 14, (h, w))
        im = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
        return im.filter(ImageFilter.GaussianBlur(radius=4))


@app.get("/api/samples")
def list_samples():
    items = []
    for s in get_all_samples():
        p = SAMPLES_DIR / s["filename"]
        thumb = ""
        if p.exists():
            im = Image.open(p).convert("RGB").resize((160, 160), Image.Resampling.LANCZOS)
            thumb = pil_to_base64(im, quality=82)
        items.append({**s, "thumbnail": thumb})
    return {
        "samples": items,
        "backgrounds": [
            {
                "id": k,
                "label": v["label"],
                "description": v["description"],
                "custom": v.get("custom", False),
                "thumbnail": pil_to_base64(
                    v["image"].resize((120, 120), Image.Resampling.LANCZOS), quality=80
                ),
            }
            for k, v in investigator.backgrounds.items()
        ],
        "models": investigator.list_models_status(),
        "categories": [
            {"idx": i, "label": lbl} for i, lbl in enumerate(investigator.categories)
        ],
    }


@app.post("/api/samples/add")
def api_add_sample(req: AddSampleRequest):
    if req.image_base64:
        img = base64_to_pil(req.image_base64)
    elif req.image_url:
        try:
            img = fetch_image_from_url(req.image_url)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to fetch image from URL: {e}"
            )
    else:
        raise HTTPException(
            status_code=400, detail="Provide either image_base64 or image_url"
        )

    slug = re.sub(r"[^a-z0-9]+", "_", req.title.lower()).strip("_") or "custom"
    sample_id = f"{slug}_{int(time.time()) % 100000}"
    filename = f"{sample_id}.jpg"

    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img_cropped = img.crop((left, top, left + s, top + s)).resize(
        (512, 512), Image.Resampling.LANCZOS
    )
    img_cropped.save(SAMPLES_DIR / filename, quality=92)

    new_entry = {
        "id": sample_id,
        "filename": filename,
        "title": req.title.strip() or "Custom Subject",
        "subtitle": req.subtitle.strip() or "Custom Experimental Image",
        "badge": req.badge.strip() or "Custom Case",
        "badge_color": req.badge_color or "sky",
        "custom": True,
        "description": req.description,
    }
    custom_list = _load_custom_samples()
    custom_list.append(new_entry)
    _save_custom_samples(custom_list)

    thumb = pil_to_base64(
        img_cropped.resize((160, 160), Image.Resampling.LANCZOS), quality=82
    )
    return {"sample": {**new_entry, "thumbnail": thumb}}


@app.delete("/api/samples/{sample_id}")
def api_delete_sample(sample_id: str):
    custom_list = _load_custom_samples()
    target = next((s for s in custom_list if s["id"] == sample_id), None)
    if not target:
        raise HTTPException(
            status_code=404, detail="Custom sample not found (preset samples cannot be deleted)."
        )
    p = SAMPLES_DIR / target["filename"]
    if p.exists():
        p.unlink()
    updated = [s for s in custom_list if s["id"] != sample_id]
    _save_custom_samples(updated)
    return {"deleted": sample_id}


@app.post("/api/backgrounds/add")
def api_add_background(req: AddBackgroundRequest):
    if req.image_base64:
        img = base64_to_pil(req.image_base64)
    elif req.image_url:
        try:
            img = fetch_image_from_url(req.image_url)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to fetch background URL: {e}"
            )
    elif req.procedural_type:
        img = _generate_procedural_background(req.procedural_type)
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide image_base64, image_url, or procedural_type",
        )

    slug = re.sub(r"[^a-z0-9]+", "_", req.label.lower()).strip("_") or "habitat"
    bg_id = f"{slug}_{int(time.time()) % 10000}"
    bg_meta = investigator.add_background(
        bg_id=bg_id,
        label=req.label.strip() or "Custom Habitat",
        description=req.description.strip() or "Custom counterfactual background",
        img=img,
    )
    return {"background": bg_meta}


@app.post("/api/models/load")
def api_load_model(req: LoadModelRequest):
    try:
        investigator.load_model(req.model_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"models": investigator.list_models_status()}


@app.post("/api/investigate")
def api_investigate(req: InvestigateRequest):
    img = _resolve_image(req.sample_id, req.image_base64)
    grid_size = max(4, min(14, req.grid_size))
    return investigator.investigate(
        img,
        grid_size=grid_size,
        model_id=req.model_id,
        target_idx=req.target_idx,
    )


@app.post("/api/perturb")
def api_perturb(req: PerturbRequest):
    img = _resolve_image(req.sample_id, req.image_base64)
    grid_size = max(4, min(14, req.grid_size))
    return investigator.run_custom_perturbation(
        img=img,
        occluded_cells=req.occluded_cells,
        grid_size=grid_size,
        fill_mode=req.fill_mode,
        bg_swap_id=req.bg_swap_id,
        target_idx=req.target_idx,
        model_id=req.model_id,
    )


@app.post("/api/analyze/cross-model")
def api_analyze_cross_model(req: CrossModelRequest):
    img = _resolve_image(req.sample_id, req.image_base64)
    return investigator.run_cross_model_analysis(
        img=img, model_ids=req.model_ids, target_idx=req.target_idx
    )


@app.post("/api/analyze/curves")
def api_analyze_curves(req: CurvesAnalysisRequest):
    img = _resolve_image(req.sample_id, req.image_base64)
    grid_size = max(4, min(14, req.grid_size))
    return investigator.run_insertion_deletion_analysis(
        img=img,
        grid_size=grid_size,
        model_id=req.model_id,
        target_idx=req.target_idx,
    )


@app.post("/api/analyze/probes")
def api_analyze_probes(req: ProbesAnalysisRequest):
    img = _resolve_image(req.sample_id, req.image_base64)
    return investigator.run_feature_bias_probes(
        img=img, model_id=req.model_id, target_idx=req.target_idx
    )


@app.post("/api/upload")
async def api_upload(
    file: UploadFile = File(...),
    grid_size: int = Form(8),
    model_id: str = Form("vit_b_16"),
    save_to_library: bool = Form(False),
    title: Optional[str] = Form(None),
):
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    saved_sample = None
    if save_to_library:
        sample_title = title or Path(file.filename or "Uploaded Image").stem.replace("_", " ").title()
        add_res = api_add_sample(
            AddSampleRequest(
                title=sample_title,
                subtitle="Uploaded Custom Subject",
                badge="Uploaded",
                badge_color="sky",
                image_base64=pil_to_base64(img),
            )
        )
        saved_sample = add_res["sample"]

    inv_result = investigator.investigate(
        img, grid_size=max(4, min(14, grid_size)), model_id=model_id
    )
    return {**inv_result, "saved_sample": saved_sample}


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
FRONTEND_DIR.mkdir(exist_ok=True)


@app.get("/")
def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
