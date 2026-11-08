import base64
import io
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.models as models
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

# Ensure local weights directory is used
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
SAMPLES_DIR = ASSETS_DIR / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TORCH_HOME"] = str(WEIGHTS_DIR)


def pil_to_base64(img: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
    buf = io.BytesIO()
    if fmt.upper() == "JPEG" and img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buf, format=fmt, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


def base64_to_pil(data_url: str) -> Image.Image:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def fetch_image_from_url(url: str) -> Image.Image:
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) ShortcutLab/2.0"
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = resp.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s)).resize(
        (512, 512), Image.Resampling.LANCZOS
    )


def create_heatmap_overlay(
    base_img: Image.Image,
    grid: np.ndarray,
    alpha: float = 0.55,
) -> Image.Image:
    """Render a 2D normalized numpy grid [0, 1] as a smooth RGB heatmap blended over base_img."""
    w, h = base_img.size
    norm = np.clip(grid, 0.0, 1.0).astype(np.float32)
    grid_img = Image.fromarray((norm * 255).astype(np.uint8)).resize(
        (w, h), Image.Resampling.BICUBIC
    )
    v = np.array(grid_img).astype(np.float32) / 255.0

    # High-contrast scientific colormap (dark navy -> blue -> magenta -> orange -> bright yellow)
    r = np.clip(np.where(v < 0.5, v * 1.6, 0.8 + (v - 0.5) * 0.4) * 255, 0, 255)
    g = np.clip(np.where(v < 0.4, v * 0.3, (v - 0.4) / 0.6) * 255, 0, 255)
    b = np.clip(
        np.where(
            v < 0.35,
            0.25 + v * 2.1,
            np.where(v < 0.7, 1.0 - (v - 0.35) * 2.2, (v - 0.7) * 0.8),
        )
        * 255,
        0,
        255,
    )
    heat_rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    base_rgb = np.array(base_img.convert("RGB")).astype(np.float32)

    dynamic_alpha = alpha * (0.35 + 0.65 * v[..., None])
    blended = base_rgb * (1.0 - dynamic_alpha) + heat_rgb * dynamic_alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def create_segmentation_overlay(
    base_img: Image.Image, fg_mask: np.ndarray
) -> Image.Image:
    """Create a visual overlay highlighting foreground vs background regions and boundary contour."""
    w, h = base_img.size
    mask_pil = Image.fromarray((np.clip(fg_mask, 0, 1) * 255).astype(np.uint8)).resize(
        (w, h), Image.Resampling.BILINEAR
    )
    edges = mask_pil.filter(ImageFilter.FIND_EDGES).filter(
        ImageFilter.MaxFilter(size=3)
    )
    m = np.array(mask_pil).astype(np.float32) / 255.0
    e = np.array(edges).astype(np.float32) / 255.0

    base_rgb = np.array(base_img.convert("RGB")).astype(np.float32)
    bg_tint = np.array([14, 165, 233], dtype=np.float32)  # Sky blue for background
    fg_tint = np.array([16, 185, 129], dtype=np.float32)  # Emerald for subject

    out = (
        base_rgb * (0.68 + 0.22 * m[..., None])
        + bg_tint * (0.32 * (1.0 - m[..., None]))
        + fg_tint * (0.10 * m[..., None])
    )
    contour_color = np.array([56, 255, 180], dtype=np.float32)
    edge_mask = np.clip(e * 1.8, 0, 1)[..., None]
    out = out * (1.0 - edge_mask) + contour_color * edge_mask
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# Registry of supported torchvision architectures
MODEL_CATALOG: Dict[str, Dict] = {
    "vit_b_16": {
        "id": "vit_b_16",
        "name": "ViT-B/16 (ImageNet-1K)",
        "short_name": "ViT-B/16",
        "family": "Vision Transformer",
        "params": "86.6M",
        "receptive_field": "Global 16×16 Token Self-Attention",
        "builder": lambda: (
            models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT),
            models.ViT_B_16_Weights.DEFAULT,
        ),
        "gradcam_type": "vit_tokens",
        "target_layer": lambda m: m.encoder.layers[-1].ln_1,
        "default_loaded": True,
    },
    "swin_t": {
        "id": "swin_t",
        "name": "Swin-T (Shifted Window ViT)",
        "short_name": "Swin-T",
        "family": "Hierarchical Transformer",
        "params": "28.3M",
        "receptive_field": "Hierarchical Windowed Attention",
        "builder": lambda: (
            models.swin_t(weights=models.Swin_T_Weights.DEFAULT),
            models.Swin_T_Weights.DEFAULT,
        ),
        "gradcam_type": "swin_channels_last",
        "target_layer": lambda m: m.features[-1][-1].norm1,
        "default_loaded": True,
    },
    "convnext_tiny": {
        "id": "convnext_tiny",
        "name": "ConvNeXt-Tiny (Modern ConvNet)",
        "short_name": "ConvNeXt-T",
        "family": "Modern 7×7 ConvNet",
        "params": "28.6M",
        "receptive_field": "7×7 Depthwise Separable Conv",
        "builder": lambda: (
            models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT),
            models.ConvNeXt_Tiny_Weights.DEFAULT,
        ),
        "gradcam_type": "cnn_spatial",
        "target_layer": lambda m: m.features[-1],
        "default_loaded": True,
    },
    "resnet50": {
        "id": "resnet50",
        "name": "ResNet-50 (Residual ConvNet)",
        "short_name": "ResNet-50",
        "family": "Classic CNN",
        "params": "25.6M",
        "receptive_field": "3×3 Bottleneck Residual Stack",
        "builder": lambda: (
            models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
            models.ResNet50_Weights.DEFAULT,
        ),
        "gradcam_type": "cnn_spatial",
        "target_layer": lambda m: m.layer4[-1],
        "default_loaded": True,
    },
    "mobilenet_v3_large": {
        "id": "mobilenet_v3_large",
        "name": "MobileNetV3-Large (Lightweight CNN)",
        "short_name": "MobileNetV3",
        "family": "Mobile CNN",
        "params": "5.5M",
        "receptive_field": "Inverted Residual + Squeeze-Excitation",
        "builder": lambda: (
            models.mobilenet_v3_large(
                weights=models.MobileNet_V3_Large_Weights.DEFAULT
            ),
            models.MobileNet_V3_Large_Weights.DEFAULT,
        ),
        "gradcam_type": "cnn_spatial",
        "target_layer": lambda m: m.features[-1],
        "default_loaded": False,
    },
    "densenet121": {
        "id": "densenet121",
        "name": "DenseNet-121 (Dense ConvNet)",
        "short_name": "DenseNet-121",
        "family": "Dense CNN",
        "params": "8.0M",
        "receptive_field": "Concatenated Multi-Scale Features",
        "builder": lambda: (
            models.densenet121(weights=models.DenseNet121_Weights.DEFAULT),
            models.DenseNet121_Weights.DEFAULT,
        ),
        "gradcam_type": "cnn_spatial",
        "target_layer": lambda m: m.features.denseblock4,
        "default_loaded": False,
    },
}


class ModelInvestigator:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[ModelInvestigator] Initializing on device: {self.device}")

        # Loaded models cache: model_id -> {"model": nn.Module, "preprocess": Callable, "spec": Dict}
        self.models: Dict[str, Dict] = {}
        self.categories: List[str] = []

        # Preload default architectures cached on disk
        for model_id, spec in MODEL_CATALOG.items():
            if spec["default_loaded"]:
                try:
                    self.load_model(model_id)
                except Exception as e:
                    print(f"[ModelInvestigator] Warning loading {model_id}: {e}")

        # Semantic Segmentation Model (DeepLabV3-MobileNetV3) for Foreground/Background isolation
        self.seg_weights = (
            models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        )
        self.seg = (
            models.segmentation.deeplabv3_mobilenet_v3_large(weights=self.seg_weights)
            .to(self.device)
            .eval()
        )
        self.seg_preprocess = self.seg_weights.transforms()
        self.seg_categories: List[str] = self.seg_weights.meta["categories"]

        # Load background templates for counterfactual synthesis
        self.backgrounds: Dict[str, Dict] = {}
        bg_specs = [
            ("snow", "bg_snow.jpg", "Arctic Snowfield", "High-albedo snowy tundra"),
            ("forest", "bg_forest.jpg", "Temperate Forest", "Dense green woodland foliage"),
            ("desert", "bg_desert.jpg", "Arid Desert Dunes", "Warm sand & rocky canyon"),
            ("city", "bg_city.jpg", "Urban Environment", "Concrete & architectural textures"),
        ]
        for bg_id, filename, label, desc in bg_specs:
            p = SAMPLES_DIR / filename
            if p.exists():
                self.backgrounds[bg_id] = {
                    "id": bg_id,
                    "label": label,
                    "description": desc,
                    "image": Image.open(p).convert("RGB").resize((512, 512)),
                    "custom": False,
                }

        # Load any persisted custom backgrounds
        self._load_custom_backgrounds()

    def _load_custom_backgrounds(self):
        meta_path = ASSETS_DIR / "custom_backgrounds.json"
        if not meta_path.exists():
            return
        try:
            items = json.loads(meta_path.read_text())
            for item in items:
                p = SAMPLES_DIR / item["filename"]
                if p.exists():
                    self.backgrounds[item["id"]] = {
                        "id": item["id"],
                        "label": item["label"],
                        "description": item["description"],
                        "image": Image.open(p).convert("RGB").resize((512, 512)),
                        "custom": True,
                        "filename": item["filename"],
                    }
        except Exception as e:
            print(f"[ModelInvestigator] Failed loading custom backgrounds: {e}")

    def add_background(
        self, bg_id: str, label: str, description: str, img: Image.Image
    ) -> Dict:
        filename = f"bg_custom_{bg_id}.jpg"
        out_path = SAMPLES_DIR / filename
        img_resized = img.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
        img_resized.save(out_path, quality=92)

        self.backgrounds[bg_id] = {
            "id": bg_id,
            "label": label,
            "description": description,
            "image": img_resized,
            "custom": True,
            "filename": filename,
        }

        # Persist metadata
        meta_path = ASSETS_DIR / "custom_backgrounds.json"
        custom_list = [
            {
                "id": k,
                "label": v["label"],
                "description": v["description"],
                "filename": v["filename"],
            }
            for k, v in self.backgrounds.items()
            if v.get("custom")
        ]
        meta_path.write_text(json.dumps(custom_list, indent=2))
        return {
            "id": bg_id,
            "label": label,
            "description": description,
            "custom": True,
            "thumbnail": pil_to_base64(
                img_resized.resize((120, 120), Image.Resampling.LANCZOS), quality=80
            ),
        }

    def load_model(self, model_id: str) -> Dict:
        if model_id in self.models:
            return self.models[model_id]
        if model_id not in MODEL_CATALOG:
            raise ValueError(f"Unsupported model_id '{model_id}'.")
        spec = MODEL_CATALOG[model_id]
        print(f"[ModelInvestigator] Loading architecture: {spec['name']}...")
        net, weights = spec["builder"]()
        net = net.to(self.device).eval()
        preprocess = weights.transforms()
        if not self.categories:
            self.categories = weights.meta["categories"]
        entry = {
            "id": model_id,
            "model": net,
            "preprocess": preprocess,
            "spec": spec,
        }
        self.models[model_id] = entry
        return entry

    def get_model_entry(self, model_id: Optional[str] = None) -> Dict:
        target = model_id or "vit_b_16"
        if target not in self.models:
            return self.load_model(target)
        return self.models[target]

    def list_models_status(self) -> List[Dict]:
        out = []
        for m_id, spec in MODEL_CATALOG.items():
            out.append(
                {
                    "id": m_id,
                    "name": spec["name"],
                    "short_name": spec["short_name"],
                    "family": spec["family"],
                    "params": spec["params"],
                    "receptive_field": spec["receptive_field"],
                    "loaded": m_id in self.models,
                }
            )
        return out

    def predict_probs(
        self, img: Image.Image, model_id: str = "vit_b_16"
    ) -> torch.Tensor:
        entry = self.get_model_entry(model_id)
        x = entry["preprocess"](img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = entry["model"](x)[0]
            probs = torch.softmax(logits, dim=0)
        return probs.cpu()

    def predict_batch_probs(
        self, imgs: List[Image.Image], model_id: str = "vit_b_16"
    ) -> torch.Tensor:
        entry = self.get_model_entry(model_id)
        prep = entry["preprocess"]
        tensors = torch.stack([prep(im) for im in imgs]).to(self.device)
        with torch.no_grad():
            logits = entry["model"](tensors)
            probs = torch.softmax(logits, dim=-1)
        return probs.cpu()

    def format_top_k(
        self, probs: torch.Tensor, k: int = 5, target_idx: Optional[int] = None
    ) -> Dict:
        top = torch.topk(probs, k)
        predictions = [
            {
                "class_idx": int(idx.item()),
                "label": self.categories[int(idx.item())],
                "confidence": round(float(probs[int(idx.item())].item()), 4),
            }
            for idx in top.indices
        ]
        resolved_target = (
            target_idx if target_idx is not None else predictions[0]["class_idx"]
        )
        target_conf = round(float(probs[resolved_target].item()), 4)
        return {
            "top_predictions": predictions,
            "top1_label": predictions[0]["label"],
            "top1_idx": predictions[0]["class_idx"],
            "top1_confidence": predictions[0]["confidence"],
            "target_idx": resolved_target,
            "target_label": self.categories[resolved_target],
            "target_confidence": target_conf,
        }

    def compute_gradcam(
        self,
        img: Image.Image,
        target_idx: Optional[int] = None,
        model_id: str = "vit_b_16",
    ) -> Tuple[np.ndarray, int]:
        """Compute spatial Grad-CAM attribution across ViT, Swin, ConvNeXt, or ResNet."""
        entry = self.get_model_entry(model_id)
        net = entry["model"]
        prep = entry["preprocess"]
        spec = entry["spec"]

        acts = {}
        grads = {}

        def fwd_hook(m, inp, out):
            acts["val"] = out

        def bwd_hook(m, grad_in, grad_out):
            grads["val"] = grad_out[0]

        target_layer = spec["target_layer"](net)
        h1 = target_layer.register_forward_hook(fwd_hook)
        h2 = target_layer.register_full_backward_hook(bwd_hook)

        try:
            x = prep(img).unsqueeze(0).to(self.device)
            x.requires_grad_(True)
            logits = net(x)
            if target_idx is None:
                target_idx = int(logits[0].argmax().item())
            net.zero_grad()
            logits[0, target_idx].backward()

            g_type = spec["gradcam_type"]
            if g_type == "vit_tokens":
                # (1, 197, 768) -> spatial tokens 1..196 -> (14, 14)
                a = acts["val"][0, 1:].detach().cpu()
                g = grads["val"][0, 1:].detach().cpu()
                cam = torch.relu((a * g).sum(dim=-1)).reshape(14, 14).numpy()
            elif g_type == "swin_channels_last":
                # (1, H, W, C) -> e.g. (1, 7, 7, 768)
                a = acts["val"][0].detach().cpu()  # (H, W, C)
                g = grads["val"][0].detach().cpu()  # (H, W, C)
                weights = g.mean(dim=(0, 1), keepdim=True)  # (1, 1, C)
                cam_7 = torch.relu((a * weights).sum(dim=-1)).numpy()
                cam = (
                    np.array(
                        Image.fromarray((cam_7 / (cam_7.max() + 1e-8) * 255).astype(np.uint8)).resize(
                            (14, 14), Image.Resampling.BICUBIC
                        )
                    ).astype(np.float32)
                    / 255.0
                )
            else:
                # Standard CNN spatial feature map: (1, C, H, W) -> e.g. (1, 2048, 7, 7)
                a = acts["val"][0].detach().cpu()  # (C, H, W)
                g = grads["val"][0].detach().cpu()  # (C, H, W)
                weights = g.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
                cam_hw = torch.relu((a * weights).sum(dim=0)).numpy()
                cam = (
                    np.array(
                        Image.fromarray(
                            (cam_hw / (cam_hw.max() + 1e-8) * 255).astype(np.uint8)
                        ).resize((14, 14), Image.Resampling.BICUBIC)
                    ).astype(np.float32)
                    / 255.0
                )

            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            return cam, target_idx
        finally:
            h1.remove()
            h2.remove()

    def segment_foreground(self, img: Image.Image) -> Tuple[np.ndarray, str, float]:
        """Segment subject vs background using DeepLabV3 with smooth alpha feathering."""
        w, h = img.size
        x = self.seg_preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.seg(x)["out"][0].cpu()
        pred_mask = out.argmax(0).numpy()
        fg_binary = (pred_mask > 0).astype(np.float32)

        unique, counts = np.unique(pred_mask[pred_mask > 0], return_counts=True)
        if len(unique) > 0:
            dom_cat = self.seg_categories[int(unique[np.argmax(counts)])]
        else:
            dom_cat = "subject"

        fg_ratio = float(fg_binary.mean())
        if fg_ratio < 0.03:
            fallback = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(fallback)
            draw.ellipse(
                [int(w * 0.2), int(h * 0.18), int(w * 0.8), int(h * 0.85)], fill=255
            )
            fallback = fallback.filter(ImageFilter.GaussianBlur(radius=10))
            alpha = np.array(fallback).astype(np.float32) / 255.0
            return alpha, "central_subject", float(alpha.mean())

        mask_pil = Image.fromarray((fg_binary * 255).astype(np.uint8)).resize(
            (w, h), Image.Resampling.BILINEAR
        )
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=4))
        alpha = np.array(mask_pil).astype(np.float32) / 255.0
        return alpha, dom_cat, float(alpha.mean())

    def run_patch_occlusion_sweep(
        self,
        img: Image.Image,
        target_idx: int,
        base_conf: float,
        fg_alpha: np.ndarray,
        grid_size: int = 8,
        model_id: str = "vit_b_16",
    ) -> Dict:
        """Systematically occlude an NxN grid of patches in a single GPU batch and measure confidence drop."""
        w, h = img.size
        pw, ph = w // grid_size, h // grid_size
        img_np = np.array(img).astype(np.float32)

        occluded_imgs: List[Image.Image] = []
        patch_coords: List[Tuple[int, int]] = []

        blurred_np = np.array(img.filter(ImageFilter.GaussianBlur(radius=18))).astype(
            np.float32
        )
        neutral_fill = blurred_np * 0.35 + 127.0 * 0.65

        for r in range(grid_size):
            for c in range(grid_size):
                y0, y1 = r * ph, (r + 1) * ph if r < grid_size - 1 else h
                x0, x1 = c * pw, (c + 1) * pw if c < grid_size - 1 else w
                perturbed = img_np.copy()
                perturbed[y0:y1, x0:x1] = neutral_fill[y0:y1, x0:x1]
                occluded_imgs.append(Image.fromarray(perturbed.astype(np.uint8)))
                patch_coords.append((r, c))

        batch_probs = self.predict_batch_probs(occluded_imgs, model_id=model_id)
        target_probs = batch_probs[:, target_idx].numpy()
        top1_indices = batch_probs.argmax(dim=-1).numpy()

        conf_drops = base_conf - target_probs
        drop_grid = conf_drops.reshape(grid_size, grid_size)

        pos_drops = np.maximum(drop_grid, 0.0)
        norm_drop_grid = pos_drops / (pos_drops.max() + 1e-8)

        fg_grid = (
            np.array(
                Image.fromarray((fg_alpha * 255).astype(np.uint8)).resize(
                    (grid_size, grid_size), Image.Resampling.BILINEAR
                )
            ).astype(np.float32)
            / 255.0
        )

        patches_detail = []
        bg_importance_sum = 0.0
        fg_importance_sum = 0.0

        for idx, (r, c) in enumerate(patch_coords):
            drop = float(conf_drops[idx])
            new_conf = float(target_probs[idx])
            new_top_idx = int(top1_indices[idx])
            fg_overlap = float(fg_grid[r, c])
            region_type = "foreground" if fg_overlap >= 0.45 else "background"

            pos_impact = max(0.0, drop)
            if region_type == "background":
                bg_importance_sum += pos_impact
            else:
                fg_importance_sum += pos_impact

            patches_detail.append(
                {
                    "row": r,
                    "col": c,
                    "region": region_type,
                    "fg_overlap": round(fg_overlap, 3),
                    "target_confidence": round(new_conf, 4),
                    "confidence_drop": round(drop, 4),
                    "new_top1_label": self.categories[new_top_idx],
                }
            )

        total_pos_impact = bg_importance_sum + fg_importance_sum + 1e-8
        bg_reliance_share = bg_importance_sum / total_pos_impact

        bg_patches_sorted = sorted(
            [p for p in patches_detail if p["region"] == "background"],
            key=lambda x: x["confidence_drop"],
            reverse=True,
        )
        fg_patches_sorted = sorted(
            [p for p in patches_detail if p["region"] == "foreground"],
            key=lambda x: x["confidence_drop"],
            reverse=True,
        )

        return {
            "grid_size": grid_size,
            "drop_grid": np.round(drop_grid, 4).tolist(),
            "norm_drop_grid": np.round(norm_drop_grid, 4).tolist(),
            "fg_grid": np.round(fg_grid, 3).tolist(),
            "patches": patches_detail,
            "bg_reliance_share": round(float(bg_reliance_share), 4),
            "top_bg_patches": bg_patches_sorted[:5],
            "top_fg_patches": fg_patches_sorted[:5],
            "heatmap_image": pil_to_base64(
                create_heatmap_overlay(img, norm_drop_grid, alpha=0.58)
            ),
        }

    def run_counterfactual_suite(
        self,
        img: Image.Image,
        target_idx: int,
        base_conf: float,
        fg_alpha: np.ndarray,
        model_id: str = "vit_b_16",
    ) -> List[Dict]:
        """Design and execute automated counterfactual experiments isolating subject vs background."""
        w, h = img.size
        img_np = np.array(img).astype(np.float32)
        alpha_3d = fg_alpha[..., None]

        experiments_spec = []

        # Experiment 1: Background Removed (Uniform Neutral Context)
        no_bg_np = img_np * alpha_3d + 127.0 * (1.0 - alpha_3d)
        experiments_spec.append(
            {
                "id": "remove_bg",
                "bg_id": "remove_bg",
                "title": "Isolate Subject (Remove Background)",
                "hypothesis": "If the model relies on genuine subject anatomy, removing the background should preserve the prediction.",
                "category": "ablation",
                "image": Image.fromarray(no_bg_np.astype(np.uint8)),
            }
        )

        # Experiment 2: Subject Occluded (Background Context Only)
        only_bg_np = img_np * (1.0 - alpha_3d) + 180.0 * alpha_3d
        experiments_spec.append(
            {
                "id": "only_bg",
                "bg_id": "only_bg",
                "title": "Background Only (Subject Removed)",
                "hypothesis": "Tests if background textures alone trigger wildlife/context priors without the subject.",
                "category": "ablation",
                "image": Image.fromarray(only_bg_np.astype(np.uint8)),
            }
        )

        # Experiment 3: Blurred Background (Remove High-Frequency Context)
        blurred_bg_np = np.array(
            img.filter(ImageFilter.GaussianBlur(radius=22))
        ).astype(np.float32)
        blur_comp_np = img_np * alpha_3d + blurred_bg_np * (1.0 - alpha_3d)
        experiments_spec.append(
            {
                "id": "blur_bg",
                "bg_id": "blur_bg",
                "title": "Defocus Background Texture",
                "hypothesis": "Tests whether high-frequency background textures (e.g., snow crystals, ripples) drive tokens.",
                "category": "ablation",
                "image": Image.fromarray(blur_comp_np.astype(np.uint8)),
            }
        )

        # Experiments 4..N: Counterfactual Environment Swaps
        for bg_id, bg_info in self.backgrounds.items():
            bg_resized = bg_info["image"].resize((w, h), Image.Resampling.LANCZOS)
            bg_np = np.array(bg_resized).astype(np.float32)
            comp_np = img_np * alpha_3d + bg_np * (1.0 - alpha_3d)
            experiments_spec.append(
                {
                    "id": f"swap_{bg_id}",
                    "bg_id": bg_id,
                    "title": f"Counterfactual: {bg_info['label']}",
                    "hypothesis": f"Transplant subject into {bg_info['description'].lower()} to test domain shift invariance.",
                    "category": "counterfactual_swap",
                    "image": Image.fromarray(comp_np.astype(np.uint8)),
                }
            )

        exp_imgs = [e["image"] for e in experiments_spec]
        batch_probs = self.predict_batch_probs(exp_imgs, model_id=model_id)

        results = []
        for i, spec in enumerate(experiments_spec):
            probs = batch_probs[i]
            formatted = self.format_top_k(probs, k=4, target_idx=target_idx)
            delta = round(formatted["target_confidence"] - base_conf, 4)
            flipped = formatted["top1_idx"] != target_idx

            results.append(
                {
                    "id": spec["id"],
                    "bg_id": spec["bg_id"],
                    "title": spec["title"],
                    "hypothesis": spec["hypothesis"],
                    "category": spec["category"],
                    "image": pil_to_base64(spec["image"], quality=88),
                    "target_confidence": formatted["target_confidence"],
                    "confidence_delta": delta,
                    "prediction_flipped": flipped,
                    "top1_label": formatted["top1_label"],
                    "top1_confidence": formatted["top1_confidence"],
                    "top_predictions": formatted["top_predictions"],
                }
            )

        return results

    def _compute_diagnosis(
        self,
        target_label: str,
        base_conf: float,
        seg_category: str,
        bg_gradcam_energy: float,
        occlusion_data: Dict,
        counterfactuals: List[Dict],
    ) -> Dict:
        remove_bg_exp = next(
            (e for e in counterfactuals if e["id"] == "remove_bg"), counterfactuals[0]
        )
        only_bg_exp = next(
            (e for e in counterfactuals if e["id"] == "only_bg"), counterfactuals[1]
        )
        swap_exps = [
            e for e in counterfactuals if e["category"] == "counterfactual_swap"
        ]
        worst_swap = (
            min(swap_exps, key=lambda x: x["target_confidence"])
            if swap_exps
            else remove_bg_exp
        )

        bg_removal_drop = max(0.0, -remove_bg_exp["confidence_delta"])
        worst_swap_drop = max(0.0, -worst_swap["confidence_delta"])
        any_flip = any(
            e["prediction_flipped"]
            for e in counterfactuals
            if e["id"] != "only_bg"
        )

        max_bg_patch_drop = (
            max(0.0, occlusion_data["top_bg_patches"][0]["confidence_drop"])
            if occlusion_data.get("top_bg_patches")
            else 0.0
        )
        effective_bg_occlusion = occlusion_data["bg_reliance_share"] * min(
            1.0, max_bg_patch_drop / 0.08
        )

        raw_score = (
            bg_removal_drop * 115.0
            + worst_swap_drop * 85.0
            + effective_bg_occlusion * 22.0
            + bg_gradcam_energy * 18.0
            + (22.0 if any_flip else 0.0)
        )
        shortcut_score = int(round(min(98.0, max(5.0, raw_score))))

        if shortcut_score >= 58 or any_flip or bg_removal_drop >= 0.25:
            verdict_level = "HIGH SHORTCUT RELIANCE"
            verdict_color = "rose"
            summary = (
                f"Exposes strong spurious correlation! While the model predicts '{target_label}' with "
                f"{base_conf*100:.1f}% confidence, removing the background drops confidence by "
                f"{bg_removal_drop*100:.1f}% (down to {remove_bg_exp['target_confidence']*100:.1f}%). "
                f"In counterfactual testing, placing the subject in '{worst_swap['title'].replace('Counterfactual: ', '')}' "
                + (
                    f"flips the top prediction to '{worst_swap['top1_label']}' ({worst_swap['top1_confidence']*100:.1f}%)."
                    if worst_swap["prediction_flipped"]
                    else f"reduces '{target_label}' confidence to {worst_swap['target_confidence']*100:.1f}%."
                )
            )
        elif shortcut_score >= 28 or bg_removal_drop >= 0.10 or worst_swap_drop >= 0.10:
            verdict_level = "MODERATE CONTEXT SENSITIVITY"
            verdict_color = "amber"
            summary = (
                f"The model attends to the {seg_category} subject for '{target_label}' ({base_conf*100:.1f}%), "
                f"with noticeable environmental coupling: removing the background changes confidence by "
                f"{remove_bg_exp['confidence_delta']*100:+.1f}%, and counterfactual habitat swaps cause up to a "
                f"{worst_swap_drop*100:.1f}% drop (in '{worst_swap['title'].replace('Counterfactual: ', '')}')."
            )
        else:
            verdict_level = "ROBUST FOREGROUND REASONING"
            verdict_color = "emerald"
            summary = (
                f"The model's prediction of '{target_label}' ({base_conf*100:.1f}%) is strongly anchored on foreground subject anatomy. "
                f"Removing the background changes confidence by only {remove_bg_exp['confidence_delta']*100:+.1f}%, and the prediction "
                f"remains stable across counterfactual habitats (max shift {-worst_swap_drop*100:.1f}%)."
            )

        return {
            "shortcut_score": shortcut_score,
            "verdict_level": verdict_level,
            "verdict_color": verdict_color,
            "summary": summary,
            "bg_removal_drop": round(bg_removal_drop, 4),
            "worst_swap_drop": round(worst_swap_drop, 4),
            "worst_swap_env": worst_swap["title"],
            "only_bg_top1": only_bg_exp["top1_label"],
            "only_bg_conf": only_bg_exp["top1_confidence"],
            "only_bg_top2": (
                only_bg_exp["top_predictions"][1]["label"]
                if len(only_bg_exp["top_predictions"]) > 1
                else ""
            ),
            "only_bg_top2_conf": (
                only_bg_exp["top_predictions"][1]["confidence"]
                if len(only_bg_exp["top_predictions"]) > 1
                else 0.0
            ),
            "prediction_flipped": any_flip,
        }

    def investigate(
        self,
        img: Image.Image,
        grid_size: int = 8,
        model_id: str = "vit_b_16",
        target_idx: Optional[int] = None,
    ) -> Dict:
        """Run full automated model investigation pipeline for a given model and optional target hypothesis."""
        t0 = time.time()
        img = img.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
        entry = self.get_model_entry(model_id)

        # Step 1: Baseline Prediction
        base_probs = self.predict_probs(img, model_id=model_id)
        baseline = self.format_top_k(base_probs, k=5, target_idx=target_idx)
        resolved_target_idx = baseline["target_idx"]
        target_label = baseline["target_label"]
        base_conf = baseline["target_confidence"]

        # Step 2: Spatial Grad-CAM Attribution
        gradcam_grid, _ = self.compute_gradcam(
            img, target_idx=resolved_target_idx, model_id=model_id
        )
        gradcam_overlay = create_heatmap_overlay(img, gradcam_grid, alpha=0.58)

        # Step 3: Semantic Foreground/Background Segmentation
        fg_alpha, seg_category, fg_area_ratio = self.segment_foreground(img)
        seg_overlay = create_segmentation_overlay(img, fg_alpha)

        fg_14 = (
            np.array(
                Image.fromarray((fg_alpha * 255).astype(np.uint8)).resize(
                    (14, 14), Image.Resampling.BILINEAR
                )
            ).astype(np.float32)
            / 255.0
        )
        bg_gradcam_energy = float((gradcam_grid * (1.0 - fg_14)).sum()) / float(
            gradcam_grid.sum() + 1e-8
        )

        # Step 4: Systematic Patch Occlusion Sweep
        occlusion_data = self.run_patch_occlusion_sweep(
            img,
            target_idx=resolved_target_idx,
            base_conf=base_conf,
            fg_alpha=fg_alpha,
            grid_size=grid_size,
            model_id=model_id,
        )

        # Step 5: Automated Counterfactual Experiment Suite
        counterfactuals = self.run_counterfactual_suite(
            img,
            target_idx=resolved_target_idx,
            base_conf=base_conf,
            fg_alpha=fg_alpha,
            model_id=model_id,
        )

        # Step 6: Synthesize Spurious Correlation & Shortcut Verdict
        diagnosis = self._compute_diagnosis(
            target_label=target_label,
            base_conf=base_conf,
            seg_category=seg_category,
            bg_gradcam_energy=bg_gradcam_energy,
            occlusion_data=occlusion_data,
            counterfactuals=counterfactuals,
        )

        elapsed_ms = round((time.time() - t0) * 1000, 1)

        return {
            "elapsed_ms": elapsed_ms,
            "device": self.device.upper(),
            "model_id": model_id,
            "model_name": entry["spec"]["name"],
            "model_family": entry["spec"]["family"],
            "original_image": pil_to_base64(img),
            "baseline": baseline,
            "gradcam": {
                "grid": np.round(gradcam_grid, 4).tolist(),
                "bg_attention_share": round(bg_gradcam_energy, 4),
                "overlay_image": pil_to_base64(gradcam_overlay),
            },
            "segmentation": {
                "detected_category": seg_category,
                "foreground_area_ratio": round(fg_area_ratio, 4),
                "background_area_ratio": round(1.0 - fg_area_ratio, 4),
                "overlay_image": pil_to_base64(seg_overlay),
            },
            "occlusion": occlusion_data,
            "counterfactuals": counterfactuals,
            "diagnosis": diagnosis,
        }

    def run_custom_perturbation(
        self,
        img: Image.Image,
        occluded_cells: List[Tuple[int, int]],
        grid_size: int = 8,
        fill_mode: str = "grey",
        bg_swap_id: Optional[str] = None,
        target_idx: Optional[int] = None,
        model_id: str = "vit_b_16",
    ) -> Dict:
        """Interactive sandbox: apply custom user patch masks and/or background swaps in real time."""
        t0 = time.time()
        img = img.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
        w, h = img.size

        base_probs = self.predict_probs(img, model_id=model_id)
        if target_idx is None:
            target_idx = int(base_probs.argmax().item())
        base_target_conf = float(base_probs[target_idx].item())

        work_np = np.array(img).astype(np.float32)

        if bg_swap_id and bg_swap_id != "original":
            fg_alpha, _, _ = self.segment_foreground(img)
            alpha_3d = fg_alpha[..., None]
            if bg_swap_id == "remove_bg":
                work_np = work_np * alpha_3d + 127.0 * (1.0 - alpha_3d)
            elif bg_swap_id == "only_bg":
                work_np = work_np * (1.0 - alpha_3d) + 180.0 * alpha_3d
            elif bg_swap_id == "blur_bg":
                blurred_bg = np.array(
                    img.filter(ImageFilter.GaussianBlur(radius=22))
                ).astype(np.float32)
                work_np = work_np * alpha_3d + blurred_bg * (1.0 - alpha_3d)
            elif bg_swap_id in self.backgrounds:
                bg_np = np.array(
                    self.backgrounds[bg_swap_id]["image"].resize((w, h))
                ).astype(np.float32)
                work_np = work_np * alpha_3d + bg_np * (1.0 - alpha_3d)

        if occluded_cells:
            pw, ph = w // grid_size, h // grid_size
            if fill_mode == "blur":
                fill_source = np.array(
                    Image.fromarray(work_np.astype(np.uint8)).filter(
                        ImageFilter.GaussianBlur(radius=20)
                    )
                ).astype(np.float32)
            elif fill_mode == "snow":
                fill_source = np.full_like(work_np, 235.0)
            elif fill_mode == "dark":
                fill_source = np.full_like(work_np, 30.0)
            else:
                fill_source = np.full_like(work_np, 127.0)

            for r, c in occluded_cells:
                if 0 <= r < grid_size and 0 <= c < grid_size:
                    y0, y1 = r * ph, (r + 1) * ph if r < grid_size - 1 else h
                    x0, x1 = c * pw, (c + 1) * pw if c < grid_size - 1 else w
                    work_np[y0:y1, x0:x1] = fill_source[y0:y1, x0:x1]

        perturbed_img = Image.fromarray(np.clip(work_np, 0, 255).astype(np.uint8))
        new_probs = self.predict_probs(perturbed_img, model_id=model_id)
        formatted = self.format_top_k(new_probs, k=5, target_idx=target_idx)

        return {
            "elapsed_ms": round((time.time() - t0) * 1000, 1),
            "perturbed_image": pil_to_base64(perturbed_img, quality=90),
            "target_idx": target_idx,
            "target_label": self.categories[target_idx],
            "base_target_confidence": round(base_target_conf, 4),
            "new_target_confidence": formatted["target_confidence"],
            "confidence_delta": round(
                formatted["target_confidence"] - base_target_conf, 4
            ),
            "prediction_flipped": formatted["top1_idx"] != target_idx,
            "top_predictions": formatted["top_predictions"],
        }

    # =========================================================================
    # NEW ANALYSIS 1: Cross-Model Shortcut Benchmark
    # =========================================================================
    def run_cross_model_analysis(
        self,
        img: Image.Image,
        model_ids: Optional[List[str]] = None,
        target_idx: Optional[int] = None,
    ) -> Dict:
        """Compare multiple vision architectures on the same image across counterfactual habitats."""
        t0 = time.time()
        img = img.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
        if not model_ids:
            model_ids = list(self.models.keys())

        fg_alpha, seg_category, fg_area_ratio = self.segment_foreground(img)
        fg_14 = (
            np.array(
                Image.fromarray((fg_alpha * 255).astype(np.uint8)).resize(
                    (14, 14), Image.Resampling.BILINEAR
                )
            ).astype(np.float32)
            / 255.0
        )

        comparisons = []
        for m_id in model_ids:
            entry = self.get_model_entry(m_id)
            spec = entry["spec"]

            base_probs = self.predict_probs(img, model_id=m_id)
            baseline = self.format_top_k(base_probs, k=4, target_idx=target_idx)
            m_target_idx = baseline["target_idx"]
            m_base_conf = baseline["target_confidence"]

            gradcam_grid, _ = self.compute_gradcam(
                img, target_idx=m_target_idx, model_id=m_id
            )
            gradcam_overlay = create_heatmap_overlay(img, gradcam_grid, alpha=0.58)
            bg_gradcam_energy = float((gradcam_grid * (1.0 - fg_14)).sum()) / float(
                gradcam_grid.sum() + 1e-8
            )

            cf_results = self.run_counterfactual_suite(
                img,
                target_idx=m_target_idx,
                base_conf=m_base_conf,
                fg_alpha=fg_alpha,
                model_id=m_id,
            )

            remove_bg = next(
                (e for e in cf_results if e["id"] == "remove_bg"), cf_results[0]
            )
            only_bg = next(
                (e for e in cf_results if e["id"] == "only_bg"), cf_results[1]
            )
            swaps = [e for e in cf_results if e["category"] == "counterfactual_swap"]
            worst_swap = (
                min(swaps, key=lambda x: x["target_confidence"]) if swaps else remove_bg
            )

            bg_removal_drop = max(0.0, -remove_bg["confidence_delta"])
            worst_swap_drop = max(0.0, -worst_swap["confidence_delta"])
            any_flip = any(
                e["prediction_flipped"] for e in cf_results if e["id"] != "only_bg"
            )

            raw_score = (
                bg_removal_drop * 115.0
                + worst_swap_drop * 85.0
                + bg_gradcam_energy * 28.0
                + (22.0 if any_flip else 0.0)
            )
            shortcut_score = int(round(min(98.0, max(5.0, raw_score))))

            if shortcut_score >= 58 or any_flip or bg_removal_drop >= 0.25:
                verdict = "HIGH SHORTCUT"
                color = "rose"
            elif shortcut_score >= 28 or bg_removal_drop >= 0.10 or worst_swap_drop >= 0.10:
                verdict = "MODERATE CONTEXT"
                color = "amber"
            else:
                verdict = "ROBUST SUBJECT"
                color = "emerald"

            comparisons.append(
                {
                    "model_id": m_id,
                    "model_name": spec["name"],
                    "short_name": spec["short_name"],
                    "family": spec["family"],
                    "params": spec["params"],
                    "receptive_field": spec["receptive_field"],
                    "top1_label": baseline["top1_label"],
                    "top1_confidence": baseline["top1_confidence"],
                    "target_label": baseline["target_label"],
                    "target_confidence": m_base_conf,
                    "remove_bg_conf": remove_bg["target_confidence"],
                    "remove_bg_delta": remove_bg["confidence_delta"],
                    "worst_swap_env": worst_swap["title"].replace(
                        "Counterfactual: ", ""
                    ),
                    "worst_swap_conf": worst_swap["target_confidence"],
                    "worst_swap_delta": worst_swap["confidence_delta"],
                    "worst_swap_top1": worst_swap["top1_label"],
                    "worst_swap_flipped": worst_swap["prediction_flipped"],
                    "only_bg_top1": only_bg["top1_label"],
                    "only_bg_conf": only_bg["top1_confidence"],
                    "bg_gradcam_share": round(bg_gradcam_energy, 4),
                    "gradcam_image": pil_to_base64(gradcam_overlay, quality=85),
                    "shortcut_score": shortcut_score,
                    "verdict": verdict,
                    "verdict_color": color,
                    "any_flip": any_flip,
                }
            )

        most_robust = min(comparisons, key=lambda x: x["shortcut_score"])
        most_vulnerable = max(comparisons, key=lambda x: x["shortcut_score"])

        return {
            "elapsed_ms": round((time.time() - t0) * 1000, 1),
            "detected_subject": seg_category,
            "foreground_area_ratio": round(fg_area_ratio, 4),
            "models": comparisons,
            "most_robust_model": most_robust["short_name"],
            "most_vulnerable_model": most_vulnerable["short_name"],
            "insight_summary": (
                f"Across {len(comparisons)} architectures, {most_robust['short_name']} is the most robust "
                f"(Shortcut Risk {most_robust['shortcut_score']}/100, Remove-BG shift {most_robust['remove_bg_delta']*100:+.1f}%), "
                f"whereas {most_vulnerable['short_name']} exhibits the highest context sensitivity "
                f"(Shortcut Risk {most_vulnerable['shortcut_score']}/100, worst habitat shift {most_vulnerable['worst_swap_delta']*100:+.1f}% in {most_vulnerable['worst_swap_env']})."
            ),
        }

    # =========================================================================
    # NEW ANALYSIS 2: Necessity & Sufficiency Curves (Deletion / Insertion / BG)
    # =========================================================================
    def run_insertion_deletion_analysis(
        self,
        img: Image.Image,
        grid_size: int = 8,
        model_id: str = "vit_b_16",
        target_idx: Optional[int] = None,
    ) -> Dict:
        """Compute progressive Deletion (Necessity), Insertion (Sufficiency), and Background Ablation curves."""
        t0 = time.time()
        img = img.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
        w, h = img.size
        pw, ph = w // grid_size, h // grid_size

        base_probs = self.predict_probs(img, model_id=model_id)
        baseline = self.format_top_k(base_probs, k=4, target_idx=target_idx)
        resolved_target = baseline["target_idx"]
        target_label = baseline["target_label"]
        base_conf = baseline["target_confidence"]

        fg_alpha, _, _ = self.segment_foreground(img)
        occ = self.run_patch_occlusion_sweep(
            img,
            target_idx=resolved_target,
            base_conf=base_conf,
            fg_alpha=fg_alpha,
            grid_size=grid_size,
            model_id=model_id,
        )

        # Rank all patches by importance (confidence_drop descending)
        all_ranked = sorted(
            occ["patches"], key=lambda p: p["confidence_drop"], reverse=True
        )
        bg_ranked = sorted(
            [p for p in occ["patches"] if p["region"] == "background"],
            key=lambda p: p["confidence_drop"],
            reverse=True,
        )

        img_np = np.array(img).astype(np.float32)
        blurred_np = np.array(img.filter(ImageFilter.GaussianBlur(radius=22))).astype(
            np.float32
        )
        substrate_np = blurred_np * 0.4 + 127.0 * 0.6

        fractions = [0.0, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.0]
        total_patches = len(all_ranked)
        total_bg = max(1, len(bg_ranked))

        del_imgs: List[Image.Image] = []
        ins_imgs: List[Image.Image] = []
        bg_del_imgs: List[Image.Image] = []

        for frac in fractions:
            k_all = int(round(frac * total_patches))
            k_bg = int(round(frac * len(bg_ranked)))

            # 1. Deletion image: start with original, replace top k_all patches with substrate
            del_arr = img_np.copy()
            for p in all_ranked[:k_all]:
                r, c = p["row"], p["col"]
                y0, y1 = r * ph, (r + 1) * ph if r < grid_size - 1 else h
                x0, x1 = c * pw, (c + 1) * pw if c < grid_size - 1 else w
                del_arr[y0:y1, x0:x1] = substrate_np[y0:y1, x0:x1]
            del_imgs.append(Image.fromarray(del_arr.astype(np.uint8)))

            # 2. Insertion image: start with substrate, reveal top k_all patches from original
            ins_arr = substrate_np.copy()
            for p in all_ranked[:k_all]:
                r, c = p["row"], p["col"]
                y0, y1 = r * ph, (r + 1) * ph if r < grid_size - 1 else h
                x0, x1 = c * pw, (c + 1) * pw if c < grid_size - 1 else w
                ins_arr[y0:y1, x0:x1] = img_np[y0:y1, x0:x1]
            ins_imgs.append(Image.fromarray(ins_arr.astype(np.uint8)))

            # 3. Background progressive ablation: remove top k_bg background patches only
            bg_arr = img_np.copy()
            for p in bg_ranked[:k_bg]:
                r, c = p["row"], p["col"]
                y0, y1 = r * ph, (r + 1) * ph if r < grid_size - 1 else h
                x0, x1 = c * pw, (c + 1) * pw if c < grid_size - 1 else w
                bg_arr[y0:y1, x0:x1] = substrate_np[y0:y1, x0:x1]
            bg_del_imgs.append(Image.fromarray(bg_arr.astype(np.uint8)))

        # Single batched GPU pass for all 27 images
        combined_batch = del_imgs + ins_imgs + bg_del_imgs
        all_probs = self.predict_batch_probs(combined_batch, model_id=model_id)
        n = len(fractions)

        del_probs = all_probs[:n, resolved_target].numpy()
        ins_probs = all_probs[n : 2 * n, resolved_target].numpy()
        bg_probs = all_probs[2 * n :, resolved_target].numpy()

        del_auc = float(np.trapezoid(del_probs, fractions))
        ins_auc = float(np.trapezoid(ins_probs, fractions))
        bg_auc = float(np.trapezoid(bg_probs, fractions))

        keyframes = []
        for idx in [2, 4, 6]:  # 25%, 50%, 75%
            keyframes.append(
                {
                    "percentage": int(fractions[idx] * 100),
                    "deletion_image": pil_to_base64(
                        del_imgs[idx].resize((220, 220)), quality=82
                    ),
                    "deletion_conf": round(float(del_probs[idx]), 4),
                    "insertion_image": pil_to_base64(
                        ins_imgs[idx].resize((220, 220)), quality=82
                    ),
                    "insertion_conf": round(float(ins_probs[idx]), 4),
                    "bg_ablation_image": pil_to_base64(
                        bg_del_imgs[idx].resize((220, 220)), quality=82
                    ),
                    "bg_ablation_conf": round(float(bg_probs[idx]), 4),
                }
            )

        return {
            "elapsed_ms": round((time.time() - t0) * 1000, 1),
            "model_id": model_id,
            "target_label": target_label,
            "base_confidence": base_conf,
            "fractions": [round(f * 100, 1) for f in fractions],
            "deletion_curve": np.round(del_probs, 4).tolist(),
            "insertion_curve": np.round(ins_probs, 4).tolist(),
            "bg_ablation_curve": np.round(bg_probs, 4).tolist(),
            "deletion_auc": round(del_auc, 4),
            "insertion_auc": round(ins_auc, 4),
            "bg_ablation_auc": round(bg_auc, 4),
            "keyframes": keyframes,
            "summary": (
                f"Deletion AUC is {del_auc:.3f} (lower means top patches are strictly necessary) and "
                f"Insertion AUC is {ins_auc:.3f} (higher means top patches alone suffice to recover '{target_label}'). "
                f"Ablating 100% of background patches while keeping the subject intact shifts confidence from "
                f"{base_conf*100:.1f}% to {bg_probs[-1]*100:.1f}%."
            ),
        }

    # =========================================================================
    # NEW ANALYSIS 3: Feature Bias & Invariance Probes (Texture vs Shape, etc.)
    # =========================================================================
    def run_feature_bias_probes(
        self,
        img: Image.Image,
        model_id: str = "vit_b_16",
        target_idx: Optional[int] = None,
    ) -> Dict:
        """Synthesize 8 diagnostic feature probes to test Texture vs Shape bias, Color, Patch Order, and Scale."""
        t0 = time.time()
        img = img.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
        w, h = img.size
        img_np = np.array(img).astype(np.float32)

        base_probs = self.predict_probs(img, model_id=model_id)
        baseline = self.format_top_k(base_probs, k=4, target_idx=target_idx)
        resolved_target = baseline["target_idx"]
        target_label = baseline["target_label"]
        base_conf = baseline["target_confidence"]

        fg_alpha, _, _ = self.segment_foreground(img)

        probes_spec = []

        # 1. High-Pass Shape Sketch (Edges / Contours Only)
        gray = img.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges_inv = ImageOps.invert(edges)
        edges_contrast = ImageEnhance.Contrast(edges_inv).enhance(2.8)
        probes_spec.append(
            {
                "id": "shape_sketch",
                "title": "Shape-Only Edge Sketch",
                "category": "Shape vs. Texture",
                "description": "Strips color and surface texture, leaving only high-pass structural contours.",
                "image": edges_contrast.convert("RGB"),
            }
        )

        # 2. Monochrome Grayscale (0% Color Saturation)
        mono = ImageEnhance.Color(img).enhance(0.0)
        probes_spec.append(
            {
                "id": "grayscale",
                "title": "Monochrome (No Color)",
                "category": "Chromatic Cue",
                "description": "Removes all color channels to test if hue/albedo tint drives the prediction.",
                "image": mono,
            }
        )

        # 3. Warm Desert Color Cast Shift
        warm_np = img_np.copy()
        warm_np[..., 0] = np.clip(warm_np[..., 0] * 1.18 + 18, 0, 255)  # boost red
        warm_np[..., 1] = np.clip(warm_np[..., 1] * 1.04 + 6, 0, 255)   # slight green
        warm_np[..., 2] = np.clip(warm_np[..., 2] * 0.72 - 15, 0, 255)  # reduce blue
        probes_spec.append(
            {
                "id": "warm_tint",
                "title": "Warm Color Cast Shift",
                "category": "Chromatic Cue",
                "description": "Shifts color temperature from cool/neutral toward warm amber/desert tones.",
                "image": Image.fromarray(warm_np.astype(np.uint8)),
            }
        )

        # 4. Low-Pass Frequency Filter (Remove Fine Texture)
        low_pass = img.filter(ImageFilter.GaussianBlur(radius=5.5))
        probes_spec.append(
            {
                "id": "low_pass",
                "title": "Low-Pass Blur (Coarse Shape)",
                "category": "Frequency Bias",
                "description": "Removes high-frequency fur and snow-grain texture while preserving coarse silhouette.",
                "image": low_pass,
            }
        )

        # 5. 2x2 Global Patch Permutation
        rng = np.random.default_rng(42)
        p2_np = img_np.copy()
        half = w // 2
        quads = [
            img_np[0:half, 0:half],
            img_np[0:half, half:w],
            img_np[half:h, 0:half],
            img_np[half:h, half:w],
        ]
        perm2 = [2, 0, 3, 1]
        p2_np[0:half, 0:half] = quads[perm2[0]]
        p2_np[0:half, half:w] = quads[perm2[1]]
        p2_np[half:h, 0:half] = quads[perm2[2]]
        p2_np[half:h, half:w] = quads[perm2[3]]
        probes_spec.append(
            {
                "id": "shuffle_2x2",
                "title": "2×2 Global Patch Shuffle",
                "category": "Spatial Topology",
                "description": "Permutes 4 quadrants to disrupt global object geometry while keeping local parts intact.",
                "image": Image.fromarray(p2_np.astype(np.uint8)),
            }
        )

        # 6. 4x4 Fine Texture Bag-of-Patches Shuffle
        p4_np = np.zeros_like(img_np)
        step = w // 4
        tiles = [
            img_np[r * step : (r + 1) * step, c * step : (c + 1) * step]
            for r in range(4)
            for c in range(4)
        ]
        perm4 = rng.permutation(16)
        for idx, src_idx in enumerate(perm4):
            r, c = divmod(idx, 4)
            p4_np[r * step : (r + 1) * step, c * step : (c + 1) * step] = tiles[
                src_idx
            ]
        probes_spec.append(
            {
                "id": "shuffle_4x4",
                "title": "4×4 Texture Patch Scramble",
                "category": "Shape vs. Texture",
                "description": "Scrambles 16 patches to destroy global shape while preserving local texture statistics.",
                "image": Image.fromarray(p4_np.astype(np.uint8)),
            }
        )

        # 7. High-Frequency Gaussian Sensor Noise
        noise = rng.normal(0, 26.0, size=img_np.shape)
        noisy_np = np.clip(img_np + noise, 0, 255)
        probes_spec.append(
            {
                "id": "gaussian_noise",
                "title": "High-Frequency Noise (σ=26)",
                "category": "Frequency Bias",
                "description": "Injects zero-mean Gaussian pixel noise to test robustness against high-frequency corruptions.",
                "image": Image.fromarray(noisy_np.astype(np.uint8)),
            }
        )

        # 8. Tight Subject Bounding-Box Crop
        ys, xs = np.where(fg_alpha > 0.35)
        if len(ys) > 20 and len(xs) > 20:
            y0, y1 = max(0, int(ys.min()) - 20), min(h, int(ys.max()) + 20)
            x0, x1 = max(0, int(xs.min()) - 20), min(w, int(xs.max()) + 20)
            cropped = img.crop((x0, y0, x1, y1)).resize(
                (w, h), Image.Resampling.LANCZOS
            )
        else:
            cropped = img.crop((int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85))).resize(
                (w, h), Image.Resampling.LANCZOS
            )
        probes_spec.append(
            {
                "id": "tight_crop",
                "title": "Tight Subject Crop (Zoom)",
                "category": "Scale & Framing",
                "description": "Crops tightly around the segmented foreground subject to minimize peripheral background.",
                "image": cropped,
            }
        )

        probe_imgs = [p["image"] for p in probes_spec]
        batch_probs = self.predict_batch_probs(probe_imgs, model_id=model_id)

        results = []
        by_id = {}
        for i, spec in enumerate(probes_spec):
            formatted = self.format_top_k(
                batch_probs[i], k=3, target_idx=resolved_target
            )
            delta = round(formatted["target_confidence"] - base_conf, 4)
            item = {
                "id": spec["id"],
                "title": spec["title"],
                "category": spec["category"],
                "description": spec["description"],
                "image": pil_to_base64(
                    spec["image"].resize((256, 256), Image.Resampling.LANCZOS),
                    quality=85,
                ),
                "target_confidence": formatted["target_confidence"],
                "confidence_delta": delta,
                "prediction_flipped": formatted["top1_idx"] != resolved_target,
                "top1_label": formatted["top1_label"],
                "top1_confidence": formatted["top1_confidence"],
                "top_predictions": formatted["top_predictions"],
            }
            results.append(item)
            by_id[spec["id"]] = item

        # Compute Texture vs Shape Dominance metrics
        tex_conf = by_id["shuffle_4x4"]["target_confidence"]
        shape_conf = by_id["shape_sketch"]["target_confidence"]
        lowpass_conf = by_id["low_pass"]["target_confidence"]
        color_retention = by_id["grayscale"]["target_confidence"] / (base_conf + 1e-8)

        shape_signal = (shape_conf + lowpass_conf) / 2.0
        tex_dominance = round(
            float((tex_conf + (base_conf - lowpass_conf)) / (tex_conf + shape_signal + (base_conf - lowpass_conf) + 1e-8)),
            4,
        )

        return {
            "elapsed_ms": round((time.time() - t0) * 1000, 1),
            "model_id": model_id,
            "target_label": target_label,
            "base_confidence": base_conf,
            "probes": results,
            "metrics": {
                "texture_dominance_ratio": round(tex_dominance, 4),
                "shape_only_confidence": shape_conf,
                "patch_scramble_confidence": tex_conf,
                "color_invariance_ratio": round(float(min(1.2, color_retention)), 4),
                "tight_crop_confidence": by_id["tight_crop"]["target_confidence"],
            },
            "summary": (
                f"Under a 4×4 patch scramble (destroying global shape), the model retains "
                f"{tex_conf*100:.1f}% confidence for '{target_label}', compared to {shape_conf*100:.1f}% on the "
                f"high-pass edge sketch and {lowpass_conf*100:.1f}% on low-pass coarse shape. "
                f"Stripping color (Grayscale) yields {by_id['grayscale']['target_confidence']*100:.1f}%, while a warm desert color cast "
                f"shifts confidence to {by_id['warm_tint']['target_confidence']*100:.1f}% ({by_id['warm_tint']['top1_label']})."
            ),
        }
