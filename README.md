# ShortcutLab — Interactive AI Model & Spurious Correlation Investigator

An interactive machine learning interpretability and experimentation suite that automatically designs perturbation, counterfactual, and representation-probing experiments to expose **spurious correlations**, **background shortcuts**, and **feature biases** across Vision Transformers and Convolutional Networks.

## Core Pipeline (4-Stage Automated Investigation)

When a vision model (e.g., `ViT-B/16`, `Swin-T`, `ConvNeXt-Tiny`, or `ResNet-50` pretrained on ImageNet-1K) makes a high-confidence prediction (such as `ViT-B/16` predicting `"timber wolf"` at **81.1%** on a wolf standing in snow), `ShortcutLab` automatically executes a 4-stage experimental investigation on Apple Silicon (`MPS`) / GPU:

1. **Baseline & Spatial Grad-CAM Attribution**: Computes spatial gradients on the final encoder/feature block (`14 × 14` token grid for ViT, or upsampled spatial feature maps for Swin/ConvNeXt/ResNet) for either the Top-1 predicted class or any user-selected **competing hypothesis class**.
2. **Semantic Subject Isolation**: Segments the foreground subject from its environment using `DeepLabV3-MobileNetV3` with feathered alpha matting, quantifying how much of the model's attribution falls on the background vs. the subject.
3. **Systematic Patch Occlusion Sweep**: Generates an `N × N` (`6×6`, `8×8`, or `10×10`) batch of occluded images in a single GPU forward pass, measuring the exact confidence drop when each spatial patch is removed.
4. **Automated Counterfactual Synthesis**: Synthesizes controlled interventions (`7+` environments, extensible with custom habitats):
   - **Isolate Subject (Remove Background)**: Replaces the snowy background with uniform neutral grey (`-38.6%` confidence drop on the blizzard wolf, down to `42.5%`).
   - **Background Only (Subject Removed)**: Masks out the wolf completely—revealing that the snowy background alone triggers `"white wolf"` (`67.9%`) and `"Arctic fox"` (`21.2%`).
   - **Defocus Background Texture**: Blurs high-frequency background cues (`-8.1%` shift).
   - **Counterfactual Habitat Swaps**: Transplants the segmented subject into **Arctic Snowfield**, **Temperate Forest**, **Arid Desert Dunes** (flips `ViT-B/16`'s top prediction from `"timber wolf"` to `"coyote"` at `38.5%`!), **Urban Environment**, plus any user-added custom or procedural habitats.

## Extended Model & Image Diagnostic Suite

Beyond the primary 4-stage pipeline, `ShortcutLab` includes three deep-dive analysis modules and dataset builders:

- **A. Cross-Architecture Shortcut Benchmark (`/api/analyze/cross-model`)**: Audits multiple architectures side-by-side (`ViT-B/16`, `Swin-T`, `ConvNeXt-Tiny`, `ResNet-50`, `MobileNetV3-Large`, `DenseNet-121`) on the same image to compare how global self-attention vs. local convolutional receptive fields respond to background removal and counterfactual domain shifts.
- **B. Necessity & Sufficiency Curves (`/api/analyze/curves`)**: Evaluates 27 progressive perturbation steps (`0% → 100%` of patches) in a single batched pass to plot **Deletion Curves (Necessity AUC)**, **Insertion Curves (Sufficiency AUC)**, and **Progressive Background-Only Ablation Curves**.
- **C. Feature Bias & Invariance Probes (`/api/analyze/probes`)**: Synthesizes 8 diagnostic probes (`Shape-Only Edge Sketch`, `Monochrome Grayscale`, `Warm Color Cast Shift`, `Low-Pass Blur`, `2×2 Global Patch Shuffle`, `4×4 Texture Patch Scramble`, `High-Frequency Gaussian Noise`, and `Tight Subject Zoom`) to quantify texture vs. shape bias and chromatic sensitivity.
- **D. Custom Dataset, Habitat & Experiment Logbook**:
  - Add new subject images via file upload (`/api/upload`), URL import (`/api/samples/add`), or **one-click saving of any perturbed/counterfactual canvas**.
  - Add new counterfactual background habitats (`/api/backgrounds/add`) via file upload, URL, or procedural generators (*Studio White*, *Deep Shadow Void*, *High-Freq Checkerboard*, *Autumn Amber Woodland*).
  - Pin runs across models and images into the **Experiment Logbook** and export reproducible **Markdown (`.md`)** or **JSON (`.json`)** reports.

## Quick Start

```bash
# Start the FastAPI server using the local Python 3.13 virtual environment
.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000** in your browser.

## Project Structure

- [`backend/investigator.py`](file:///Users/ethanxie/Documents/VSCode%20Projects/gdg/backend/investigator.py): Core PyTorch interpretability engine (`ModelInvestigator`) supporting 6 torchvision architectures, architecture-aware Grad-CAM hooks, DeepLabV3 segmentation, batched patch occlusion sweeps, counterfactual synthesis, deletion/insertion AUC curves, and 8x feature bias probes.
- [`backend/app.py`](file:///Users/ethanxie/Documents/VSCode%20Projects/gdg/backend/app.py): FastAPI server exposing `/api/samples`, `/api/samples/add`, `/api/backgrounds/add`, `/api/models/load`, `/api/investigate`, `/api/perturb`, `/api/upload`, `/api/analyze/cross-model`, `/api/analyze/curves`, and `/api/analyze/probes`.
- [`frontend/index.html`](file:///Users/ethanxie/Documents/VSCode%20Projects/gdg/frontend/index.html) & [`frontend/app.js`](file:///Users/ethanxie/Documents/VSCode%20Projects/gdg/frontend/app.js): Interactive dark-mode laboratory UI with live patch painting, model & hypothesis switching, custom image/habitat modals, SVG attribution curve plotting, and experiment logbook export.
