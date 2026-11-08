// ShortcutLab v2.0 — Interactive AI Model & Spurious Correlation Investigator

const state = {
  samples: [],
  backgrounds: [],
  models: [],
  categories: [],

  activeSampleId: 'wolf_snow',
  customImageBase64: null,
  activeModelId: 'vit_b_16',
  targetClassIdx: null, // null = auto top-1
  gridSize: 8,

  activeViewMode: 'occlusion', // 'occlusion' | 'gradcam' | 'segmentation' | 'sandbox'
  activeCounterfactualId: null,
  investigation: null,

  occludedSet: new Set(), // stores "r,c" strings
  isDragging: false,
  dragAction: 'add',
  perturbTimer: null,

  // Extended analyses results cache
  selectedCrossModels: new Set(['vit_b_16', 'swin_t', 'convnext_tiny', 'resnet50']),
  crossModelData: null,
  curvesData: null,
  probesData: null,
  logbook: [],
};

// DOM References
const els = {
  modelSelect: document.getElementById('model-select'),
  metaDevice: document.getElementById('meta-device'),
  metaLatency: document.getElementById('meta-latency'),
  btnPinLogbook: document.getElementById('btn-pin-logbook'),
  logbookCountBadge: document.getElementById('logbook-count-badge'),
  tabLogbookCount: document.getElementById('tab-logbook-count'),

  sampleCards: document.getElementById('sample-cards'),
  activeSampleDescPill: document.getElementById('active-sample-desc-pill'),
  gridSizeSelect: document.getElementById('grid-size-select'),
  fileUpload: document.getElementById('file-upload'),
  btnOpenAddImage: document.getElementById('btn-open-add-image'),
  btnOpenAddBg: document.getElementById('btn-open-add-bg'),

  targetClassSelect: document.getElementById('target-class-select'),
  btnResetTarget: document.getElementById('btn-reset-target'),
  loadingIndicator: document.getElementById('loading-indicator'),
  loadingText: document.getElementById('loading-text'),

  canvasImage: document.getElementById('canvas-image'),
  canvasActivePill: document.getElementById('canvas-active-pill'),
  patchGridOverlay: document.getElementById('patch-grid-overlay'),
  hoveredPatchReadout: document.getElementById('hovered-patch-readout'),
  modeTabs: document.querySelectorAll('.mode-tab'),

  sandboxMaskCount: document.getElementById('sandbox-mask-count'),
  btnMaskTopBg: document.getElementById('btn-mask-top-bg'),
  btnMaskAllBg: document.getElementById('btn-mask-all-bg'),
  btnMaskSubject: document.getElementById('btn-mask-subject'),
  btnResetMask: document.getElementById('btn-reset-mask'),
  sandboxFillMode: document.getElementById('sandbox-fill-mode'),
  sandboxBgSwap: document.getElementById('sandbox-bg-swap'),
  sandboxDeltaBadge: document.getElementById('sandbox-delta-badge'),
  sandboxTargetLabel: document.getElementById('sandbox-target-label'),
  sandboxTargetConf: document.getElementById('sandbox-target-conf'),
  sandboxTargetBar: document.getElementById('sandbox-target-bar'),
  sandboxTopList: document.getElementById('sandbox-top-list'),
  btnSaveCanvasSample: document.getElementById('btn-save-canvas-sample'),

  verdictCard: document.getElementById('verdict-card'),
  verdictBadge: document.getElementById('verdict-badge'),
  verdictHeadline: document.getElementById('verdict-headline'),
  verdictScore: document.getElementById('verdict-score'),
  verdictSummary: document.getElementById('verdict-summary'),
  metricBgDrop: document.getElementById('metric-bg-drop'),
  metricSwapDrop: document.getElementById('metric-swap-drop'),
  metricSwapName: document.getElementById('metric-swap-name'),
  metricOnlyBgLabel: document.getElementById('metric-only-bg-label'),
  metricOnlyBgConf: document.getElementById('metric-only-bg-conf'),

  segInfoBadge: document.getElementById('seg-info-badge'),
  baselineBars: document.getElementById('baseline-bars'),
  occFgPct: document.getElementById('occ-fg-pct'),
  occBgPct: document.getElementById('occ-bg-pct'),
  occFgBar: document.getElementById('occ-fg-bar'),
  occBgBar: document.getElementById('occ-bg-bar'),
  camFgPct: document.getElementById('cam-fg-pct'),
  camBgPct: document.getElementById('cam-bg-pct'),
  camFgBar: document.getElementById('cam-fg-bar'),
  camBgBar: document.getElementById('cam-bg-bar'),

  counterfactualCountBadge: document.getElementById('counterfactual-count-badge'),
  counterfactualGrid: document.getElementById('counterfactual-grid'),

  // Extended Analysis Tabs & Panels
  analysisTabs: document.querySelectorAll('.analysis-tab'),
  panelCrossModel: document.getElementById('panel-cross-model'),
  panelCurves: document.getElementById('panel-curves'),
  panelProbes: document.getElementById('panel-probes'),
  panelLogbook: document.getElementById('panel-logbook'),

  modelCheckboxes: document.getElementById('model-checkboxes'),
  btnRunCrossModel: document.getElementById('btn-run-cross-model'),
  crossModelResults: document.getElementById('cross-model-results'),

  btnRunCurves: document.getElementById('btn-run-curves'),
  curvesResults: document.getElementById('curves-results'),

  btnRunProbes: document.getElementById('btn-run-probes'),
  probesResults: document.getElementById('probes-results'),

  btnPinCurrentInTab: document.getElementById('btn-pin-current-in-tab'),
  btnExportMarkdown: document.getElementById('btn-export-markdown'),
  btnExportJson: document.getElementById('btn-export-json'),
  logbookResults: document.getElementById('logbook-results'),

  // Modals
  modalAddImage: document.getElementById('modal-add-image'),
  btnCloseAddImage: document.getElementById('btn-close-add-image'),
  btnCancelAddImage: document.getElementById('btn-cancel-add-image'),
  btnSubmitAddImage: document.getElementById('btn-submit-add-image'),
  newSampleTitle: document.getElementById('new-sample-title'),
  newSampleSubtitle: document.getElementById('new-sample-subtitle'),
  newSampleColor: document.getElementById('new-sample-color'),
  newSampleFile: document.getElementById('new-sample-file'),
  newSampleUrl: document.getElementById('new-sample-url'),
  addSampleError: document.getElementById('add-sample-error'),

  modalAddBg: document.getElementById('modal-add-bg'),
  btnCloseAddBg: document.getElementById('btn-close-add-bg'),
  btnCancelAddBg: document.getElementById('btn-cancel-add-bg'),
  btnSubmitAddBg: document.getElementById('btn-submit-add-bg'),
  newBgLabel: document.getElementById('new-bg-label'),
  newBgDesc: document.getElementById('new-bg-desc'),
  newBgProcedural: document.getElementById('new-bg-procedural'),
  newBgFile: document.getElementById('new-bg-file'),
  newBgUrl: document.getElementById('new-bg-url'),
  addBgError: document.getElementById('add-bg-error'),
};

function setLoading(loading, message = 'Running PyTorch experiments...') {
  els.loadingText.textContent = message;
  if (loading) {
    els.loadingIndicator.classList.remove('hidden');
    els.loadingIndicator.classList.add('flex');
  } else {
    els.loadingIndicator.classList.add('hidden');
    els.loadingIndicator.classList.remove('flex');
  }
}

function getBadgeStyle(color) {
  const styles = {
    rose: 'bg-rose-500/20 text-rose-300 border-rose-500/40',
    amber: 'bg-amber-500/20 text-amber-300 border-amber-500/40',
    emerald: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40',
    sky: 'bg-sky-500/20 text-sky-300 border-sky-500/40',
  };
  return styles[color] || styles.sky;
}

function populateModelControls() {
  if (!state.models.length) return;
  els.modelSelect.innerHTML = '';
  state.models.forEach((m) => {
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = `${m.short_name} — ${m.family} (${m.params})`;
    if (m.id === state.activeModelId) opt.selected = true;
    els.modelSelect.appendChild(opt);
  });

  els.modelCheckboxes.innerHTML = '';
  state.models.forEach((m) => {
    const checked = state.selectedCrossModels.has(m.id);
    const lbl = document.createElement('label');
    lbl.className = `inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg border cursor-pointer transition ${
      checked
        ? 'bg-sky-500/15 border-sky-500/50 text-sky-200'
        : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
    }`;
    lbl.innerHTML = `
      <input type="checkbox" value="${m.id}" ${checked ? 'checked' : ''} class="accent-sky-500 rounded" />
      <span>${m.short_name}</span>
    `;
    const cb = lbl.querySelector('input');
    cb.addEventListener('change', () => {
      if (cb.checked) state.selectedCrossModels.add(m.id);
      else if (state.selectedCrossModels.size > 1) state.selectedCrossModels.delete(m.id);
      else cb.checked = true;
      populateModelControls();
    });
    els.modelCheckboxes.appendChild(lbl);
  });
}

function populateBackgroundDropdown() {
  const currentVal = els.sandboxBgSwap.value || 'original';
  els.sandboxBgSwap.innerHTML = `
    <option value="original">Original Habitat</option>
    <option value="remove_bg">Isolate Subject (Neutral Grey)</option>
    <option value="only_bg">Background Only (Occlude Subject)</option>
    <option value="blur_bg">Defocus Background (Gaussian Blur)</option>
  `;
  state.backgrounds.forEach((bg) => {
    const opt = document.createElement('option');
    opt.value = bg.id;
    opt.textContent = `Swap: ${bg.label}`;
    els.sandboxBgSwap.appendChild(opt);
  });
  if ([...els.sandboxBgSwap.options].some((o) => o.value === currentVal)) {
    els.sandboxBgSwap.value = currentVal;
  }
}

function renderSampleCards() {
  els.sampleCards.innerHTML = '';
  const activeSample = state.samples.find((s) => s.id === state.activeSampleId);
  if (activeSample && !state.customImageBase64) {
    els.activeSampleDescPill.textContent = activeSample.description || '';
  } else if (state.customImageBase64) {
    els.activeSampleDescPill.textContent = 'Investigating ad-hoc uploaded image';
  }

  state.samples.forEach((s) => {
    const isSelected = !state.customImageBase64 && state.activeSampleId === s.id;
    const card = document.createElement('div');
    card.className = `relative group flex items-center gap-3 p-2.5 rounded-xl border text-left transition cursor-pointer ${
      isSelected
        ? 'bg-slate-800/95 border-sky-500 shadow-lg shadow-sky-500/10 ring-1 ring-sky-500/50'
        : 'bg-slate-900/70 border-slate-800 hover:border-slate-700 hover:bg-slate-900'
    }`;
    card.innerHTML = `
      <img src="${s.thumbnail}" alt="${s.title}" class="w-14 h-14 rounded-lg object-cover shrink-0 border border-slate-700/80" />
      <div class="min-w-0 flex-1">
        <div class="flex items-center justify-between gap-1">
          <span class="text-xs font-bold text-white truncate">${s.title}</span>
        </div>
        <div class="text-[11px] text-slate-400 truncate mt-0.5">${s.subtitle}</div>
        <span class="inline-block mt-1 text-[10px] font-mono px-1.5 py-0.2 rounded border ${getBadgeStyle(s.badge_color)}">
          ${s.badge}
        </span>
      </div>
      ${
        s.custom
          ? `<button data-delete-id="${s.id}" class="delete-sample-btn opacity-0 group-hover:opacity-100 absolute top-1.5 right-1.5 w-5 h-5 rounded-full bg-rose-500/20 hover:bg-rose-500 text-rose-300 hover:text-white text-xs flex items-center justify-center transition" title="Delete custom sample">&times;</button>`
          : ''
      }
    `;
    card.addEventListener('click', (e) => {
      if (e.target.closest('.delete-sample-btn')) return;
      state.activeSampleId = s.id;
      state.customImageBase64 = null;
      state.targetClassIdx = null;
      renderSampleCards();
      runInvestigation();
    });

    const delBtn = card.querySelector('.delete-sample-btn');
    if (delBtn) {
      delBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        await deleteCustomSample(s.id);
      });
    }

    els.sampleCards.appendChild(card);
  });
}

async function deleteCustomSample(sampleId) {
  try {
    const res = await fetch(`/api/samples/${sampleId}`, { method: 'DELETE' });
    if (!res.ok) return;
    state.samples = state.samples.filter((s) => s.id !== sampleId);
    if (state.activeSampleId === sampleId) {
      state.activeSampleId = state.samples[0]?.id || 'wolf_snow';
      runInvestigation();
    }
    renderSampleCards();
  } catch (err) {
    console.error('Failed deleting sample:', err);
  }
}

function updateTargetHypothesisSelector(data) {
  els.targetClassSelect.innerHTML = '';
  const autoOpt = document.createElement('option');
  autoOpt.value = 'auto';
  autoOpt.textContent = `Auto: Top-1 ("${data.baseline.top1_label}" - ${(data.baseline.top1_confidence * 100).toFixed(1)}%)`;
  if (state.targetClassIdx === null) autoOpt.selected = true;
  els.targetClassSelect.appendChild(autoOpt);

  // Add Top-5 predictions as quick hypothesis targets
  const topGroup = document.createElement('optgroup');
  topGroup.label = 'Top-5 Competing Hypotheses';
  const seen = new Set();
  data.baseline.top_predictions.forEach((p, idx) => {
    seen.add(p.class_idx);
    const opt = document.createElement('option');
    opt.value = String(p.class_idx);
    opt.textContent = `#${idx + 1}: ${p.label} (${(p.confidence * 100).toFixed(1)}%)`;
    if (state.targetClassIdx === p.class_idx) opt.selected = true;
    topGroup.appendChild(opt);
  });
  els.targetClassSelect.appendChild(topGroup);

  // Also include counterfactual flip classes if not in top-5
  const cfGroup = document.createElement('optgroup');
  cfGroup.label = 'Counterfactual & Habitat Prior Classes';
  data.counterfactuals.forEach((cf) => {
    cf.top_predictions.forEach((p) => {
      if (!seen.has(p.class_idx)) {
        seen.add(p.class_idx);
        const opt = document.createElement('option');
        opt.value = String(p.class_idx);
        opt.textContent = `${p.label} (seen in ${cf.title.replace('Counterfactual: ', '')})`;
        if (state.targetClassIdx === p.class_idx) opt.selected = true;
        cfGroup.appendChild(opt);
      }
    });
  });
  if (cfGroup.children.length > 0) {
    els.targetClassSelect.appendChild(cfGroup);
  }

  if (state.targetClassIdx !== null) {
    els.btnResetTarget.classList.remove('hidden');
  } else {
    els.btnResetTarget.classList.add('hidden');
  }
}

async function runInvestigation() {
  setLoading(true, `Running ${state.activeModelId} investigation...`);
  try {
    const payload = {
      sample_id: state.customImageBase64 ? null : state.activeSampleId,
      image_base64: state.customImageBase64,
      grid_size: state.gridSize,
      model_id: state.activeModelId,
      target_idx: state.targetClassIdx,
    };
    const res = await fetch('/api/investigate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`Investigation failed: ${res.statusText}`);
    const data = await res.json();
    state.investigation = data;
    state.occludedSet.clear();
    state.activeCounterfactualId = null;
    els.sandboxBgSwap.value = 'original';
    renderInvestigation(data);
  } catch (err) {
    console.error(err);
  } finally {
    setLoading(false);
  }
}

function renderInvestigation(data) {
  els.metaDevice.textContent = data.device;
  els.metaLatency.textContent = `(${data.elapsed_ms} ms)`;

  updateTargetHypothesisSelector(data);
  updateCanvasImage();
  buildPatchGridOverlay(data);
  syncGridVisualSelection();

  renderSandboxReadout({
    target_label: data.baseline.target_label,
    base_target_confidence: data.baseline.target_confidence,
    new_target_confidence: data.baseline.target_confidence,
    confidence_delta: 0,
    prediction_flipped: data.baseline.target_idx !== data.baseline.top1_idx,
    top_predictions: data.baseline.top_predictions,
  });

  const diag = data.diagnosis;
  const colorMap = {
    rose: {
      border: 'border-l-rose-500',
      badge: 'bg-rose-500/20 text-rose-300 border-rose-500/40',
      score: 'text-rose-400',
    },
    amber: {
      border: 'border-l-amber-500',
      badge: 'bg-amber-500/20 text-amber-300 border-amber-500/40',
      score: 'text-amber-400',
    },
    emerald: {
      border: 'border-l-emerald-500',
      badge: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40',
      score: 'text-emerald-400',
    },
  };
  const cStyle = colorMap[diag.verdict_color] || colorMap.rose;
  els.verdictCard.className = `glass-panel rounded-xl p-5 border-l-4 ${cStyle.border} space-y-4`;
  els.verdictBadge.className = `text-[11px] font-mono font-bold uppercase px-2.5 py-0.5 rounded-full border ${cStyle.badge}`;
  els.verdictBadge.textContent = diag.verdict_level;

  if (data.baseline.target_idx === data.baseline.top1_idx) {
    els.verdictHeadline.textContent = `${data.model_name.split(' ')[0]} Predicts "${data.baseline.top1_label}" (${(data.baseline.top1_confidence * 100).toFixed(1)}%)`;
  } else {
    els.verdictHeadline.textContent = `Hypothesis "${data.baseline.target_label}" (${(data.baseline.target_confidence * 100).toFixed(1)}%) vs Top-1 "${data.baseline.top1_label}"`;
  }

  els.verdictScore.className = `text-2xl font-mono font-extrabold ${cStyle.score}`;
  els.verdictScore.textContent = diag.shortcut_score;
  els.verdictSummary.textContent = diag.summary;

  const removeBgExp = data.counterfactuals.find((e) => e.id === 'remove_bg');
  const removeBgDelta = removeBgExp ? removeBgExp.confidence_delta : -diag.bg_removal_drop;
  els.metricBgDrop.textContent = `${removeBgDelta >= 0 ? '+' : ''}${(removeBgDelta * 100).toFixed(1)}%`;
  els.metricBgDrop.className = `text-base font-bold mt-0.5 ${
    removeBgDelta < -0.15 ? 'text-rose-400' : removeBgDelta < -0.05 ? 'text-amber-400' : 'text-emerald-400'
  }`;

  els.metricSwapDrop.textContent = `-${(diag.worst_swap_drop * 100).toFixed(1)}%`;
  els.metricSwapDrop.className = `text-base font-bold mt-0.5 ${
    diag.worst_swap_drop > 0.2 ? 'text-rose-400' : diag.worst_swap_drop > 0.08 ? 'text-amber-400' : 'text-emerald-400'
  }`;
  els.metricSwapName.textContent = diag.worst_swap_env.replace('Counterfactual: ', '');
  els.metricOnlyBgLabel.textContent = diag.only_bg_top1;
  const top2Str = diag.only_bg_top2
    ? ` (${(diag.only_bg_conf * 100).toFixed(0)}%) / ${diag.only_bg_top2} (${(diag.only_bg_top2_conf * 100).toFixed(0)}%)`
    : `${(diag.only_bg_conf * 100).toFixed(1)}% conf`;
  els.metricOnlyBgConf.textContent = top2Str;

  // Baseline Top-5 Bars (Clickable to switch target class hypothesis)
  els.segInfoBadge.textContent = `Subject (${data.segmentation.detected_category}): ${(data.segmentation.foreground_area_ratio * 100).toFixed(1)}% area`;
  els.baselineBars.innerHTML = '';
  data.baseline.top_predictions.forEach((pred, idx) => {
    const pct = (pred.confidence * 100).toFixed(1);
    const isTarget = pred.class_idx === data.baseline.target_idx;
    const row = document.createElement('div');
    row.className = `space-y-1 p-1.5 rounded-lg cursor-pointer transition ${
      isTarget ? 'bg-sky-500/10 border border-sky-500/30' : 'hover:bg-slate-900/80'
    }`;
    row.title = `Click to investigate "${pred.label}" as the target hypothesis`;
    row.innerHTML = `
      <div class="flex justify-between text-xs font-mono">
        <span class="${isTarget ? 'text-sky-300 font-bold' : idx === 0 ? 'text-white font-semibold' : 'text-slate-400'} flex items-center gap-1.5">
          <span>${idx + 1}. ${pred.label}</span>
          ${isTarget ? '<span class="text-[9px] uppercase px-1.5 py-0.2 rounded bg-sky-500/20 text-sky-300">Target</span>' : ''}
        </span>
        <span class="${isTarget ? 'text-sky-400 font-bold' : 'text-slate-400'}">${pct}%</span>
      </div>
      <div class="w-full h-1.5 rounded-full bg-slate-900 overflow-hidden">
        <div class="h-full ${isTarget ? 'bg-sky-400' : idx === 0 ? 'bg-indigo-500' : 'bg-slate-600'}" style="width: ${Math.max(2, pred.confidence * 100)}%"></div>
      </div>
    `;
    row.addEventListener('click', () => {
      state.targetClassIdx = pred.class_idx === data.baseline.top1_idx ? null : pred.class_idx;
      runInvestigation();
    });
    els.baselineBars.appendChild(row);
  });

  // Attribution Split Bars
  const bgOccPct = Math.round(data.occlusion.bg_reliance_share * 100);
  const fgOccPct = 100 - bgOccPct;
  els.occFgPct.textContent = `${fgOccPct}% Subject`;
  els.occBgPct.textContent = `${bgOccPct}% Background`;
  els.occFgBar.style.width = `${fgOccPct}%`;
  els.occBgBar.style.width = `${bgOccPct}%`;

  const bgCamPct = Math.round(data.gradcam.bg_attention_share * 100);
  const fgCamPct = 100 - bgCamPct;
  els.camFgPct.textContent = `${fgCamPct}% Subject`;
  els.camBgPct.textContent = `${bgCamPct}% Background`;
  els.camFgBar.style.width = `${fgCamPct}%`;
  els.camBgBar.style.width = `${bgCamPct}%`;

  els.counterfactualCountBadge.textContent = `${data.counterfactuals.length} Controlled Interventions`;
  renderCounterfactualCards(data.counterfactuals, data.baseline.target_label);
}

function updateCanvasImage() {
  if (!state.investigation) return;
  const data = state.investigation;
  const modeLabels = {
    occlusion: 'Occlusion Heatmap',
    gradcam: `${data.model_name.split(' ')[0]} Grad-CAM`,
    segmentation: 'DeepLabV3 Subject Mask',
    sandbox: 'Interactive Sandbox',
  };
  els.canvasActivePill.textContent = modeLabels[state.activeViewMode] || 'Canvas';

  if (state.activeViewMode === 'occlusion') {
    els.canvasImage.src = data.occlusion.heatmap_image;
  } else if (state.activeViewMode === 'gradcam') {
    els.canvasImage.src = data.gradcam.overlay_image;
  } else if (state.activeViewMode === 'segmentation') {
    els.canvasImage.src = data.segmentation.overlay_image;
  } else {
    els.canvasImage.src = data.original_image;
  }
}

function buildPatchGridOverlay(data) {
  const g = data.occlusion.grid_size;
  els.patchGridOverlay.style.gridTemplateColumns = `repeat(${g}, minmax(0, 1fr))`;
  els.patchGridOverlay.style.gridTemplateRows = `repeat(${g}, minmax(0, 1fr))`;
  els.patchGridOverlay.innerHTML = '';

  data.occlusion.patches.forEach((p) => {
    const key = `${p.row},${p.col}`;
    const cell = document.createElement('div');
    cell.dataset.key = key;
    cell.dataset.row = p.row;
    cell.dataset.col = p.col;
    cell.className = `patch-cell border border-slate-400/15 cursor-pointer relative flex items-center justify-center ${
      state.occludedSet.has(key) ? 'patch-occluded' : ''
    }`;

    cell.addEventListener('mouseenter', () => {
      const sign = p.confidence_drop >= 0 ? '-' : '+';
      const dropPct = Math.abs(p.confidence_drop * 100).toFixed(1);
      els.hoveredPatchReadout.textContent = `Patch (${p.row},${p.col}) [${p.region.toUpperCase()}]: ${sign}${dropPct}% conf when occluded`;

      if (state.isDragging) {
        if (state.dragAction === 'add') {
          state.occludedSet.add(key);
          cell.classList.add('patch-occluded');
        } else {
          state.occludedSet.delete(key);
          cell.classList.remove('patch-occluded');
        }
        scheduleLivePerturbation();
      }
    });

    cell.addEventListener('mousedown', (e) => {
      e.preventDefault();
      state.isDragging = true;
      if (state.occludedSet.has(key)) {
        state.dragAction = 'remove';
        state.occludedSet.delete(key);
        cell.classList.remove('patch-occluded');
      } else {
        state.dragAction = 'add';
        state.occludedSet.add(key);
        cell.classList.add('patch-occluded');
      }
      scheduleLivePerturbation();
    });

    els.patchGridOverlay.appendChild(cell);
  });
}

window.addEventListener('mouseup', () => {
  state.isDragging = false;
});

function syncGridVisualSelection() {
  const cells = els.patchGridOverlay.querySelectorAll('.patch-cell');
  cells.forEach((cell) => {
    const key = cell.dataset.key;
    if (state.occludedSet.has(key)) {
      cell.classList.add('patch-occluded');
    } else {
      cell.classList.remove('patch-occluded');
    }
  });
  els.sandboxMaskCount.textContent = `${state.occludedSet.size} patches masked`;
}

function scheduleLivePerturbation() {
  syncGridVisualSelection();
  if (state.perturbTimer) clearTimeout(state.perturbTimer);
  state.perturbTimer = setTimeout(() => {
    executeLivePerturbation();
  }, 85);
}

async function executeLivePerturbation() {
  if (!state.investigation) return;
  const occludedCells = Array.from(state.occludedSet).map((s) => {
    const [r, c] = s.split(',').map(Number);
    return [r, c];
  });
  const bgSwap = els.sandboxBgSwap.value;

  if (occludedCells.length === 0 && bgSwap === 'original') {
    state.activeCounterfactualId = null;
    highlightActiveCounterfactualCard();
    updateCanvasImage();
    renderSandboxReadout({
      target_label: state.investigation.baseline.target_label,
      base_target_confidence: state.investigation.baseline.target_confidence,
      new_target_confidence: state.investigation.baseline.target_confidence,
      confidence_delta: 0,
      prediction_flipped: state.investigation.baseline.target_idx !== state.investigation.baseline.top1_idx,
      top_predictions: state.investigation.baseline.top_predictions,
    });
    return;
  }

  try {
    const payload = {
      sample_id: state.customImageBase64 ? null : state.activeSampleId,
      image_base64: state.customImageBase64,
      occluded_cells: occludedCells,
      grid_size: state.gridSize,
      fill_mode: els.sandboxFillMode.value,
      bg_swap_id: bgSwap,
      target_idx: state.investigation.baseline.target_idx,
      model_id: state.activeModelId,
    };
    const res = await fetch('/api/perturb', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) return;
    const data = await res.json();
    els.canvasImage.src = data.perturbed_image;
    els.canvasActivePill.textContent = `Live Perturbation (${occludedCells.length} masked, ${bgSwap})`;
    renderSandboxReadout(data);
  } catch (err) {
    console.error('Perturbation error:', err);
  }
}

function renderSandboxReadout(data) {
  els.sandboxTargetLabel.textContent = data.target_label;
  els.sandboxTargetConf.textContent = `${(data.new_target_confidence * 100).toFixed(1)}%`;
  els.sandboxTargetBar.style.width = `${Math.max(2, data.new_target_confidence * 100)}%`;

  const deltaPct = (data.confidence_delta * 100).toFixed(1);
  if (Math.abs(data.confidence_delta) < 0.001 && !data.prediction_flipped) {
    els.sandboxDeltaBadge.textContent = 'Baseline';
    els.sandboxDeltaBadge.className = 'font-mono font-bold px-2 py-0.5 rounded bg-slate-800 text-slate-300';
  } else if (data.prediction_flipped) {
    els.sandboxDeltaBadge.textContent = `FLIPPED (${deltaPct >= 0 ? '+' : ''}${deltaPct}%)`;
    els.sandboxDeltaBadge.className = 'font-mono font-bold px-2 py-0.5 rounded bg-rose-500/25 text-rose-300 border border-rose-500/40';
  } else if (data.confidence_delta < 0) {
    els.sandboxDeltaBadge.textContent = `${deltaPct}% Drop`;
    els.sandboxDeltaBadge.className = 'font-mono font-bold px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30';
  } else {
    els.sandboxDeltaBadge.textContent = `+${deltaPct}% Gain`;
    els.sandboxDeltaBadge.className = 'font-mono font-bold px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30';
  }

  els.sandboxTopList.innerHTML = '';
  data.top_predictions.slice(0, 3).forEach((p, idx) => {
    const row = document.createElement('div');
    row.className = 'flex items-center justify-between';
    const isFlippedTop = idx === 0 && data.prediction_flipped;
    row.innerHTML = `
      <span class="${isFlippedTop ? 'text-rose-400 font-bold' : idx === 0 ? 'text-white font-semibold' : 'text-slate-400'} truncate">
        ${idx + 1}. ${p.label}
      </span>
      <span class="text-slate-300 font-semibold">${(p.confidence * 100).toFixed(1)}%</span>
    `;
    els.sandboxTopList.appendChild(row);
  });
}

function highlightActiveCounterfactualCard() {
  const cards = els.counterfactualGrid.querySelectorAll('[data-cf-id]');
  cards.forEach((c) => {
    if (c.dataset.cfId === state.activeCounterfactualId) {
      c.classList.add('ring-2', 'ring-sky-400');
    } else {
      c.classList.remove('ring-2', 'ring-sky-400');
    }
  });
}

function renderCounterfactualCards(experiments, baselineLabel) {
  els.counterfactualGrid.innerHTML = '';
  experiments.forEach((exp) => {
    const deltaPct = (exp.confidence_delta * 100).toFixed(1);
    const deltaStr = `${exp.confidence_delta >= 0 ? '+' : ''}${deltaPct}%`;
    const isFlipped = exp.prediction_flipped;

    const card = document.createElement('div');
    card.dataset.cfId = exp.id;
    card.className = `flex flex-col justify-between rounded-xl bg-slate-900/80 border ${
      isFlipped ? 'border-rose-500/60 shadow-lg shadow-rose-500/5' : 'border-slate-800'
    } overflow-hidden hover:border-sky-500/60 transition cursor-pointer group`;

    card.innerHTML = `
      <div>
        <div class="relative aspect-square w-full bg-slate-950 overflow-hidden">
          <img src="${exp.image}" alt="${exp.title}" class="w-full h-full object-cover group-hover:scale-105 transition duration-300" />
          <div class="absolute top-2 right-2">
            <span class="text-[11px] font-mono font-bold px-2 py-0.5 rounded-md shadow backdrop-blur-md ${
              isFlipped
                ? 'bg-rose-500/90 text-white'
                : exp.confidence_delta < -0.10
                ? 'bg-amber-500/90 text-slate-950'
                : 'bg-slate-900/85 text-emerald-400 border border-slate-700'
            }">
              ${isFlipped ? 'FLIPPED' : deltaStr}
            </span>
          </div>
          <div class="absolute bottom-0 inset-x-0 bg-gradient-to-t from-slate-950 via-slate-950/80 to-transparent p-2.5 pt-6">
            <div class="text-xs font-bold text-white leading-tight">${exp.title}</div>
          </div>
        </div>

        <div class="p-3 space-y-2">
          <p class="text-[11px] text-slate-400 leading-snug line-clamp-3">
            ${exp.hypothesis}
          </p>
        </div>
      </div>

      <div class="px-3 pb-3 pt-2 border-t border-slate-800/80 bg-slate-950/50 space-y-1 font-mono text-[11px]">
        <div class="flex justify-between">
          <span class="text-slate-400 truncate">Top-1:</span>
          <span class="${isFlipped ? 'text-rose-400 font-bold' : 'text-emerald-400 font-semibold'} truncate max-w-[110px]">
            ${exp.top1_label}
          </span>
        </div>
        <div class="flex justify-between">
          <span class="text-slate-500 truncate">"${baselineLabel}":</span>
          <span class="text-slate-200 font-semibold">${(exp.target_confidence * 100).toFixed(1)}% (${deltaStr})</span>
        </div>
      </div>
    `;

    card.addEventListener('click', () => {
      state.activeCounterfactualId = exp.id;
      highlightActiveCounterfactualCard();

      // Sync the Habitat Intervention dropdown with this counterfactual
      if (exp.bg_id && [...els.sandboxBgSwap.options].some((o) => o.value === exp.bg_id)) {
        els.sandboxBgSwap.value = exp.bg_id;
      }

      if (state.occludedSet.size > 0) {
        scheduleLivePerturbation();
      } else {
        els.canvasImage.src = exp.image;
        els.canvasActivePill.textContent = exp.title;
        renderSandboxReadout({
          target_label: baselineLabel,
          base_target_confidence: state.investigation.baseline.target_confidence,
          new_target_confidence: exp.target_confidence,
          confidence_delta: exp.confidence_delta,
          prediction_flipped: exp.prediction_flipped,
          top_predictions: exp.top_predictions,
        });
      }
    });

    els.counterfactualGrid.appendChild(card);
  });
}

// ============================================================================
// EXTENDED ANALYSIS A: Cross-Model Shortcut Benchmark
// ============================================================================
async function runCrossModelAudit() {
  setLoading(true, 'Running Cross-Architecture Shortcut Audit...');
  try {
    const payload = {
      sample_id: state.customImageBase64 ? null : state.activeSampleId,
      image_base64: state.customImageBase64,
      model_ids: Array.from(state.selectedCrossModels),
      target_idx: state.targetClassIdx,
    };
    const res = await fetch('/api/analyze/cross-model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error('Cross-model audit failed');
    const data = await res.json();
    state.crossModelData = data;
    renderCrossModelResults(data);
  } catch (err) {
    console.error(err);
  } finally {
    setLoading(false);
  }
}

function renderCrossModelResults(data) {
  const cardsHtml = data.models
    .map((m) => {
      const badgeCls = getBadgeStyle(m.verdict_color);
      const isCurrent = m.model_id === state.activeModelId;
      return `
        <div class="rounded-xl bg-slate-900/85 border ${
          isCurrent ? 'border-sky-500 ring-1 ring-sky-500/40' : 'border-slate-800'
        } p-4 flex flex-col justify-between space-y-3">
          <div class="space-y-2.5">
            <div class="flex items-start justify-between gap-2">
              <div>
                <div class="flex items-center gap-1.5">
                  <span class="text-sm font-bold text-white font-mono">${m.short_name}</span>
                  <span class="text-[10px] font-mono px-1.5 py-0.2 rounded bg-slate-800 text-slate-400">${m.params}</span>
                </div>
                <div class="text-[11px] text-slate-400">${m.receptive_field}</div>
              </div>
              <span class="text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${badgeCls}">
                Risk: ${m.shortcut_score}/100
              </span>
            </div>

            <div class="relative aspect-square w-full rounded-lg overflow-hidden border border-slate-800 bg-slate-950">
              <img src="${m.gradcam_image}" alt="${m.short_name} Grad-CAM" class="w-full h-full object-cover" />
              <div class="absolute bottom-1.5 left-1.5 right-1.5 bg-slate-950/85 backdrop-blur-sm rounded px-2 py-1 flex justify-between text-[10px] font-mono">
                <span class="text-slate-300">BG Attention Share:</span>
                <span class="text-sky-400 font-bold">${(m.bg_gradcam_share * 100).toFixed(1)}%</span>
              </div>
            </div>

            <div class="space-y-1.5 font-mono text-xs pt-1">
              <div class="flex justify-between">
                <span class="text-slate-400">Baseline Top-1:</span>
                <span class="text-white font-semibold truncate max-w-[140px]">${m.top1_label} (${(m.top1_confidence * 100).toFixed(1)}%)</span>
              </div>
              <div class="flex justify-between">
                <span class="text-slate-400">Remove BG Shift:</span>
                <span class="${m.remove_bg_delta < -0.15 ? 'text-rose-400' : 'text-emerald-400'} font-bold">
                  ${m.remove_bg_delta >= 0 ? '+' : ''}${(m.remove_bg_delta * 100).toFixed(1)}%
                </span>
              </div>
              <div class="flex justify-between">
                <span class="text-slate-400">Worst Swap (${m.worst_swap_env.split(' ')[0]}):</span>
                <span class="${m.worst_swap_flipped ? 'text-rose-400' : 'text-amber-400'} font-bold">
                  ${m.worst_swap_delta >= 0 ? '+' : ''}${(m.worst_swap_delta * 100).toFixed(1)}% ${m.worst_swap_flipped ? `→ ${m.worst_swap_top1}` : ''}
                </span>
              </div>
              <div class="flex justify-between">
                <span class="text-slate-400">BG-Only Prior:</span>
                <span class="text-sky-300 truncate max-w-[140px]">${m.only_bg_top1} (${(m.only_bg_conf * 100).toFixed(1)}%)</span>
              </div>
            </div>
          </div>

          <button data-switch-model="${m.model_id}" class="switch-model-btn w-full py-1.5 rounded-lg font-mono text-xs font-semibold transition ${
            isCurrent
              ? 'bg-sky-500/20 text-sky-300 border border-sky-500/40'
              : 'bg-slate-800 hover:bg-slate-700 text-slate-200'
          }">
            ${isCurrent ? 'Active in Main Canvas' : `Switch Main View to ${m.short_name}`}
          </button>
        </div>
      `;
    })
    .join('');

  els.crossModelResults.innerHTML = `
    <div class="bg-slate-900/90 border border-slate-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
      <div class="space-y-1 max-w-4xl">
        <div class="text-xs font-mono uppercase text-sky-400 font-bold">Cross-Architecture Synthesis (${data.elapsed_ms} ms)</div>
        <p class="text-xs text-slate-200 leading-relaxed">${data.insight_summary}</p>
      </div>
      <div class="flex gap-3 font-mono text-xs">
        <div class="bg-emerald-500/10 border border-emerald-500/30 rounded-lg px-3 py-2">
          <div class="text-[10px] text-emerald-400 uppercase">Most Robust</div>
          <div class="font-bold text-white">${data.most_robust_model}</div>
        </div>
        <div class="bg-rose-500/10 border border-rose-500/30 rounded-lg px-3 py-2">
          <div class="text-[10px] text-rose-400 uppercase">Most Shortcut-Prone</div>
          <div class="font-bold text-white">${data.most_vulnerable_model}</div>
        </div>
      </div>
    </div>
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      ${cardsHtml}
    </div>
  `;

  els.crossModelResults.querySelectorAll('.switch-model-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const mId = btn.dataset.switchModel;
      state.activeModelId = mId;
      els.modelSelect.value = mId;
      runInvestigation();
      renderCrossModelResults(state.crossModelData);
    });
  });
}

// ============================================================================
// EXTENDED ANALYSIS B: Deletion / Insertion / Background Ablation Curves
// ============================================================================
async function runCurvesAnalysis() {
  setLoading(true, 'Computing 27-step Deletion & Insertion Curves...');
  try {
    const payload = {
      sample_id: state.customImageBase64 ? null : state.activeSampleId,
      image_base64: state.customImageBase64,
      grid_size: state.gridSize,
      model_id: state.activeModelId,
      target_idx: state.targetClassIdx,
    };
    const res = await fetch('/api/analyze/curves', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error('Curves analysis failed');
    const data = await res.json();
    state.curvesData = data;
    renderCurvesResults(data);
  } catch (err) {
    console.error(err);
  } finally {
    setLoading(false);
  }
}

function buildCurvesSvg(data) {
  const w = 680;
  const h = 280;
  const padL = 48;
  const padR = 24;
  const padT = 20;
  const padB = 36;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const toPoints = (arr) =>
    arr
      .map((v, i) => {
        const x = padL + (i / (arr.length - 1)) * plotW;
        const y = padT + (1.0 - Math.max(0, Math.min(1, v))) * plotH;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');

  const delPts = toPoints(data.deletion_curve);
  const insPts = toPoints(data.insertion_curve);
  const bgPts = toPoints(data.bg_ablation_curve);

  const gridLines = [0, 0.25, 0.5, 0.75, 1.0]
    .map((tick) => {
      const y = padT + (1.0 - tick) * plotH;
      return `
        <line x1="${padL}" y1="${y}" x2="${w - padR}" y2="${y}" stroke="#1e293b" stroke-dasharray="3,3" />
        <text x="${padL - 8}" y="${y + 4}" text-anchor="end" fill="#64748b" font-size="10" font-family="JetBrains Mono">${Math.round(tick * 100)}%</text>
      `;
    })
    .join('');

  const xLabels = data.fractions
    .map((pct, i) => {
      const x = padL + (i / (data.fractions.length - 1)) * plotW;
      return `<text x="${x}" y="${h - 10}" text-anchor="middle" fill="#64748b" font-size="10" font-family="JetBrains Mono">${Math.round(pct)}%</text>`;
    })
    .join('');

  return `
    <svg viewBox="0 0 ${w} ${h}" class="w-full h-auto bg-slate-950/90 rounded-xl border border-slate-800 p-2">
      ${gridLines}
      ${xLabels}
      <!-- Background-Only Ablation Curve (Sky Blue) -->
      <polyline fill="none" stroke="#38bdf8" stroke-width="2.5" stroke-dasharray="5,3" points="${bgPts}" />
      <!-- Insertion Sufficiency Curve (Emerald) -->
      <polyline fill="none" stroke="#10b981" stroke-width="3" points="${insPts}" />
      <!-- Deletion Necessity Curve (Rose) -->
      <polyline fill="none" stroke="#f43f5e" stroke-width="3" points="${delPts}" />
    </svg>
  `;
}

function renderCurvesResults(data) {
  const svgChart = buildCurvesSvg(data);
  const keyframesHtml = data.keyframes
    .map(
      (kf) => `
      <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-3 space-y-2 font-mono text-xs">
        <div class="text-slate-300 font-bold border-b border-slate-800 pb-1">
          At ${kf.percentage}% Patches Perturbed
        </div>
        <div class="grid grid-cols-3 gap-2 text-center">
          <div>
            <img src="${kf.deletion_image}" class="w-full aspect-square object-cover rounded border border-rose-500/40" />
            <div class="text-[10px] text-rose-400 mt-1 font-bold">Del: ${(kf.deletion_conf * 100).toFixed(1)}%</div>
          </div>
          <div>
            <img src="${kf.insertion_image}" class="w-full aspect-square object-cover rounded border border-emerald-500/40" />
            <div class="text-[10px] text-emerald-400 mt-1 font-bold">Ins: ${(kf.insertion_conf * 100).toFixed(1)}%</div>
          </div>
          <div>
            <img src="${kf.bg_ablation_image}" class="w-full aspect-square object-cover rounded border border-sky-500/40" />
            <div class="text-[10px] text-sky-400 mt-1 font-bold">BG-Abl: ${(kf.bg_ablation_conf * 100).toFixed(1)}%</div>
          </div>
        </div>
      </div>
    `
    )
    .join('');

  els.curvesResults.innerHTML = `
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-4 items-center">
      <div class="lg:col-span-7 space-y-3">
        ${svgChart}
        <div class="flex flex-wrap items-center justify-center gap-6 text-xs font-mono">
          <div class="flex items-center gap-2">
            <span class="w-3 h-1 bg-rose-500 inline-block rounded"></span>
            <span class="text-rose-300">Deletion Curve (AUC: ${data.deletion_auc.toFixed(3)})</span>
          </div>
          <div class="flex items-center gap-2">
            <span class="w-3 h-1 bg-emerald-500 inline-block rounded"></span>
            <span class="text-emerald-300">Insertion Curve (AUC: ${data.insertion_auc.toFixed(3)})</span>
          </div>
          <div class="flex items-center gap-2">
            <span class="w-3 h-1 bg-sky-400 inline-block rounded"></span>
            <span class="text-sky-300">BG-Only Ablation (AUC: ${data.bg_ablation_auc.toFixed(3)})</span>
          </div>
        </div>
      </div>

      <div class="lg:col-span-5 space-y-3">
        <div class="bg-slate-900/90 border border-slate-800 rounded-xl p-4 space-y-2">
          <div class="text-xs font-mono uppercase text-sky-400 font-bold">Necessity &amp; Sufficiency Interpretation</div>
          <p class="text-xs text-slate-300 leading-relaxed">${data.summary}</p>
        </div>
        <div class="space-y-2.5">
          ${keyframesHtml}
        </div>
      </div>
    </div>
  `;
}

// ============================================================================
// EXTENDED ANALYSIS C: Feature Bias & Invariance Probes (8x)
// ============================================================================
async function runProbesAnalysis() {
  setLoading(true, 'Synthesizing & testing 8 Feature Bias Probes...');
  try {
    const payload = {
      sample_id: state.customImageBase64 ? null : state.activeSampleId,
      image_base64: state.customImageBase64,
      model_id: state.activeModelId,
      target_idx: state.targetClassIdx,
    };
    const res = await fetch('/api/analyze/probes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error('Probes analysis failed');
    const data = await res.json();
    state.probesData = data;
    renderProbesResults(data);
  } catch (err) {
    console.error(err);
  } finally {
    setLoading(false);
  }
}

function renderProbesResults(data) {
  const m = data.metrics;
  const probeCards = data.probes
    .map((p) => {
      const deltaStr = `${p.confidence_delta >= 0 ? '+' : ''}${(p.confidence_delta * 100).toFixed(1)}%`;
      return `
        <div data-probe-img="${p.image}" data-probe-title="${p.title}" class="probe-card rounded-xl bg-slate-900/85 border ${
          p.prediction_flipped ? 'border-rose-500/50' : 'border-slate-800'
        } overflow-hidden hover:border-sky-500/60 transition cursor-pointer flex flex-col justify-between">
          <div>
            <div class="relative aspect-square bg-slate-950">
              <img src="${p.image}" alt="${p.title}" class="w-full h-full object-cover" />
              <span class="absolute top-2 left-2 text-[10px] font-mono px-2 py-0.5 rounded bg-slate-950/85 text-sky-300 border border-slate-700">
                ${p.category}
              </span>
              <span class="absolute top-2 right-2 text-[10px] font-mono font-bold px-2 py-0.5 rounded ${
                p.prediction_flipped ? 'bg-rose-500 text-white' : 'bg-slate-950/85 text-emerald-400 border border-slate-700'
              }">
                ${p.prediction_flipped ? 'FLIPPED' : deltaStr}
              </span>
            </div>
            <div class="p-3 space-y-1">
              <div class="text-xs font-bold text-white">${p.title}</div>
              <p class="text-[11px] text-slate-400 leading-snug">${p.description}</p>
            </div>
          </div>
          <div class="px-3 pb-2.5 pt-2 border-t border-slate-800/80 bg-slate-950/50 font-mono text-[11px] space-y-0.5">
            <div class="flex justify-between">
              <span class="text-slate-400">Top-1:</span>
              <span class="${p.prediction_flipped ? 'text-rose-400 font-bold' : 'text-emerald-400'} truncate max-w-[120px]">${p.top1_label} (${(p.top1_confidence * 100).toFixed(1)}%)</span>
            </div>
            <div class="flex justify-between">
              <span class="text-slate-500">"${data.target_label}":</span>
              <span class="text-slate-200 font-semibold">${(p.target_confidence * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>
      `;
    })
    .join('');

  els.probesResults.innerHTML = `
    <div class="bg-slate-900/90 border border-slate-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
      <div class="max-w-3xl space-y-1">
        <div class="text-xs font-mono uppercase text-sky-400 font-bold">Representation Bias Diagnosis (${data.elapsed_ms} ms)</div>
        <p class="text-xs text-slate-200 leading-relaxed">${data.summary}</p>
      </div>
      <div class="grid grid-cols-2 sm:grid-cols-4 gap-2.5 font-mono text-xs">
        <div class="bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-center">
          <div class="text-[10px] text-slate-400">4×4 Patch Scramble</div>
          <div class="font-bold text-amber-400 text-sm">${(m.patch_scramble_confidence * 100).toFixed(1)}%</div>
        </div>
        <div class="bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-center">
          <div class="text-[10px] text-slate-400">Shape Edge Sketch</div>
          <div class="font-bold text-sky-400 text-sm">${(m.shape_only_confidence * 100).toFixed(1)}%</div>
        </div>
        <div class="bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-center">
          <div class="text-[10px] text-slate-400">Color Invariance</div>
          <div class="font-bold text-emerald-400 text-sm">${(m.color_invariance_ratio * 100).toFixed(0)}%</div>
        </div>
        <div class="bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-center">
          <div class="text-[10px] text-slate-400">Tight Subject Zoom</div>
          <div class="font-bold text-indigo-400 text-sm">${(m.tight_crop_confidence * 100).toFixed(1)}%</div>
        </div>
      </div>
    </div>

    <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3.5">
      ${probeCards}
    </div>
  `;

  els.probesResults.querySelectorAll('.probe-card').forEach((card) => {
    card.addEventListener('click', () => {
      els.canvasImage.src = card.dataset.probeImg;
      els.canvasActivePill.textContent = `Probe: ${card.dataset.probeTitle}`;
    });
  });
}

// ============================================================================
// EXTENDED ANALYSIS D: Experiment Logbook & Report Export
// ============================================================================
function pinCurrentRunToLogbook() {
  if (!state.investigation) return;
  const inv = state.investigation;
  const diag = inv.diagnosis;
  const sampleTitle = state.customImageBase64
    ? 'Uploaded Custom Subject'
    : state.samples.find((s) => s.id === state.activeSampleId)?.title || state.activeSampleId;

  const entry = {
    id: `run_${Date.now()}`,
    timestamp: new Date().toLocaleTimeString(),
    sample_title: sampleTitle,
    model_name: inv.model_name,
    target_label: inv.baseline.target_label,
    base_confidence: inv.baseline.target_confidence,
    top1_label: inv.baseline.top1_label,
    top1_confidence: inv.baseline.top1_confidence,
    shortcut_score: diag.shortcut_score,
    verdict_level: diag.verdict_level,
    verdict_color: diag.verdict_color,
    bg_removal_drop: diag.bg_removal_drop,
    worst_swap_drop: diag.worst_swap_drop,
    worst_swap_env: diag.worst_swap_env.replace('Counterfactual: ', ''),
    only_bg_top1: diag.only_bg_top1,
    only_bg_conf: diag.only_bg_conf,
    bg_occlusion_share: inv.occlusion.bg_reliance_share,
    bg_gradcam_share: inv.gradcam.bg_attention_share,
    summary: diag.summary,
  };

  state.logbook.unshift(entry);
  els.logbookCountBadge.textContent = String(state.logbook.length);
  els.tabLogbookCount.textContent = String(state.logbook.length);
  renderLogbook();
}

function renderLogbook() {
  if (state.logbook.length === 0) {
    els.logbookResults.innerHTML = `
      <div class="text-center py-8 text-xs font-mono text-slate-400 bg-slate-900/40 rounded-xl border border-dashed border-slate-800">
        No runs pinned yet. Click <span class="text-indigo-300 font-semibold">"Pin to Logbook"</span> in the top navigation bar or above to record snapshots across models and images.
      </div>
    `;
    return;
  }

  const rows = state.logbook
    .map(
      (item, idx) => `
      <tr class="border-b border-slate-800/80 hover:bg-slate-900/60">
        <td class="py-2.5 px-3 text-slate-400">${idx + 1} (${item.timestamp})</td>
        <td class="py-2.5 px-3 font-semibold text-white">${item.sample_title}</td>
        <td class="py-2.5 px-3 text-sky-300">${item.model_name.split(' ')[0]}</td>
        <td class="py-2.5 px-3 text-slate-200">${item.target_label} (${(item.base_confidence * 100).toFixed(1)}%)</td>
        <td class="py-2.5 px-3 text-rose-400">-${(item.bg_removal_drop * 100).toFixed(1)}%</td>
        <td class="py-2.5 px-3 text-amber-400">-${(item.worst_swap_drop * 100).toFixed(1)}% (${item.worst_swap_env})</td>
        <td class="py-2.5 px-3 text-sky-400">${item.only_bg_top1} (${(item.only_bg_conf * 100).toFixed(1)}%)</td>
        <td class="py-2.5 px-3">
          <span class="px-2 py-0.5 rounded border ${getBadgeStyle(item.verdict_color)} font-bold">
            ${item.shortcut_score}/100 (${item.verdict_level})
          </span>
        </td>
      </tr>
    `
    )
    .join('');

  els.logbookResults.innerHTML = `
    <div class="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/60">
      <table class="w-full text-left font-mono text-xs">
        <thead class="bg-slate-900 text-slate-400 uppercase text-[10px]">
          <tr>
            <th class="py-2.5 px-3">#</th>
            <th class="py-2.5 px-3">Image / Case</th>
            <th class="py-2.5 px-3">Model</th>
            <th class="py-2.5 px-3">Target Class</th>
            <th class="py-2.5 px-3">Remove BG Drop</th>
            <th class="py-2.5 px-3">Worst Habitat Shift</th>
            <th class="py-2.5 px-3">BG-Only Prior</th>
            <th class="py-2.5 px-3">Shortcut Risk</th>
          </tr>
        </thead>
        <tbody>
          ${rows}
        </tbody>
      </table>
    </div>
  `;
}

function downloadBlob(filename, content, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// ============================================================================
// UI EVENT LISTENERS & MODAL HANDLERS
// ============================================================================
els.modelSelect.addEventListener('change', (e) => {
  state.activeModelId = e.target.value;
  runInvestigation();
});

els.targetClassSelect.addEventListener('change', (e) => {
  const val = e.target.value;
  state.targetClassIdx = val === 'auto' ? null : parseInt(val, 10);
  runInvestigation();
});

els.btnResetTarget.addEventListener('click', () => {
  state.targetClassIdx = null;
  runInvestigation();
});

els.modeTabs.forEach((btn) => {
  btn.addEventListener('click', () => {
    state.activeViewMode = btn.dataset.mode;
    els.modeTabs.forEach((b) => {
      if (b.dataset.mode === state.activeViewMode) {
        b.className = 'mode-tab px-2.5 py-1 rounded-md bg-sky-500 text-slate-950 font-semibold transition';
      } else {
        b.className = 'mode-tab px-2.5 py-1 rounded-md text-slate-300 hover:text-white transition';
      }
    });
    if (state.activeViewMode === 'sandbox' && (state.occludedSet.size > 0 || els.sandboxBgSwap.value !== 'original')) {
      executeLivePerturbation();
    } else {
      updateCanvasImage();
    }
  });
});

els.analysisTabs.forEach((btn) => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.analysisTab;
    els.analysisTabs.forEach((b) => {
      if (b.dataset.analysisTab === tab) {
        b.className = 'analysis-tab px-3 py-1.5 rounded-lg bg-sky-500 text-slate-950 font-bold transition';
      } else {
        b.className = 'analysis-tab px-3 py-1.5 rounded-lg text-slate-300 hover:text-white transition';
      }
    });
    els.panelCrossModel.classList.toggle('hidden', tab !== 'cross-model');
    els.panelCurves.classList.toggle('hidden', tab !== 'curves');
    els.panelProbes.classList.toggle('hidden', tab !== 'probes');
    els.panelLogbook.classList.toggle('hidden', tab !== 'logbook');
  });
});

els.btnRunCrossModel.addEventListener('click', runCrossModelAudit);
els.btnRunCurves.addEventListener('click', runCurvesAnalysis);
els.btnRunProbes.addEventListener('click', runProbesAnalysis);

els.btnPinLogbook.addEventListener('click', pinCurrentRunToLogbook);
els.btnPinCurrentInTab.addEventListener('click', pinCurrentRunToLogbook);

els.btnExportJson.addEventListener('click', () => {
  if (state.logbook.length === 0) pinCurrentRunToLogbook();
  downloadBlob('shortcutlab_experiments.json', JSON.stringify(state.logbook, null, 2), 'application/json');
});

els.btnExportMarkdown.addEventListener('click', () => {
  if (state.logbook.length === 0) pinCurrentRunToLogbook();
  const lines = [
    '# ShortcutLab — AI Model Shortcut & Spurious Correlation Report',
    '',
    '| Image Case | Model | Target Class | Remove BG Drop | Worst Habitat Shift | BG-Only Prior | Shortcut Score |',
    '|---|---|---|---|---|---|---|',
    ...state.logbook.map(
      (r) =>
        `| ${r.sample_title} | ${r.model_name} | ${r.target_label} (${(r.base_confidence * 100).toFixed(1)}%) | -${(r.bg_removal_drop * 100).toFixed(1)}% | -${(r.worst_swap_drop * 100).toFixed(1)}% (${r.worst_swap_env}) | ${r.only_bg_top1} (${(r.only_bg_conf * 100).toFixed(1)}%) | **${r.shortcut_score}/100** (${r.verdict_level}) |`
    ),
    '',
    '## Detailed Run Summaries',
    ...state.logbook.map((r, i) => `### ${i + 1}. ${r.sample_title} — ${r.model_name}\n- ${r.summary}\n`),
  ];
  downloadBlob('shortcutlab_report.md', lines.join('\n'), 'text/markdown');
});

// Quick Mask Preset Buttons
els.btnMaskTopBg.addEventListener('click', () => {
  if (!state.investigation) return;
  state.occludedSet.clear();
  const bgSorted = state.investigation.occlusion.patches
    .filter((p) => p.region === 'background')
    .sort((a, b) => b.confidence_drop - a.confidence_drop)
    .slice(0, 10);
  bgSorted.forEach((p) => state.occludedSet.add(`${p.row},${p.col}`));
  scheduleLivePerturbation();
});

els.btnMaskAllBg.addEventListener('click', () => {
  if (!state.investigation) return;
  state.occludedSet.clear();
  state.investigation.occlusion.patches
    .filter((p) => p.region === 'background')
    .forEach((p) => state.occludedSet.add(`${p.row},${p.col}`));
  scheduleLivePerturbation();
});

els.btnMaskSubject.addEventListener('click', () => {
  if (!state.investigation) return;
  state.occludedSet.clear();
  state.investigation.occlusion.patches
    .filter((p) => p.region === 'foreground')
    .forEach((p) => state.occludedSet.add(`${p.row},${p.col}`));
  scheduleLivePerturbation();
});

els.btnResetMask.addEventListener('click', () => {
  state.occludedSet.clear();
  els.sandboxBgSwap.value = 'original';
  scheduleLivePerturbation();
});

els.sandboxFillMode.addEventListener('change', scheduleLivePerturbation);
els.sandboxBgSwap.addEventListener('change', scheduleLivePerturbation);

els.gridSizeSelect.addEventListener('change', (e) => {
  state.gridSize = parseInt(e.target.value, 10);
  runInvestigation();
});

// Save Current Perturbed/Counterfactual Canvas as a New Sample in Library
els.btnSaveCanvasSample.addEventListener('click', async () => {
  if (!els.canvasImage.src) return;
  const activeSample = state.samples.find((s) => s.id === state.activeSampleId);
  const baseTitle = activeSample ? activeSample.title.split(' ')[0] : 'Subject';
  const swapLabel = els.sandboxBgSwap.options[els.sandboxBgSwap.selectedIndex]?.text || 'Perturbed';
  const title = `${baseTitle} (${swapLabel.replace('Swap: ', '')})`;

  setLoading(true, 'Saving canvas as new sample...');
  try {
    const res = await fetch('/api/samples/add', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title,
        subtitle: `Saved from ${state.activeModelId} canvas`,
        badge: 'Synthesized',
        badge_color: 'amber',
        description: `Synthesized counterfactual/perturbed sample from ${baseTitle}.`,
        image_base64: els.canvasImage.src,
      }),
    });
    if (!res.ok) throw new Error('Failed saving sample');
    const data = await res.json();
    state.samples.push(data.sample);
    state.activeSampleId = data.sample.id;
    state.customImageBase64 = null;
    renderSampleCards();
    await runInvestigation();
  } catch (err) {
    console.error(err);
  } finally {
    setLoading(false);
  }
});

// Quick Upload via /api/upload
els.fileUpload.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  setLoading(true, 'Uploading & investigating image...');
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('grid_size', String(state.gridSize));
    formData.append('model_id', state.activeModelId);
    formData.append('save_to_library', 'true');
    const res = await fetch('/api/upload', { method: 'POST', body: formData });
    if (!res.ok) throw new Error('Upload failed');
    const data = await res.json();
    if (data.saved_sample) {
      state.samples.push(data.saved_sample);
      state.activeSampleId = data.saved_sample.id;
      state.customImageBase64 = null;
      renderSampleCards();
    }
    state.investigation = data;
    state.occludedSet.clear();
    renderInvestigation(data);
  } catch (err) {
    console.error(err);
  } finally {
    setLoading(false);
  }
});

// Modal 1: Add New Subject Image
function closeAddImageModal() {
  els.modalAddImage.classList.add('hidden');
  els.modalAddImage.classList.remove('flex');
  els.addSampleError.classList.add('hidden');
}
els.btnOpenAddImage.addEventListener('click', () => {
  els.modalAddImage.classList.remove('hidden');
  els.modalAddImage.classList.add('flex');
});
els.btnCloseAddImage.addEventListener('click', closeAddImageModal);
els.btnCancelAddImage.addEventListener('click', closeAddImageModal);

els.btnSubmitAddImage.addEventListener('click', async () => {
  const title = els.newSampleTitle.value.trim() || 'Custom Experimental Subject';
  const subtitle = els.newSampleSubtitle.value.trim() || 'Custom Shortcut Hypothesis';
  const badgeColor = els.newSampleColor.value;
  const file = els.newSampleFile.files[0];
  const url = els.newSampleUrl.value.trim();

  if (!file && !url) {
    els.addSampleError.textContent = 'Please select an image file or enter an image URL.';
    els.addSampleError.classList.remove('hidden');
    return;
  }

  let imageBase64 = null;
  if (file) {
    imageBase64 = await new Promise((resolve) => {
      const r = new FileReader();
      r.onload = (ev) => resolve(ev.target.result);
      r.readAsDataURL(file);
    });
  }

  setLoading(true, 'Adding new image to gallery...');
  try {
    const res = await fetch('/api/samples/add', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title,
        subtitle,
        badge: 'Custom Case',
        badge_color: badgeColor,
        description: `${title} — ${subtitle}`,
        image_base64: imageBase64,
        image_url: url || null,
      }),
    });
    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || 'Failed to add image');
    }
    const data = await res.json();
    state.samples.push(data.sample);
    state.activeSampleId = data.sample.id;
    state.customImageBase64 = null;
    closeAddImageModal();
    renderSampleCards();
    await runInvestigation();
  } catch (err) {
    els.addSampleError.textContent = err.message;
    els.addSampleError.classList.remove('hidden');
  } finally {
    setLoading(false);
  }
});

// Modal 2: Add New Counterfactual Background Environment
function closeAddBgModal() {
  els.modalAddBg.classList.add('hidden');
  els.modalAddBg.classList.remove('flex');
  els.addBgError.classList.add('hidden');
}
els.btnOpenAddBg.addEventListener('click', () => {
  els.modalAddBg.classList.remove('hidden');
  els.modalAddBg.classList.add('flex');
});
els.btnCloseAddBg.addEventListener('click', closeAddBgModal);
els.btnCancelAddBg.addEventListener('click', closeAddBgModal);

els.newBgProcedural.addEventListener('change', (e) => {
  const map = {
    studio_white: ['Studio White Backdrop', 'High-albedo uniform bright background'],
    dark_void: ['Deep Shadow Void', 'Low-luminance dark studio backdrop'],
    checkerboard: ['High-Freq Checkerboard', 'High-contrast spatial frequency grid'],
    autumn_foliage: ['Autumn Amber Woodland', 'Warm orange and amber foliage texture'],
  };
  if (map[e.target.value]) {
    els.newBgLabel.value = map[e.target.value][0];
    els.newBgDesc.value = map[e.target.value][1];
  }
});

els.btnSubmitAddBg.addEventListener('click', async () => {
  const label = els.newBgLabel.value.trim() || 'Custom Habitat';
  const description = els.newBgDesc.value.trim() || 'Custom counterfactual background';
  const proc = els.newBgProcedural.value;
  const file = els.newBgFile.files[0];
  const url = els.newBgUrl.value.trim();

  if (!proc && !file && !url) {
    els.addBgError.textContent = 'Select a procedural generator, upload a file, or enter a URL.';
    els.addBgError.classList.remove('hidden');
    return;
  }

  let imageBase64 = null;
  if (file) {
    imageBase64 = await new Promise((resolve) => {
      const r = new FileReader();
      r.onload = (ev) => resolve(ev.target.result);
      r.readAsDataURL(file);
    });
  }

  setLoading(true, 'Adding habitat & re-running counterfactual suite...');
  try {
    const res = await fetch('/api/backgrounds/add', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        label,
        description,
        procedural_type: proc || null,
        image_base64: imageBase64,
        image_url: url || null,
      }),
    });
    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || 'Failed to add background');
    }
    const data = await res.json();
    state.backgrounds.push(data.background);
    populateBackgroundDropdown();
    closeAddBgModal();
    await runInvestigation();
  } catch (err) {
    els.addBgError.textContent = err.message;
    els.addBgError.classList.remove('hidden');
  } finally {
    setLoading(false);
  }
});

// Initialize Application
async function enableStaticPagesFallback() {
  // Intercept fetch calls when hosted on static GitHub Pages (where /api/* is unavailable)
  const origFetch = window.fetch.bind(window);
  let clientAssets = null;

  window.fetch = async (url, options = {}) => {
    try {
      const res = await origFetch(url, options);
      if (res.ok || !String(url).startsWith('/api/')) return res;
    } catch (_) {
      // Network error on /api/* -> use static bundle
    }

    const urlStr = String(url);
    const body = options.body && typeof options.body === 'string' ? JSON.parse(options.body) : {};
    const sid = body.sample_id || state.activeSampleId || 'wolf_snow';
    const mid = body.model_id || state.activeModelId || 'vit_b_16';

    if (urlStr === '/api/samples') {
      return origFetch('./data/samples.json');
    }
    if (urlStr === '/api/investigate') {
      return origFetch(`./data/inv_${sid}_${mid}.json`);
    }
    if (urlStr === '/api/analyze/cross-model') {
      return origFetch(`./data/cm_${sid}.json`);
    }
    if (urlStr === '/api/analyze/curves') {
      return origFetch(`./data/curves_${sid}.json`);
    }
    if (urlStr === '/api/analyze/probes') {
      return origFetch(`./data/probes_${sid}.json`);
    }
    if (urlStr === '/api/perturb' && state.investigation) {
      // Perform client-side HTML5 canvas patch masking + precomputed sensitivity lookup
      if (!clientAssets) {
        const caRes = await origFetch('./data/client_assets.json');
        if (caRes.ok) clientAssets = await caRes.json();
      }
      const inv = state.investigation;
      const g = body.grid_size || 8;
      const cells = body.occluded_cells || [];
      const bgSwap = body.bg_swap_id || 'original';

      // Determine base image & starting confidence from counterfactuals
      let baseImgSrc = inv.original_image;
      let startConf = inv.baseline.target_confidence;
      let topPreds = inv.baseline.top_predictions.map((p) => ({ ...p }));

      if (bgSwap !== 'original') {
        const cfMatch = inv.counterfactuals.find((c) => c.bg_id === bgSwap || c.id === bgSwap);
        if (cfMatch) {
          baseImgSrc = cfMatch.image;
          startConf = cfMatch.target_confidence;
          topPreds = cfMatch.top_predictions.map((p) => ({ ...p }));
        }
      }

      // Draw occluded patches on an offscreen HTML5 canvas
      const imgEl = new Image();
      await new Promise((resolve) => {
        imgEl.onload = resolve;
        imgEl.src = baseImgSrc;
      });
      const canvas = document.createElement('canvas');
      canvas.width = 512;
      canvas.height = 512;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(imgEl, 0, 0, 512, 512);

      const pw = 512 / g;
      const ph = 512 / g;
      const fillColors = {
        grey: 'rgb(127, 127, 127)',
        snow: 'rgb(235, 240, 245)',
        dark: 'rgb(30, 35, 45)',
        blur: 'rgba(135, 145, 158, 0.92)',
      };
      ctx.fillStyle = fillColors[body.fill_mode] || fillColors.grey;

      let totalDrop = 0;
      cells.forEach(([r, c]) => {
        ctx.fillRect(c * pw, r * ph, pw, ph);
        const patchInfo = inv.occlusion.patches.find((p) => p.row === r && p.col === c);
        if (patchInfo) {
          totalDrop += patchInfo.confidence_drop * 0.72;
        }
      });

      const newConf = Math.max(0.01, Math.min(0.99, startConf - totalDrop));
      const delta = newConf - inv.baseline.target_confidence;
      topPreds[0].confidence = newConf;
      const flipped = newConf < 0.36 && topPreds.length > 1;
      if (flipped) {
        const second = { ...topPreds[1], confidence: Math.min(0.85, 1.0 - newConf - 0.1) };
        topPreds = [second, topPreds[0], ...topPreds.slice(2)];
      }

      return new Response(
        JSON.stringify({
          elapsed_ms: 4.2,
          perturbed_image: canvas.toDataURL('image/jpeg', 0.9),
          target_idx: inv.baseline.target_idx,
          target_label: inv.baseline.target_label,
          base_target_confidence: inv.baseline.target_confidence,
          new_target_confidence: newConf,
          confidence_delta: delta,
          prediction_flipped: flipped,
          top_predictions: topPreds,
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      );
    }

    return new Response(JSON.stringify({ detail: 'Static demo mode' }), { status: 200 });
  };
}

async function init() {
  await enableStaticPagesFallback();
  try {
    const res = await fetch('/api/samples');
    const data = await res.json();
    state.samples = data.samples;
    state.backgrounds = data.backgrounds;
    state.models = data.models || [];
    state.categories = data.categories || [];
    populateModelControls();
    populateBackgroundDropdown();
    renderSampleCards();
    renderLogbook();
    await runInvestigation();
  } catch (err) {
    console.error('Initialization failed:', err);
  }
}

init();
