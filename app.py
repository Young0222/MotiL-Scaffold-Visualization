#!/usr/bin/env python3

from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, render_template_string, request, send_file, send_from_directory, url_for
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "web_uploads"
JOB_DIR = BASE_DIR / "web_jobs"
DEMO_DIR = BASE_DIR / "demo_assets"
SCRIPT_PATH = BASE_DIR / "plot_scaffold_tsne.py"
ROOT_DIR = SCRIPT_PATH.resolve().parents[1]

UPLOAD_DIR.mkdir(exist_ok=True)
JOB_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()

IGNORED_LOG_FRAGMENTS = [
    "WARNING: not removing hydrogen atom without neighbors",
]


PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MotiL Scaffold Visualization Local App</title>
  <style>
    :root {
      --bg: #f5f1ea;
      --paper: #fffdf8;
      --ink: #1a1a1a;
      --muted: #5f6470;
      --line: #ddd5c8;
      --accent: #ff4b6e;
      --accent-2: #2148ef;
      --good: #1a8a59;
      --bad: #a24a28;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background: linear-gradient(180deg, #faf7f1 0%, #f1ece4 100%);
    }
    .page {
      max-width: 1200px;
      margin: 0 auto;
      padding: 28px 20px 56px;
    }
    .hero, .grid {
      display: grid;
      gap: 22px;
    }
    .hero {
      grid-template-columns: 1.05fr 0.95fr;
      margin-bottom: 24px;
    }
    .grid {
      grid-template-columns: 1fr 1fr;
    }
    .card {
      background: rgba(255, 253, 248, 0.95);
      border: 1px solid rgba(169, 154, 129, 0.2);
      border-radius: 28px;
      padding: 22px;
      box-shadow: 0 18px 48px rgba(61, 46, 25, 0.07);
    }
    h1 {
      margin: 0 0 10px;
      font-size: clamp(2.2rem, 4vw, 4rem);
      line-height: 0.96;
      letter-spacing: -0.04em;
    }
    h2 {
      margin: 0 0 14px;
      font-size: 1.3rem;
    }
    h3 {
      margin: 16px 0 8px;
      font-size: 1rem;
    }
    p, li, small {
      color: var(--muted);
      line-height: 1.7;
    }
    ul { margin: 10px 0 0 18px; }
    .pill-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }
    .flow-row {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
      margin-top: 18px;
    }
    .pill {
      border: 1px solid var(--line);
      background: white;
      border-radius: 999px;
      padding: 10px 14px;
      font-size: 0.92rem;
    }
    .flow-arrow {
      color: var(--muted);
      font-size: 1rem;
      font-weight: 700;
      line-height: 1;
    }
    .title-mark {
      display: inline-flex;
      align-items: center;
      gap: 10px;
    }
    .title-icon {
      font-size: 0.88em;
      transform: translateY(-1px);
    }
    .preview {
      width: 100%;
      border-radius: 22px;
      border: 1px solid rgba(110, 110, 110, 0.16);
      background: white;
    }
    .field { margin-bottom: 16px; }
    .field label {
      display: block;
      font-weight: 700;
      margin-bottom: 8px;
    }
    .field input[type="text"],
    .field input[type="number"],
    .field select,
    .field input[type="file"] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      font: inherit;
      background: white;
    }
    .inline {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 12px;
    }
    .format-box, .status-box, .result-box, .log-box {
      border-radius: 18px;
      border: 1px solid var(--line);
      background: #fffdfa;
      padding: 14px 16px;
    }
    .hint {
      font-size: 0.92rem;
      color: var(--muted);
      margin-top: 6px;
    }
    .status-box.good {
      color: var(--good);
      border-color: rgba(26,138,89,0.25);
      background: rgba(26,138,89,0.08);
    }
    .status-box.bad {
      color: var(--bad);
      border-color: rgba(162,74,40,0.25);
      background: rgba(162,74,40,0.08);
    }
    .actions {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 16px;
    }
    button, .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      text-decoration: none;
      border: 0;
      border-radius: 14px;
      padding: 12px 16px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }
    button {
      background: var(--ink);
      color: white;
    }
    button:disabled {
      opacity: 0.6;
      cursor: wait;
    }
    .btn {
      background: white;
      color: var(--ink);
      border: 1px solid var(--line);
    }
    code, pre {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.92rem;
    }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0;
      line-height: 1.6;
      color: #222;
    }
    .progress-shell {
      width: 100%;
      height: 14px;
      border-radius: 999px;
      overflow: hidden;
      background: #ece5d7;
      border: 1px solid rgba(169, 154, 129, 0.3);
      margin-top: 10px;
    }
    .progress-bar {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--accent), #ff8b38);
      transition: width 0.35s ease;
    }
    .progress-meta {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      margin-top: 10px;
      font-size: 0.94rem;
      color: var(--muted);
    }
    .log-box {
      max-height: 280px;
      overflow: auto;
      margin-top: 16px;
    }
    .result-links {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 12px;
    }
    .preview-shell {
      display: flex;
      align-items: stretch;
      gap: 0;
      min-height: 520px;
      overflow: hidden;
      border-radius: 18px;
      background: #f7f3ec;
    }
    .legend-pane {
      flex: 0 0 34%;
      min-width: 180px;
      max-width: 70%;
      padding: 10px;
      overflow: auto;
      background: #f7f3ec;
    }
    .splitter {
      width: 14px;
      flex: 0 0 14px;
      cursor: col-resize;
      position: relative;
      z-index: 3;
      touch-action: none;
      background: linear-gradient(180deg, #ede4d7 0%, #e2d7c5 100%);
      border-left: 1px solid rgba(169, 154, 129, 0.25);
      border-right: 1px solid rgba(169, 154, 129, 0.25);
    }
    .splitter::before {
      content: "";
      position: absolute;
      inset: 50% 3px auto 3px;
      height: 36px;
      transform: translateY(-50%);
      border-radius: 999px;
      background:
        linear-gradient(90deg,
          transparent 0,
          transparent 28%,
          rgba(95, 100, 112, 0.75) 28%,
          rgba(95, 100, 112, 0.75) 36%,
          transparent 36%,
          transparent 46%,
          rgba(95, 100, 112, 0.75) 46%,
          rgba(95, 100, 112, 0.75) 54%,
          transparent 54%,
          transparent 64%,
          rgba(95, 100, 112, 0.75) 64%,
          rgba(95, 100, 112, 0.75) 72%,
          transparent 72%,
          transparent 100%);
    }
    .scatter-pane {
      flex: 1 1 auto;
      min-width: 0;
      padding: 10px;
      overflow: auto;
      background: #fffdfa;
    }
    .scatter-stage {
      position: relative;
      width: 100%;
      min-height: 520px;
      border-radius: 18px;
      border: 1px solid rgba(110, 110, 110, 0.16);
      background: white;
      overflow: hidden;
    }
    .scatter-canvas {
      width: 100%;
      height: 100%;
      min-height: 520px;
      display: block;
      background: white;
    }
    .hover-card {
      position: absolute;
      min-width: 240px;
      max-width: 300px;
      padding: 12px;
      border-radius: 16px;
      background: rgba(255, 253, 248, 0.98);
      border: 1px solid rgba(169, 154, 129, 0.35);
      box-shadow: 0 18px 42px rgba(61, 46, 25, 0.16);
      pointer-events: none;
      z-index: 5;
    }
    .hover-title {
      margin: 0 0 8px;
      font-size: 0.85rem;
      font-weight: 700;
      color: var(--ink);
    }
    .hover-title-row {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 8px;
    }
    .hover-color-dot {
      width: 14px;
      height: 14px;
      border-radius: 50%;
      border: 2px solid white;
      box-shadow: 0 0 0 1px rgba(26, 26, 26, 0.12);
      flex: 0 0 14px;
    }
    .hover-smiles {
      margin: 0 0 8px;
      font-size: 0.78rem;
      color: var(--muted);
      line-height: 1.45;
      word-break: break-word;
    }
    .hover-image {
      width: 100%;
      height: auto;
      display: block;
      border-radius: 12px;
      background: white;
      border: 1px solid rgba(110, 110, 110, 0.14);
    }
    .hover-target {
      margin-top: 8px;
      font-size: 0.78rem;
      color: var(--muted);
    }
    .pane-title {
      margin: 0 0 8px;
      font-size: 0.92rem;
      color: var(--muted);
      font-weight: 700;
    }
    .preview-fit {
      width: 100%;
      height: auto;
      display: block;
      border-radius: 18px;
      border: 1px solid rgba(110, 110, 110, 0.16);
      background: white;
    }
    .hidden {
      display: none;
    }
    @media (max-width: 940px) {
      .hero, .grid, .inline { grid-template-columns: 1fr; }
      .preview-shell {
        display: block;
        min-height: 0;
      }
      .legend-pane, .scatter-pane {
        width: 100%;
        max-width: none;
      }
      .splitter {
        display: none;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="card">
        <h1><span class="title-mark"><span class="title-icon">⌬</span><span>Scaffold Visualization<br>Local App</span></span></h1>
        <p>
          Upload your own molecular dataset CSV and pretrained checkpoint,
          choose a few display parameters, and run the local scaffold visualization tool directly from this page.
        </p>
        <div class="flow-row">
          <div class="pill">🧪 Upload CSV</div>
          <div class="flow-arrow">→</div>
          <div class="pill">🧠 Upload checkpoint</div>
          <div class="flow-arrow">→</div>
          <div class="pill">🎛 Set panel label, style, top-k</div>
          <div class="flow-arrow">→</div>
          <div class="pill">🖼 Get PNG + CSV</div>
        </div>
        <div class="actions">
          <a class="btn" href="{{ url_for('download_demo', filename='bace_demo_10.csv') }}">⬇️ Demo CSV</a>
          <a class="btn" href="{{ url_for('download_demo', filename='bace_full.csv') }}">⬇️ Full BACE CSV</a>
          <a class="btn" href="{{ url_for('download_demo', filename='bbbp_full.csv') }}">⬇️ Full BBBP CSV</a>
          <a class="btn" href="{{ url_for('download_demo', filename='original_CMPN_0707_0800_12000th_epoch.pkl') }}">⬇️ Sample checkpoint</a>
        </div>
      </div>
      <div class="card">
        <img class="preview" src="{{ url_for('download_example_output', filename='bace_scaffold_tsne.png') }}" alt="BACE preview">
        <p>Example output from the bundled BACE demo.</p>
      </div>
    </section>

    <section class="grid">
      <div class="card">
        <h2>📥 Required Input Format</h2>
        <h3>Dataset CSV</h3>
        <div class="format-box">
          <ul>
            <li>The file must be a <code>.csv</code>.</li>
            <li>The first column should contain molecule SMILES strings.</li>
            <li>The first column header should be <code>smiles</code>.</li>
            <li>Extra columns are allowed, for example a class label or a regression target.</li>
          </ul>
        </div>

        <h3>Checkpoint</h3>
        <div class="format-box">
          <ul>
            <li>Use a MotiL-compatible PyTorch checkpoint such as <code>.pkl</code> or <code>.pt</code>.</li>
            <li>It should load through <code>torch.load(..., map_location="cpu")</code>.</li>
            <li>It should contain encoder weights compatible with the micromolecule CMPNN encoder.</li>
          </ul>
        </div>

        <h3>📘 Data Type Support</h3>
        <div class="format-box">
          <ul>
            <li><code>classification</code> is supported.</li>
            <li><code>regression</code> is supported.</li>
            <li><code>multiclass</code> is supported. Please also set the number of classes.</li>
          </ul>
        </div>
      </div>

      <div class="card">
        <h2>🎛 Run the Tool</h2>
        <form id="run-form" method="post" enctype="multipart/form-data">
          <div class="field">
            <label for="csv_file">Dataset CSV</label>
            <input id="csv_file" name="csv_file" type="file" accept=".csv" required>
          </div>
          <div class="field">
            <label for="ckpt_file">Checkpoint file</label>
            <input id="ckpt_file" name="ckpt_file" type="file" accept=".pkl,.pt,.ckpt" required>
          </div>
          <div class="inline">
            <div class="field">
              <label for="dataset_type">Dataset type</label>
              <select id="dataset_type" name="dataset_type">
                <option value="classification">classification</option>
                <option value="regression">regression</option>
                <option value="multiclass">multiclass</option>
              </select>
            </div>
            <div class="field">
              <label for="panel_label">Panel label</label>
              <input id="panel_label" name="panel_label" type="text" value="BACE">
            </div>
            <div class="field">
              <label for="style">Style</label>
              <select id="style" name="style">
                <option value="reference">reference</option>
                <option value="basic">basic</option>
              </select>
            </div>
          </div>
          <div class="inline">
            <div class="field">
              <label for="top_k">Top-k scaffolds</label>
              <input id="top_k" name="top_k" type="number" min="1" value="6">
            </div>
            <div class="field">
              <label for="min_scaffold_size">Min scaffold size</label>
              <input id="min_scaffold_size" name="min_scaffold_size" type="number" min="1" value="8">
            </div>
            <div class="field">
              <label for="max_points">Max points (optional)</label>
              <input id="max_points" name="max_points" type="number" min="1" placeholder="leave empty">
            </div>
          </div>
          <div class="inline">
            <div class="field">
              <label for="multiclass_num_classes">Multiclass number of classes</label>
              <input id="multiclass_num_classes" name="multiclass_num_classes" type="number" min="2" value="3">
              <div class="hint">Used only when dataset type is <code>multiclass</code>.</div>
            </div>
            <div class="field">
              <label>
                <input name="show_counts" type="checkbox">
                show scaffold counts in the left panel
              </label>
            </div>
          </div>
          <button id="run-button" type="submit">Run visualization</button>
        </form>
      </div>
    </section>

    <section id="run-status" class="card hidden" style="margin-top: 22px;">
      <h2 id="status-title">⏳ Running visualization</h2>
      <div id="status-box" class="status-box">
        <pre id="status-message">Waiting to start...</pre>
      </div>
      <div class="progress-shell">
        <div id="progress-bar" class="progress-bar"></div>
      </div>
      <div class="progress-meta">
        <span id="progress-stage">Preparing...</span>
        <span id="progress-percent">0%</span>
      </div>
      <div class="log-box">
        <pre id="log-output">No logs yet.</pre>
      </div>
      <div id="result-links" class="result-links hidden">
        <a id="legend-link" class="btn hidden" href="#" target="_blank">🧩 Open legend PNG</a>
        <a id="scatter-link" class="btn hidden" href="#" target="_blank">📈 Open scatter PNG</a>
        <a id="png-link" class="btn hidden" href="#" target="_blank">🖼 Open PNG</a>
        <a id="csv-link" class="btn" href="#" download>📄 Download CSV</a>
      </div>
      <div id="result-preview" class="result-box hidden" style="margin-top: 16px;">
        <div id="preview-shell" class="preview-shell">
          <div id="legend-pane" class="legend-pane hidden">
            <div class="pane-title">Legend</div>
            <img id="legend-preview-image" class="preview-fit hidden" src="" alt="Generated legend">
          </div>
          <div id="preview-splitter" class="splitter hidden" aria-label="Resize legend panel" title="Drag to resize legend"></div>
          <div id="scatter-pane" class="scatter-pane">
            <div class="pane-title">Scatter Plot</div>
            <div id="scatter-stage" class="scatter-stage">
              <svg id="interactive-scatter" class="scatter-canvas hidden" viewBox="0 0 900 640" preserveAspectRatio="xMidYMid meet"></svg>
              <img id="preview-image" class="preview-fit hidden" src="" alt="Generated figure">
              <div id="hover-card" class="hover-card hidden">
                <div class="hover-title-row">
                  <span id="hover-color-dot" class="hover-color-dot"></span>
                  <div id="hover-title" class="hover-title"></div>
                </div>
                <div id="hover-smiles" class="hover-smiles"></div>
                <img id="hover-image" class="hover-image" src="" alt="Hovered molecule">
                <div id="hover-target" class="hover-target hidden"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  </div>

  <script>
    const form = document.getElementById("run-form");
    const runButton = document.getElementById("run-button");
    const statusSection = document.getElementById("run-status");
    const statusTitle = document.getElementById("status-title");
    const statusBox = document.getElementById("status-box");
    const statusMessage = document.getElementById("status-message");
    const progressBar = document.getElementById("progress-bar");
    const progressStage = document.getElementById("progress-stage");
    const progressPercent = document.getElementById("progress-percent");
    const logOutput = document.getElementById("log-output");
    const resultLinks = document.getElementById("result-links");
    const legendLink = document.getElementById("legend-link");
    const scatterLink = document.getElementById("scatter-link");
    const pngLink = document.getElementById("png-link");
    const csvLink = document.getElementById("csv-link");
    const resultPreview = document.getElementById("result-preview");
    const previewShell = document.getElementById("preview-shell");
    const legendPane = document.getElementById("legend-pane");
    const previewSplitter = document.getElementById("preview-splitter");
    const scatterPane = document.getElementById("scatter-pane");
    const legendPreviewImage = document.getElementById("legend-preview-image");
    const previewImage = document.getElementById("preview-image");
    const interactiveScatter = document.getElementById("interactive-scatter");
    const scatterStage = document.getElementById("scatter-stage");
    const hoverCard = document.getElementById("hover-card");
    const hoverColorDot = document.getElementById("hover-color-dot");
    const hoverTitle = document.getElementById("hover-title");
    const hoverSmiles = document.getElementById("hover-smiles");
    const hoverImage = document.getElementById("hover-image");
    const hoverTarget = document.getElementById("hover-target");

    let currentJobId = null;
    let pollTimer = null;
    let isResizing = false;
    let scatterPoints = [];
    let activeCircle = null;

    function setStatus(state, message) {
      statusBox.classList.remove("good", "bad");
      if (state === "done") {
        statusBox.classList.add("good");
        statusTitle.textContent = "✅ Run complete";
      } else if (state === "error") {
        statusBox.classList.add("bad");
        statusTitle.textContent = "⚠️ Run failed";
      } else {
        statusTitle.textContent = "⏳ Running visualization";
      }
      statusMessage.textContent = message;
    }

    function updateProgress(percent, stage) {
      progressBar.style.width = `${percent}%`;
      progressPercent.textContent = `${percent}%`;
      progressStage.textContent = stage || "Running...";
    }

    function hideHoverCard() {
      hoverCard.classList.add("hidden");
      if (activeCircle) {
        activeCircle.setAttribute("r", activeCircle.dataset.baseRadius || "8.5");
        activeCircle.setAttribute("stroke", "white");
        activeCircle.setAttribute("stroke-width", "2");
        activeCircle.setAttribute("fill-opacity", "0.96");
        activeCircle.removeAttribute("filter");
        activeCircle = null;
      }
    }

    function highlightCircle(circle) {
      if (activeCircle && activeCircle !== circle) {
        activeCircle.setAttribute("r", activeCircle.dataset.baseRadius || "8.5");
        activeCircle.setAttribute("stroke", "white");
        activeCircle.setAttribute("stroke-width", "2");
        activeCircle.setAttribute("fill-opacity", "0.96");
        activeCircle.removeAttribute("filter");
      }
      activeCircle = circle;
      if (!circle) {
        return;
      }
      circle.setAttribute("r", "11.5");
      circle.setAttribute("stroke", "#111111");
      circle.setAttribute("stroke-width", "2.8");
      circle.setAttribute("fill-opacity", "1");
      circle.setAttribute("filter", "drop-shadow(0px 0px 7px rgba(17,17,17,0.22))");
    }

    function showHoverCard(event, point, circle) {
      highlightCircle(circle);
      hoverTitle.textContent = point.scaffold;
      hoverColorDot.style.background = String(point.color || "#4b6ff2");
      hoverSmiles.textContent = point.smiles;
      hoverImage.src = `/molecule_image?smiles=${encodeURIComponent(point.smiles)}`;
      if (point.target) {
        hoverTarget.textContent = `Target: ${point.target}`;
        hoverTarget.classList.remove("hidden");
      } else {
        hoverTarget.classList.add("hidden");
      }

      const stageBounds = scatterStage.getBoundingClientRect();
      hoverCard.classList.remove("hidden");
      const cardWidth = hoverCard.offsetWidth || 260;
      const cardHeight = hoverCard.offsetHeight || 220;
      const margin = 14;
      let left = event.clientX - stageBounds.left + 16;
      let top = event.clientY - stageBounds.top + 16;
      if (left + cardWidth + margin > stageBounds.width) {
        left = event.clientX - stageBounds.left - cardWidth - 16;
      }
      if (left < margin) {
        left = margin;
      }
      if (top + cardHeight + margin > stageBounds.height) {
        top = event.clientY - stageBounds.top - cardHeight - 16;
      }
      if (top < margin) {
        top = margin;
      }
      hoverCard.style.left = `${left}px`;
      hoverCard.style.top = `${top}px`;
    }

    function buildColorMap(selectedScaffolds, colors, points) {
      const colorMap = {};
      selectedScaffolds.forEach((scaffold, index) => {
        const rawColor = colors[index];
        colorMap[scaffold] = typeof rawColor === "string" ? rawColor : "#4b6ff2";
      });
      if (Object.keys(colorMap).length === 0) {
        const palette = ["#ff245a", "#b3b3b3", "#ffdd2d", "#4b6ff2", "#ff8b38", "#9f1ae2"];
        let colorIndex = 0;
        points.forEach((point) => {
          if (!colorMap[point.scaffold]) {
            colorMap[point.scaffold] = palette[colorIndex % palette.length];
            colorIndex += 1;
          }
        });
      }
      return colorMap;
    }

    function renderInteractiveScatter(payload) {
      const points = payload.points || [];
      if (!points.length) {
        interactiveScatter.classList.add("hidden");
        previewImage.classList.remove("hidden");
        return;
      }

      const width = 900;
      const height = 640;
      const padding = 52;
      const xs = points.map((point) => point.x);
      const ys = points.map((point) => point.y);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      const spanX = Math.max(maxX - minX, 1e-6);
      const spanY = Math.max(maxY - minY, 1e-6);
      const colorMap = buildColorMap(payload.selected_scaffolds || [], payload.colors || [], points);

      interactiveScatter.innerHTML = "";
      interactiveScatter.setAttribute("viewBox", `0 0 ${width} ${height}`);

      for (let i = 0; i < 6; i += 1) {
        const x = padding + ((width - 2 * padding) * i) / 5;
        const y = padding + ((height - 2 * padding) * i) / 5;

        const vLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
        vLine.setAttribute("x1", x);
        vLine.setAttribute("y1", padding);
        vLine.setAttribute("x2", x);
        vLine.setAttribute("y2", height - padding);
        vLine.setAttribute("stroke", "#cfd4dd");
        vLine.setAttribute("stroke-opacity", "0.45");
        vLine.setAttribute("stroke-width", "1.2");
        interactiveScatter.appendChild(vLine);

        const hLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
        hLine.setAttribute("x1", padding);
        hLine.setAttribute("y1", y);
        hLine.setAttribute("x2", width - padding);
        hLine.setAttribute("y2", y);
        hLine.setAttribute("stroke", "#cfd4dd");
        hLine.setAttribute("stroke-opacity", "0.45");
        hLine.setAttribute("stroke-width", "1.2");
        interactiveScatter.appendChild(hLine);
      }

      const title = document.createElementNS("http://www.w3.org/2000/svg", "text");
      title.setAttribute("x", width / 2);
      title.setAttribute("y", 34);
      title.setAttribute("text-anchor", "middle");
      title.setAttribute("font-size", "28");
      title.setAttribute("font-family", "Avenir Next, Segoe UI, sans-serif");
      title.textContent = payload.db_index !== null && payload.db_index !== undefined
        ? `DB index: ${Number(payload.db_index).toFixed(2)}`
        : "Scatter Plot";
      interactiveScatter.appendChild(title);

      scatterPoints = points.map((point) => {
        const cx = padding + ((point.x - minX) / spanX) * (width - 2 * padding);
        const cy = height - padding - ((point.y - minY) / spanY) * (height - 2 * padding);
        return { ...point, cx, cy, color: colorMap[point.scaffold] || "#4b6ff2" };
      });

      scatterPoints.forEach((point) => {
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", point.cx);
        circle.setAttribute("cy", point.cy);
        circle.setAttribute("r", "8.5");
        circle.dataset.baseRadius = "8.5";
        circle.setAttribute("fill", String(point.color || "#4b6ff2"));
        circle.setAttribute("stroke", "white");
        circle.setAttribute("stroke-width", "2");
        circle.setAttribute("fill-opacity", "0.96");
        circle.addEventListener("mousemove", (event) => showHoverCard(event, point, circle));
        circle.addEventListener("mouseleave", hideHoverCard);
        interactiveScatter.appendChild(circle);
      });

      interactiveScatter.classList.remove("hidden");
      previewImage.classList.add("hidden");
    }

    async function loadInteractiveScatter(pointsUrl, fallbackImageUrl) {
      try {
        const response = await fetch(pointsUrl);
        if (!response.ok) {
          throw new Error("Could not load scatter points.");
        }
        const payload = await response.json();
        renderInteractiveScatter(payload);
      } catch (error) {
        interactiveScatter.classList.add("hidden");
        previewImage.classList.remove("hidden");
        previewImage.src = `${fallbackImageUrl}?t=${Date.now()}`;
      }
    }

    function startResize(event) {
      if (window.innerWidth <= 940) {
        return;
      }
      isResizing = true;
      document.documentElement.style.cursor = "col-resize";
      document.documentElement.style.userSelect = "none";
      event.preventDefault();
    }

    function stopResize() {
      if (!isResizing) {
        return;
      }
      isResizing = false;
      document.documentElement.style.cursor = "";
      document.documentElement.style.userSelect = "";
    }

    function handleResize(event) {
      if (!isResizing || window.innerWidth <= 940) {
        return;
      }
      const bounds = previewShell.getBoundingClientRect();
      const left = Math.max(180, Math.min(event.clientX - bounds.left, bounds.width * 0.7));
      const ratio = (left / bounds.width) * 100;
      legendPane.style.flexBasis = `${ratio}%`;
    }

    async function pollJobStatus() {
      if (!currentJobId) {
        return;
      }

      const response = await fetch(`/job_status/${currentJobId}`);
      const data = await response.json();

      setStatus(data.state, data.message);
      updateProgress(data.progress, data.stage);
      logOutput.textContent = data.logs || "No logs yet.";

      if (data.state === "done") {
        runButton.disabled = false;
        resultLinks.classList.remove("hidden");
        resultPreview.classList.remove("hidden");
        if (data.legend_url) {
          legendLink.classList.remove("hidden");
          legendLink.href = data.legend_url;
          legendPane.classList.remove("hidden");
          previewSplitter.classList.remove("hidden");
          legendPreviewImage.classList.remove("hidden");
          legendPreviewImage.src = `${data.legend_url}?t=${Date.now()}`;
        } else {
          legendLink.classList.add("hidden");
          legendPane.classList.add("hidden");
          previewSplitter.classList.add("hidden");
          legendPreviewImage.classList.add("hidden");
        }
        if (data.scatter_url) {
          scatterLink.classList.remove("hidden");
          scatterLink.href = data.scatter_url;
          loadInteractiveScatter(data.points_url, data.scatter_url);
          pngLink.classList.add("hidden");
        } else if (data.png_url) {
          pngLink.classList.remove("hidden");
          pngLink.href = data.png_url;
          loadInteractiveScatter(data.points_url, data.png_url);
          legendPane.classList.add("hidden");
          previewSplitter.classList.add("hidden");
          scatterLink.classList.add("hidden");
        } else {
          interactiveScatter.classList.add("hidden");
          previewImage.classList.add("hidden");
          legendPane.classList.add("hidden");
          previewSplitter.classList.add("hidden");
          scatterLink.classList.add("hidden");
          pngLink.classList.add("hidden");
        }
        csvLink.href = data.csv_url;
        clearInterval(pollTimer);
        pollTimer = null;
      } else if (data.state === "error") {
        runButton.disabled = false;
        clearInterval(pollTimer);
        pollTimer = null;
      }
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      runButton.disabled = true;
      statusSection.classList.remove("hidden");
      resultLinks.classList.add("hidden");
      resultPreview.classList.add("hidden");
      setStatus("running", "Uploading files and starting the job...");
      updateProgress(0, "Preparing...");
      logOutput.textContent = "Starting job...";

      const formData = new FormData(form);
      const response = await fetch("/start", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        runButton.disabled = false;
        setStatus("error", data.message || "Failed to start the job.");
        logOutput.textContent = data.logs || "No logs available.";
        return;
      }

      currentJobId = data.job_id;
      if (pollTimer) {
        clearInterval(pollTimer);
      }
      pollTimer = setInterval(pollJobStatus, 1000);
      pollJobStatus();
    });

    previewSplitter.addEventListener("pointerdown", startResize);
    window.addEventListener("pointermove", handleResize);
    window.addEventListener("pointerup", stopResize);
    scatterStage.addEventListener("mouseleave", hideHoverCard);
  </script>
</body>
</html>
"""


def safe_filename(original_name: str) -> str:
    name = secure_filename(original_name)
    return name or f"upload_{uuid.uuid4().hex}"


def check_csv_header(csv_path: Path) -> tuple[bool, str]:
    try:
        with csv_path.open() as f:
            reader = csv.reader(f)
            header = next(reader)
    except Exception as exc:
        return False, f"Could not read CSV header: {exc}"

    if not header:
        return False, "CSV header is empty."
    if header[0].strip() != "smiles":
        return False, f'The first column header must be "smiles". Found "{header[0]}".'
    return True, "CSV format looks good."


def build_command(
    csv_path: Path,
    ckpt_path: Path,
    output_dir: Path,
    form_data,
) -> list[str]:
    dataset_type = form_data.get("dataset_type", "classification").strip()
    panel_label = form_data.get("panel_label", "").strip() or "BACE"
    style = form_data.get("style", "reference").strip()
    top_k = form_data.get("top_k", "6").strip() or "6"
    min_scaffold_size = form_data.get("min_scaffold_size", "8").strip() or "8"
    max_points = form_data.get("max_points", "").strip()
    multiclass_num_classes = form_data.get("multiclass_num_classes", "3").strip() or "3"
    show_counts = form_data.get("show_counts") == "on"

    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--data-path", str(csv_path),
        "--checkpoint-path", str(ckpt_path),
        "--dataset-type", dataset_type,
        "--panel-label", panel_label,
        "--style", style,
        "--top-k", str(top_k),
        "--min-scaffold-size", str(min_scaffold_size),
        "--multiclass-num-classes", str(multiclass_num_classes),
        "--output-dir", str(output_dir),
    ]

    if max_points:
        command.extend(["--max-points", str(max_points)])
    if show_counts:
        command.append("--show-counts")
    return command


def append_job_log(job_id: str, message: str) -> None:
    with JOBS_LOCK:
        if job_id not in JOBS:
            return
        JOBS[job_id]["logs"].append(message)


def update_job(job_id: str, **kwargs) -> None:
    with JOBS_LOCK:
        if job_id not in JOBS:
            return
        JOBS[job_id].update(kwargs)


def parse_progress_line(line: str) -> tuple[int | None, str | None]:
    if not line.startswith("[PROGRESS] "):
        return None, None
    body = line[len("[PROGRESS] ") :].strip()
    parts = body.split(" ", 1)
    if not parts:
        return None, None
    try:
        percent = int(parts[0])
    except ValueError:
        return None, None
    stage = parts[1] if len(parts) > 1 else ""
    return percent, stage


def should_hide_log_line(line: str) -> bool:
    if not line.strip():
        return False
    if any(fragment in line for fragment in IGNORED_LOG_FRAGMENTS):
        return True
    stripped = line.strip()
    if "it/s" in stripped or "%|" in stripped:
        return True
    return False


def run_job(job_id: str, command: list[str], output_dir: Path, csv_stem: str) -> None:
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(BASE_DIR / ".mplcache")

    process = subprocess.Popen(
        command,
        cwd=str(ROOT_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    update_job(job_id, state="running", message="Running visualization locally...")

    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        if should_hide_log_line(line):
            continue
        progress, stage = parse_progress_line(line)
        if progress is not None:
            update_job(job_id, progress=progress, stage=stage, message=stage)
            continue
        append_job_log(job_id, line)

    return_code = process.wait()
    png_name = f"{csv_stem}_scaffold_tsne.png"
    legend_name = f"{csv_stem}_scaffold_legend.png"
    scatter_name = f"{csv_stem}_scaffold_scatter.png"
    csv_name = f"{csv_stem}_scaffold_tsne.csv"
    meta_name = f"{csv_stem}_scaffold_meta.json"
    png_path = output_dir / png_name
    legend_path = output_dir / legend_name
    scatter_path = output_dir / scatter_name
    csv_path = output_dir / csv_name
    meta_path = output_dir / meta_name

    has_reference_outputs = legend_path.exists() and scatter_path.exists()
    has_basic_outputs = png_path.exists()
    if return_code != 0 or (not has_reference_outputs and not has_basic_outputs) or not csv_path.exists() or not meta_path.exists():
        update_job(
            job_id,
            state="error",
            progress=100,
            stage="Run failed",
            message="The script did not finish successfully.",
        )
        return

    update_job(
        job_id,
        state="done",
        progress=100,
        stage="Finished",
        message="Visualization finished successfully.",
        png_url=f"/jobs/{job_id}/{png_name}" if has_basic_outputs else None,
        legend_url=f"/jobs/{job_id}/{legend_name}" if has_reference_outputs else None,
        scatter_url=f"/jobs/{job_id}/{scatter_name}" if has_reference_outputs else None,
        csv_url=f"/jobs/{job_id}/{csv_name}",
        meta_url=f"/jobs/{job_id}/{meta_name}",
        points_url=f"/job_points/{job_id}",
    )


@app.route("/", methods=["GET"])
def index():
    return render_template_string(PAGE_TEMPLATE)


@app.route("/start", methods=["POST"])
def start_job():
    csv_file = request.files.get("csv_file")
    ckpt_file = request.files.get("ckpt_file")

    if not csv_file or not ckpt_file:
        return jsonify({"message": "Please upload both a CSV file and a checkpoint file."}), 400

    job_id = uuid.uuid4().hex[:10]
    upload_dir = UPLOAD_DIR / job_id
    output_dir = JOB_DIR / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_name = safe_filename(csv_file.filename)
    ckpt_name = safe_filename(ckpt_file.filename)
    csv_path = upload_dir / csv_name
    ckpt_path = upload_dir / ckpt_name
    csv_file.save(csv_path)
    ckpt_file.save(ckpt_path)

    ok, msg = check_csv_header(csv_path)
    if not ok:
        return jsonify({"message": msg}), 400

    command = build_command(csv_path, ckpt_path, output_dir, request.form)

    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "state": "queued",
            "progress": 0,
            "stage": "Queued",
            "message": "The job has been queued.",
            "logs": [],
            "png_url": None,
            "legend_url": None,
            "scatter_url": None,
            "csv_url": None,
            "meta_url": None,
            "points_url": None,
        }

    worker = threading.Thread(
        target=run_job,
        args=(job_id, command, output_dir, csv_path.stem),
        daemon=True,
    )
    worker.start()

    return jsonify({"job_id": job_id})


@app.route("/job_status/<job_id>")
def job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return jsonify({"message": "Job not found."}), 404
        payload = {
            "job_id": job["id"],
            "state": job["state"],
            "progress": job["progress"],
            "stage": job["stage"],
            "message": job["message"],
            "logs": "\n".join(job["logs"]).strip(),
            "png_url": job["png_url"],
            "legend_url": job.get("legend_url"),
            "scatter_url": job.get("scatter_url"),
            "csv_url": job["csv_url"],
            "meta_url": job.get("meta_url"),
            "points_url": job.get("points_url"),
        }
    return jsonify(payload)


@app.route("/job_points/<job_id>")
def job_points(job_id: str):
    job_dir = JOB_DIR / job_id
    if not job_dir.exists():
        abort(404)

    csv_files = sorted(job_dir.glob("*_scaffold_tsne.csv"))
    meta_files = sorted(job_dir.glob("*_scaffold_meta.json"))
    if not csv_files or not meta_files:
        abort(404)

    csv_path = csv_files[0]
    meta_path = meta_files[0]
    metadata = json.loads(meta_path.read_text())

    points = []
    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            points.append(
                {
                    "smiles": row["smiles"],
                    "scaffold": row["scaffold"],
                    "x": float(row["tsne_x"]),
                    "y": float(row["tsne_y"]),
                    "target": row.get("target", ""),
                }
            )

    return jsonify(
        {
            "points": points,
            "selected_scaffolds": metadata.get("selected_scaffolds", []),
            "colors": metadata.get("colors", []),
            "dataset_name": metadata.get("dataset_name", ""),
            "db_index": metadata.get("db_index"),
        }
    )


@app.route("/molecule_image")
def molecule_image():
    smiles = request.args.get("smiles", "").strip()
    if not smiles:
        abort(400)

    from rdkit import Chem
    from rdkit.Chem import Draw

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        abort(404)

    image = Draw.MolToImage(mol, size=(260, 160))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return send_file(buffer, mimetype="image/png")


@app.route("/demo/<path:filename>")
def download_demo(filename: str):
    return send_from_directory(DEMO_DIR, filename, as_attachment=True)


@app.route("/example/<path:filename>")
def download_example_output(filename: str):
    return send_from_directory(BASE_DIR / "outputs_bace_demo", filename, as_attachment=False)


@app.route("/jobs/<job_id>/<path:filename>")
def download_job_file(job_id: str, filename: str):
    job_dir = JOB_DIR / job_id
    if not job_dir.exists():
        abort(404)
    return send_from_directory(job_dir, filename, as_attachment=filename.endswith(".csv"))


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8000)
