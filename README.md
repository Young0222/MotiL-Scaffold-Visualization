# Scaffold Visualization with MotiL

This folder provides a small local web tool for scaffold-based visualization of molecular embeddings from MotiL.

It is designed for easy demos and quick reuse.

## ✨ What This Tool Does

With this tool, a user can:

- 🧪 upload a molecular dataset CSV
- 🧠 upload a pretrained MotiL checkpoint
- 🎛 choose a few display parameters
- 📈 generate scaffold-colored visualization results
- 💾 download the output CSV

For `reference` style, the output is split into two images:

- 🧩 `*_scaffold_legend.png`
- 📈 `*_scaffold_scatter.png`

For `basic` style, the output is:

- 🖼 `*_scaffold_tsne.png`

In all cases, the tool also exports:

- 📄 `*_scaffold_tsne.csv`

## 🌱 Upstream Code

This tool works together with the original MotiL repository:

- MotiL repository: [Young0222/MotiL](https://github.com/Young0222/MotiL)

It directly reuses code from:

- Scaffold extraction: [scaffold.py](https://github.com/Young0222/MotiL/blob/main/MotiL_micromolecule/chemprop/data/scaffold.py)
- Data loading: [utils.py](https://github.com/Young0222/MotiL/blob/main/MotiL_micromolecule/chemprop/data/utils.py)
- Model building: [model.py](https://github.com/Young0222/MotiL/blob/main/MotiL_micromolecule/chemprop/models/model.py)
- CMPNN encoder: [cmpn.py](https://github.com/Young0222/MotiL/blob/main/MotiL_micromolecule/chemprop/models/cmpn.py)
- Featurization: [featurization.py](https://github.com/Young0222/MotiL/blob/main/MotiL_micromolecule/chemprop/features/featurization.py)

## 📁 Main Files

- `app.py`
  Local web app. This is the main entry point.
- `plot_scaffold_tsne.py`
  Core plotting script used by the web app.
- `requirements.txt`
  Extra Python packages needed by this folder.
- `demo_assets/`
  Demo CSV files and a sample checkpoint.
- `outputs_bace_demo/`
  Example BACE output files.

## 🧭 Folder Placement

Place `scaffold_visualization_v2` next to `MotiL_micromolecule` inside the original MotiL repository.

Expected structure:

```text
MotiL-main/
├── MotiL_micromolecule/
└── scaffold_visualization_v2/
```

## ⚙️ Setup

Use the same Python environment as the MotiL micromolecule code.

Install the original MotiL requirements first if needed:

```bash
cd MotiL_micromolecule
pip install -r requirements.txt
```

Then install the extra packages for this tool:

```bash
pip install -r ../scaffold_visualization_v2/requirements.txt
```

## 🚀 Start the Web App

From the repository root:

```bash
python3 scaffold_visualization_v2/app.py
```

Then open:

```text
http://127.0.0.1:8000
```

## 🪜 Simple Workflow

The web app follows this flow:

- 🧪 Upload CSV
- 🧠 Upload checkpoint
- 🎛 Set `panel-label`, `style`, `top-k`, and other options
- ▶️ Run visualization
- 🖼 Preview results
- 💾 Download output files

## 📥 Required Input Format

### Dataset CSV

- The file must be a `.csv`.
- The first column must contain molecule SMILES strings.
- The first column header must be `smiles`.
- Extra columns are allowed, such as class labels or regression targets.

Example files:

- [bace_demo_10.csv](/Users/ziyangliu/Desktop/MotiL-main/scaffold_visualization_v2/demo_assets/bace_demo_10.csv)
- [bace_full.csv](/Users/ziyangliu/Desktop/MotiL-main/scaffold_visualization_v2/demo_assets/bace_full.csv)
- [bbbp_full.csv](/Users/ziyangliu/Desktop/MotiL-main/scaffold_visualization_v2/demo_assets/bbbp_full.csv)

### Checkpoint

- Use a MotiL-compatible PyTorch checkpoint such as `.pkl` or `.pt`.
- It should load through `torch.load(..., map_location="cpu")`.
- It should contain encoder weights compatible with the micromolecule CMPNN encoder.

Example file:

- [original_CMPN_0707_0800_12000th_epoch.pkl](/Users/ziyangliu/Desktop/MotiL-main/scaffold_visualization_v2/demo_assets/original_CMPN_0707_0800_12000th_epoch.pkl)

## 🧩 Supported Data Types

- `classification`
- `regression`
- `multiclass`

For `multiclass`, also set the number of classes.

## 🎛 Main Options

- `Dataset type`
  Choose `classification`, `regression`, or `multiclass`.
- `Panel label`
  Controls the vertical label in the legend panel, for example `BACE`.
- `Style`
  Choose `reference` or `basic`.
- `Top-k scaffolds`
  Number of scaffold groups to show when automatic selection is used.
- `Min scaffold size`
  Minimum scaffold frequency for automatic selection.
- `Max points`
  Optional cap on the number of loaded molecules for faster testing.
- `Multiclass number of classes`
  Used only when `dataset type` is `multiclass`.
- `Show scaffold counts`
  Shows `n=...` next to each scaffold in the legend panel.

## 🖼 Output Files

### Reference Style

The tool generates:

- 🧩 `*_scaffold_legend.png`
- 📈 `*_scaffold_scatter.png`
- 📄 `*_scaffold_tsne.csv`

Notes:

- The legend and scatter plot are saved as separate images.
- This keeps the scaffold images clean and avoids layout conflicts.
- In the web app, the legend panel can be resized by dragging the divider.

### Basic Style

The tool generates:

- 🖼 `*_scaffold_tsne.png`
- 📄 `*_scaffold_tsne.csv`

## 📄 CSV Output Columns

The exported CSV contains:

- `smiles`
- `scaffold`
- `tsne_x`
- `tsne_y`
- `target`

## 🧪 Command Line Examples

### BACE, reference style

```bash
python3 scaffold_visualization_v2/plot_scaffold_tsne.py \
  --data-path MotiL_micromolecule/data/bace.csv \
  --dataset-type classification \
  --panel-label BACE \
  --style reference \
  --top-k 6 \
  --min-scaffold-size 8 \
  --output-dir scaffold_visualization_v2/outputs_bace_demo
```

### BACE, basic style

```bash
python3 scaffold_visualization_v2/plot_scaffold_tsne.py \
  --data-path MotiL_micromolecule/data/bace.csv \
  --dataset-type classification \
  --panel-label BACE \
  --style basic \
  --output-dir scaffold_visualization_v2/outputs_bace_demo
```

### Show scaffold counts

```bash
python3 scaffold_visualization_v2/plot_scaffold_tsne.py \
  --data-path MotiL_micromolecule/data/bace.csv \
  --dataset-type classification \
  --panel-label BACE \
  --style reference \
  --top-k 6 \
  --show-counts \
  --output-dir scaffold_visualization_v2/outputs_bace_demo
```

## 🎬 Demo Tips

If you are making a demo video, these are good features to show:

- 🌐 upload a CSV and checkpoint from the browser
- ⏳ real-time progress updates
- 🧾 clean running logs
- 🧩 separate legend and scatter outputs
- ↔️ draggable legend panel in the result viewer
- 💾 direct download of PNG and CSV outputs

## 📝 Notes

- The scaffold structures are drawn automatically by RDKit from scaffold SMILES.
- The `reference` style is best for polished figures and demos.
- The `basic` style is useful for quick checks.
- If automatic scaffold selection is too strict for a small demo dataset, lower `Min scaffold size`.
