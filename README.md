# Scaffold Visualization with MotiL

This folder provides a small plotting tool for making scaffold-colored t-SNE figures from MotiL micromolecule embeddings.

It is built for fast reuse and demo videos.

## Upstream Code

This script works together with the original MotiL repository:

- MotiL repository: [Young0222/MotiL](https://github.com/Young0222/MotiL)

It directly reuses code from:

- Scaffold extraction: [scaffold.py](https://github.com/Young0222/MotiL/blob/main/MotiL_micromolecule/chemprop/data/scaffold.py)
- Data loading: [utils.py](https://github.com/Young0222/MotiL/blob/main/MotiL_micromolecule/chemprop/data/utils.py)
- Model building: [model.py](https://github.com/Young0222/MotiL/blob/main/MotiL_micromolecule/chemprop/models/model.py)
- CMPNN encoder: [cmpn.py](https://github.com/Young0222/MotiL/blob/main/MotiL_micromolecule/chemprop/models/cmpn.py)
- Featurization: [featurization.py](https://github.com/Young0222/MotiL/blob/main/MotiL_micromolecule/chemprop/features/featurization.py)

## Files

- `plot_scaffold_tsne.py`
  Main plotting script.
- `requirements.txt`
  Extra plotting dependencies.

## Setup

Place this `scaffold_visualization` folder next to `MotiL_micromolecule` inside the original MotiL repository.

Use the same environment as the micromolecule code:

```bash
cd MotiL_micromolecule
pip install -r requirements.txt
```

If needed, also install:

```bash
pip install matplotlib scikit-learn
```

## Quick Start

From the repository root:

```bash
python3 scaffold_visualization/plot_scaffold_tsne.py
```

This uses:

- dataset: `MotiL_micromolecule/data/esol.csv`
- checkpoint: `MotiL_micromolecule/dumped/pre-train/1-model/original_CMPN_0707_0800_12000th_epoch.pkl`
- style: `reference`

## Demo-Friendly Commands

Reference-style ESOL figure:

```bash
python3 scaffold_visualization/plot_scaffold_tsne.py \
  --data-path MotiL_micromolecule/data/esol.csv \
  --dataset-type regression \
  --panel-label ESOL \
  --style reference \
  --output-dir scaffold_visualization/outputs_esol_demo
```

Reference-style BACE figure:

```bash
python3 scaffold_visualization/plot_scaffold_tsne.py \
  --data-path MotiL_micromolecule/data/bace.csv \
  --dataset-type classification \
  --panel-label BACE \
  --style reference \
  --top-k 6 \
  --min-scaffold-size 8 \
  --output-dir scaffold_visualization/outputs_bace_demo
```

Reference-style figure with counts shown in the left panel:

```bash
python3 scaffold_visualization/plot_scaffold_tsne.py \
  --data-path MotiL_micromolecule/data/bace.csv \
  --dataset-type classification \
  --panel-label BACE \
  --style reference \
  --top-k 6 \
  --show-counts \
  --output-dir scaffold_visualization/outputs_bace_counts
```

Simple scatter style:

```bash
python3 scaffold_visualization/plot_scaffold_tsne.py \
  --data-path MotiL_micromolecule/data/bace.csv \
  --dataset-type classification \
  --panel-label BACE \
  --style basic \
  --output-dir scaffold_visualization/outputs_bace_basic
```

If your working environment uses a specific Python executable, run the script with that exact path. Example:

```bash
MPLCONFIGDIR=/absolute/path/to/scaffold_visualization/.mplcache \
/opt/homebrew/bin/python3.10 scaffold_visualization/plot_scaffold_tsne.py \
  --data-path MotiL_micromolecule/data/bace.csv \
  --dataset-type classification \
  --panel-label BACE \
  --style reference
```

## Main Arguments

- `--panel-label`
  Overrides the vertical label in the left panel, for example `BACE`.
- `--style`
  Plot style. Use `reference` for the demo-style layout and `basic` for a simple scatter plot.
- `--show-counts`
  Shows `n=...` for each scaffold in the left panel.
- `--top-k`
  Number of scaffold groups to display when scaffolds are selected automatically.
- `--min-scaffold-size`
  Minimum scaffold frequency for automatic selection.
- `--scaffolds`
  Manually specify scaffold SMILES strings and their display order.
- `--max-points`
  Limit the number of molecules for faster testing.

## Output Files

The script saves:

- `*_scaffold_tsne.png`
- `*_scaffold_tsne.csv`

The CSV contains:

- `smiles`
- `scaffold`
- `tsne_x`
- `tsne_y`
- `target`

## Notes

- The left scaffold structures are drawn automatically with RDKit from scaffold SMILES.
- The `reference` style is intended for polished figures and demo videos.
- For paper-style figures, `--scaffolds` gives the most control over scaffold order and appearance.
