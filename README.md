# Scaffold Visualization with MotiL

This folder turns the scaffold-related parts of the MotiL micromolecule code into a small, reusable workflow for plotting scaffold-colored t-SNE figures.

The goal is simple: help other researchers reuse the same scaffold idea from the paper without having to trace the full training code first.

## 🔗 Upstream code

This visualization script is designed to work together with the original MotiL repository:

- MotiL repository: [Young0222/MotiL](https://github.com/Young0222/MotiL)

Please download or clone that repository first, because this script directly reuses several functions from the MotiL micromolecule code.

## ✨ What this folder helps you do

With one script, you can:

1. Load a molecular dataset from a CSV file.
2. Extract molecular embeddings from a pretrained MotiL encoder.
3. Compute Bemis-Murcko scaffolds for each molecule.
4. Select a small set of scaffolds to visualize.
5. Run t-SNE on the molecular embeddings.
6. Save a scaffold-colored figure and a CSV file of 2D coordinates.

## 🧩 What this reuses from the original codebase

This workflow reuses the existing MotiL implementation directly.

- [`MotiL_micromolecule/chemprop/data/scaffold.py`](https://github.com/Young0222/MotiL/blob/main/MotiL_micromolecule/chemprop/data/scaffold.py)
  - Uses `generate_scaffold(mol)` to compute Bemis-Murcko scaffolds with RDKit.
- [`MotiL_micromolecule/chemprop/data/utils.py`](https://github.com/Young0222/MotiL/blob/main/MotiL_micromolecule/chemprop/data/utils.py)
  - Uses `get_data(...)` to load molecules from a CSV file.
- [`MotiL_micromolecule/chemprop/models/model.py`](https://github.com/Young0222/MotiL/blob/main/MotiL_micromolecule/chemprop/models/model.py)
  - Uses `build_pretrain_model(...)` to build the MotiL encoder for embedding extraction.
- [`MotiL_micromolecule/chemprop/train/predict.py`](https://github.com/Young0222/MotiL/blob/main/MotiL_micromolecule/chemprop/train/predict.py)
  - Uses `get_emb(...)` to extract molecular embeddings from the pretrained encoder.

In other words, this folder is a lightweight visualization layer built on top of MotiL, not a reimplementation of the full micromolecule pipeline.

## 🧠 What Figure 2a means

Figure 2a in the paper is not a direct drawing of scaffold structures.

It is a 2D embedding plot:

1. Each point is one molecule.
2. The point position comes from t-SNE on molecular embeddings.
3. The point color shows the scaffold label of that molecule.
4. A lower Davies-Bouldin index suggests cleaner separation between scaffold groups.

## 📁 Files in this folder

- `plot_scaffold_tsne.py`
  Ready-to-run script for loading a dataset, extracting MotiL embeddings, computing scaffolds, running t-SNE, and saving outputs.
- `requirements.txt`
  Extra plotting dependencies used by this folder.

## 📄 Expected input format

The script expects a CSV file whose first column is `smiles`.

Examples already included in this repository:

- `MotiL_micromolecule/data/esol.csv`
- `MotiL_micromolecule/data/bbbp.csv`

## ⚙️ Environment setup

Use the same environment as the micromolecule code.

```bash
cd MotiL_micromolecule
pip install -r requirements.txt
```

The plotting script also expects `matplotlib` and `scikit-learn`. In many environments they are already installed. If not, install them manually:

```bash
pip install matplotlib scikit-learn
```

## 📦 What you need to download

To run this script directly, make sure the following files are available from the original MotiL repository:

- MotiL repository: [Young0222/MotiL](https://github.com/Young0222/MotiL)
- Scaffold function: [scaffold.py](https://github.com/Young0222/MotiL/blob/main/MotiL_micromolecule/chemprop/data/scaffold.py)
- Data loading utilities: [utils.py](https://github.com/Young0222/MotiL/blob/main/MotiL_micromolecule/chemprop/data/utils.py)
- Pretrain model builder: [model.py](https://github.com/Young0222/MotiL/blob/main/MotiL_micromolecule/chemprop/models/model.py)
- Embedding extraction code: [predict.py](https://github.com/Young0222/MotiL/blob/main/MotiL_micromolecule/chemprop/train/predict.py)
- Example dataset: [esol.csv](https://github.com/Young0222/MotiL/blob/main/MotiL_micromolecule/data/esol.csv)
- Example dataset: [bbbp.csv](https://github.com/Young0222/MotiL/blob/main/MotiL_micromolecule/data/bbbp.csv)
- Example pretrained checkpoint folder: [dumped/pre-train/1-model](https://github.com/Young0222/MotiL/tree/main/MotiL_micromolecule/dumped/pre-train/1-model)

The simplest setup is to place this `scaffold_visualization` folder next to `MotiL_micromolecule` inside the original MotiL repository structure.

## ✅ Tested environment

This workflow has been tested in this repository with:

- macOS
- `/opt/homebrew/bin/python3.10`
- `torch`
- `rdkit`
- `scikit-learn`
- `matplotlib`
- `pandas`
- `Unidecode`

Tested command:

```bash
/opt/homebrew/bin/python3.10 scaffold_visualization/plot_scaffold_tsne.py \
  --data-path MotiL_micromolecule/data/esol.csv \
  --dataset-type regression \
  --output-dir scaffold_visualization/outputs_test
```

Tested output:

- dataset: `ESOL`
- valid molecules loaded: `1128`
- selected scaffolds: `4`
- Davies-Bouldin index: `1.896`
- generated files:
  - `scaffold_visualization/outputs_test/esol_scaffold_tsne.png`
  - `scaffold_visualization/outputs_test/esol_scaffold_tsne.csv`

If your default `python3` does not have RDKit, use the exact Python executable from your working MotiL environment.

## 🚀 Quick start

From the repository root:

```bash
python3 scaffold_visualization/plot_scaffold_tsne.py
```

This default command:

- loads `MotiL_micromolecule/data/esol.csv`
- loads the included pretrained checkpoint
- computes scaffolds with the original MotiL scaffold code
- keeps the top 4 frequent scaffolds with at least 10 molecules
- runs t-SNE on the selected molecules
- saves a `.png` figure and a `.csv` table of 2D coordinates

## ▶️ Direct run commands

If you are using the original MotiL repository structure, these are the simplest commands.

Run the default ESOL example:

```bash
python3 scaffold_visualization/plot_scaffold_tsne.py
```

Run ESOL explicitly:

```bash
python3 scaffold_visualization/plot_scaffold_tsne.py \
  --data-path MotiL_micromolecule/data/esol.csv \
  --checkpoint-path MotiL_micromolecule/dumped/pre-train/1-model/original_CMPN_0707_0800_12000th_epoch.pkl \
  --dataset-type regression
```

Run BBBP explicitly:

```bash
python3 scaffold_visualization/plot_scaffold_tsne.py \
  --data-path MotiL_micromolecule/data/bbbp.csv \
  --checkpoint-path MotiL_micromolecule/dumped/pre-train/1-model/original_CMPN_0707_0800_12000th_epoch.pkl \
  --dataset-type classification
```

If your working environment uses a specific Python executable, run the script with that exact Python path. For example:

```bash
/opt/homebrew/bin/python3.10 scaffold_visualization/plot_scaffold_tsne.py \
  --data-path MotiL_micromolecule/data/esol.csv \
  --dataset-type regression
```

## 🔬 Example commands

Plot the default ESOL example:

```bash
python3 scaffold_visualization/plot_scaffold_tsne.py \
  --data-path MotiL_micromolecule/data/esol.csv \
  --dataset-type regression
```

Plot BBBP:

```bash
python3 scaffold_visualization/plot_scaffold_tsne.py \
  --data-path MotiL_micromolecule/data/bbbp.csv \
  --dataset-type classification \
  --top-k 4 \
  --min-scaffold-size 15
```

Plot user-selected scaffolds instead of automatic top-k selection:

```bash
python3 scaffold_visualization/plot_scaffold_tsne.py \
  --data-path MotiL_micromolecule/data/esol.csv \
  --dataset-type regression \
  --scaffolds "c1ccc2ccccc2c1" "O=C1NC(=O)NC(=O)N1"
```

Save outputs to a custom folder:

```bash
python3 scaffold_visualization/plot_scaffold_tsne.py \
  --output-dir scaffold_visualization/my_outputs
```

## 📤 Output files

The script writes two files into the output directory:

- `*_scaffold_tsne.png`
  A scaffold-colored t-SNE figure.
- `*_scaffold_tsne.csv`
  One row per plotted molecule with:
  - `smiles`
  - `scaffold`
  - `tsne_x`
  - `tsne_y`
  - `target`

## 🎯 How scaffold selection works

In the paper text, the authors describe selecting four chemically diverse scaffolds for visualization. So the plotting step is not only "compute scaffold for every molecule", but also "choose which scaffolds to display".

This script supports two practical choices:

- Automatic selection with `--top-k`
- Manual selection with `--scaffolds`

For paper-style figures, manual selection is often better because it gives more control over which chemotypes appear in the figure.

## 🪜 Recommended workflow for collaborators

If a lab member wants to make a scaffold plot on a new dataset:

1. Prepare a CSV file with `smiles` in the first column.
2. Use a pretrained MotiL checkpoint.
3. Run `plot_scaffold_tsne.py`.
4. Start with automatic top-k selection.
5. If needed, rerun with `--scaffolds` to highlight specific scaffold families.

## 🛠 Quick troubleshooting

- If you see `ModuleNotFoundError: rdkit`, your current Python does not have RDKit installed.
- If your default `python3` fails but another Python works, run the script with that exact Python path.
- If you see an error related to `chemprop`, make sure this folder is placed next to the original `MotiL_micromolecule` folder from [Young0222/MotiL](https://github.com/Young0222/MotiL).
- If no scaffold passes the filter, lower `--min-scaffold-size`.
- If you want a more paper-like figure, use `--scaffolds` and choose the scaffold families manually.

## 📝 Notes and limitations

- This folder reuses the scaffold logic already present in the repository. It does not replace the original implementation.
- The repository does not include the exact internal plotting script used to generate Figure 2a in the paper.
- This folder provides a clean and reproducible approximation of that workflow using the published codebase.
