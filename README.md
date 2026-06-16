# Scaffold Visualization with MotiL

This folder packages the scaffold-related pieces of the MotiL micromolecule code into a small, reusable workflow for plotting scaffold-colored t-SNE figures.

The goal is to help other researchers reuse the same scaffold logic from the paper without having to trace the full training code.

## What this uses from the original codebase

The visualization workflow reuses the existing MotiL implementation directly:

- `MotiL_micromolecule/chemprop/data/scaffold.py`
  - `generate_scaffold(mol)` computes the Bemis-Murcko scaffold with RDKit.
- `MotiL_micromolecule/chemprop/data/utils.py`
  - `get_data(...)` loads molecules from a CSV file.
- `MotiL_micromolecule/chemprop/train/predict.py`
  - `get_emb(...)` extracts molecular embeddings from the pretrained encoder.
- `MotiL_micromolecule/chemprop/models/model.py`
  - `build_pretrain_model(...)` builds the MotiL encoder used for embedding extraction.

## What Figure 2a does

Figure 2a in the paper is not a direct drawing of scaffold structures. It is a 2D embedding plot:

1. Extract a molecular embedding for each molecule.
2. Compute the Bemis-Murcko scaffold for each molecule.
3. Select a small set of scaffolds to visualize.
4. Run t-SNE on the molecular embeddings.
5. Plot one point per molecule and color the points by scaffold label.
6. Optionally report the Davies-Bouldin index to measure cluster separation.

## Files in this folder

- `plot_scaffold_tsne.py`
  - A ready-to-run script that loads a dataset, extracts MotiL embeddings, computes scaffolds, runs t-SNE, and saves a scaffold-colored figure.

## Expected input format

The script expects a CSV file whose first column is `smiles`.

Examples already included in this repository:

- `MotiL_micromolecule/data/esol.csv`
- `MotiL_micromolecule/data/bbbp.csv`

## Environment setup

Use the same environment as the micromolecule code.

```bash
cd MotiL_micromolecule
pip install -r requirements.txt
```

The plotting script also expects `matplotlib` and `scikit-learn`. In many environments they are already installed. If not, install them manually:

```bash
pip install matplotlib scikit-learn
```

## Quick start

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

## Example commands

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

## Output files

The script writes two files into the output directory:

- `*_scaffold_tsne.png`
  - the scaffold-colored t-SNE figure
- `*_scaffold_tsne.csv`
  - one row per plotted molecule with:
    - `smiles`
    - `scaffold`
    - `tsne_x`
    - `tsne_y`
    - `target`

## Important note about scaffold selection

In the paper text, the authors describe selecting four chemically diverse scaffolds for visualization. That means the plotting step is not only "compute scaffold for every molecule", but also "choose which scaffolds to display".

This script supports two practical choices:

- automatic selection with `--top-k`
- manual selection with `--scaffolds`

For paper-style figures, manual scaffold selection is often better because it gives more control over which chemotypes appear in the figure.

## Recommended workflow for collaborators

If a lab member wants to make a scaffold plot on a new dataset:

1. Prepare a CSV file with `smiles` in the first column.
2. Use a pretrained MotiL checkpoint.
3. Run `plot_scaffold_tsne.py`.
4. Start with automatic top-k selection.
5. If needed, rerun with `--scaffolds` to highlight specific scaffold families.

## Notes and limitations

- This folder reuses the scaffold logic already present in the repository. It does not replace the original implementation.
- The repository does not include the exact internal plotting script used to generate Figure 2a in the paper.
- This folder provides a clean and reproducible approximation of that workflow using the published codebase.
