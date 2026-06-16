#!/usr/bin/env python3

"""Plot scaffold-colored t-SNE figures from MotiL embeddings."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import types

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score


ROOT_DIR = Path(__file__).resolve().parents[1]
MICRO_DIR = ROOT_DIR / "MotiL_micromolecule"

# `chemprop/models/model.py` imports names from `turtle`, although they are not
# used by the scaffold plotting workflow. On headless machines this can fail
# because `turtle` pulls in `tkinter`. A small stub keeps the workflow portable.
if "turtle" not in sys.modules:
    turtle_stub = types.ModuleType("turtle")
    turtle_stub.forward = None
    turtle_stub.hideturtle = None
    turtle_stub.up = None
    sys.modules["turtle"] = turtle_stub

# The micromolecule code expects to be imported from inside MotiL_micromolecule
# because some files are loaded through relative paths at import time.
os.chdir(MICRO_DIR)
if str(MICRO_DIR) not in sys.path:
    sys.path.insert(0, str(MICRO_DIR))

from chemprop.data.scaffold import generate_scaffold  # noqa: E402
from chemprop.data.utils import get_data  # noqa: E402
from chemprop.models.model import build_pretrain_model  # noqa: E402
from chemprop.train.predict import get_emb  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a scaffold-colored t-SNE plot from MotiL micromolecule embeddings."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=ROOT_DIR / "MotiL_micromolecule" / "data" / "esol.csv",
        help="Path to a CSV file whose first column is SMILES.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=ROOT_DIR
        / "MotiL_micromolecule"
        / "dumped"
        / "pre-train"
        / "1-model"
        / "original_CMPN_0707_0800_12000th_epoch.pkl",
        help="Path to a pretrained MotiL encoder checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "scaffold_visualization" / "outputs",
        help="Directory for the figure and exported coordinates.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["classification", "regression", "multiclass"],
        default="regression",
        help="Only used to build the Chemprop model config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size used during embedding extraction.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of scaffolds to keep when --scaffolds is not provided.",
    )
    parser.add_argument(
        "--min-scaffold-size",
        type=int,
        default=10,
        help="Minimum number of molecules in a scaffold to keep it for plotting.",
    )
    parser.add_argument(
        "--scaffolds",
        nargs="+",
        default=None,
        help="Optional explicit scaffold SMILES strings to visualize.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Optional cap on the number of valid molecules loaded from the CSV.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity. It will be clipped automatically if the dataset is small.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Random seed for t-SNE.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for embedding extraction.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title.",
    )
    return parser.parse_args()


def build_runtime_args(cli_args: argparse.Namespace) -> SimpleNamespace:
    if cli_args.device == "auto":
        use_cuda = torch.cuda.is_available()
    elif cli_args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        use_cuda = True
    else:
        use_cuda = False

    return SimpleNamespace(
        data_path=str(cli_args.data_path.resolve()),
        dataset_type=cli_args.dataset_type,
        features_generator=None,
        features_path=None,
        max_data_size=cli_args.max_points,
        use_compound_names=False,
        cuda=use_cuda,
        hidden_size=300,
        bias=False,
        depth=3,
        undirected=False,
        atom_messages=False,
        features_only=False,
        use_input_features=False,
        activation="ReLU",
        dropout=0.0,
        ffn_num_layers=2,
        multiclass_num_classes=3,
    )


def normalize_path(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()


def load_model(runtime_args: SimpleNamespace, checkpoint_path: Path) -> torch.nn.Module:
    model = build_pretrain_model(runtime_args, encoder_name="CMPNN")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.encoder.load_state_dict(state_dict, strict=False)
    if runtime_args.cuda:
        model = model.cuda()
    model.eval()
    return model


def select_scaffolds(
    scaffold_labels: list[str],
    explicit_scaffolds: list[str] | None,
    top_k: int,
    min_scaffold_size: int,
) -> list[str]:
    counts = Counter(scaffold_labels)

    if explicit_scaffolds:
        selected = [scaffold for scaffold in explicit_scaffolds if scaffold in counts]
        missing = [scaffold for scaffold in explicit_scaffolds if scaffold not in counts]
        if missing:
            print("Warning: these scaffolds were not found in the dataset:")
            for scaffold in missing:
                print(f"  {scaffold}")
        if not selected:
            raise ValueError("None of the requested scaffolds were found in the dataset.")
        return selected

    frequent_scaffolds = [
        scaffold
        for scaffold, count in counts.most_common()
        if count >= min_scaffold_size and scaffold
    ]
    selected = frequent_scaffolds[:top_k]
    if not selected:
        raise ValueError(
            "No scaffold passed the current filter. Try lowering --min-scaffold-size."
        )
    return selected


def clip_perplexity(perplexity: float, n_samples: int) -> float:
    if n_samples < 3:
        raise ValueError("At least 3 molecules are required for a t-SNE plot.")
    max_valid = max(2.0, float(n_samples - 1))
    return min(perplexity, max_valid)


def export_coordinates(
    output_path: Path,
    smiles_list: list[str],
    scaffolds: list[str],
    coordinates: np.ndarray,
    target_values: list[str],
) -> None:
    with output_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["smiles", "scaffold", "tsne_x", "tsne_y", "target"])
        for smile, scaffold, coord, target in zip(
            smiles_list, scaffolds, coordinates, target_values
        ):
            writer.writerow([smile, scaffold, coord[0], coord[1], target])


def format_legend_label(scaffold: str, count: int) -> str:
    short = scaffold if len(scaffold) <= 30 else scaffold[:27] + "..."
    return f"{short} (n={count})"


def main() -> None:
    cli_args = parse_args()
    cli_args.data_path = normalize_path(cli_args.data_path)
    cli_args.checkpoint_path = normalize_path(cli_args.checkpoint_path)
    cli_args.output_dir = normalize_path(cli_args.output_dir)
    cli_args.output_dir.mkdir(parents=True, exist_ok=True)

    runtime_args = build_runtime_args(cli_args)
    data = get_data(
        path=str(cli_args.data_path),
        args=runtime_args,
        max_data_size=cli_args.max_points,
    )
    if len(data) == 0:
        raise ValueError("No valid molecules were loaded from the input CSV.")

    print(f"Loaded {len(data)} valid molecules from {cli_args.data_path}.")

    model = load_model(runtime_args, cli_args.checkpoint_path)
    embeddings = get_emb(model, data, batch_size=cli_args.batch_size)

    smiles_list = data.smiles()
    scaffold_labels = [generate_scaffold(smile) for smile in smiles_list]
    target_values = [
        "" if not datapoint.targets else datapoint.targets[0]
        for datapoint in data.data
    ]

    selected_scaffolds = select_scaffolds(
        scaffold_labels,
        cli_args.scaffolds,
        cli_args.top_k,
        cli_args.min_scaffold_size,
    )

    selected_set = set(selected_scaffolds)
    keep_indices = [
        index
        for index, scaffold in enumerate(scaffold_labels)
        if scaffold in selected_set
    ]
    if len(keep_indices) < 3:
        raise ValueError("Fewer than 3 molecules remain after scaffold filtering.")

    filtered_embeddings = embeddings[keep_indices]
    filtered_smiles = [smiles_list[index] for index in keep_indices]
    filtered_scaffolds = [scaffold_labels[index] for index in keep_indices]
    filtered_targets = [str(target_values[index]) for index in keep_indices]

    perplexity = clip_perplexity(cli_args.perplexity, len(filtered_embeddings))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=cli_args.seed,
    )
    coordinates = tsne.fit_transform(filtered_embeddings)

    db_index = None
    if len(set(filtered_scaffolds)) > 1:
        label_ids = [selected_scaffolds.index(scaffold) for scaffold in filtered_scaffolds]
        db_index = davies_bouldin_score(filtered_embeddings, label_ids)

    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("tab10", len(selected_scaffolds))
    scaffold_counts = Counter(filtered_scaffolds)
    for color_id, scaffold in enumerate(selected_scaffolds):
        scaffold_indices = [
            index for index, label in enumerate(filtered_scaffolds) if label == scaffold
        ]
        points = coordinates[scaffold_indices]
        plt.scatter(
            points[:, 0],
            points[:, 1],
            s=28,
            alpha=0.85,
            color=cmap(color_id),
            label=format_legend_label(scaffold, scaffold_counts[scaffold]),
        )

    title = cli_args.title
    if title is None:
        title = f"Scaffold-colored t-SNE: {cli_args.data_path.stem}"
    if db_index is not None:
        title += f" | DB index = {db_index:.3f}"

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()

    png_path = cli_args.output_dir / f"{cli_args.data_path.stem}_scaffold_tsne.png"
    csv_path = cli_args.output_dir / f"{cli_args.data_path.stem}_scaffold_tsne.csv"
    plt.savefig(png_path, dpi=cli_args.dpi)
    plt.close()

    export_coordinates(
        csv_path,
        filtered_smiles,
        filtered_scaffolds,
        coordinates,
        filtered_targets,
    )

    print(f"Selected scaffolds ({len(selected_scaffolds)}):")
    for scaffold in selected_scaffolds:
        print(f"  {scaffold}: {scaffold_counts[scaffold]} molecules")
    if db_index is not None:
        print(f"Davies-Bouldin index: {db_index:.3f}")
    print(f"Saved figure to {png_path}")
    print(f"Saved coordinates to {csv_path}")


if __name__ == "__main__":
    main()
