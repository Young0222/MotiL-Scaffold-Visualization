#!/usr/bin/env python3

"""Plot scaffold-colored t-SNE figures from MotiL embeddings."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import types

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import to_hex
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch
from rdkit import RDLogger
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
MICRO_DIR = ROOT_DIR / "MotiL_micromolecule"
CHEMPROP_DIR = MICRO_DIR / "chemprop"

matplotlib.use("Agg")

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

from rdkit import Chem  # noqa: E402
from rdkit.Chem import Draw  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def ensure_package(name: str, path: Path) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return module


def load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def bootstrap_chemprop():
    ensure_package("chemprop", CHEMPROP_DIR)
    features_pkg = ensure_package("chemprop.features", CHEMPROP_DIR / "features")
    data_pkg = ensure_package("chemprop.data", CHEMPROP_DIR / "data")
    ensure_package("chemprop.models", CHEMPROP_DIR / "models")

    features_generators_module = load_module(
        "chemprop.features.features_generators",
        CHEMPROP_DIR / "features" / "features_generators.py",
    )
    features_utils_module = load_module(
        "chemprop.features.utils",
        CHEMPROP_DIR / "features" / "utils.py",
    )
    featurization_module = load_module(
        "chemprop.features.featurization",
        CHEMPROP_DIR / "features" / "featurization.py",
    )

    features_pkg.get_available_features_generators = (
        features_generators_module.get_available_features_generators
    )
    features_pkg.get_features_generator = features_generators_module.get_features_generator
    features_pkg.atom_features = featurization_module.atom_features
    features_pkg.bond_features = featurization_module.bond_features
    features_pkg.BatchMolGraph = featurization_module.BatchMolGraph
    features_pkg.get_atom_fdim = featurization_module.get_atom_fdim
    features_pkg.get_bond_fdim = featurization_module.get_bond_fdim
    features_pkg.mol2graph = featurization_module.mol2graph
    features_pkg.clear_cache = featurization_module.clear_cache
    features_pkg.load_features = features_utils_module.load_features
    features_pkg.save_features = features_utils_module.save_features

    scaler_module = load_module("chemprop.data.scaler", CHEMPROP_DIR / "data" / "scaler.py")
    data_data_module = load_module("chemprop.data.data", CHEMPROP_DIR / "data" / "data.py")
    data_pkg.MoleculeDatapoint = data_data_module.MoleculeDatapoint
    data_pkg.MoleculeDataset = data_data_module.MoleculeDataset
    data_pkg.StandardScaler = scaler_module.StandardScaler

    load_module("chemprop.nn_utils", CHEMPROP_DIR / "nn_utils.py")
    scaffold_module = load_module("chemprop.data.scaffold", CHEMPROP_DIR / "data" / "scaffold.py")
    data_utils_module = load_module("chemprop.data.utils", CHEMPROP_DIR / "data" / "utils.py")

    data_pkg.scaffold_to_smiles = scaffold_module.scaffold_to_smiles

    load_module("chemprop.models.cmpn", CHEMPROP_DIR / "models" / "cmpn.py")
    load_module("chemprop.models.mpn", CHEMPROP_DIR / "models" / "mpn.py")
    model_module = load_module("chemprop.models.model", CHEMPROP_DIR / "models" / "model.py")

    return scaffold_module.generate_scaffold, data_utils_module.get_data, model_module.build_pretrain_model


generate_scaffold, get_data, build_pretrain_model = bootstrap_chemprop()


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
        default=SCRIPT_DIR / "outputs",
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
    parser.add_argument(
        "--panel-label",
        type=str,
        default=None,
        help="Optional label shown vertically in the left scaffold panel.",
    )
    parser.add_argument(
        "--style",
        choices=["reference", "basic"],
        default="reference",
        help="Plot style. Use 'reference' for the demo-style figure.",
    )
    parser.add_argument(
        "--show-counts",
        action="store_true",
        help="Show scaffold counts in the left panel.",
    )
    parser.add_argument(
        "--multiclass-num-classes",
        type=int,
        default=3,
        help="Number of classes when --dataset-type is multiclass.",
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
        multiclass_num_classes=cli_args.multiclass_num_classes,
    )


def normalize_path(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()


def emit_progress(percent: int, message: str) -> None:
    print(f"[PROGRESS] {percent} {message}", flush=True)


def load_model(runtime_args: SimpleNamespace, checkpoint_path: Path) -> torch.nn.Module:
    model = build_pretrain_model(runtime_args, encoder_name="CMPNN")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.encoder.load_state_dict(state_dict, strict=False)
    if runtime_args.cuda:
        model = model.cuda()
    model.eval()
    return model


def get_emb(model: torch.nn.Module, data, batch_size: int) -> np.ndarray:
    model.eval()
    embs = []
    num_iters = len(data)
    dataset_cls = data.__class__
    total_batches = max(1, (num_iters + batch_size - 1) // batch_size)

    for batch_index, start in enumerate(range(0, num_iters, batch_size), start=1):
        mol_batch = dataset_cls(data[start : start + batch_size])
        smiles_batch = mol_batch.smiles()
        features_batch = mol_batch.features()

        with torch.no_grad():
            batch_embs = model.encoder(
                "pretrain",
                "contrast_mol",
                3,
                0.0,
                smiles_batch,
                features_batch,
            )

        embs.append(batch_embs.detach().cpu().numpy())
        progress = 30 + int(40 * batch_index / total_batches)
        emit_progress(
            progress,
            f"Extracting embeddings ({batch_index}/{total_batches} batches)",
        )

    return np.vstack(embs)


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
        fallback_scaffolds = [scaffold for scaffold, _ in counts.most_common() if scaffold][:top_k]
        if not fallback_scaffolds:
            raise ValueError("No valid scaffolds were found in the dataset.")
        print(
            "Warning: no scaffold passed the current min-scaffold-size filter. "
            "Falling back to the most frequent scaffolds instead."
        )
        selected = fallback_scaffolds
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


def export_metadata(
    output_path: Path,
    selected_scaffolds: list[str],
    scaffold_counts: Counter,
    db_index: float | None,
    dataset_name: str,
) -> None:
    payload = {
        "dataset_name": dataset_name,
        "selected_scaffolds": selected_scaffolds,
        "scaffold_counts": {scaffold: scaffold_counts[scaffold] for scaffold in selected_scaffolds},
        "db_index": db_index,
        "colors": build_color_palette(len(selected_scaffolds)),
    }
    output_path.write_text(json.dumps(payload, indent=2))


def scaffold_to_image(scaffold: str, size: tuple[int, int] = (220, 120)):
    mol = Chem.MolFromSmiles(scaffold)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)


def dataset_display_name(path: Path) -> str:
    return path.stem.upper()


def build_color_palette(num_colors: int) -> list[str]:
    base_palette = [
        "#ff245a",
        "#b3b3b3",
        "#ffdd2d",
        "#4b6ff2",
        "#ff8b38",
        "#9f1ae2",
    ]
    if num_colors <= len(base_palette):
        return base_palette[:num_colors]
    cmap = plt.get_cmap("tab20", num_colors)
    return [to_hex(cmap(i)) for i in range(num_colors)]


def reference_layout_settings(num_scaffolds: int) -> dict[str, object]:
    compact_layout = num_scaffolds >= 9
    top_margin = 0.90 if compact_layout else 0.86
    bottom_margin = 0.06 if compact_layout else 0.10
    y_positions = np.linspace(top_margin, bottom_margin, num_scaffolds)
    return {
        "compact_layout": compact_layout,
        "legend_fig_height": max(6.6, 1.8 + 1.08 * num_scaffolds),
        "scatter_fig_height": max(6.0, 5.2 + 0.22 * max(0, num_scaffolds - 6)),
        "point_size": 440 if compact_layout else 520,
        "scatter_size": 185 if compact_layout else 230,
        "panel_fontsize": 23 if compact_layout else 27,
        "count_fontsize": 8 if compact_layout else 10,
        "scaffold_image_size": (170, 92) if compact_layout else (220, 120),
        "image_zoom": 0.46 if compact_layout else 0.62,
        "image_x": 0.70,
        "y_positions": y_positions,
    }


def add_scaffold_image(ax, image, x: float, y: float, zoom: float) -> None:
    image_box = OffsetImage(np.asarray(image), zoom=zoom)
    annotation = AnnotationBbox(
        image_box,
        (x, y),
        frameon=False,
        xycoords="axes fraction",
        box_alignment=(0.5, 0.5),
        zorder=2,
    )
    ax.add_artist(annotation)


def plot_reference_legend_figure(
    selected_scaffolds: list[str],
    scaffold_counts: Counter,
    dataset_name: str,
    output_path: Path,
    dpi: int,
    show_counts: bool,
) -> None:
    colors = build_color_palette(len(selected_scaffolds))
    color_map = {
        scaffold: colors[index]
        for index, scaffold in enumerate(selected_scaffolds)
    }
    layout = reference_layout_settings(len(selected_scaffolds))

    fig, ax_panel = plt.subplots(
        figsize=(3.6, layout["legend_fig_height"]),
        facecolor="white",
    )
    ax_panel.set_xlim(0, 1)
    ax_panel.set_ylim(0, 1)
    ax_panel.axis("off")

    panel_bg = FancyBboxPatch(
        (0.04, 0.04),
        0.9,
        0.92,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0.0,
        facecolor="#f3f3f3",
    )
    ax_panel.add_patch(panel_bg)

    for scaffold, y_pos in zip(selected_scaffolds, layout["y_positions"]):
        color = color_map[scaffold]
        ax_panel.scatter(
            [0.37],
            [y_pos],
            s=layout["point_size"],
            color=color,
            edgecolors="white",
            linewidths=1.4,
            zorder=3,
        )
        img = scaffold_to_image(scaffold, size=layout["scaffold_image_size"])
        if img is not None:
            add_scaffold_image(
                ax_panel,
                img,
                x=layout["image_x"],
                y=y_pos,
                zoom=layout["image_zoom"],
            )
        if show_counts:
            ax_panel.text(
                0.49,
                y_pos + 0.040,
                f"n={scaffold_counts[scaffold]}",
                fontsize=layout["count_fontsize"],
                color="#4d5563",
                ha="left",
                va="bottom",
            )

    ax_panel.text(
        0.17,
        0.50,
        dataset_name,
        rotation=90,
        va="center",
        ha="center",
        fontsize=layout["panel_fontsize"],
        color="black",
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.01)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_reference_scatter_figure(
    coordinates: np.ndarray,
    filtered_scaffolds: list[str],
    selected_scaffolds: list[str],
    db_index: float | None,
    output_path: Path,
    dpi: int,
) -> None:
    colors = build_color_palette(len(selected_scaffolds))
    color_map = {
        scaffold: colors[index]
        for index, scaffold in enumerate(selected_scaffolds)
    }
    layout = reference_layout_settings(len(selected_scaffolds))
    fig, ax_plot = plt.subplots(
        figsize=(8.2, layout["scatter_fig_height"]),
        facecolor="white",
    )

    for scaffold in selected_scaffolds:
        indices = [
            idx for idx, label in enumerate(filtered_scaffolds) if label == scaffold
        ]
        points = coordinates[indices]
        ax_plot.scatter(
            points[:, 0],
            points[:, 1],
            s=layout["scatter_size"],
            color=color_map[scaffold],
            edgecolors="white",
            linewidths=1.5,
            alpha=0.96,
            zorder=3,
        )

    title = "DB index"
    if db_index is not None:
        title = f"DB index: {db_index:.2f}"
    ax_plot.set_title(title, fontsize=30, pad=6)

    ax_plot.grid(True, color="#cfd4dd", linewidth=1.0, alpha=0.45)
    ax_plot.set_axisbelow(True)
    ax_plot.tick_params(labelbottom=False, labelleft=False, length=3, colors="#b7bcc7")
    ax_plot.set_xlabel("")
    ax_plot.set_ylabel("")

    for spine in ax_plot.spines.values():
        spine.set_color("#8f96a3")
        spine.set_linewidth(1.2)

    fig.subplots_adjust(left=0.08, right=0.995, top=0.92, bottom=0.06)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_basic_style_figure(
    coordinates: np.ndarray,
    filtered_scaffolds: list[str],
    selected_scaffolds: list[str],
    db_index: float | None,
    dataset_name: str,
    output_path: Path,
    dpi: int,
) -> None:
    colors = build_color_palette(len(selected_scaffolds))
    color_map = {scaffold: colors[index] for index, scaffold in enumerate(selected_scaffolds)}

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    for scaffold in selected_scaffolds:
        indices = [idx for idx, label in enumerate(filtered_scaffolds) if label == scaffold]
        points = coordinates[indices]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=36,
            color=color_map[scaffold],
            edgecolors="white",
            linewidths=0.8,
            alpha=0.92,
        )

    title = dataset_name if db_index is None else f"{dataset_name} | DB index: {db_index:.2f}"
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def main() -> None:
    cli_args = parse_args()
    cli_args.data_path = normalize_path(cli_args.data_path)
    cli_args.checkpoint_path = normalize_path(cli_args.checkpoint_path)
    cli_args.output_dir = normalize_path(cli_args.output_dir)
    cli_args.output_dir.mkdir(parents=True, exist_ok=True)

    emit_progress(5, "Preparing runtime")
    runtime_args = build_runtime_args(cli_args)
    emit_progress(10, "Loading dataset")
    data = get_data(
        path=str(cli_args.data_path),
        args=runtime_args,
        max_data_size=cli_args.max_points,
    )
    if len(data) == 0:
        raise ValueError("No valid molecules were loaded from the input CSV.")

    print(f"Loaded {len(data)} valid molecules from {cli_args.data_path}.")

    emit_progress(22, "Loading checkpoint")
    model = load_model(runtime_args, cli_args.checkpoint_path)
    emit_progress(30, "Starting embedding extraction")
    embeddings = get_emb(model, data, batch_size=cli_args.batch_size)

    emit_progress(72, "Computing scaffolds")
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

    emit_progress(80, "Running t-SNE")
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

    scaffold_counts = Counter(filtered_scaffolds)

    png_path = cli_args.output_dir / f"{cli_args.data_path.stem}_scaffold_tsne.png"
    legend_png_path = cli_args.output_dir / f"{cli_args.data_path.stem}_scaffold_legend.png"
    scatter_png_path = cli_args.output_dir / f"{cli_args.data_path.stem}_scaffold_scatter.png"
    csv_path = cli_args.output_dir / f"{cli_args.data_path.stem}_scaffold_tsne.csv"
    meta_path = cli_args.output_dir / f"{cli_args.data_path.stem}_scaffold_meta.json"
    panel_label = cli_args.panel_label if cli_args.panel_label else dataset_display_name(cli_args.data_path)
    emit_progress(92, "Rendering figure")
    if cli_args.style == "reference":
        plot_reference_legend_figure(
            selected_scaffolds=selected_scaffolds,
            scaffold_counts=scaffold_counts,
            dataset_name=panel_label,
            output_path=legend_png_path,
            dpi=cli_args.dpi,
            show_counts=cli_args.show_counts,
        )
        plot_reference_scatter_figure(
            coordinates=coordinates,
            filtered_scaffolds=filtered_scaffolds,
            selected_scaffolds=selected_scaffolds,
            db_index=db_index,
            output_path=scatter_png_path,
            dpi=cli_args.dpi,
        )
    else:
        plot_basic_style_figure(
            coordinates=coordinates,
            filtered_scaffolds=filtered_scaffolds,
            selected_scaffolds=selected_scaffolds,
            db_index=db_index,
            dataset_name=panel_label,
            output_path=png_path,
            dpi=cli_args.dpi,
        )

    export_coordinates(
        csv_path,
        filtered_smiles,
        filtered_scaffolds,
        coordinates,
        filtered_targets,
    )
    export_metadata(
        meta_path,
        selected_scaffolds,
        scaffold_counts,
        db_index,
        panel_label,
    )
    emit_progress(100, "Finished")

    print(f"Selected scaffolds ({len(selected_scaffolds)}):")
    for scaffold in selected_scaffolds:
        print(f"  {scaffold}: {scaffold_counts[scaffold]} molecules")
    if db_index is not None:
        print(f"Davies-Bouldin index: {db_index:.3f}")
    if cli_args.style == "reference":
        print(f"Saved legend figure to {legend_png_path}")
        print(f"Saved scatter figure to {scatter_png_path}")
    else:
        print(f"Saved figure to {png_path}")
    print(f"Saved coordinates to {csv_path}")
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
