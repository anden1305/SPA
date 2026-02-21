from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_pair_labels(state_labels: list[str]) -> list[tuple[int, int, str]]:
    pairs: list[tuple[int, int, str]] = []
    for left in range(len(state_labels)):
        for right in range(left + 1, len(state_labels)):
            pairs.append((left, right, f"{state_labels[left]}–{state_labels[right]}"))
    return pairs


def style_axes(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", alpha=0.25)


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_average_subject_vs_population(
    pop_data: dict[str, Any],
    subject_data_list: list[dict[str, Any]],
    output_path: Path,
) -> None:
    pop_color = "#1f7391"
    subj_color = "#c42373"

    labels = pop_data["state_labels"]
    pairs = get_pair_labels(labels)

    # Pairwise ED: population vs subjects (avg)
    pop_vals = np.array([pop_data["pairwise_energy"][i][j] for i, j, _ in pairs], dtype=float)
    subj_vals_all = np.array(
        [[sd["pairwise_energy"][i][j] for i, j, _ in pairs] for sd in subject_data_list],
        dtype=float,
    )
    subj_avg = subj_vals_all.mean(axis=0)

    # Sort by population separation (descending)
    order = np.argsort(-pop_vals)
    pairs = [pairs[k] for k in order]
    pair_names = [p[2] for p in pairs]
    pop_vals = pop_vals[order]
    subj_avg = subj_avg[order]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    x = np.arange(len(pair_names))
    width = 0.36

    ax.bar(x - width / 2, pop_vals, width=width, color=pop_color, label="Population")
    ax.bar(x + width / 2, subj_avg, width=width, color=subj_color, label="Subjects Avg")

    ax.set_xticks(x)
    ax.set_xticklabels(pair_names)
    ax.set_ylabel("Energy Distance")
    ax.set_title("Population vs Average Subject Distinctness", pad=8)
    style_axes(ax)

    # Tighter y-limits
    ymax = float(max(pop_vals.max(), subj_avg.max()))
    ax.set_ylim(0, ymax * 1.10)

    # Value labels + deltas (delta just above the taller bar)
    y_pad = 0.03
    delta_y = 0.20
    for i, (pv, sv) in enumerate(zip(pop_vals, subj_avg)):
        ax.text(i - width / 2, pv + y_pad, f"{pv:.2f}", ha="center", va="bottom", fontsize=11)
        ax.text(i + width / 2, sv + y_pad, f"{sv:.2f}", ha="center", va="bottom", fontsize=11)
        ax.text(
            i,
            max(pv, sv) + delta_y,
            f"Δ {sv - pv:+.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Legend (keep compact)
    ax.legend(frameon=False, loc="upper right")

    save_figure(fig, output_path)


def find_subject_jsons(state_distinctness_dir: Path) -> list[Path]:
    subject_files = sorted(state_distinctness_dir.glob("pairwise_ed_raw_numbers_subject_*.json"))
    if not subject_files:
        raise FileNotFoundError(
            f"No subject JSON files found in {state_distinctness_dir}. "
            "Expected files like pairwise_ed_raw_numbers_subject_39.json"
        )
    return subject_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the average subject vs population distinctness plot.")
    parser.add_argument(
        "--population_json",
        type=Path,
        default=Path("results/state_distinctness/pairwise_ed_raw_numbers_population.json"),
        help="Population raw numbers JSON path.",
    )
    parser.add_argument(
        "--state_distinctness_dir",
        type=Path,
        default=Path("results/state_distinctness"),
        help="Directory containing subject raw numbers JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/state_distinctness/report_plots/average_subject_vs_population.png"),
        help="Output file path for the figure.",
    )
    args = parser.parse_args()

    pop_data = load_json(args.population_json)
    subject_jsons = find_subject_jsons(args.state_distinctness_dir)
    subject_data_list = [load_json(path) for path in subject_jsons]

    if not subject_data_list:
        raise ValueError("No subject data available for averaging.")

    plt.style.use("seaborn-v0_8-whitegrid")
    plot_average_subject_vs_population(pop_data, subject_data_list, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
