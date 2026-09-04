"""Render Appendix A3--A5 from the accepted, full-precision results.

Run with the project figure environment; no external data or TeX is required.
"""

import json

from appendix_style import (
    BLUE, GRAY, INK, MUTED, ORANGE, ROOT, TEAL,
    apply_style, save_figure,
)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
import numpy as np


def read_results():
    with (ROOT / "data" / "ownership_summary.json").open(encoding="utf-8") as f:
        ownership = json.load(f)
    with (ROOT / "data" / "accepted_results.json").open(encoding="utf-8") as f:
        closure = json.load(f)["input_closure"]
    return ownership, closure


def numeric_axis(ax, limits, ticks, xlabel):
    ax.set_xlim(*limits)
    ax.set_xticks(ticks)
    ax.set_xlabel(xlabel, labelpad=7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3)
    ax.set_axisbelow(True)
    ax.grid(axis="x", linewidth=0.5, alpha=0.65)
    ax.axvline(0, color=MUTED, linewidth=0.8, zorder=1)


def panel_title(ax, letter, title):
    ax.set_title(f"{letter}  {title}", loc="left", pad=14)


def number(value, digits):
    return f"{value:.{digits}f}".replace("-", "−")


def plot_ownership(ownership):
    concepts = ownership["concepts"]
    assert ownership["matrix_orientation"] == (
        "rows are steering directions; columns are clinical questions"
    )
    matrix = np.array([
        [ownership["effect_matrix"][direction][question] for question in concepts]
        for direction in concepts
    ])
    winners = [concepts.index(ownership["columns"][q]["winner"]) for q in concepts]
    assert np.array_equal(np.argmax(matrix, axis=0), winners)
    assert np.allclose(np.diag(matrix), [
        ownership["columns"][q]["diagonal_effect"] for q in concepts
    ])

    fig = plt.figure(figsize=(6.4, 5.8))
    ax = fig.add_axes([0.225, 0.265, 0.63, 0.695])
    cax = fig.add_axes([0.895, 0.265, 0.025, 0.695])
    # A linear numerical scale puts white at zero, retaining signed magnitude.
    vmin, vmax = -0.1, 0.55
    cmap = LinearSegmentedColormap.from_list(
        "signed_effect", [(0, ORANGE), (-vmin / (vmax - vmin), "white"),
                          (1, BLUE)], N=1024,
    )
    norm = Normalize(vmin=vmin, vmax=vmax)
    mesh = ax.pcolormesh(
        np.arange(7) - 0.5, np.arange(7) - 0.5, matrix,
        cmap=cmap, norm=norm, edgecolors="white", linewidth=1.0,
        rasterized=False,
    )
    ax.set(xlim=(-0.5, 5.5), ylim=(5.5, -0.5), aspect="equal")
    ax.set_xticks(np.arange(6), concepts, rotation=43, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(6), concepts)
    ax.tick_params(axis="both", length=0, pad=6)
    ax.set_xlabel("Clinical question", labelpad=9)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Steering direction", labelpad=9)
    ax.spines[:].set_visible(False)
    for row, col in np.ndindex(matrix.shape):
        rgb = np.array(cmap(norm(matrix[row, col]))[:3])
        color = "white" if np.dot(rgb, [0.2126, 0.7152, 0.0722]) < 0.53 else INK
        ax.text(col, row, number(matrix[row, col], 3), ha="center", va="center",
                color=color, fontsize=9)
    for i in range(6):
        outline = Rectangle((i - 0.44, i - 0.44), 0.88, 0.88, fill=False,
                            edgecolor=INK, linewidth=1.1, zorder=4)
        outline.set_path_effects([pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()])
        ax.add_patch(outline)
    for col, row in enumerate(winners):
        ax.plot(col + 0.32, row - 0.30, marker="^", markersize=6,
                markerfacecolor=ORANGE, markeredgecolor="white", markeredgewidth=0.6,
                linestyle="none", zorder=5)
    colorbar = fig.colorbar(mesh, cax=cax, ticks=[-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5])
    colorbar.solids.set_rasterized(False)
    colorbar.solids.set_edgecolor("face")
    colorbar.set_label(r"$W_{q,d}$", labelpad=6)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=3)
    handles = [
        Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=INK, linewidth=1.1,
                  label="Named diagonal"),
        Line2D([], [], marker="^", color=ORANGE, linestyle="none", markersize=6,
               label="Column winner"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.54, 0.060),
               ncol=2, frameon=False, fontsize=8, handlelength=1.2, columnspacing=2)
    fig.text(0.54, 0.023,
             f"{ownership['n_off_diagonal_winners']}/6 off-diagonal winners"
             f"   ·   {ownership['n_owned_concepts']}/6 named owners"
             f"   ·   {ownership['n_eval_patients']} patients",
             ha="center", fontsize=8, color=MUTED)
    save_figure(fig, "figA3_ownership_matrix")


def dot_stem(ax, value, y, color, marker="o", digits=4):
    ax.plot([0, value], [y, y], color=color, linewidth=1.5, alpha=0.7, zorder=2)
    ax.plot(value, y, marker=marker, markersize=5.5, color=color,
            linestyle="none", zorder=3)
    ax.annotate(number(value, digits), (max(value, 0), y), xytext=(6, 0),
                textcoords="offset points", ha="left", va="center", fontsize=8)


def plot_alias(ownership):
    alias = ownership["shared_alias"]
    concepts = ownership["concepts"]
    fig = plt.figure(figsize=(6.4, 3.9))
    left = fig.add_axes([0.19, 0.31, 0.305, 0.57])
    right = fig.add_axes([0.68, 0.31, 0.305, 0.57], sharex=left)
    for ax in [left, right]:
        numeric_axis(ax, (-0.05, 0.46), [0, 0.1, 0.2, 0.3, 0.4],
                     r"Change in $P(\mathrm{yes})$")
        ax.set_ylim(-0.55, 5.55)
    panel_title(left, "A", "Clinical directions")
    panel_title(right, "B", "Global controls")
    left.set_yticks(np.arange(6)[::-1], concepts)
    for y, direction in zip(np.arange(6)[::-1], concepts):
        color = ORANGE if direction == alias["candidate_direction"] else BLUE
        dot_stem(left, alias["by_direction"][direction]["off_diagonal_effect"], y, color)
    right.set_yticks([5, 3, 1], [f"{alias['candidate_direction']}\ncandidate",
                                "Random\nmaximum", "|Sham|\nmaximum"])
    for y, key, color, marker in [
        (5, "off_diagonal_effect", ORANGE, "o"),
        (3, "global_random_effect_max", GRAY, "D"),
        (1, "global_absolute_sham_effect_max", TEAL, "s"),
    ]:
        dot_stem(right, alias[key], y, color, marker=marker)
    fig.text(0.19, 0.155, "Effusion simultaneous 95% lower bounds:", fontsize=8, color=MUTED)
    fig.text(0.19, 0.108,
             f"effect {alias['off_diagonal_effect_simultaneous_lower_95']:.4f}"
             f"; relative dominance {alias['relative_dominance_simultaneous_lower_95']:.4f}",
             fontsize=8, color=MUTED)
    fig.text(0.19, 0.035,
             "Shared alias: " + ("detected" if alias["detected"] else "not detected"),
             fontsize=9, color=INK)
    save_figure(fig, "figA4_shared_alias")


def lower_bound_plot(ax, estimate, lower, limits, ticks, xlabel, color, digits):
    numeric_axis(ax, limits, ticks, xlabel)
    ax.set(ylim=(0, 1), yticks=[])
    # The rightward arrow is an unbounded confidence set, not a two-sided CI.
    endpoint = limits[1] - 0.03 * (limits[1] - limits[0])
    ax.annotate("", xy=(endpoint, 0.53), xytext=(lower, 0.53),
                arrowprops=dict(arrowstyle="->", color=color, linewidth=1.1))
    ax.plot(lower, 0.53, marker="|", markersize=10, color=color,
            markeredgewidth=1.3, linestyle="none")
    ax.plot(estimate, 0.53, marker="o", color=color, markersize=6, linestyle="none")
    ax.annotate(number(estimate, digits), (estimate, 0.53), xytext=(0, 10),
                textcoords="offset points", ha="center", fontsize=9)
    ax.text(0.02, 0.12, f"95% lower bound {number(lower, digits)}",
            transform=ax.transAxes, fontsize=8, color=MUTED)


def plot_closure(closure):
    fig = plt.figure(figsize=(6.4, 5.35))
    reader = fig.add_axes([0.095, 0.655, 0.36, 0.225])
    displacement = fig.add_axes([0.625, 0.655, 0.36, 0.225])
    gains = fig.add_axes([0.185, 0.21, 0.30, 0.28])
    margin = fig.add_axes([0.625, 0.21, 0.36, 0.28])

    panel_title(reader, "A", "Reader eligibility")
    probe = closure["probe"]
    selectivity = probe["selectivity"]
    lo, hi = probe["selectivity_ci95"]
    numeric_axis(reader, (-0.005, 0.12), [0, 0.04, 0.08, 0.12], r"Reader selectivity, $S$")
    reader.set(ylim=(0, 1), yticks=[])
    reader.errorbar(selectivity, 0.53, xerr=[[selectivity - lo], [hi - selectivity]],
                    fmt="o", color=BLUE, markersize=6, capsize=3, elinewidth=1.3)
    reader.annotate(f"{selectivity:.4f}", (selectivity, 0.53), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=9)
    reader.text(0.02, 0.12, f"95% CI [{lo:.4f}, {hi:.4f}]", transform=reader.transAxes,
                fontsize=8, color=MUTED)
    reader.text(0.02, 0.88, f"AUROC {probe['real_auroc']:.4f}", transform=reader.transAxes,
                fontsize=8, color=MUTED)

    panel_title(displacement, "B", "Paired displacement")
    displacement_data = closure["input_displacement"]
    lower_bound_plot(displacement, displacement_data["estimate"],
                     displacement_data["one_sided_lower_95"], (-0.06, 0.19),
                     [-0.05, 0, 0.05, 0.10, 0.15], "Mean displacement", TEAL, 4)
    displacement.text(0.02, 0.88, f"{closure['n_pairs']} patient pairs",
                      transform=displacement.transAxes, fontsize=8, color=MUTED)

    panel_title(gains, "C", "Closure gains")
    numeric_axis(gains, (-1.1, 3.7), [-1, 0, 1, 2, 3], r"Closure gain ($\times 10^{-3}$)")
    gains.set_ylim(-0.55, 3.65)
    gains.set_yticks([3, 2, 1, 0], ["Consolidation", "Random max.",
                                  f"Best clinical\n({closure['maximum_unrelated']['concept']})", "Sham"])
    for y, value, color, marker in [
        (3, closure["concept_closure_gain"], ORANGE, "o"),
        (2, closure["random_closure_gain_max"], GRAY, "D"),
        (1, closure["maximum_unrelated"]["closure_gain"], BLUE, "o"),
        (0, closure["sham_closure_gain"], TEAL, "s"),
    ]:
        dot_stem(gains, value * 1000, y, color, marker, digits=2)

    panel_title(margin, "D", "Clinical margin")
    margin_data = closure["clinical_familywise_margin"]
    lower_bound_plot(margin, margin_data["estimate"] * 1000,
                     margin_data["one_sided_lower_95"] * 1000,
                     (-2.2, 3.4), [-2, -1, 0, 1, 2, 3],
                     r"Clinical margin ($\times 10^{-3}$)", ORANGE, 2)
    margin.text(0.02, 0.87, "Exact random-rank " + rf"$p={closure['exact_random_rank_p']:.3f}$"
                + "\n(descriptive)", transform=margin.transAxes, fontsize=8, color=MUTED,
                linespacing=1.4)
    fig.text(0.095, 0.086, "Arrows indicate one-sided 95% confidence sets extending to the right.",
             fontsize=8, color=MUTED)
    fig.text(0.095, 0.035,
             "Input closure: " + ("confirmed" if closure["input_closure"] else "not confirmed"),
             fontsize=9, color=INK)
    save_figure(fig, "figA5_input_closure")


def main():
    apply_style()
    ownership, closure = read_results()
    plot_ownership(ownership)
    plot_alias(ownership)
    plot_closure(closure)


if __name__ == "__main__":
    main()
