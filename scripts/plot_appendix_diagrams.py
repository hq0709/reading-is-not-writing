"""Draw Appendix A1, A2 and A6 as reproducible vector diagrams.

Run from any directory with the paper's Python environment. Content follows
sections/A_appendix.tex; typography and export settings use appendix_style.
"""

from appendix_style import (
    BLUE, GRID, INK, LIGHT, MUTED, ORANGE, TEAL, apply_style, save_figure,
)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


# With the shared export padding, the finished pages are 6.4 inches wide.
WIDTH = 6.28


def canvas(height):
    fig = plt.figure(figsize=(WIDTH, height))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, WIDTH), ylim=(height, 0))
    ax.set_axis_off()
    return fig, ax


def text(ax, x, y, value, *, size=7.8, color=INK, weight="normal",
         ha="left", va="top", **kwargs):
    return ax.text(x, y, value, fontsize=size, color=color,
                   fontweight=weight, ha=ha, va=va, linespacing=1.4,
                   **kwargs)


def rule(ax, x1, x2, y, *, color=GRID, width=0.7):
    ax.plot((x1, x2), (y, y), color=color, linewidth=width,
            solid_capstyle="butt")


def arrow(ax, start, end, *, color=MUTED, style="-", width=0.8):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=8,
        linewidth=width, color=color, linestyle=style,
        shrinkA=0, shrinkB=0,
    ))


def label(ax, x, y, title):
    text(ax, x, y, title, size=8.5, weight="bold")


def prompt_score():
    fig, ax = canvas(2.90)
    label(ax, 0.12, 0.10, "Fixed clinical question")
    ax.add_patch(Rectangle((0.12, 0.34), 6.04, 0.55,
                           facecolor=LIGHT, edgecolor="none"))
    ax.plot((0.12, 0.12), (0.34, 0.89), color=BLUE, linewidth=1.2)
    text(ax, 0.25, 0.44,
         "Is there <finding> in this chest radiograph?", size=8.0)
    text(ax, 0.25, 0.66, "Answer yes or no.", size=8.0)

    label(ax, 0.12, 1.06, "Six ownership findings")
    text(ax, 6.15, 1.06, "Only the finding phrase changes", size=7.2,
         ha="right")
    findings = (
        ("Effusion", "a pleural effusion"),
        ("Atelectasis", "atelectasis"),
        ("Pneumothorax", "a pneumothorax"),
        ("Cardiomegaly", "cardiomegaly"),
        ("Mass", "a lung mass"),
        ("Nodule", "a lung nodule"),
    )
    for i, (name, phrase) in enumerate(findings):
        col, row = divmod(i, 3)
        x, y = 0.12 + col * 3.17, 1.35 + row * 0.26
        text(ax, x, y, name, size=7.8)
        text(ax, x + 1.22, y, phrase, size=7.8)
        rule(ax, x, x + 2.87, y + 0.20)

    label(ax, 0.12, 2.17, "First-answer scoring")
    centers = (0.59, 1.94, 3.57, 5.40)
    entries = (
        "Image +\nfixed question",
        "First-answer-\nposition logits",
        "Max log probability\nfor each answer over\nvalid token variants",
        r"$P(\mathrm{yes}) = \sigma(\ell_{\mathrm{yes}}-\ell_{\mathrm{no}})$",
    )
    for x, entry in zip(centers, entries):
        text(ax, x, 2.62, entry, size=7.5, ha="center", va="center")
    for start, end in ((1.08, 1.34), (2.54, 2.78), (4.38, 4.60)):
        arrow(ax, (start, 2.62), (end, 2.62))
    save_figure(fig, "figA1_prompt_score")


def provenance():
    fig, ax = canvas(2.35)
    columns = ((0.12, 1.67, BLUE), (1.94, 3.75, TEAL),
               (4.02, 6.16, ORANGE))
    headings = ("Registered design", "Patient-level\nestimates",
                "Registered\nstatistical decision")
    for (left, right, color), heading in zip(columns, headings):
        rule(ax, left, right, 0.10, color=color, width=1.0)
        text(ax, left, 0.21, heading, size=8.5, weight="bold")
    arrow(ax, (1.69, 0.35), (1.85, 0.35), color=INK)
    arrow(ax, (3.77, 0.35), (3.93, 0.35), color=INK)

    text(ax, 0.12, 0.70, "Model + locus", size=8.0, weight="bold")
    text(ax, 0.12, 0.95, "Qwen\nFinal visual block", size=7.8)
    text(ax, 0.12, 1.55, "Dose + cohort", size=8.0, weight="bold")
    text(ax, 0.12, 1.80, "Prespecified dose rule\nand patient selection", size=7.8)

    text(ax, 1.94, 0.70, "Ownership / alias", size=8.0, weight="bold")
    text(ax, 1.94, 0.95, "400 patients\n304,800 evaluations", size=7.8)
    text(ax, 4.02, 0.70, "Patient bootstrap +\nregistered comparators", size=7.8)
    text(ax, 4.02, 1.10, "0/6 owned\nNo shared alias", weight="bold", size=8.0)
    arrow(ax, (3.68, 1.05), (3.92, 1.05), color=INK)

    rule(ax, 1.94, 6.16, 1.43)
    text(ax, 1.94, 1.55, "Input closure", size=8.0, weight="bold")
    text(ax, 1.94, 1.80, "50 patient pairs\n6,450 evaluations", size=7.8)
    text(ax, 4.02, 1.55, "Paired bootstrap +\nclosure conjunction", size=7.8)
    text(ax, 4.02, 1.95, "No input closure", weight="bold", size=8.0)
    arrow(ax, (3.68, 1.90), (3.92, 1.90), color=INK)

    rule(ax, 0.12, 6.16, 2.20)
    save_figure(fig, "figA2_provenance")


def forward_path():
    fig, ax = canvas(2.97)
    text(ax, 0.12, 0.10, "Consumed visual path", size=8.5, weight="bold")
    text(ax, 6.16, 0.11, "Capture → intervene → measure", size=7.2,
         ha="right")

    # The shaded region identifies the intervention locus within one forward.
    ax.add_patch(Rectangle((1.48, 0.38), 1.70, 0.57,
                           facecolor=LIGHT, edgecolor="none"))
    stages = ((0.61, "Visual\ntokens", BLUE),
              (2.33, "Consumed\nfinal block", BLUE),
              (4.09, "Connector /\nmerger", TEAL),
              (5.66, "Answer\nlogits", ORANGE))
    for x, title, color in stages:
        text(ax, x, 0.48, title, size=8.0, weight="bold", ha="center")
        rule(ax, x - 0.40, x + 0.40, 0.86, color=color, width=1.2)
    for start, end in ((1.04, 1.44), (3.22, 3.55), (4.65, 5.10)):
        arrow(ax, (start, 0.67), (end, 0.67))
    text(ax, 2.33, 1.07, "Capture + intervention", size=7.2, ha="center")
    text(ax, 4.09, 1.07, "Downstream witness", size=7.2, ha="center")
    text(ax, 5.66, 1.07, "Endpoint witness", size=7.2, ha="center")

    # Dose branches annotate checks of the same hook, not extra model stages.
    ax.plot((2.33, 2.33), (1.28, 1.42), color=MUTED, linewidth=0.7)
    ax.plot((1.21, 4.53), (1.42, 1.42), color=MUTED, linewidth=0.7)
    for x in (1.21, 4.53):
        arrow(ax, (x, 1.42), (x, 1.59), color=MUTED)
    text(ax, 1.21, 1.70, r"$\alpha = 0$  ·  bitwise no-op",
         size=8.0, ha="center", weight="bold")
    text(ax, 4.53, 1.70, r"$\alpha \ne 0$  ·  downstream change",
         size=8.0, ha="center", weight="bold")
    text(ax, 1.21, 1.95, "Baseline preserved", size=7.2, ha="center")
    text(ax, 4.53, 1.95, "Connector / merger and answer logits", size=7.2,
         ha="center")

    rule(ax, 0.12, 6.16, 2.25)
    text(ax, 0.12, 2.42, "LLaVA", size=7.8, weight="bold")
    text(ax, 0.90, 2.42, "encoder.layers.22", size=7.5,
         fontfamily="DejaVu Sans Mono")
    text(ax, 6.16, 2.42, "captured once", size=7.2, ha="right")
    text(ax, 0.12, 2.72, "Qwen", size=7.8, weight="bold")
    text(ax, 0.90, 2.72, "model.visual.blocks.31", size=7.5,
         fontfamily="DejaVu Sans Mono")
    text(ax, 6.16, 2.72, "32 blocks verified", size=7.2, ha="right")
    save_figure(fig, "figA6_forward_path")


def main():
    apply_style()
    prompt_score()
    provenance()
    forward_path()


if __name__ == "__main__":
    main()
