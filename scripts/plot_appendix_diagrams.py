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


def text(ax, x, y, value, *, size=9, color=INK, weight="normal",
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


def label(ax, x, y, number, title, color):
    text(ax, x, y, number, size=8, color=color, weight="bold")
    text(ax, x + 0.28, y - 0.015, title, size=9, weight="semibold")


def prompt_score():
    fig, ax = canvas(4.00)
    label(ax, 0.12, 0.13, "01", "Fixed clinical question", BLUE)
    ax.add_patch(Rectangle((0.12, 0.45), 6.04, 0.80,
                           facecolor=LIGHT, edgecolor="none"))
    ax.plot((0.12, 0.12), (0.45, 1.25), color=BLUE, linewidth=1.5)
    text(ax, 0.29, 0.59,
         "Is there <finding> in this chest radiograph?", size=10)
    text(ax, 0.29, 0.91, "Answer yes or no.", size=10)

    label(ax, 0.12, 1.49, "02", "Six ownership findings", TEAL)
    text(ax, 6.15, 1.49, "Only the finding phrase changes", size=8,
         color=MUTED, ha="right")
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
        x, y = 0.12 + col * 3.17, 1.88 + row * 0.35
        text(ax, x, y, name, size=8.5, weight="semibold")
        text(ax, x + 1.22, y, phrase, size=8.5, color=TEAL)
        rule(ax, x, x + 2.87, y + 0.25)

    label(ax, 0.12, 3.00, "03", "First-answer scoring", ORANGE)
    centers = (0.59, 1.94, 3.57, 5.40)
    entries = (
        "Image +\nfixed question",
        "First-answer-\nposition logits",
        "Max log probability\nfor each answer over\nvalid token variants",
        r"$P(\mathrm{yes}) = \sigma(\ell_{\mathrm{yes}}-\ell_{\mathrm{no}})$",
    )
    for x, entry in zip(centers, entries):
        text(ax, x, 3.60, entry, size=8.1, ha="center", va="center")
    for start, end in ((1.08, 1.34), (2.54, 2.78), (4.38, 4.60)):
        arrow(ax, (start, 3.60), (end, 3.60))
    rule(ax, 4.65, 6.15, 3.91, color=ORANGE, width=1.2)
    save_figure(fig, "figA1_prompt_score")


def provenance():
    fig, ax = canvas(3.65)
    columns = ((0.12, 1.67, BLUE), (1.94, 3.75, TEAL),
               (4.02, 6.16, ORANGE))
    headings = ("Registered design", "Patient-level\nestimates",
                "Registered\nstatistical decision")
    for i, ((left, right, color), heading) in enumerate(zip(columns, headings)):
        text(ax, left, 0.13, f"0{i + 1}", size=8.5, weight="bold")
        rule(ax, left + 0.27, right, 0.19, color=color)
        text(ax, left, 0.37, heading, size=9.4, weight="semibold")
    arrow(ax, (1.69, 0.53), (1.85, 0.53), color=INK)
    arrow(ax, (3.77, 0.53), (3.93, 0.53), color=INK)

    text(ax, 0.12, 1.04, "Model + locus", weight="semibold", size=8.8)
    text(ax, 0.12, 1.35, "Qwen\nFinal visual block", size=8.8)
    text(ax, 0.12, 2.06, "Dose + cohort", weight="semibold", size=8.8)
    text(ax, 0.12, 2.37, "Prespecified dose rule\nand patient selection", size=8.8)

    text(ax, 1.94, 1.04, "Ownership / alias", weight="semibold", size=8.8)
    text(ax, 1.94, 1.38, "400 patients\n304,800 evaluations", size=9)
    text(ax, 4.02, 1.04, "Patient bootstrap +\nregistered comparators", size=8.8)
    text(ax, 4.02, 1.63, "0/6 owned\nNo shared alias", weight="semibold", size=9)
    arrow(ax, (3.68, 1.48), (3.92, 1.48), color=INK)

    rule(ax, 1.94, 6.16, 2.20)
    text(ax, 1.94, 2.40, "Input closure", weight="semibold", size=8.8)
    text(ax, 1.94, 2.74, "50 patient pairs\n6,450 evaluations", size=9)
    text(ax, 4.02, 2.40, "Paired bootstrap +\nclosure conjunction", size=8.8)
    text(ax, 4.02, 2.99, "No input closure", weight="semibold", size=9)
    arrow(ax, (3.68, 2.85), (3.92, 2.85), color=INK)

    rule(ax, 0.12, 6.16, 3.39)
    save_figure(fig, "figA2_provenance")


def forward_path():
    fig, ax = canvas(3.88)
    text(ax, 0.12, 0.13, "Consumed visual path", size=9.5, weight="semibold")
    text(ax, 6.16, 0.15, "Capture → intervene → measure", size=8,
         color=MUTED, ha="right")

    # The shaded region identifies the intervention locus within one forward.
    ax.add_patch(Rectangle((1.48, 0.61), 1.70, 0.86,
                           facecolor=LIGHT, edgecolor="none"))
    stages = ((0.61, "Visual\ntokens", BLUE),
              (2.33, "Consumed\nfinal block", BLUE),
              (4.09, "Connector /\nmerger", TEAL),
              (5.66, "Answer\nlogits", ORANGE))
    for x, title, color in stages:
        text(ax, x, 0.72, title, size=9, weight="semibold", ha="center")
        rule(ax, x - 0.40, x + 0.40, 1.31, color=color, width=1.2)
    for start, end in ((1.04, 1.44), (3.22, 3.55), (4.65, 5.10)):
        arrow(ax, (start, 1.04), (end, 1.04))
    text(ax, 2.33, 1.60, "Capture + intervention", size=8, color=BLUE, ha="center")
    text(ax, 4.09, 1.60, "Downstream witness", size=8, color=TEAL, ha="center")
    text(ax, 5.66, 1.60, "Endpoint witness", size=8, color=ORANGE, ha="center")

    # Dose branches annotate checks of the same hook, not extra model stages.
    ax.plot((2.33, 2.33), (1.86, 2.02), color=MUTED, linewidth=0.7)
    ax.plot((1.21, 4.53), (2.02, 2.02), color=MUTED, linewidth=0.7)
    for x in (1.21, 4.53):
        arrow(ax, (x, 2.02), (x, 2.22), color=MUTED)
    text(ax, 1.21, 2.33, r"$\alpha = 0$  ·  bitwise no-op",
         size=9, color=BLUE, ha="center", weight="semibold")
    text(ax, 4.53, 2.33, r"$\alpha \ne 0$  ·  downstream change",
         size=9, color=TEAL, ha="center", weight="semibold")
    text(ax, 1.21, 2.66, "Baseline preserved", size=8, color=MUTED, ha="center")
    text(ax, 4.53, 2.66, "Connector / merger and answer logits", size=8,
         color=MUTED, ha="center")

    rule(ax, 0.12, 6.16, 3.02)
    text(ax, 0.12, 3.20, "LLaVA", size=8.5, weight="semibold")
    text(ax, 0.90, 3.20, "encoder.layers.22", size=8.2,
         fontfamily="DejaVu Sans Mono")
    text(ax, 6.16, 3.20, "captured once", size=8, color=MUTED, ha="right")
    text(ax, 0.12, 3.58, "Qwen", size=8.5, weight="semibold")
    text(ax, 0.90, 3.58, "model.visual.blocks.31", size=8.2,
         fontfamily="DejaVu Sans Mono")
    text(ax, 6.16, 3.58, "32 blocks verified", size=8, color=MUTED, ha="right")
    save_figure(fig, "figA6_forward_path")


def main():
    apply_style()
    prompt_score()
    provenance()
    forward_path()


if __name__ == "__main__":
    main()
