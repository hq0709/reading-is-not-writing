"""One pass over the generated tables that makes every header sit the same way in its cell.

Two defects the tables carried, both from headers being written inline at forty-nine places rather than
built by one function:

  vertical    A two-row header holds group labels on the top row and their sub-labels underneath. The
              columns with no group -- Dataset, blocks, cells -- have a label on one row and a blank on
              the other, and the two builders disagree about which: table_cf_refit puts them on the top
              row, table_cf_scale on the bottom. Either way the label hugs one edge of a two-line header
              instead of sitting in the middle of it. Wrapping it in \\multirow{2}{*}{...} centres it and
              makes the two builders agree.

  horizontal  A header over a numeric column inherits the column's r and sets flush right, so a wide
              label such as NIH ChestX-ray14 stands over numbers pinned to its right edge. Centring the
              label over the column reads as a heading for the column rather than as another right-aligned
              entry in it.

The pass is deliberately conservative: it only touches a header row that it can account for column by
column, it never moves a cell that carries content on both rows (those are wrapped two-line labels, not
defects), and it leaves the body untouched.
"""
from __future__ import annotations

import re

CMIDRULE = re.compile(r"^\s*\\cmidrule")
CMIDRULE_RUN = re.compile(r"\s*(?:\\cmidrule(?:\([lr]+\))?\{\d+-\d+\}\s*)+")
MULTICOL = re.compile(r"^\\multicolumn\{(\d+)\}")


def split_cells(row: str) -> list[str]:
    """Split a LaTeX row on the ampersands that are not inside braces."""
    out, depth, cur = [], 0, ""
    for ch in row:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "&" and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return out


def _spans(cells: list[str]) -> list[tuple[int, int, str]]:
    """(first column, width, text) for each cell, following \\multicolumn spans."""
    out, col = [], 0
    for c in cells:
        t = c.strip()
        m = MULTICOL.match(t)
        w = int(m.group(1)) if m else 1
        out.append((col, w, t))
        col += w
    return out


def _ws(line: str) -> str:
    """The newline a row opens with, which rebuilding the cells drops and \\toprule needs to stay on its own line.

    Normalised to exactly one newline rather than carried through verbatim: a row whose first cell is empty
    regains a space from the join every time, so preserving the original run makes the pass grow the indent by
    one space per run instead of leaving the file alone."""
    lead = line[:len(line) - len(line.lstrip())]
    return "\n" if "\n" in lead else ""


def _join(cells) -> str:
    """Join header cells, without the space the join puts in front when the first cell is empty."""
    return " & ".join(cells).lstrip(" ")


def _centre(text: str, align: str) -> str:
    """Centre a plain header label over a column that is not already left-aligned."""
    if not text or align == "l" or text.startswith("\\multicolumn") or text.startswith("\\multirow"):
        return text
    return r"\multicolumn{1}{c}{" + text + "}"


def restyle(tex: str) -> str:
    """Return the table source with its header rows centred horizontally and spanned vertically."""
    out = []
    for block in re.split(r"(?=\\toprule)", tex):
        m = re.search(r"\\toprule(.*?)\\midrule", block, re.S)
        spec_m = re.search(r"\\begin\{(?:tabular|longtable)\}(?:\[[a-z]\])?\{([^}]*)\}", tex)
        if not m or not spec_m:
            out.append(block)
            continue
        aligns = [c for c in spec_m.group(1) if c in "lcr"]
        head = m.group(1)
        raw = head.split("\\\\")
        # a \cmidrule run carries no \\ of its own, so it arrives glued to the front of the next header row;
        # peel it off or the row behind it is mistaken for a separator and the header looks like one row
        lines, lead = [], []
        for chunk in raw:
            mm = CMIDRULE_RUN.match(chunk)
            lead.append(chunk[:mm.end()] if mm else "")
            lines.append(chunk[mm.end():] if mm else chunk)
        # index the rows that are real header rows, keeping the separators (\cmidrule, blank) in place
        idx = [i for i, ln in enumerate(lines) if ln.strip() and not CMIDRULE.match(ln) and "\\endfirsthead" not in ln
               and "\\endhead" not in ln and "\\multicolumn{" + str(len(aligns)) not in ln]
        if len(idx) == 2:
            a, b = (split_cells(lines[i]) for i in idx)
            sa, sb = _spans(a), _spans(b)
            if sa and sb and sa[-1][0] + sa[-1][1] == sb[-1][0] + sb[-1][1] == len(aligns):
                by_col_b = {c: (w, t) for c, w, t in sb}
                new_a, new_b = [], {c: t for c, _, t in sb}
                for col, w, t in sa:
                    if w == 1 and col in by_col_b and by_col_b[col][0] == 1 and not t.startswith("\\multirow"):
                        top, bot = t, by_col_b[col][1]
                        if top and not bot:                      # label on the top row, blank underneath
                            new_a.append(r"\multirow{2}{*}{" + top.strip() + "}")
                            new_b[col] = ""
                            continue
                        if bot and not top:                      # blank on top, label underneath
                            new_a.append(r"\multirow{2}{*}{" + bot.strip() + "}")
                            new_b[col] = ""
                            continue
                    new_a.append(t)
                lines[idx[0]] = _ws(lines[idx[0]]) + _join(new_a)
                lines[idx[1]] = _ws(lines[idx[1]]) + _join(new_b[c] for c, _, _ in sb)
        # centre every plain label over a column that is not left-aligned
        for i in idx:
            cells = split_cells(lines[i])
            sp = _spans(cells)
            if not sp or sp[-1][0] + sp[-1][1] != len(aligns):
                continue
            lines[i] = _ws(lines[i]) + _join(_centre(t, aligns[col]) if w == 1 else t for col, w, t in sp)
        out.append(block.replace(head, "\\\\".join(l + r for l, r in zip(lead, lines)), 1))
    return "".join(out)


def restyle_file(path) -> bool:
    """Rewrite one generated table in place; returns whether anything changed."""
    src = path.read_text()
    new = restyle(src)
    if new != src:
        path.write_text(new)
    return new != src
