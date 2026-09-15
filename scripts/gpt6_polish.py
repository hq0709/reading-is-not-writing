"""Paragraph-by-paragraph writing pass with GPT-6-Astra through the side-by-side Codex CLI (codex154 exec).

Every prose paragraph of the listed sections is sent with the author's constraints; a rewrite is accepted only if it
keeps every number, citation key, reference label, and math span, adds no hedging phrase, and is not longer than the
original. Accepted rewrites are applied in place; a before/after log is written to review-artifacts/.
"""
import re, subprocess, sys, json, time, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODEX = "/rodata/azradonc_dev/m253405/fake_home/.local/bin/codex154"
SECTIONS = sys.argv[1:] or ["0_abstract", "1_introduction", "2_related_work", "3_framework", "4_setup", "5_results", "6_implications", "7_conclusion"]
LOG = ROOT / "review-artifacts" / f"gpt6_polish_{time.strftime('%Y%m%d_%H%M')}.md"
BANNED = ["we do not claim", "limitation", "cannot rule out", "may not", "might not", "we acknowledge", "caveat", "however, we", "it is possible that", "does not establish", "we cannot"]
SKIP_START = ("\\section", "\\subsection", "\\begin{", "\\end{", "\\input", "\\label", "\\centering", "\\includegraphics", "\\caption", "\\FloatBarrier", "%", "\\item", "\\end")

CONSTRAINTS = """You are editing one paragraph of an ICLR submission (LaTeX). Rewrite it for clarity, precision, and force, paragraph-internal only. Return ONLY the rewritten LaTeX paragraph: no preamble, no commentary, no code fences.

Hard constraints:
- Keep every number, percentage, interval, model name, dataset name, \\cite key, \\ref/\\eqref/\\label, \\emph and \\textbf markup, \\paragraph{...} heading, and every math span exactly as given. Do not add, remove, or change any factual claim or any number.
- The paper has ONE core thesis and it stays at full strength: a readable clinical probe direction is not a handle on the concept; written back it moves other findings' answers as much as its own; the same write is a handle for natural objects (the deficit is specific to clinical concepts); the direction the model answers along exists in the same representation and is nearly orthogonal to the probe direction on chest radiographs; and what a written direction does is decided by the reader of the representation (connector and language model), not by the representation. Never weaken, hedge, or narrow it.
- No hedges, no limitation sentences, no "we do not claim", no defensive framing, no apologies, no filler ("note that", "it is worth noting", "importantly").
- Plain declarative sentences, one idea each; prefer active voice; British spelling as in the source; keep LaTeX valid.
- Length: the same or shorter (at most 5% longer is rejected).

Paragraph:
"""


def prose_blocks(text):
    blocks = text.split("\n\n")
    out = []
    for b in blocks:
        s = b.strip()
        if not s or s.startswith(SKIP_START) or "\\begin{" in s or "\\end{" in s or len(s) < 200:
            out.append((b, False))
        else:
            out.append((b, True))
    return out


def tokens(s):
    nums = re.findall(r"(?<![A-Za-z\\])-?\d+(?:[.,]\d+)*(?:\\%)?", s)
    cites = re.findall(r"\\cite[pt]?\{([^}]*)\}", s)
    refs = re.findall(r"\\(?:eq)?ref\{([^}]*)\}", s)
    labels = re.findall(r"\\label\{([^}]*)\}", s)
    return sorted(nums), sorted(cites), sorted(refs), sorted(labels)


def polish(paragraph):
    prompt = CONSTRAINTS + paragraph.strip()
    out = ROOT / "review-artifacts" / ".gpt6_last.txt"
    cmd = [CODEX, "exec", "-m", "gpt-6-astra", "-c", 'model_reasoning_effort="xhigh"', "--sandbox", "read-only", "--skip-git-repo-check",
           "--output-last-message", str(out), prompt]
    try:
        subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return None
    if not out.exists():
        return None
    new = out.read_text().strip()
    out.unlink(missing_ok=True)
    new = re.sub(r"^```(?:latex)?\s*|\s*```$", "", new).strip()
    return new or None


def main():
    LOG.parent.mkdir(exist_ok=True)
    log = [f"# GPT-6-Astra paragraph pass {time.strftime('%Y-%m-%d %H:%M')}\n"]
    stats = {"sent": 0, "accepted": 0, "rejected": 0}
    for sec in SECTIONS:
        p = ROOT / "sections" / f"{sec}.tex"; text = p.read_text()
        blocks = prose_blocks(text); new_blocks = []
        for b, is_prose in blocks:
            if not is_prose:
                new_blocks.append(b); continue
            stats["sent"] += 1
            new = polish(b)
            reason = None
            if new is None:
                reason = "no response"
            elif tokens(new) != tokens(b):
                reason = "tokens changed"
            elif len(new) > 1.05 * len(b.strip()):
                reason = "longer"
            elif any(ph in new.lower() and ph not in b.lower() for ph in BANNED):
                reason = "hedge added"
            elif new.count("{") != new.count("}") or new.count("$") % 2:
                reason = "unbalanced LaTeX"
            log.append(f"\n## {sec} paragraph {stats['sent']} — {'ACCEPTED' if reason is None else 'REJECTED: ' + reason}\n\n**before**\n\n{b.strip()}\n\n**after**\n\n{new or ''}\n")
            if reason is None:
                new_blocks.append(new); stats["accepted"] += 1
            else:
                new_blocks.append(b); stats["rejected"] += 1
            LOG.write_text("".join(log))
            print(f"[{sec}] paragraph {stats['sent']}: {'accepted' if reason is None else 'rejected (' + reason + ')'}", flush=True)
        p.write_text("\n\n".join(new_blocks))
    log.append(f"\n\n{json.dumps(stats)}\n"); LOG.write_text("".join(log))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
