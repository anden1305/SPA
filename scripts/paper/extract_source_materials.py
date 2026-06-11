#!/usr/bin/env python3
"""Extract and cross-check thesis PDF, project docs, plans, and framing docx.

Runs multiple PDF engines and flags disagreements. Output is for paper drafting /
gap analysis — not a substitute for reading the thesis.

Usage:
  PYTHONPATH=. python3 scripts/paper/extract_source_materials.py
  PYTHONPATH=. python3 scripts/paper/extract_source_materials.py --thesis-only
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "docs/paper/source_extraction"

# Anchors we expect in the MSc thesis (cross-engine consensus check).
THESIS_ANCHORS = [
    "Characterizing Sleep Patterns",
    "Oliver Rosbæk Elmgreen",
    "Andreas Holmer Bigom",
    "population NMI of about 0.51",
    "population NMI of about 0.70",
    "conditional Gaussian mixture variational autoencoder",
    "leave one subject out",
    "Mouse Sleep Staging Validation",
    "Table 10",
    "decoder-only",
    "subject conditioned",
]

# Numbers / claims used in plos_my_paper.tex — checked against thesis extraction.
PAPER_CLAIM_ANCHORS = [
    ("thesis_hmm_nmi", r"0\.51"),
    ("thesis_cgmvae_nmi", r"0\.70"),
    ("decoder_only", r"decoder.?only|decoder only"),
    ("zero_shot", r"zero.?shot|leave.?one.?subject|leave one subject out"),
    ("mssv", r"Mouse Sleep Staging Validation|ds006366|MSSV"),
]

DOCX_STALE_NOTE = (
    "Take with a grain of salt — early special-course framing; superseded by "
    "holdout ladder results and docs/paper/*.md."
)

PLAN_GLOBS = [
    ".cursor/plans/5_page_paper_plan.md",
    ".cursor/plans/plos_manuscript_draft_f5977e3b.plan.md",
]

DOC_PATHS = [
    "docs/paper/deep_research_synthesis.md",
    "docs/paper/notes/holdout_experiments.md",
    "docs/paper/related_work_novelty.md",
    "docs/paper/k_sweep_experiments.md",
    "docs/cv4fold/unified_holdout_paper_line.md",
    "paper/overleaf/tables/holdout_ladder.csv",
    "paper/overleaf/tables/holdout_ladder_summary.json",
]


@dataclass
class ExtractResult:
    engine: str
    ok: bool
    page_count: int = 0
    char_count: int = 0
    word_count: int = 0
    text_path: str | None = None
    error: str | None = None
    anchors_found: list[str] = field(default_factory=list)


def _norm(s: str) -> str:
    s = s.replace("\u00ad", "")  # soft hyphen
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _words(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s))


def _anchors_in(text: str, anchors: list[str]) -> list[str]:
    low = text.lower()
    found = []
    for a in anchors:
        if a.lower() in low:
            found.append(a)
    return found


def extract_pymupdf(pdf: Path, out_dir: Path) -> ExtractResult:
    try:
        import fitz  # pymupdf
    except ImportError as e:
        return ExtractResult("pymupdf", False, error=str(e))

    try:
        doc = fitz.open(pdf)
        pages = [_norm(doc[i].get_text("text")) for i in range(len(doc))]
        text = "\n\n".join(pages)
        path = out_dir / "thesis_pymupdf.txt"
        path.write_text(text, encoding="utf-8")
        return ExtractResult(
            engine="pymupdf",
            ok=True,
            page_count=len(doc),
            char_count=len(text),
            word_count=_words(text),
            text_path=str(path.relative_to(REPO)),
            anchors_found=_anchors_in(text, THESIS_ANCHORS),
        )
    except Exception as e:
        return ExtractResult("pymupdf", False, error=str(e))


def extract_pdfplumber(pdf: Path, out_dir: Path) -> ExtractResult:
    try:
        import pdfplumber
    except ImportError as e:
        return ExtractResult("pdfplumber", False, error=str(e))

    try:
        pages: list[str] = []
        with pdfplumber.open(pdf) as doc:
            for page in doc.pages:
                pages.append(_norm(page.extract_text() or ""))
        text = "\n\n".join(pages)
        path = out_dir / "thesis_pdfplumber.txt"
        path.write_text(text, encoding="utf-8")
        return ExtractResult(
            engine="pdfplumber",
            ok=True,
            page_count=len(pages),
            char_count=len(text),
            word_count=_words(text),
            text_path=str(path.relative_to(REPO)),
            anchors_found=_anchors_in(text, THESIS_ANCHORS),
        )
    except Exception as e:
        return ExtractResult("pdfplumber", False, error=str(e))


def extract_pypdf(pdf: Path, out_dir: Path) -> ExtractResult:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        return ExtractResult("pypdf", False, error=str(e))

    try:
        reader = PdfReader(str(pdf))
        pages = [_norm(p.extract_text() or "") for p in reader.pages]
        text = "\n\n".join(pages)
        path = out_dir / "thesis_pypdf.txt"
        path.write_text(text, encoding="utf-8")
        return ExtractResult(
            engine="pypdf",
            ok=True,
            page_count=len(pages),
            char_count=len(text),
            word_count=_words(text),
            text_path=str(path.relative_to(REPO)),
            anchors_found=_anchors_in(text, THESIS_ANCHORS),
        )
    except Exception as e:
        return ExtractResult("pypdf", False, error=str(e))


def cross_check_pdf(results: list[ExtractResult]) -> dict[str, Any]:
    ok = [r for r in results if r.ok]
    report: dict[str, Any] = {"engines_ok": len(ok), "engines_total": len(results)}

    if not ok:
        report["status"] = "FAIL — no PDF engine succeeded"
        return report

    pages = [r.page_count for r in ok]
    chars = [r.char_count for r in ok]
    report["page_counts"] = {r.engine: r.page_count for r in ok}
    report["char_counts"] = {r.engine: r.char_count for r in ok}
    report["page_count_agree"] = len(set(pages)) == 1
    report["char_count_spread"] = max(chars) - min(chars)
    report["char_count_spread_pct"] = round(100 * report["char_count_spread"] / max(max(chars), 1), 2)

    # Per-anchor: how many engines found it
    anchor_votes: dict[str, int] = {}
    for r in ok:
        for a in r.anchors_found:
            anchor_votes[a] = anchor_votes.get(a, 0) + 1
    report["anchor_votes"] = anchor_votes
    report["anchors_missing"] = [a for a in THESIS_ANCHORS if anchor_votes.get(a, 0) == 0]
    report["anchors_weak"] = [a for a in THESIS_ANCHORS if 0 < anchor_votes.get(a, 0) < len(ok)]

    # Pairwise text similarity vs first OK engine (canonical)
    canonical = ok[0]
    if canonical.text_path:
        canon_text = (REPO / canonical.text_path).read_text(encoding="utf-8", errors="replace")
        sims = {}
        for r in ok[1:]:
            if r.text_path:
                other = (REPO / r.text_path).read_text(encoding="utf-8", errors="replace")
                sims[r.engine] = round(SequenceMatcher(None, canon_text[:500_000], other[:500_000]).ratio(), 4)
        report["text_similarity_to_canonical"] = sims

    legacy = REPO / "Master Thesis.txt"
    if legacy.is_file() and canonical.text_path:
        legacy_text = legacy.read_text(encoding="utf-8", errors="replace")
        canon_text = (REPO / canonical.text_path).read_text(encoding="utf-8", errors="replace")
        report["legacy_txt_similarity"] = round(
            SequenceMatcher(None, legacy_text[:500_000], canon_text[:500_000]).ratio(), 4
        )

    report["status"] = "PASS" if report["page_count_agree"] and report["char_count_spread_pct"] < 15 else "WARN"
    return report


def extract_docx(docx: Path, out_dir: Path) -> dict[str, Any]:
    try:
        from docx import Document
    except ImportError as e:
        return {"ok": False, "error": str(e)}

    if not docx.is_file():
        return {"ok": False, "error": f"not found: {docx}"}

    doc = Document(docx)
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    text = "\n\n".join(paras)
    path = out_dir / "special_course_framing.txt"
    path.write_text(text, encoding="utf-8")
    return {
        "ok": True,
        "path": str(docx.relative_to(REPO)),
        "paragraphs": len(paras),
        "char_count": len(text),
        "word_count": _words(text),
        "text_path": str(path.relative_to(REPO)),
        "stale_warning": DOCX_STALE_NOTE,
        "preview": text[:2000],
    }


def load_text_sources() -> dict[str, Any]:
    out: dict[str, Any] = {"plans": {}, "docs": {}, "manuscript": {}}
    for rel in PLAN_GLOBS:
        p = REPO / rel
        if p.is_file():
            out["plans"][rel] = {"words": _words(p.read_text(encoding="utf-8")), "exists": True}
    for rel in DOC_PATHS:
        p = REPO / rel
        if p.is_file():
            t = p.read_text(encoding="utf-8")
            out["docs"][rel] = {"words": _words(t), "exists": True}
    ms = REPO / "docs/paper/plos_my_paper.tex"
    if ms.is_file():
        t = ms.read_text(encoding="utf-8")
        out["manuscript"] = {"words": _words(t), "has_author_summary": "Author summary" in t}
    return out


def check_paper_claims(thesis_text: str) -> list[dict[str, Any]]:
    low = thesis_text.lower()
    rows = []
    for name, pattern in PAPER_CLAIM_ANCHORS:
        rows.append({
            "claim": name,
            "pattern": pattern,
            "in_thesis": bool(re.search(pattern, low, re.I)),
        })
    return rows


def extract_thesis_sections(pymupdf_text: str, out_dir: Path) -> dict[str, str]:
    """Heuristic section splits from pymupdf full text."""
    sections = {}
    markers = [
        ("abstract", r"\bAbstract\b"),
        ("introduction", r"\bIntroduction\b"),
        ("discussion", r"\bDiscussion\b"),
        ("conclusion", r"\bConclusion\b"),
    ]
    for name, pat in markers:
        m = re.search(pat, pymupdf_text)
        if m:
            start = m.start()
            sections[name] = pymupdf_text[start : start + 8000]
    path = out_dir / "thesis_sections.json"
    path.write_text(json.dumps({k: len(v) for k, v in sections.items()}, indent=2), encoding="utf-8")
    return {k: str(len(v)) for k, v in sections.items()}


def write_markdown_report(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Source material extraction report",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Honest answer: was the thesis PDF read for `plos_my_paper.tex`?",
        "",
        "The first manuscript draft was built from **holdout CSV/JSON results**, ",
        "`docs/paper/*.md`, cursor plans, and **cited thesis numbers** (HMM NMI ≈ 0.51, ",
        "cGMVAE ≈ 0.70). It did **not** run multi-engine PDF extraction or a ",
        "section-by-section thesis→paper mapping. Use this report to close that gap.",
        "",
        "## Thesis PDF cross-check",
        "",
        f"**File:** `{payload['thesis_pdf']}`",
        "",
        f"**Status:** {payload['pdf_cross_check'].get('status', 'n/a')}",
        "",
    ]

    cc = payload["pdf_cross_check"]
    if cc.get("page_counts"):
        lines.append("| Engine | Pages | Chars | Words |")
        lines.append("|--------|-------|-------|-------|")
        for r in payload["pdf_results"]:
            if r["ok"]:
                lines.append(f"| {r['engine']} | {r['page_count']} | {r['char_count']} | {r['word_count']} |")
        lines.append("")
        lines.append(f"- Page counts agree: **{cc.get('page_count_agree')}**")
        lines.append(f"- Char spread: **{cc.get('char_count_spread_pct')}%**")
        if cc.get("legacy_txt_similarity") is not None:
            lines.append(f"- Similarity to `Master Thesis.txt`: **{cc['legacy_txt_similarity']}**")
        lines.append("")

    missing = cc.get("anchors_missing") or []
    weak = cc.get("anchors_weak") or []
    if missing:
        lines.append("### Anchors missing from all engines")
        for a in missing:
            lines.append(f"- {a}")
        lines.append("")
    if weak:
        lines.append("### Anchors found by only some engines (OCR/layout drift)")
        for a in weak:
            lines.append(f"- {a} ({cc['anchor_votes'].get(a, 0)} engines)")
        lines.append("")

    lines.extend(["## Paper claims vs thesis text", ""])
    for row in payload.get("paper_claim_checks", []):
        mark = "yes" if row["in_thesis"] else "**NO**"
        lines.append(f"- `{row['claim']}`: {mark}")
    lines.append("")

    docx = payload.get("docx", {})
    lines.extend([
        "## Special Course framing (docx)",
        "",
        f"**Stale warning:** {docx.get('stale_warning', DOCX_STALE_NOTE)}",
        "",
    ])
    if docx.get("ok"):
        lines.append(f"- Paragraphs: {docx['paragraphs']}, words: {docx['word_count']}")
        lines.append(f"- Extract: `{docx.get('text_path')}`")
        lines.append("")
        lines.append("### Preview (first ~500 chars)")
        lines.append("```")
        lines.append(docx.get("preview", "")[:500])
        lines.append("```")
    else:
        lines.append(f"- Extract failed: {docx.get('error')}")
    lines.append("")

    lines.extend(["## Cursor plans loaded", ""])
    for rel, meta in payload.get("text_sources", {}).get("plans", {}).items():
        lines.append(f"- `{rel}` ({meta['words']} words)")
    lines.append("")

    lines.extend([
        "## Re-run",
        "",
        "```bash",
        "pip install -r scripts/paper/requirements-extract.txt",
        "PYTHONPATH=. python3 scripts/paper/extract_source_materials.py",
        "```",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thesis", type=Path, default=REPO / "Master Thesis.pdf")
    parser.add_argument("--docx", type=Path, default=REPO / "docs/google_sheets/Special Course - Project framing.docx")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--thesis-only", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    pdf_results = [
        extract_pymupdf(args.thesis, args.out),
        extract_pdfplumber(args.thesis, args.out),
        extract_pypdf(args.thesis, args.out),
    ]
    cross = cross_check_pdf(pdf_results)

    canonical_text = ""
    for r in pdf_results:
        if r.ok and r.text_path:
            canonical_text = (REPO / r.text_path).read_text(encoding="utf-8", errors="replace")
            break

    sections = extract_thesis_sections(canonical_text, args.out) if canonical_text else {}
    paper_checks = check_paper_claims(canonical_text) if canonical_text else []

    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "thesis_pdf": str(args.thesis.relative_to(REPO)) if args.thesis.is_relative_to(REPO) else str(args.thesis),
        "pdf_results": [asdict(r) for r in pdf_results],
        "pdf_cross_check": cross,
        "thesis_sections_chars": sections,
        "paper_claim_checks": paper_checks,
    }

    if not args.thesis_only:
        payload["docx"] = extract_docx(args.docx, args.out)
        payload["text_sources"] = load_text_sources()

    json_path = args.out / "extraction_report.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown_report(payload, args.out / "extraction_report.md")

    print(f"Wrote {json_path.relative_to(REPO)}")
    print(f"Wrote {(args.out / 'extraction_report.md').relative_to(REPO)}")
    print(f"PDF cross-check: {cross.get('status')}")
    for r in pdf_results:
        if r.ok:
            print(f"  {r.engine}: {r.page_count} pages, {r.word_count} words, {len(r.anchors_found)}/{len(THESIS_ANCHORS)} anchors")
        else:
            print(f"  {r.engine}: FAILED — {r.error}")
    return 0 if cross.get("engines_ok", 0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
