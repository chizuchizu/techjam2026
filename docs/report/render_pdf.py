#!/usr/bin/env python3
"""Render TECHNICAL_REPORT.md to PDF through Typst, without an HTML intermediate."""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import tempfile


HERE = pathlib.Path(__file__).resolve().parent
SOURCE = HERE / "TECHNICAL_REPORT.md"
OUTPUT = HERE / "notgpu-attention-technical-report.pdf"

TYPST_HEADER = r'''#let ink = rgb("#101817")
#let muted = rgb("#566360")
#let rule = rgb("#d7dfdd")
#let soft = rgb("#eef3f1")
#let accent = rgb("#ee5725")
#let green = rgb("#087c62")

#set document(title: "NotGPU Attention — Concise Technical Project Report",
  author: "TinyCluster")
#set page(
  paper: "a4",
  margin: (top: 17mm, bottom: 17mm, left: 18mm, right: 18mm),
  header: context [
    #set text(font: "Noto Sans", size: 7.5pt, fill: muted)
    NOTGPU ATTENTION #h(1fr) TECHJAM 2026
  ],
  footer: context [
    #set text(font: "Noto Sans", size: 7.5pt, fill: muted)
    #h(1fr) #counter(page).display("1")
  ],
)
#set text(font: "Noto Sans", size: 9.5pt, fill: ink)
#set par(justify: true, leading: 0.68em)
#set heading(numbering: none)
#set list(indent: 14pt, body-indent: 6pt, spacing: 4pt)
#set enum(indent: 14pt, body-indent: 6pt, spacing: 4pt)
#set table(inset: (x: 5pt, y: 4pt), stroke: 0.45pt + rule)
#show link: set text(fill: green)
#show raw: set text(font: "DejaVu Sans Mono", size: 8.2pt)
#show heading.where(level: 1): it => block(above: 6pt, below: 10pt)[
  #text(size: 27pt, weight: "bold", fill: ink)[#it.body]
  #v(4pt)
  #line(length: 100%, stroke: 2.2pt + accent)
]
#show heading.where(level: 2): it => block(above: 13pt, below: 6pt,
  breakable: false)[
  #text(size: 17pt, weight: "bold", fill: ink)[#it.body]
]
#show heading.where(level: 3): it => block(above: 9pt, below: 4pt,
  breakable: false)[
  #text(size: 11.5pt, weight: "bold", fill: accent)[#it.body]
]
#show figure.caption: set text(size: 8pt, fill: muted)
'''


def escape_text(value: str) -> str:
    """Escape characters that have special meaning in Typst markup."""
    for old, new in (("\\", "\\\\"), ("#", "\\#"), ("$", "\\$"),
                     ("_", "\\_"), ("[", "\\["), ("]", "\\]")):
        value = value.replace(old, new)
    return value


INLINE = re.compile(r"`([^`]+)`|\[([^\]]+)\]\(([^)]+)\)|\*\*([^*]+)\*\*")


def inline(value: str) -> str:
    """Convert the small inline-Markdown subset used by the report."""
    out: list[str] = []
    cursor = 0
    for match in INLINE.finditer(value):
        out.append(escape_text(value[cursor:match.start()]))
        code, label, url, bold = match.groups()
        if code is not None:
            out.append(f"`{code}`")
        elif label is not None and url is not None:
            safe_url = url.replace("\\", "\\\\").replace('"', '\\"')
            out.append(f'#link("{safe_url}")[{inline(label)}]')
        else:
            out.append(f"*{inline(bold)}*")
        cursor = match.end()
    out.append(escape_text(value[cursor:]))
    return "".join(out)


def is_table_rule(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def table_to_typst(lines: list[str]) -> str:
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in lines
    ]
    header, body = rows[0], rows[2:]
    count = len(header)
    widths = {
        2: "(0.9fr, 2.4fr)",
        5: "(0.55fr, 1.15fr, 1.15fr, 1.35fr, 0.9fr)",
    }.get(count, str(count))
    cells = [f"[{inline(cell)}]" for row in body for cell in row]
    heads = ", ".join(f"[*{inline(cell)}*]" for cell in header)
    body_text = ",\n  ".join(cells)
    return (
        "#table(\n"
        f"  columns: {widths},\n"
        "  fill: (_, y) => if y == 0 { accent } else if calc.even(y) { soft },\n"
        "  table.header(\n"
        f"    {heads},\n"
        "  ),\n"
        f"  {body_text},\n"
        ")"
    )


def markdown_to_typst(markdown: str) -> str:
    lines = markdown.splitlines()
    out = [TYPST_HEADER]
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            out.append("")
            i += 1
            continue
        if stripped == "---":
            out.append("#pagebreak()")
            i += 1
            continue
        image_match = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if image_match:
            caption, path = image_match.groups()
            safe_path = path.replace("\\", "/").replace('"', '\\"')
            out.append(
                f'#figure(image("{safe_path}", width: 100%), '
                f'caption: [{inline(caption)}])'
            )
            i += 1
            continue
        heading_match = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading_match:
            marks, title = heading_match.groups()
            out.append(f"{'=' * len(marks)} {inline(title)}")
            i += 1
            continue
        if stripped.startswith("|") and i + 1 < len(lines) and is_table_rule(lines[i + 1]):
            table_lines = [line, lines[i + 1]]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            out.append(table_to_typst(table_lines))
            continue
        if stripped.startswith("- "):
            while i < len(lines) and lines[i].strip().startswith("- "):
                out.append(f"- {inline(lines[i].strip()[2:])}")
                i += 1
            continue
        if re.match(r"^\d+\.\s+", stripped):
            while i < len(lines) and re.match(r"^\d+\.\s+", lines[i].strip()):
                item = re.sub(r"^\d+\.\s+", "", lines[i].strip())
                out.append(f"+ {inline(item)}")
                i += 1
            continue
        if stripped.startswith("```"):
            block = [stripped]
            i += 1
            while i < len(lines):
                block.append(lines[i])
                if lines[i].strip() == "```":
                    i += 1
                    break
                i += 1
            out.append("\n".join(block))
            continue

        paragraph = [stripped]
        i += 1
        while i < len(lines):
            candidate = lines[i].strip()
            if not candidate or candidate == "---" or candidate.startswith(("#", "- ", "|", "```", "![")):
                break
            if re.match(r"^\d+\.\s+", candidate):
                break
            paragraph.append(candidate)
            i += 1
        out.append(inline(" ".join(paragraph)))
    return "\n".join(out) + "\n"


def main() -> None:
    typst = shutil.which("typst")
    if not typst:
        raise SystemExit("Typst is required: https://typst.app/open-source/")
    generated = markdown_to_typst(SOURCE.read_text(encoding="utf-8"))
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".typ", prefix=".technical-report-", dir=HERE,
        encoding="utf-8", delete=False,
    ) as handle:
        handle.write(generated)
        temporary = pathlib.Path(handle.name)
    try:
        subprocess.run(
            [typst, "compile", "--root", str(HERE.parent.parent),
             str(temporary), str(OUTPUT)],
            check=True,
        )
    finally:
        temporary.unlink(missing_ok=True)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
