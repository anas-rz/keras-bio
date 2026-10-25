"""Generate Colab notebooks from the Python snippets in the documentation.

Each documentation page with ``python`` code blocks gets a notebook under
``notebooks/`` mirroring the page path. The first cell installs ``keras-bio``
from PyPI, and every snippet becomes a code cell under its section heading.

Usage:
    python scripts/colab_notebooks.py          # regenerate notebooks
    python scripts/colab_notebooks.py --check  # fail if notebooks are stale

The file also acts as an MkDocs hook (see ``hooks`` in ``mkdocs.yml``): it adds
an "Open in Colab" badge to each page with a notebook, and warns (failing
``mkdocs build --strict``) when a committed notebook is out of date.
"""

import argparse
import json
import logging
import re
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
NOTEBOOKS_DIR = ROOT / "notebooks"
REPO = "anas-rz/keras-bio"
BRANCH = "main"

INSTALL_CELL = "!pip install keras-bio"
COLAB_BADGE = "https://colab.research.google.com/assets/colab-badge.svg"

FENCE_RE = re.compile(r"^(?P<indent>[ \t]*)```(?P<lang>[\w+-]*)")
HEADING_RE = re.compile(r"^(?P<level>#{1,6})\s+(?P<text>.+?)\s*$")
TAB_RE = re.compile(r'^===\s+"(?P<text>.+)"\s*$')

log = logging.getLogger("mkdocs.hooks.colab")


def _clean(text):
    """Strip markdown emphasis so headings read well in notebook titles."""
    return re.sub(r"[*_`]", "", text).strip()


def extract_sections(markdown):
    """Return the page title and a list of ``(heading, code)`` python snippets."""
    title = None
    heading = None
    sections = []
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        fence = FENCE_RE.match(line)
        if fence:
            indent = fence.group("indent")
            body = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                body.append(lines[i])
                i += 1
            if fence.group("lang") == "python":
                code = textwrap.dedent("\n".join(body)).strip("\n")
                sections.append((heading, code))
            i += 1
            continue
        match = HEADING_RE.match(line)
        if match:
            text = _clean(match.group("text"))
            if len(match.group("level")) == 1 and title is None:
                title = text
            else:
                heading = text
        match = TAB_RE.match(line)
        if match:
            heading = _clean(match.group("text"))
        i += 1
    return title, sections


def _source(text):
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def _markdown_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def _code_cell(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


def build_notebook(doc_path, markdown):
    """Build the notebook JSON for a doc page, or ``None`` if it has no snippets."""
    title, sections = extract_sections(markdown)
    if not sections:
        return None
    title = title or "keras-bio"
    page = doc_path.with_suffix("").as_posix()
    if page == "index":
        page = ""
    cells = [
        _markdown_cell(
            f"# {title}\n\n"
            f"Runnable examples from the [keras-bio documentation]"
            f"(https://{REPO.split('/')[0]}.github.io/{REPO.split('/')[1]}/{page}{'/' if page else ''}).\n\n"
            f"This notebook is generated from `docs/{doc_path.as_posix()}`; "
            f"edit the documentation page rather than this file."
        ),
        _code_cell(INSTALL_CELL),
    ]
    last_heading = None
    for heading, code in sections:
        if heading and heading != last_heading:
            cells.append(_markdown_cell(f"## {heading}"))
            last_heading = heading
        cells.append(_code_cell(code))
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


def notebook_path(doc_path):
    return NOTEBOOKS_DIR / doc_path.with_suffix(".ipynb")


def colab_url(doc_path):
    rel = notebook_path(doc_path).relative_to(ROOT).as_posix()
    return f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/{rel}"


def expected_notebooks():
    """Map each notebook path to its expected content for every doc page."""
    notebooks = {}
    for md_file in sorted(DOCS_DIR.rglob("*.md")):
        doc_path = md_file.relative_to(DOCS_DIR)
        content = build_notebook(doc_path, md_file.read_text(encoding="utf-8"))
        if content is not None:
            notebooks[notebook_path(doc_path)] = content
    return notebooks


def stale_notebooks():
    """Return notebooks that are missing, outdated, or no longer generated."""
    expected = expected_notebooks()
    stale = [
        path
        for path, content in expected.items()
        if not path.exists() or path.read_text(encoding="utf-8") != content
    ]
    if NOTEBOOKS_DIR.exists():
        stale += [p for p in NOTEBOOKS_DIR.rglob("*.ipynb") if p not in expected]
    return sorted(stale)


def write_notebooks():
    expected = expected_notebooks()
    if NOTEBOOKS_DIR.exists():
        for path in NOTEBOOKS_DIR.rglob("*.ipynb"):
            if path not in expected:
                path.unlink()
    for path, content in expected.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return sorted(expected)


# --- MkDocs hooks -----------------------------------------------------------


def on_pre_build(config):
    stale = stale_notebooks()
    if stale:
        names = ", ".join(p.relative_to(ROOT).as_posix() for p in stale)
        log.warning(
            "Colab notebooks are out of date (%s). "
            "Run `python scripts/colab_notebooks.py` and commit the result.",
            names,
        )


def on_page_markdown(markdown, page, config, files):
    doc_path = Path(page.file.src_uri)
    if not notebook_path(doc_path).exists():
        return markdown
    badge = (
        f'<a href="{colab_url(doc_path)}" target="_blank" class="colab-badge">'
        f'<img src="{COLAB_BADGE}" alt="Open In Colab"></a>\n\n'
    )
    # Place the badge under the page title, or under the heading that
    # introduces the first python snippet when the page has no title.
    lines = markdown.split("\n")
    anchor = None
    for i, line in enumerate(lines):
        if line.startswith("# "):
            anchor = i
            break
        if HEADING_RE.match(line):
            anchor = i
        elif FENCE_RE.match(line) and FENCE_RE.match(line).group("lang") == "python":
            break
    if anchor is None:
        return badge + markdown
    lines.insert(anchor + 1, "\n" + badge.rstrip("\n"))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="exit non-zero if notebooks are stale"
    )
    args = parser.parse_args()
    if args.check:
        stale = stale_notebooks()
        for path in stale:
            print(f"stale: {path.relative_to(ROOT).as_posix()}")
        return 1 if stale else 0
    for path in write_notebooks():
        print(f"wrote: {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
