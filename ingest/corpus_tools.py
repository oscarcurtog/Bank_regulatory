"""Read-only views of the parsed corpus for judge A (PROTOCOL.md, step 3).

The four tools judge A used in DEV batch 2, as functions that return their text instead of
printing it, with the same output byte for byte: article titles, every node of an article,
nodes in full with their heading, and a case-insensitive regex over the own text (or over
article titles). The excluded PSD2 articles are marked on every line where they appear.

Nothing here writes, ranks or embeds anything: a judge sees the corpus only in document
order or in the order of a literal match. scripts/evidence_tools.py prints these texts; the
evidence broker of TEST (ingest/holdout_broker.py) pages them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from ingest import benchmark as bm
from ingest import processed as pr

EXCLUSIONS_PATH = Path("data/benchmark/psd2_excluded_articles.json")


def _number(path: str) -> int:
    return int("".join(ch for ch in path if ch.isdigit()) or 0)


class CorpusTools:
    def __init__(
        self, docs: Mapping[str, pr.ProcessedDocument], exclusions_path: Path = EXCLUSIONS_PATH
    ) -> None:
        self.texts = bm.node_texts(docs)
        self.titles_of = {(act, a.path): a.title for act, d in docs.items() for a in d.articles}
        self.order: dict[bm.NodeKey, int] = {}
        self.article_nodes: dict[bm.NodeKey, list[str]] = {}
        for act, doc in docs.items():
            for article in doc.articles:
                for node in article.nodes:
                    self.order[(act, node.path)] = len(self.order)
                    self.article_nodes.setdefault((act, article.path), []).append(node.path)
        listed = json.loads(exclusions_path.read_text(encoding="utf-8"))
        self.psd2_excluded = {a["article"] for a in listed["articles"]}

    def excluded(self, act: str, path: str) -> bool:
        return act == "PSD2" and path.split(".")[0] in self.psd2_excluded

    def heading(self, act: str, path: str) -> str:
        article = path.split(".")[0]
        return f"{act} Article {article}: {self.titles_of.get((act, article), '?')}"

    def mark(self, act: str, path: str) -> str:
        return "  [PSD2 EXCLUDED: never a candidate]" if self.excluded(act, path) else ""

    @staticmethod
    def _text(lines: Sequence[str]) -> str:
        return "".join(f"{line}\n" for line in lines)

    def titles(self, act: str, lo: int = 0, hi: int = 10**6) -> str:
        lines = [
            f"{a} {path}: {title}{'  [PSD2 EXCLUDED]' if self.excluded(a, path) else ''}"
            for (a, path), title in sorted(
                self.titles_of.items(), key=lambda kv: (_number(kv[0][1]), kv[0][1])
            )
            if a == act and lo <= _number(path) <= hi
        ]
        return self._text(lines)

    def article(self, act: str, number: str) -> str | None:
        """None when the article does not exist (the CLI then exits with its message)."""
        paths = self.article_nodes.get((act, number))
        if not paths:
            return None
        lines = [self.heading(act, number) + self.mark(act, number)]
        lines += [f"  [{act}:{p}] {self.texts.get((act, p), '')}" for p in paths]
        return self._text(lines)

    def nodes(self, refs: Sequence[str]) -> str:
        lines = []
        for ref in refs:
            act, _, path = ref.partition(":")
            text = self.texts.get((act, path))
            shown = text if text is not None else "NO SUCH NODE (or a node without own text)"
            lines.append(f"[{ref}] {self.heading(act, path)}{self.mark(act, path)}\n  {shown}\n")
        return self._text(lines)

    def search(self, scope: str, pattern: str, in_titles: bool) -> str:
        rx = re.compile(pattern, re.I)
        if in_titles:
            return self._text(
                [
                    f"{a} Article {p}: {t}{self.mark(a, p)}"
                    for (a, p), t in self.titles_of.items()
                    if scope in ("ALL", a) and rx.search(t)
                ]
            )
        hits = [
            (k, t) for k, t in self.texts.items() if scope in ("ALL", k[0]) and t and rx.search(t)
        ]
        lines = []
        for (a, p), t in sorted(hits, key=lambda kv: self.order.get(kv[0], 0)):
            m = rx.search(t)
            assert m is not None
            lo = max(0, m.start() - 90)
            title = self.heading(a, p).split(": ", 1)[1][:50]
            lines.append(f"[{a}:{p}] ({title}){self.mark(a, p)} ...{t[lo : m.end() + 110]}...")
        lines.append(f"-- {len(hits)} nodes")
        return self._text(lines)
