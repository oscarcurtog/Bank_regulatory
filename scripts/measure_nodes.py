"""T2, measurement step: own text per node, tokens and profile. It does NOT chunk.

1. Parses the corpus with ingest.hierarchy (own_text per node).
2. INVARIANT: rebuilds every article as
       "Article <path> <title>" + sum( reinserted_marker + own_text ) in document order
   and compares it word by word with Article.text, which comes from an independent
   code path (string cleaning). Reports losses, duplicates and articles with a diff.
3. INDEPENDENT CONTROL against the raw XHTML: how many "Article N(M)" references there
   are in the article's raw text and how many survive in Article.text. Detects the
   1.2.0 bug without depending on any cleaning function.
4. Tokens with cl100k_base (D-005, provisional). own_tokens and inclusive_tokens.
5. Profile. Writes data/t2_nodes.json and data/t2_node_profile.json.

Usage: .venv/bin/python scripts/measure_nodes.py
"""

from __future__ import annotations

import difflib
import json
import re
import statistics as st
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import tiktoken

from ingest import download as dl
from ingest import hierarchy as h

NODES_OUT = Path("data/t2_nodes.json")
PROFILE_OUT = Path("data/t2_node_profile.json")
REF_RX = re.compile(r"Articles? \d+[a-z]?\(\d{1,2}\)")
ENC = tiktoken.get_encoding("cl100k_base")


def reinsert(nd: h.Node) -> str:
    if nd.kind == "apartado":
        return f"{nd.marker}. {nd.own_text}"
    if nd.kind == "punto":
        return f"({nd.marker}) {nd.own_text}"
    return nd.own_text


def reconstruct(a: h.Article) -> str:
    head = f"Article {a.path} {a.title}"
    return h._normalize(" ".join([head] + [reinsert(nd) for nd in a.nodes]))


def word_diff(expected: str, got: str) -> tuple[list[str], list[str]]:
    e, g = expected.split(), got.split()
    lost, extra = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, e, g, autojunk=False).get_opcodes():
        if tag in ("delete", "replace"):
            lost.extend(e[i1:i2])
        if tag in ("insert", "replace"):
            extra.extend(g[j1:j2])
    return lost, extra


def raw_article_text(xhtml_path: str, article_id: str) -> str:
    root = ET.parse(xhtml_path).getroot()
    el = next(e for e in root.iter() if e.get("id") == article_id)
    return re.sub(r"\s+", " ", "".join(el.itertext()))


def pct(v: list[int]) -> int:
    v = sorted(v)
    return v[int(0.95 * len(v)) - 1] if v else 0


def main() -> None:
    docs = dl.load_manifest()
    # Payloads bound for JSON, heterogeneous values (str, int, list). Any is
    # the honest type: EF-3 decided that these dicts carry no schema because
    # they have no code consumer.
    records: list[dict[str, Any]] = []
    invariant: list[dict[str, Any]] = []
    rawcheck: list[dict[str, Any]] = []

    for act in ("DORA", "MiCA", "PSD2"):
        src = dl.parsed_source_for(act, docs)
        family = "consolidado" if src.role == "consolidated" else "doue"
        for a in h.parse(src.local_path, family):
            # ---- reconstruction invariant
            rec = reconstruct(a)
            lost, extra = word_diff(a.text, rec)
            invariant.append(
                {
                    "act": act,
                    "article_id": a.article_id,
                    "tokens_text": len(ENC.encode(a.text)),
                    "tokens_rec": len(ENC.encode(rec)),
                    "words_lost": len(lost),
                    "words_extra": len(extra),
                    "lost_sample": lost[:8],
                    "extra_sample": extra[:8],
                }
            )
            # ---- independent control against the raw text
            raw_refs = len(REF_RX.findall(raw_article_text(src.local_path, a.article_id)))
            rawcheck.append(
                {
                    "act": act,
                    "article_id": a.article_id,
                    "refs_raw": raw_refs,
                    "refs_text": len(REF_RX.findall(a.text)),
                }
            )
            # ---- tokens per node
            own = {nd.path: len(ENC.encode(nd.own_text)) for nd in a.nodes}
            for nd in a.nodes:
                incl = sum(t for p, t in own.items() if p == nd.path or p.startswith(nd.path + "."))
                records.append(
                    {
                        "act": act,
                        "celex": src.celex,
                        "source_sha256": src.sha256,
                        "article_id": a.article_id,
                        "path": nd.path,
                        "level": nd.level,
                        "marker": nd.marker,
                        "kind": nd.kind,
                        "parser_version": h.PARSER_VERSION,
                        "own_text": nd.own_text,
                        "own_tokens": own[nd.path],
                        "inclusive_tokens": incl,
                    }
                )

    NODES_OUT.write_text(json.dumps(records, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    # ================= INVARIANT =================
    bad = [r for r in invariant if r["words_lost"] or r["words_extra"]]
    print("=" * 78)
    print("INVARIANT: reconstruction from nodes (marker reinserted) vs Article.text")
    print("=" * 78)
    print(f"articles: {len(invariant)}   with a difference != 0: {len(bad)}")
    print(
        f"words lost (in text, not in the reconstruction): {sum(r['words_lost'] for r in invariant)}"
    )
    print(
        f"extra words (in the reconstruction, not in text): {sum(r['words_extra'] for r in invariant)}"
    )
    print(
        f"tokens text / reconstruction: {sum(r['tokens_text'] for r in invariant)} / {sum(r['tokens_rec'] for r in invariant)}"
    )
    if bad:
        print("\ndetail of ALL the articles with a difference (none hidden):")
        for r in bad:
            print(
                f"  {r['act']:5} {r['article_id']:9} lost={r['words_lost']:3d} extra={r['words_extra']:3d}"
                f"  lost={r['lost_sample']}  extra={r['extra_sample']}"
            )

    # ================= RAW CONTROL =================
    print("\n" + "=" * 78)
    print("INDEPENDENT CONTROL: 'Article N(M)' references in the raw text vs in Article.text")
    print("=" * 78)
    for act in ("DORA", "MiCA", "PSD2"):
        rs = [r for r in rawcheck if r["act"] == act]
        a_raw, a_txt = sum(r["refs_raw"] for r in rs), sum(r["refs_text"] for r in rs)
        mism = [r for r in rs if r["refs_raw"] != r["refs_text"]]
        print(
            f"  {act:5} raw={a_raw:4d}  text={a_txt:4d}  articles with a mismatch={len(mism)}"
            + (
                f"  e.g.: {[(m['article_id'], m['refs_raw'], m['refs_text']) for m in mism[:4]]}"
                if mism
                else ""
            )
        )

    # ================= PROFILE =================
    def profile(key: str) -> dict[str, Any]:
        # r[key] is Any because records is a heterogeneous JSON payload; here it
        # is always an integer token count, and the local type declares it.
        v: list[int] = [r[key] for r in records]
        by_level: dict[int, list[int]] = {}
        for r in records:
            by_level.setdefault(r["level"], []).append(r[key])
        return {
            "total_nodes": len(v),
            "median": st.median(v),
            "p95": pct(v),
            "max": max(v),
            "by_level": {
                lv: {"n": len(x), "median": st.median(x), "p95": pct(x), "max": max(x)}
                for lv, x in sorted(by_level.items())
            },
            "exceeding": {
                cap: {
                    "n": sum(1 for t in v if t > cap),
                    "pct": round(100 * sum(1 for t in v if t > cap) / len(v), 1),
                }
                for cap in (300, 400, 500)
            },
            "largest": [
                {
                    k: r[k]
                    for k in (
                        "act",
                        "article_id",
                        "path",
                        "level",
                        "own_tokens",
                        "inclusive_tokens",
                    )
                }
                for r in sorted(records, key=lambda r: r[key], reverse=True)[:12]
            ],
        }

    empty_l1 = [r for r in records if r["level"] == 1 and r["own_tokens"] <= 3]
    prof: dict[str, Any] = {
        "parser_version": h.PARSER_VERSION,
        "tokenizer": "cl100k_base (D-005, provisional)",
        "own_tokens": profile("own_tokens"),
        "inclusive_tokens": profile("inclusive_tokens"),
        "level1_own_text_empty_or_nearly": {
            "n": len(empty_l1),
            "of": sum(1 for r in records if r["level"] == 1),
            "examples": [f"{r['act']}/{r['article_id']}" for r in empty_l1[:6]],
        },
        "invariant": {
            "articles": len(invariant),
            "with_difference": len(bad),
            "total_words_lost": sum(r["words_lost"] for r in invariant),
            "total_words_extra": sum(r["words_extra"] for r in invariant),
            "detail": bad,
        },
        "raw_control": rawcheck,
    }
    PROFILE_OUT.write_text(json.dumps(prof, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"NODE PROFILE  (tokenizer {prof['tokenizer']}, parser {h.PARSER_VERSION})")
    print("=" * 78)
    for key in ("own_tokens", "inclusive_tokens"):
        P = prof[key]
        print(f"\n--- {key} ---")
        print(
            f"  total nodes {P['total_nodes']}   median {P['median']:.0f}   p95 {P['p95']}   max {P['max']}"
        )
        print(
            "  by level:   "
            + "   ".join(
                f"L{lv}: n={d['n']} med={d['median']:.0f} p95={d['p95']} max={d['max']}"
                for lv, d in P["by_level"].items()
            )
        )
        print(
            "  exceeding:  "
            + "   ".join(f">{cap}: {d['n']} ({d['pct']}%)" for cap, d in P["exceeding"].items())
        )
        print("  largest:")
        for m in P["largest"][:8]:
            print(
                f"    {m['act']:5} {m['article_id']:9} {m['path']:12} L{m['level']}  own={m['own_tokens']:5d}  incl={m['inclusive_tokens']:5d}"
            )
    e = prof["level1_own_text_empty_or_nearly"]
    print(
        f"\n--- level 1 nodes with empty or nearly empty own_text (<=3 tokens): {e['n']} of {e['of']} ---"
    )
    print(f"  e.g.: {e['examples']}")
    print(f"\nwritten: {NODES_OUT} ({len(records)} nodes), {PROFILE_OUT}")


if __name__ == "__main__":
    main()
