"""Read-only search of the parsed corpus for judge A (PROTOCOL.md, step 3).

    PYTHONPATH=. .venv/bin/python scripts/evidence_tools.py titles ACT [FROM TO]
    PYTHONPATH=. .venv/bin/python scripts/evidence_tools.py article ACT N
    PYTHONPATH=. .venv/bin/python scripts/evidence_tools.py nodes ACT:path [ACT:path ...]
    PYTHONPATH=. .venv/bin/python scripts/evidence_tools.py search ACT|ALL "regex" [--titles]

The four tools judge A used in DEV batch 2, with the same output (ingest/corpus_tools.py):
article titles, every node of an article, nodes in full with their heading, and a
case-insensitive regex over the own text (or over article titles). Nothing here writes,
ranks or embeds anything.
"""

from __future__ import annotations

import sys

from ingest import benchmark as bm
from ingest.corpus_tools import CorpusTools


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("titles", "article", "nodes", "search"):
        print(__doc__)
        return 2
    tools = CorpusTools(bm.load_corpus())
    command, args = argv[0], argv[1:]
    if command == "titles":
        text = (
            tools.titles(args[0], int(args[1]), int(args[2]))
            if len(args) > 2
            else tools.titles(args[0])
        )
    elif command == "article":
        found = tools.article(args[0], args[1])
        if found is None:
            sys.exit(f"no article {args[0]} {args[1]}")
        text = found
    elif command == "nodes":
        text = tools.nodes(args)
    else:
        text = tools.search(args[0], args[1], "--titles" in args[2:])
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
