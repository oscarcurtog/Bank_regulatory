"""D of retrieval iteration 4: the question decomposed into information requirements, each ranked
by E5 over the whole corpus, merged with E5's ranking by a fixed rule
(config/retrieval/decomposition.md).

E5 ranks the answering provision in its top 10 for most queries; what it misses is the evidence
the answer depends on: a definition, a condition, an exception, the lead-in of a list, a provision
another one refers to. A local model reads the question and E5's first 3 results and writes 2 to 4
information requirements, each a question about one point of law, guided by a neutral statement
of what a correct and complete answer can need (the task as the protocol's grades 3 and 2 define
it, with no example, article, label or result of DEV). A requirement asks; it never states what
the law says, nor an article number or a date, so that no invented text joins the search (the
failure of HyDE). E5 ranks each requirement over the 1,354 chunks, and a mechanical rule, with no
model, builds the 10:

- E5's first 6 stay;
- places 7 to 10 go, in this order, to: E5's ranks 7 to 10 that some requirement also has in its
  first K = 5 (corroborated); each requirement's best chunk outside E5's top 10, in turns; the
  rest of E5's ranks 7 to 10, in E5's order;
- the 10 are ordered by their rank in E5, and every other chunk follows in E5's order.

An answer with fewer than 2 readable requirements keeps E5's ranking, and is counted. Nothing
here reads a label.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

MIN_REQUIREMENTS = 2
MAX_REQUIREMENTS = 4
EXCERPTS = 3
K = 5
KEEP = 6
DEPTH = 10
POOL = 50  # diagnostics only: the requirements' wider pool

NECESSARY = (
    "A legally correct and complete answer can need, besides the provision that states the "
    "rule, figure or deadline asked for:\n"
    "- a definition that decides whether a rule applies or how something is classified;\n"
    "- a condition that changes the answer;\n"
    "- an exception, exemption or derogation that could change it;\n"
    "- the introductory wording that gives a listed point its legal effect;\n"
    "- a specific provision that another provision refers to and that the answer needs;\n"
    "- a provision that would apply under a fact the question does not state but that could "
    "reasonably hold."
)
TASK = (
    f"Write {MIN_REQUIREMENTS} to {MAX_REQUIREMENTS} questions that a search of the law must "
    "answer for the answer to the QUESTION to be correct and complete. Each asks about one point "
    "of law (a rule, a definition, a condition, an exception, a deadline or an amount, or a "
    "provision that another provision refers to), in English, in one self-contained sentence "
    "that names who it concerns and what. Do not answer, do not state what the law says, do not "
    "copy the excerpts, and write no article numbers and no dates."
)
SYSTEM = (
    "You plan the search for the legal evidence that a question about EU financial regulation "
    "needs."
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def requirements_prompt(question: str, excerpts: Sequence[str]) -> list[dict[str, str]]:
    shown = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(excerpts, 1))
    user = (
        f"QUESTION: {question}\n\n"
        "EXCERPTS (the first results of a search for the question; they can be incomplete or "
        f"wrong):\n{shown}\n\n{NECESSARY}\n\n{TASK}\n\n"
        "Answer in this form and nothing else:\nREQUIREMENTS:\n- <question>\n- <question>"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


@dataclass(frozen=True)
class Requirements:
    items: tuple[str, ...]  # the first MAX_REQUIREMENTS distinct ones, in the model's order
    dropped: int  # items repeated, beyond the limit, or the form's placeholder

    @property
    def readable(self) -> bool:
        return len(self.items) >= MIN_REQUIREMENTS


_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*(.+?)\s*$", re.MULTILINE)


def parse(output: str) -> Requirements:
    """The items listed after "REQUIREMENTS:" or, without that heading, every line that opens
    with "-", "*", a bullet or a number. An item repeated (ignoring case and spacing) counts once;
    the form's own placeholder is not an item."""
    m = re.search(r"REQUIREMENTS:\s*(.*)\Z", output, re.DOTALL | re.IGNORECASE)
    found = [normalize(x) for x in _ITEM.findall(m.group(1) if m else output)]
    items: list[str] = []
    for x in found:
        if x and "<question>" not in x.lower() and x.lower() not in {i.lower() for i in items}:
            items.append(x)
    kept = tuple(items[:MAX_REQUIREMENTS])
    return Requirements(kept, len(found) - len(kept))


@dataclass(frozen=True)
class Assembly:
    ranking: tuple[str, ...]  # every chunk: the 10, in E5's order, then the rest in E5's order
    corroborated: tuple[str, ...] = field(default=())  # E5 ranks 7-10 kept as corroborated
    added: tuple[str, ...] = field(default=())  # chunks from outside E5's top 10
    displaced: tuple[str, ...] = field(default=())  # E5 ranks 7-10 that left the 10


def assemble(e5: Sequence[str], hits: Sequence[Sequence[str]]) -> Assembly:
    """The rule of the module docstring. `e5` is E5's whole ranking of the question; `hits`, for
    each requirement in the model's order, the chunks of E5's ranking of that requirement (only
    the first K count). With no requirement, the ranking is E5's."""
    top = list(e5[:DEPTH])
    in_top = set(top)
    tail = top[KEEP:]
    slots = DEPTH - KEEP
    support = {c for h in hits for c in h[:K]}
    corroborated = [c for c in tail if c in support]
    fresh = [[c for c in h[:K] if c not in in_top] for h in hits]
    added: list[str] = []
    pointer = [0] * len(fresh)
    while len(corroborated) + len(added) < slots:
        moved = False
        for i, chunks in enumerate(fresh):
            while pointer[i] < len(chunks) and chunks[pointer[i]] in added:
                pointer[i] += 1
            if pointer[i] < len(chunks):
                added.append(chunks[pointer[i]])
                pointer[i] += 1
                moved = True
                if len(corroborated) + len(added) == slots:
                    break
        if not moved:
            break
    rest = [c for c in tail if c not in support]
    fill = (corroborated + added + rest)[:slots]
    rank = {c: i for i, c in enumerate(e5)}
    chosen = sorted(top[:KEEP] + fill, key=rank.__getitem__)
    placed = set(chosen)
    return Assembly(
        tuple(chosen + [c for c in e5 if c not in placed]),
        tuple(corroborated),
        tuple(c for c in fill if c not in in_top),
        tuple(c for c in tail if c not in placed),
    )


def articles_named(text: str) -> set[str]:
    """The article numbers a text names ("Article 59", "Articles 11 and 12"), for checking that a
    requirement names only articles its excerpts name."""
    out: set[str] = set()
    for m in re.finditer(r"\bArticles?\s+((?:\d+[a-z]?(?:\s*(?:,|and|or|to)\s*)?)+)", text):
        out |= set(re.findall(r"\d+[a-z]?", m.group(1)))
    return out
