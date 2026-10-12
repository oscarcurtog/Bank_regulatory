# Retrieval benchmark protocol, v1.2

How one information need goes from an idea to the DEV benchmark (`data/benchmark/dev/`).
Decisions D-019 to D-023 govern it; `ingest/benchmark.py` enforces what can be checked by code.
The three needs in `data/benchmark/fixtures/` are test data for the pipeline, not DEV: their
queries were written knowing the evidence.

Nobody here is a legal expert. Every judgment must be grounded in the words of the legal text
and explained in its rationale; a question that the text alone cannot settle is marked as
ambiguous, never turned silently into ground truth.

| Version | Since | What changed |
|---|---|---|
| v1 | 2026-10-12 | First version. Batch 1 (N004 to N013) was built with it. |
| v1.1 | 2026-10-12 | From the calibration of batch 1: one primary retrieval problem per need (section 9); grades 3 and 2 decided by the question, with a worked example (section 2); a hard negative must be wrong whatever the need leaves open (section 2); the area of the act and numeric or date answers in the coverage grid (section 10); the person's review checks the method, not the law (section 11). No data schema changed. |
| v1.2 | 2026-10-12 | From the review of batch 1 against v1.1: a definition is graded by necessity, not by mapping; a reference to whole articles marks where to look, not what is relevant; a grade 0 needs a scope check and must be definitely wrong on the stated facts (all three in section 2); every disagreement between judges A and B goes to an independent judge C, recorded in `adjudications.jsonl` (section 12); a replacement keeps its id only when it narrows the same need (section 5). No data schema changed: the adjudication record is a new file, outside the seal. |

The data files do not record the protocol version: each batch states it in `dev/CALIBRATION.md`.

## 1. Workflow

| Step | What happens | Output |
|---|---|---|
| 0. Gap | Pick the cell of the coverage grid (section 10) that is missing: type, act, area of the act, answerability, and whether a number or a date is expected in the answer. Without reading the legal text. | a cell |
| 1. Need | Write `statement` in English: the actor, the situation, and one primary question (section 9). Set type, drivers, difficulty, acts and answerability provisionally. No article numbers or legal wording the user would not know. | need, `status: draft` |
| 2. Queries, locked | At least one English query (formal or natural) and one natural Spanish query, from the statement alone. LLM variants come from `scripts/benchmark.py prompt <dir> <need_id>`, run in a fresh context that has not read the corpus; record `origin: llm_from_need` and the template. Commit the need and its queries before any evidence search: the history is the record of the lock. After that only typos are fixed; a new wording is a new query. | queries |
| 3. Evidence search | Structural (article titles, the definitions article, the articles a candidate refers to, parent and sibling nodes), lexical over the own text with the official terms, and twins (identical own text, the same article title, the parallel regime). | candidates |
| 4. Qrels, judge A | Judge A grades every candidate examined, and the answerability, with the rules of section 2, and writes them down before judge B starts. The rationale cites the node's words. `text_sha256` is `ingest.benchmark.sha256_text(own_text)` from the processed layer, never typed. A `span` only for a long node with several ideas. | qrels |
| 5. Hard negatives | One to three per need when a twin or a plausible distractor exists, graded 0 only after the scope check of section 2. The rationale says why the node tempts and why it is wrong on the facts the need states. | grade-0 qrels |
| 6. Answerability | List the elements the need asks for (section 3). Set `gap_note` when something is missing. | answerability |
| 7. Judges B and C, then review | Judge B, a fresh context, grades the same candidates and the answerability without seeing judge A or any earlier label. Every disagreement goes to judge C (section 12): the first judge never settles its own disagreement. Then a person reviews the method, not the law (section 11). | `status: reviewed`, `adjudications.jsonl` |
| 8. Validation | `scripts/benchmark.py validate data/benchmark/dev`: the seal, the dataset rules, the corpus, and the lexical overlap of each query. | report |
| 9. Ready | `status: ready`, then `scripts/benchmark.py seal data/benchmark/dev --role dev --version <next>`. Log the time spent on the need. | DEV |

## 2. Grades

A grade says what the node does for the answer to the need.

| Grade | Test |
|---|---|
| 3 | It answers a primary question of the need directly and completely: read with its article heading and the lead-in that governs it, it states the rule, figure or deadline asked for, or one element of the list asked for. |
| 2 | It is not the answer, but without it the answer would not be legally correct or complete: a definition needed to read the need or to decide whether a rule applies; an exception, condition or derogation that can change the answer; the lead-in that turns a point into a rule; a specific provision that the main rule refers to and that the answer needs. |
| 1 | Useful context that a correct and complete answer can leave out. |
| 0 | Examined explicitly and definitely not relevant on the facts the need states, after the scope check below; never "probably not relevant". Not the same as unjudged: only examined nodes are graded. |

When the answer is a list, each point that states one element of it is a 3: "completely" means
that the node holds its part of the answer, not the whole list. A need with an allowed second
question (section 9) can have a grade-3 node for each of its two questions.

Binary relevance is grade >= 2. A chunk takes the maximum grade of the nodes it owns.

### Lead-in and list: a worked example

The grade follows the question, not the position in the list. The second subparagraph of DORA
Article 28(8) is one lead-in and three points:

> Financial entities shall ensure that they are able to exit contractual arrangements without:
> (a) disruption to their business activities,
> (b) limiting compliance with regulatory requirements,
> (c) detriment to the continuity and quality of services provided to clients.

| Question of the need | Lead-in | (a) | (b) | (c) | Why |
|---|---|---|---|---|---|
| When a bank leaves an ICT provider, what must the exit avoid? | 2 | 3 | 3 | 3 | The points hold the answer, one element each. The lead-in is what makes them a duty; without it they are three nouns. |
| Does DORA require a bank to be able to leave its ICT providers? | 3 | 2 | 2 | 2 | The answer is the duty itself, and it lives in the lead-in. The points say what "able to exit" means: an answer without them is incomplete. |

The two passes of batch 1 split on this lead-in (N005); v1.1 settles such a case by the question.
The usual case is DORA Article 28(7): for "in which situations must a bank be able to end the
contract?", each of the points (a) to (d) is a 3 and the lead-in ("may be terminated in any of
the following circumstances:") is a 2. Article 19(4) is the reverse case: the duty to submit, the
addressee and the time limits are in the lead-in, and point (a) only names the initial
notification.

### Frequent cases

- **Definitions.** 2 only when the definition is necessary to read the need correctly or to
  decide whether a rule applies. The test: without this definition, could the need reasonably be
  answered wrongly or incompletely? If yes, 2; if no, usually 1. An obvious mapping from everyday
  words to the legal term is not a 2 by default; a term that is ambiguous, narrows the scope or
  separates plausible legal categories can be. For a need about the company that runs a bank's
  core banking platform, DORA 3(19) ("'ICT third-party service provider' means an undertaking
  providing ICT services") is a 1 when no other category is plausible. For a need about exit
  strategies, DORA 3(22) ("critical or important function") is a 2, because the duty of
  Article 28(8) applies only "for ICT services supporting critical or important functions".
- **References to whole articles.** A reference such as MiCA 68(7)'s plans "set up pursuant to
  Articles 11 and 12 of Regulation (EU) 2022/2554" marks where to look, not what is relevant.
  The nodes of the referred articles are judged one by one: 2 for the specific nodes without
  which the answer would be wrong or incomplete, 1 for useful context, unjudged for the rest, and
  ambiguous when it cannot be told which part is needed. A cross-reference never adds qrels by
  itself.
- **Exceptions, conditions and derogations.** 2 when they can apply to the need and would change
  the answer; 1 when the facts of the need rule them out.
- **Mandates for technical standards.** 2 when they say where the element asked for is fixed;
  otherwise 1 or left unjudged.
- **Twins and repeated wording.** Only the right article is relevant. The parallel regime and
  the same wording elsewhere are graded 0 when examined, if they pass the scope check below.

### Hard negatives and the scope check

A grade 0 means definitely not relevant on the facts the need states, not probably not relevant.
Before grading a node 0, check its scope: read the node and every provision needed to know whether
it applies (its actor, conditions, exceptions, field and eligibility, and the facts it needs).
The adjudication record names those provisions (`scope_basis`). If the node could be relevant
under a reasonable reading of facts that the need does not specify, it is not a 0:

- grade it 1 when it is potentially useful context;
- leave it unjudged when it does not deserve a judgment;
- mark it ambiguous when its applicability cannot be settled;
- clarify the need, only if it is not locked yet and the fact is essential to it.

A locked need is never rewritten to manufacture a hard negative.

| Node | Need | Scope read | Result |
|---|---|---|---|
| DORA 23: the incident chapter "shall also apply to operational or security payment-related incidents" of credit institutions, payment institutions, account information service providers and electronic money institutions | How must an insurance company report a major ICT-related incident? | The node itself names the entities it reaches | An insurer is none of them: the 0 holds. |
| DORA 42(6): authorities may require the termination of arrangements with critical ICT third-party service providers | When must a bank be able to end its contract with an ICT provider? The need does not say whether the provider is critical | The node itself: it reaches only "critical ICT third-party service providers" | Relevant if the provider is critical: not a 0. |

## 3. Answerability

- **complete**: every element the need asks for is stated by a node of grade >= 2, and at least
  one node of grade 3 answers its primary question.
- **partial**: some element lives outside the corpus (a Level 2 act, an annex, another act,
  national law), but useful evidence of grade >= 2 exists. `gap_note` names what is missing.
  It is evaluated on the evidence the corpus has, and reported apart.
- **none**: no node gives evidence of grade >= 2. The candidates examined are still graded 0
  or 1, as the record of the search. It never enters Recall, MRR or nDCG; it is kept for
  abstention.

## 4. Ambiguity

When the grade depends on an interpretation rather than on what the text says, set
`ambiguous: true` on the qrel (it then counts as unjudged), or `ambiguity_note` on the need if
the doubt affects all of it. An ambiguous need never enters TEST; it stays in DEV only if it
helps calibration.

## 5. Reject a need when

- it is vague, or depends on facts it does not state;
- it asks more than one primary question (section 9). Before the lock it is split. After the
  lock its statement can no longer change, so it is replaced, and each replacement goes through
  the workflow again with its own lock and new queries. A replacement keeps the `need_id` only
  when it narrows or clarifies the same information need; a different retrieval problem gets a
  new id. Retired ids are never reused. The data files hold only the current needs, so the
  calibration record says what replaced what;
- it is navigation only ("show me Article 28");
- it duplicates another need: the same intent and the same grade-3 evidence;
- it can only be phrased by copying the node;
- it depends on several versions of the law (out of scope in v1; a date written in the law
  is fine);
- the ambiguity is so central that no node can be graded with confidence;
- it needs a PSD2 article listed in `psd2_excluded_articles.json`. Those articles were amended
  after the snapshot Fundamento parses; the validator refuses any qrel on them. If you cannot
  tell whether a PSD2 need is safe, it does not enter.

## 6. Calibration after each batch

Before writing more, review the batch: grades by type, hard negatives per need, the share of
ambiguous judgments, the overlap distribution by language and style, and the time per need.
Since v1.1, also: the areas of the act covered and the share of numeric or date answers
(section 10), and the needs with more relevant chunks than the smallest cut-off of Recall@k
(section 9). Adjust this protocol first; a change of rules is a new protocol version.

## 7. Checked by code, reviewed by reason

| Checked by `ingest/benchmark.py` | Needs reasoned review |
|---|---|
| ids, references, enums, unknown keys | whether the statement is a real need |
| split by need; evidence-aware queries only in fixtures | whether the queries were written blind |
| evidence rules for reviewed and ready needs | every grade and its rationale |
| a complete need has a non-ambiguous grade 3 | one primary question per need |
| node exists, has text, still has the judged text | partial or none, and its gap |
| the seal: files, templates, parser, corpus snapshot | ambiguity |
| PSD2 exclusions | a flagged possible copy |
| lexical overlap, reported per query | duplicate needs with different words |
| the adjudication record: final labels are the qrels; no A/B disagreement without judge C; every grade 0 names its scope provisions, and they exist | the scope check itself, and whether each 0 holds |
| definitions graded 2 or 3, listed by `validate` | whether each one is necessary; which nodes of a referred article the answer needs |

## 8. Parameters

Fixed for v1 and unchanged in v1.1. Changing one is recorded and versioned like any other rule.

| Parameter | Value | Meaning |
|---|---|---|
| `MIN_RUN_DEPTH` (`ingest/evaluation.py`) | 100 | Results a run must return per query. A run too short to fill a token budget is refused, never truncated. |
| `COPY_RUN` (`ingest/benchmark.py`) | 6 | Consecutive query words found verbatim in the evidence that raise `POSSIBLE COPY`. A diagnostic heuristic, not a definition of leakage: it never rejects a query by itself. A legitimate match of legal terminology is approved after a reasoned review, and the reason is written in the calibration record of the batch. |
| Query template | `query_variants.v1` | Frozen since the first batch; the manifest seals its SHA-256. It receives the statement and the formulations requested, nothing else. The renderer refuses a statement that cites an article, a paragraph or point, a structural unit, a node path or a CELEX number. |

## 9. One need, one primary retrieval problem

A need asks one primary question. A second question is allowed only when all three hold:

1. it clearly depends on the same legal rule as the first;
2. the same main evidence can answer both;
3. asking it as a separate need would give two artificially redundant cases.

A need never mixes several distinct obligations, several moments of a process, or several
independent legal decisions. A primary question may still need several provisions (type
`multi_provision`): what counts is that it is one question, not how many nodes answer it.

Batch 1 showed why. Every statement asked two or three things; when they pointed to different
rules the relevant set grew (21 nodes in 6 chunks of 500 tokens for N005), and with more relevant
chunks than k, Recall@k cannot reach 1 even for a perfect ranking. The size of the relevant set is
a symptom, not the criterion: a need is narrowed so that it asks one thing, never to improve a
metric.

## 10. The coverage grid

A batch is planned as a grid before any legal text is read. Each cell fixes:

| Dimension | Values |
|---|---|
| Type | `direct`, `multi_provision`, `exception_confusion`, `partial_out_of_scope` |
| Act | DORA, MiCA, PSD2, or two of them |
| Area of the act | a row of the table below |
| Answerability | complete, partial, none |
| Number or date | whether the answer is expected to be a figure, a period or a date |
| Difficulty | easy, medium, hard |

There are no fixed quotas, only heuristics:

- numeric or date answers stay at about half of a batch or less (batch 1 had 6 of 10);
- an area is not repeated within a batch while other areas of the same act are uncovered;
- among the remaining cells, prefer the retrieval challenge the benchmark has least of: a twin,
  an exception, a cross-reference, another act, a gap.

The model that drafts the needs receives the names of the areas, not the article ranges. After the
evidence search, the area is read again from the articles of the grade 2 and 3 nodes and recorded
in the calibration record, next to the drivers.

| Act | Area | Articles |
|---|---|---|
| DORA | Scope, definitions, proportionality | 1 to 4 |
| DORA | ICT risk management, including the simplified framework | 5 to 16 |
| DORA | ICT-related incidents: management, classification, reporting | 17 to 23 |
| DORA | Digital operational resilience testing | 24 to 27 |
| DORA | ICT third-party risk: principles, contracts, exit | 28 to 30 |
| DORA | Oversight of critical ICT third-party service providers | 31 to 44 |
| DORA | Information sharing | 45 |
| DORA | Competent authorities, cooperation, penalties | 46 to 56 |
| MiCA | Scope and definitions | 1 to 3 |
| MiCA | Offers and white papers of other crypto-assets | 4 to 15 |
| MiCA | Asset-referenced tokens | 16 to 47 |
| MiCA | E-money tokens | 48 to 58 |
| MiCA | Crypto-asset service providers: authorisation and cross-border services | 59 to 65 |
| MiCA | Crypto-asset service providers: obligations of all providers | 66 to 74 |
| MiCA | Crypto-asset services: custody, trading platforms, exchange, orders, placing, advice, portfolio management, transfers | 75 to 82 |
| MiCA | Acquisitions and significant providers | 83 to 85 |
| MiCA | Market abuse | 86 to 92 |
| MiCA | Supervision: competent authorities, EBA, ESMA | 93 to 138 |
| PSD2 | Scope and definitions | 1 to 4 |
| PSD2 | Payment institutions: authorisation, exemptions, access to payment systems | 5 to 37 |
| PSD2 | Transparency and information requirements | 38 to 60 |
| PSD2 | Consent, access to accounts, liability and refunds | 61 to 77 |
| PSD2 | Execution of payment transactions | 78 to 93 |
| PSD2 | Data protection, operational and security risks, authentication | 94 to 98 |
| PSD2 | Complaints, out-of-court redress, penalties | 99 to 103 |

## 11. Two kinds of review

| | Legal validation | Methodological review |
|---|---|---|
| Who | judges A and B, and judge C for every disagreement (section 12) | a person: the owner of the benchmark |
| Question | does the grade say what the legal text says? | was the protocol applied consistently? |
| How | source-grounded: each grade quotes the words of the node; a disagreement is decided by judge C against the text and this protocol, never by majority; `ambiguous` when the text cannot settle it | the checklist below, need by need |
| Not its job | whether the queries sound real | deciding whether a substantive legal interpretation is right |

For each need, the person checks:

1. Does the need ask one primary question, or one plus an allowed second (section 9)?
2. Do the queries sound like a real person asking, in English and in Spanish?
3. Are grades 3, 2, 1 and 0 applied as section 2 says, and the same way as in the other needs?
4. Does each rationale explain its grade with the words of the node?
5. Is there duplication with another need, or something artificial in the statement or queries?
6. Is the need useful as a retrieval case: would a good retriever and a weak one score differently
   on it?

For the batch, the person also checks that it is coherent as a whole, that the queries were
committed before the evidence search, and that every `POSSIBLE COPY` has a reason. A finding goes
into the calibration record. A doubt about the law is never settled by the person, who is never
asked to choose between two legal interpretations: it goes to judges A, B and C and the text, or
the qrel becomes `ambiguous`.

## 12. Adjudication: judges A, B and C

The first judge never settles its own disagreement, and nobody counts votes.

| Judge | Who | Receives | Never receives |
|---|---|---|---|
| A | whoever searched the evidence | the need, the candidates, the rules | (writes before B starts) |
| B | a fresh context | the need, each candidate with its legal context and the provisions that fix its scope, the rules | A's judgments, any earlier label |
| C | a fresh context | the need (and a query when needed), the candidate node, the strictly necessary legal context, the rules, and the two positions as "Rationale A" and "Rationale B" in random order | which judge was first, any earlier label |

- **A and B agree** (the same grade and ambiguous flag, both unjudged, or the same answerability):
  the label is final, unless it breaks an automatic check such as a dataset rule; then it goes to
  C with the reason (`escalated`).
- **A and B disagree:** C decides. It may take A's label, B's, another, or mark the node
  ambiguous, and it justifies the label against the text and this protocol. "Two judges said so"
  is never a reason.
- **Every disagreement goes to C:** 0 against 1, 1 against 2, 2 against 3, a hard negative,
  ambiguous against judged, judged against unjudged, and answerability (complete, partial, none).

`adjudications.jsonl` holds one line per candidate of every reviewed or ready need, and one for
its answerability: judgment A, judgment B, judgment C when there was one, each with its rationale,
and the reason for an escalation. A disagreement across the relevance line (grade >= 2 on one
side only) shows in the record. The file sits next to the data but outside the seal, because
sealing it would change the manifest contract; instead, the validator requires its final labels
to be exactly the qrels and the needs' answerability, and refuses an A/B disagreement without C.

A typical flow: A grades a node 1, B grades it 2, so the record shows a disagreement across the
relevance line. C reads the node, its context and both rationales, presented as A and B in random
order, and returns 2 with a rationale that quotes the text. The qrel takes grade 2, and the record
keeps the three judgments.
