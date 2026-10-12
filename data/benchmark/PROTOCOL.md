# Retrieval benchmark protocol, v1

How one information need goes from an idea to the DEV benchmark (`data/benchmark/dev/`).
Decisions D-019 and D-020 govern it; `ingest/benchmark.py` enforces what can be checked by code.
The three needs in `data/benchmark/fixtures/` are test data for the pipeline, not DEV: their
queries were written knowing the evidence.

Nobody here is a legal expert. Every judgment must be grounded in the words of the legal text
and explained in its rationale; a question that the text alone cannot settle is marked as
ambiguous, never turned silently into ground truth.

## 1. Workflow

| Step | What happens | Output |
|---|---|---|
| 0. Gap | Pick the cell of the coverage grid (type, act, answerability) that is missing, without reading the legal text. | a cell |
| 1. Need | Write `statement` in English: the actor, the situation, what they want to know. Set type, drivers, difficulty, acts and answerability provisionally. No article numbers or legal wording the user would not know. | need, `status: draft` |
| 2. Queries, locked | At least one English query (formal or natural) and one natural Spanish query, from the statement alone. LLM variants come from `scripts/benchmark.py prompt <dir> <need_id>`, run in a fresh context that has not read the corpus; record `origin: llm_from_need` and the template. Commit the need and its queries before any evidence search: the history is the record of the lock. After that only typos are fixed; a new wording is a new query. | queries |
| 3. Evidence search | Structural (article titles, the definitions article, the articles a candidate refers to, parent and sibling nodes), lexical over the own text with the official terms, and twins (identical own text, the same article title, the parallel regime). | candidates |
| 4. Qrels | Grade every candidate examined, grade 0 included, with the rules of section 2. The rationale cites the node's words. `text_sha256` is `ingest.benchmark.sha256_text(own_text)` from the processed layer, never typed. A `span` only for a long node with several ideas. | qrels |
| 5. Hard negatives | One to three per need when a twin or a plausible distractor exists, graded 0. The rationale says why it tempts and why it is wrong. | grade-0 qrels |
| 6. Answerability | List the elements the need asks for (section 3). Set `gap_note` when something is missing. | answerability |
| 7. Second pass | A fresh pass grades the same candidates without seeing the first grades. A disagreement is settled against the text or marked `ambiguous`, never by majority. A person checks consistency, not legal validity: rationales match the words, queries were committed before the qrels, the Spanish is natural, and every `POSSIBLE COPY` has a reason. | `status: reviewed` |
| 8. Validation | `scripts/benchmark.py validate data/benchmark/dev`: the seal, the dataset rules, the corpus, and the lexical overlap of each query. | report |
| 9. Ready | `status: ready`, then `scripts/benchmark.py seal data/benchmark/dev --role dev --version <next>`. Log the time spent on the need. | DEV |

## 2. Grades

| Grade | Test |
|---|---|
| 3 | Returned alone, with its article heading, this node shows the rule, date or figure the need asks for. |
| 2 | Without this node the answer would be wrong or incomplete: a definition that maps the need's words to a legal term, an exception or derogation that changes the answer, the provision that a grade-3 node refers to. |
| 1 | Useful context that a correct and complete answer can leave out. |
| 0 | Examined and not relevant. Not the same as unjudged: only examined nodes are graded. |

Frequent cases:

- **List item and lead-in.** The item that answers is 3. The lead-in is 2 if it carries the
  operative verb or condition ("this Regulation applies to the following entities:"), 1
  otherwise. When the rule lives in the lead-in and the item only names its object, the
  grades swap (DORA 19(4) and its point (a)).
- **Definitions.** 2 when the need's own words must be mapped to a defined term; 1 when the need
  already uses the term in its legal sense.
- **Mandates for technical standards.** 2 when they say where the element asked for is fixed;
  otherwise 1 or left unjudged.
- **Twins and repeated wording.** Only the right article is relevant. The parallel regime and
  the same wording elsewhere are graded 0 when examined.

Binary relevance is grade >= 2. A chunk takes the maximum grade of the nodes it owns.

## 3. Answerability

- **complete**: every element the need asks for is stated by a node of grade >= 2.
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
- it is navigation only ("show me Article 28");
- it duplicates another need: the same intent and the same grade-3 evidence;
- it can only be phrased by copying the node;
- it depends on several versions of the law (out of scope in v1; a date written in the law
  is fine);
- the ambiguity is so central that no node can be graded with confidence;
- it needs a PSD2 article listed in `psd2_excluded_articles.json`. Those articles were amended
  after the snapshot Fundamento parses; the validator refuses any qrel on them. If you cannot
  tell whether a PSD2 need is safe, it does not enter.

## 6. Calibration after the first 10 needs

Before writing more, review the batch: grades by type, hard negatives per need, the share of
ambiguous judgments, the overlap distribution by language and style, and the time per need.
Adjust this protocol first; a change of rules is a new protocol version.

## 7. Checked by code, reviewed by reason

| Checked by `ingest/benchmark.py` | Needs reasoned review |
|---|---|
| ids, references, enums, unknown keys | whether the statement is a real need |
| split by need; evidence-aware queries only in fixtures | whether the queries were written blind |
| evidence rules for reviewed and ready needs | every grade and its rationale |
| node exists, has text, still has the judged text | partial or none, and its gap |
| the seal: files, templates, parser, corpus snapshot | ambiguity |
| PSD2 exclusions | a flagged possible copy |
| lexical overlap, reported per query | duplicate needs with different words |
