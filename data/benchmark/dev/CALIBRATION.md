# DEV calibration record

What each batch of DEV needs taught about the protocol, measured before the next batch is
written (`data/benchmark/PROTOCOL.md`, section 6). A proposed change of rules is listed here and
adopted only as a new protocol version.

## Batch 1: N004 to N013 (2026-10-12)

Protocol v1 with the parameters of its section 8. Query template `query_variants.v1` (SHA-256
`466bcfe8…`). DEV was sealed at 0.2.0 when the needs and queries were locked (commit `01e0014`)
and at 0.3.0 once judged (commit `5bd5095`).

### 1. Coverage grid

Chosen before any legal text was read. The topic of each cell was left to the model that drafted
the needs.

| Cell | Need | Act(s) | Type | Answerability | Difficulty |
|---|---|---|---|---|---|
| 1 | N004 | DORA | direct | complete | easy |
| 2 | N005 | DORA | multi_provision | complete | medium, revised to hard |
| 3 | N006 | DORA | exception_confusion | complete | hard |
| 4 | N007 | DORA | partial_out_of_scope | partial | medium |
| 5 | N008 | MiCA | direct | complete | easy |
| 6 | N009 | MiCA | exception_confusion | complete | medium |
| 7 | N010 | MiCA | partial_out_of_scope | none | hard |
| 8 | N011 | DORA and MiCA | multi_provision | complete | hard |
| 9 | N012 | PSD2 | direct | complete | easy |
| 10 | N013 | PSD2 | multi_provision | complete | medium |

N005 was revised to hard after the evidence search: its answer needs provisions from Articles 3,
28 and 30 of DORA.

### 2. How the needs and queries were written

- **Needs.** One fresh model context drafted a first choice and an alternate for each cell. It
  received the grid, the acts and versions in the corpus, what the corpus does not contain
  (recitals, annexes, Level 2 acts, guidelines, national law, other acts), the rules for a
  statement, the topics the fixtures already use, and the numbers and broad topics of the
  excluded PSD2 articles. It received no legal text and no qrel. The alternates were never
  needed and are not in the repository.
- **Queries.** For each need, `scripts/benchmark.py prompt` rendered the template from the
  statement alone, and a fresh context per need answered it: English formal, English natural
  and natural Spanish, recorded as `llm_from_need` with the template.
- **Isolation check.** The transcripts of the 11 generating contexts show no tool call other
  than returning the result, a task identical byte for byte to the rendered prompt, and no
  corpus text in what they received (searches for "shall", "Article" followed by a number,
  node paths and qrels found nothing). The transcripts stay local; they are not versioned.
- **Lock.** Needs and queries were committed in `01e0014`, pushed at 13:24 UTC and checked by
  CI run 38255430620 before the evidence search began at 13:29 UTC. `queries.jsonl` has not
  changed since: the diff from `01e0014` to `5bd5095` is empty.

### 3. Distribution

| | Count |
|---|---|
| Acts | DORA 4, MiCA 3, PSD2 2, DORA and MiCA 1 |
| Type | direct 3, multi_provision 3, exception_confusion 2, partial_out_of_scope 2 |
| Difficulty | easy 3, medium 3, hard 4 |
| Answerability | complete 8, partial 1, none 1 |
| Drivers | numeric 6, exception 4, cross_reference 4, lexical_twin 4, negation 1, cross_act 1 |

### 4. Qrels

137 qrels: grade 0: 24, grade 1: 52, grade 2: 27, grade 3: 34. Two are ambiguous.

| Need | Answerability | Qrels | Grades 0 / 1 / 2 / 3 | Relevant nodes | Relevant chunks at 500 / 300 tokens | Ambiguous |
|---|---|---|---|---|---|---|
| N004 | complete | 8 | 3 / 4 / 0 / 1 | 1 | 1 / 1 | 0 |
| N005 | complete | 26 | 1 / 4 / 7 / 14 | 21 | 6 / 8 | 0 |
| N006 | complete | 19 | 2 / 12 / 3 / 2 | 5 | 2 / 4 | 1 |
| N007 | partial | 21 | 2 / 8 / 4 / 7 | 11 | 6 / 6 | 0 |
| N008 | complete | 7 | 3 / 3 / 0 / 1 | 1 | 1 / 1 | 0 |
| N009 | complete | 10 | 2 / 4 / 2 / 2 | 4 | 2 / 3 | 0 |
| N010 | none | 6 | 2 / 4 / 0 / 0 | 0 | not evaluated | 0 |
| N011 | complete | 16 | 2 / 8 / 4 / 2 | 5 | 4 / 4 | 1 |
| N012 | complete | 7 | 4 / 1 / 1 / 1 | 2 | 1 / 1 | 0 |
| N013 | complete | 17 | 3 / 4 / 6 / 4 | 10 | 2 / 3 | 0 |

Relevant means grade 2 or 3 and not ambiguous. Over the nine evaluated needs, the median is 5
relevant nodes (from 1 to 21). Every need has grade-0 qrels: 1 to 4 per need, 24 in all, most
of them lexical twins, parallel regimes or look-alike deadlines.

### 5. The two passes

The first pass was made by the session that had searched the evidence. The second pass was a
fresh context per need, which received the statement, sections 2 and 3 of the protocol and the
candidate texts in document order, never the first grades.

- Same grade: 117 of 137 (0.85).
- Same side of the relevance line (grade 2 or more): 130 of 137 (0.95).
- Same answerability: 10 of 10.

Every disagreement was settled against the text, with no vote. When the text could not settle
it, the qrel was marked ambiguous.

| Need | Node | First | Second | Final | Why |
|---|---|---|---|---|---|
| N004 | DORA 13.2 | 1 | 0 | 1 | A review after a major incident is related context, not a look-alike. |
| N005 | DORA 28.8.sub_2 | 2 | 3 | 3 | The rule lives in the lead-in, so the list rule swaps the grades. |
| N005 | DORA 28.8.sub_2.b | 1 | 2 | 2 | A conjunct of the same exit rule as points (a) and (c). |
| N005 | DORA 28.8.sub_3 | 2 | 3 | 3 | It states a requirement on the bank's own planning. |
| N005 | DORA 28.8.sub_5 | 2 | 3 | 3 | It states what must be in place. |
| N005 | DORA 30.2.d | 2 | 3 | 3 | A contract term that the switch depends on. |
| N005 | DORA 30.3.f.i | 2 | 3 | 3 | It answers the no-disruption part directly. |
| N005 | DORA 42.6 | 0 | 1 | 1 | A real termination case, but only for critical providers. |
| N005 | DORA 33.3.f | 0 | 1 | 0 | It sets what the Lead Overseer assesses and imposes nothing on the bank. |
| N005 | DORA 30.3.b | 0 | 1 | 1 | The provider's notices feed the circumstances of Article 28(7)(b). |
| N006 | PSD2 32.1 | 1 | 2 | ambiguous | Whether the need already states its status in legal terms. |
| N006 | DORA 2.3 | 0 | 1 | 1 | Paragraph 1 applies "without prejudice" to these exclusions. |
| N007 | DORA 50.1 | 1 | 2 | 1 | Article 50(2)(c) is self-contained; paragraph 1 adds nothing asked for. |
| N007 | DORA 52.1 | 1 | 2 | 2 | A derogation that Article 50(3) refers to, and it changes the answer. |
| N007 | DORA 3.30 | 1 | 2 | 2 | "Board members" must be mapped to "members of the management body". |
| N007 | DORA 50.3 | 2 | 3 | 3 | On its own it says who sets the penalties and where the maximum is. |
| N011 | MiCA 68.7 | 2 | 1 | 1 | The aims of the policy are stated in the second subparagraph. |
| N011 | DORA 2.1.f | 2 | 1 | ambiguous | Two plausible legal routes lead to the yearly testing duty. |
| N012 | PSD2 101.3 | 1 | 0 | 0 | Nothing about the reply time. |
| N012 | PSD2 102.1 | 1 | 0 | 0 | Nothing about the reply time. |

Of the 20 disagreements, 3 kept the first grade, 15 took the second and 2 became ambiguous.
Six were between grades 2 and 3, and five of those six were in N005.

### 6. Ambiguity and rejections

- **Ambiguous qrels:** 2, both settled as unjudged (see the table above). No need has an
  ambiguity at its core.
- **Rejected needs:** none. No alternate was used.
- **Human consistency check** (protocol step 7): the automated part has been done:
  - every quoted fragment in a rationale was searched for in the judged text; the fragments not
    found there quote the need or a node the rationale names, except two, corrected before the
    seal;
  - the order of the commits shows the lock;
  - no query was flagged as a possible copy.

  The person's review of these points is still pending.

### 7. Leakage diagnostics

| Formulation | Queries | Share of content words, median / max | Longest verbatim run, median / max | Possible copy |
|---|---|---|---|---|
| English formal | 9 | 0.43 / 0.52 | 4 / 5 | 0 |
| English natural | 9 | 0.14 / 0.27 | 2 / 3 | 0 |
| Spanish natural | 9 | 0.02 / 0.03 | 1 / 1 | 0 |

N010 has no evidence, so its three queries have no overlap to measure. No query reached
`COPY_RUN` (6 words), so none needed a reasoned approval.

The Spanish figures say little: the stopword list is English, so Spanish function words count as
content, and a copy made by translation would not show up.

### 8. Duplication

- **Exact duplicates:** none; the validator refuses duplicate query texts.
- **Near duplicates:** the Jaccard similarity of content words was computed between all needs,
  fixtures included, and between all English queries of different needs. The highest values are
  0.14 for statements and 0.17 for queries. No pair reached 0.35, so none needed resolving.

### 9. Cost

| Step | Wall clock |
|---|---|
| Needs drafted: one context, 20 statements | 17 min |
| Queries: 10 contexts in parallel | 2.6 min (0.7 to 2.6 each) |
| Lock: seal 0.2.0, tests, commit, push, CI | about 5 min |
| Evidence search and first-pass notes, 10 needs | 9 min (N013 3.2, N005 1.2, the rest under 1) |
| First-pass rationales written, second-pass prompts built | about 13 min |
| Second pass: 10 contexts in parallel | 6.7 min (1.1 to 6.7 each) |
| Disagreements settled, qrels built, seal 0.3.0, tests | about 45 min |
| From the first draft to the judged commit | 104 min |

The 21 model contexts used 1.16 million tokens. A query context used 36,000 to 48,000 tokens to
return three lines, so most of that cost is the context each new agent starts with. This is
machine time; the time a person would need is not measured here.

### 10. Corpus boundaries

- **complete (8).**
- **partial (1).** N007: DORA lists the sanctions and remedial measures and allows them for
  members of the management body, but the maximum fine for a financial entity, and the
  conditions of personal liability, are set by each Member State's law.
- **none (1).** N010: the data that must travel with a crypto-asset transfer, and the checks for
  self-hosted addresses, are in Regulation (EU) 2023/1113. MiCA only requires a client agreement
  for transfer services.

### 11. PSD2

- **Needs on PSD2:** N012 and N013.
- **PSD2 qrels in other needs:** N006 (Article 32, ambiguous) and N010 (Articles 45 and 57,
  grade 0).
- **PSD2 articles used:** 4, 32, 45, 57, 71, 73, 76, 77, 99, 101 and 102. None is in
  `psd2_excluded_articles.json`, and the validator checked every qrel against that list.
- **Indirect dependencies:** the grade 2 or 3 evidence of N012 and N013 refers only to PSD2
  Articles 76, 77 and 99 to 102, none of them amended, and to Regulation (EU) No 260/2012.

### 12. Validation

DEV is sealed at 0.3.0: schema 1, parser 1.4.0, the SHA-256 of the three source documents, of the
three data files and of `query_variants.v1`. The seal re-ran:

- the dataset rules: split, languages, evidence for ready needs, acts against evidence, and
  `none` without grade 2 or more;
- the corpus check: every node exists, has text and still has the judged text;
- the PSD2 guard.

The test suite passes with the corpus (565 tests).

### 13. Critical review

- **Is the 0-3 scale working?** At the relevance line, yes: 95% agreement, and only 7 of 137
  qrels crossed it. Between 2 and 3 it is less stable, in two ways:
  - five of the six disagreements of that kind came from N005;
  - the first pass misapplied the list rule once, in Article 28(8).

  A need with several asks has many nodes that fully answer one of them, and the rules do not say
  whether each of those is a 3.
- **Are the types enough?** They covered all ten cells. The type `partial_out_of_scope` repeats
  what the answerability already says. The real gap is breadth:
  - every statement asks two or three questions (three in N007, N011 and N013). When one
    provision answers them all (N004, N008, N012) that costs nothing; when they need different
    provisions the relevant set grows, and N005 needs 21 nodes in 6 chunks at 500 tokens;
  - with more than five relevant chunks, Recall@5 cannot reach 1 even for a perfect ranking.
    That ceiling applies to N005 and N007.
- **Are the hard negatives useful?** Yes, for error analysis: twins with another frequency (DORA
  16(2), 8(2); MiCA 81(14)), another deadline (PSD2 77(2), 101(2)), another addressee (DORA
  35(8); MiCA 34(9)). They do not change Recall or MRR, only Judged@k. One distractor of the
  first pass, DORA 42(6), is relevant if the vendor is a critical provider, a fact the need does
  not state; another, DORA 30(3)(b), had been judged too strictly. A hard negative should be
  wrong whatever the unstated facts are.
- **Does the template produce natural queries?**
  - The English natural and Spanish queries read like a person asking.
  - The English formal ones repeat the statement in official terms.
  - All formulations carry every ask of the need, so they are long questions for an assistant,
    not short search-box queries.
- **Signs of lexical leakage?** None from the evidence: the query writers never saw it, and no
  query reached the copy threshold. The overlap of the English formal queries (median 0.43)
  comes from the statements: the model that drafted them knows these acts, and uses terms such
  as "ICT risk management framework". That is the vocabulary of an expert user, not a copy of
  the evidence. The natural and Spanish queries are the counterweight.
- **Bias towards some articles or acts?** The acts follow the grid. Inside them:
  - the drafting model went to well-known provisions with a crisp, often numeric answer
    (numeric driver in 6 of 10);
  - grade 2 or more evidence sits in 16 articles, mostly governance, scope and contracts in
    DORA, CASP rules in MiCA, and refunds and complaints in PSD2;
  - no evidence yet covers asset-referenced or e-money tokens, market abuse, DORA resilience
    testing (Chapter IV), or the PSD2 information requirements and liability for unauthorised
    transactions.

### 14. Proposed for protocol v1.1

1. **Breadth.** One question per need, or two that one rule is likely to answer together; a
   need whose questions point to different rules is split into separate needs.
2. **Grades 2 and 3 in a need with several asks.** State that a node which fully answers one ask
   is a 3, and that a node needed to complete an ask is a 2. Add Article 28(8) of DORA as a worked
   example of the lead-in swap.
3. **Hard negatives.** A grade-0 distractor must be wrong for the need whatever facts the need
   leaves unstated. A node that is relevant only under such a fact is graded 1.
4. **Grid.** Add the area of the act to the grid (for example asset-referenced tokens, market
   abuse, testing, payment information) and cap the numeric driver at about half of a batch.

**Adopted** on 2026-10-12 as protocol v1.1, in the wording of `PROTOCOL.md`, which differs from
these proposals in four points:

- a second question is allowed only when it depends on the same rule, the same evidence answers
  both, and a separate need would be redundant (section 9);
- a node whose relevance depends on a fact the need leaves open is graded 1, left unjudged, or the
  need is clarified before the lock (section 2);
- the numeric or date share is a heuristic, not a cap (section 10);
- the person's review checks the method, never the law (section 11).

This batch was built with v1. Any change that its review against v1.1 suggests waits for approval.
