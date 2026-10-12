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

## Batch 1, revised under protocol v1.2 (2026-10-12)

The review of batch 1 against v1.1 found six needs that ask more than one primary question, and
gaps in the grading rules that v1.2 closed. This revision applies v1.2 to every need of the
batch, the kept ones included. Query template `query_variants.v1`, unchanged. DEV was sealed at
0.4.0 when the replacements were locked (commit `82e1fea`), and at 0.5.0 once judged (commit
`4c97ae3`).

### 1. What replaced what

| Before | After | Decision | Why |
|---|---|---|---|
| N004 | N004 | kept | One question and an allowed second, both answered by DORA 6(5). |
| N005 | N014 | replaced, new id | Three primary questions. Kept: in which situations the bank must be able to end its contract with the vendor. The retrieval problem changed, so the id changed. |
| N006 | N006 | narrowed, same id | Scope and regime are two rules. Kept: whether DORA applies to the business at all. |
| N007 | N015 | replaced, new id | Three primary questions. Kept: the maximum fine the national supervisor could impose (partial). |
| N008 | N008 | kept | One rule, MiCA 75(5), answers both asks. |
| N009 | N009 | kept | Article 61(2) limits the exemption of 61(1). |
| N010 | N010 | narrowed, same id | The self-hosted wallet checks are a second obligation. Kept: the information that must accompany a transfer to another exchange. |
| N011 | N011 | narrowed, same id | The testing frequency is another obligation in another act. Kept: what the continuity policy must contain. |
| N012 | N012 | kept | One rule, PSD2 101(2), answers both asks. |
| N013 | N013 | narrowed and clarified, same id | The bank's response time is a later moment. Kept: whether, six weeks on, she can get the money back; the statement now says that the form fixed no amount. |

The data files hold only the current needs, and the schema has no field for a replacement. A
field would let a script follow N005 to N014; without one, this table and the history are the
record. A field was not added because nothing in the pipeline needs to follow the link. A test
keeps the retired ids N005 and N007, and the 18 queries written for the old statements, out of
DEV.

### 2. How it was done

- **Statements.** One fresh model context rewrote the six statements, given only the old
  statement and an instruction (what to keep, and for N013 the fact to add); no corpus text and
  no qrel. It was asked once more to drop the self-hosted wallets still named in N010's situation
  and a repeated phrase in N013.
- **Queries.** A fresh context per need wrote English formal, English natural and natural Spanish
  queries from `query_variants.v1`.
- **Lock.** Statements and queries were committed in `82e1fea`, pushed at 16:54:32 UTC and checked
  by CI run 38280656114. The evidence search began at 16:55:01 UTC. Of the 30 queries, 12 date
  from the batch 1 lock (`01e0014`) and 18 from `82e1fea`, and none has changed since.
- **Judges.**
  - Judge A, the session that searched the evidence, wrote its grades down at 17:03 UTC, before
    judge B started.
  - Judge B was a fresh context per need. It received the rules of section 2 with the batch-1
    examples removed, and each candidate with its legal context and the provisions that fix its
    scope; it saw no grade.
  - Judge C was a fresh context for each need with a disagreement. It saw the two positions as
    "Rationale A" and "Rationale B" in an order drawn at random for each item, never which judge
    was first or the earlier label.
- **Isolation check.** In the transcripts of the 26 contexts, the only tool call besides returning
  the result is reading its own file.

### 3. The batch

| Need | Act | Type | Difficulty | Answerability | Relevant nodes | Relevant chunks at 500 / 300 | Hard negatives |
|---|---|---|---|---|---|---|---|
| N004 | DORA | direct | easy | complete | 1 | 1 / 1 | 2 |
| N006 | DORA | exception_confusion | medium | complete | 2 | 1 / 1 | 2 |
| N008 | MiCA | direct | easy | complete | 1 | 1 / 1 | 3 |
| N009 | MiCA | exception_confusion | medium | complete | 4 | 2 / 3 | 2 |
| N010 | MiCA | partial_out_of_scope | hard | none | 0 | not evaluated | 5 |
| N011 | MiCA | direct | medium | complete | 1 | 1 / 1 | 2 |
| N012 | PSD2 | direct | easy | complete | 2 | 1 / 1 | 3 |
| N013 | PSD2 | multi_provision | medium | complete | 8 | 2 / 3 | 2 |
| N014 | DORA | direct | medium | complete | 5 | 1 / 1 | 1 |
| N015 | DORA | partial_out_of_scope | medium | partial | 3 | 2 / 2 | 5 |

- Acts: DORA 4, MiCA 4, PSD2 2.
- Types: direct 5, exception_confusion 2, partial_out_of_scope 2, multi_provision 1.
- Difficulty: easy 3, medium 6, hard 1.
- Answerability: 8 complete, 1 partial, 1 none.
- A numeric driver in 5 needs of 10, a date driver in none.
- The grade 2 and 3 evidence covers 9 areas of the grid, each once.

### 4. Qrels and adjudication

There are 118 qrels: 27 of grade 0, 64 of grade 1, 13 of grade 2 and 14 of grade 3. One is
ambiguous: DORA 42(6) in N014, which reaches only critical providers, and the need does not say
whether the vendor is one.

| | Count |
|---|---|
| Grades on which A and B agree | 100 of 118 (0.85) |
| Answerabilities on which A and B agree | 10 of 10 |
| Disagreements, all sent to judge C | 18: 7 between 0 and 1, 7 between 1 and 2, 2 between 2 and 3, 2 ambiguous against judged |
| Judge C's decision | A's label 10, B's label 7, ambiguous 1, another label 0 |
| Disagreements across the relevance line | 5, all of which ended below it |
| Agreements escalated to C | 0 |

`adjudications.jsonl` keeps the three judgments of each label, and the provisions read for every
grade 0.

### 5. Against batch 1 as judged under v1

74 nodes were judged under the same need in both versions. Eleven labels changed, six of them
across the relevance line:

- **The need was narrowed:**
  - DORA 16(1) in N006 went from 3 to 1;
  - DORA 11(6)(a) in N011 from 3 to 1;
  - PSD2 77(2) in N013 from 3 to 1, and its second subparagraph from 2 to 1.
- **The reference rule:** DORA 11(3) and 11(4) in N011 went from 2 to 1. MiCA 68(7) names the
  plans itself, so the DORA articles it refers to are context, and N011 now rests on MiCA alone.
- **The hard-negative rule:** three former grade 0s became 1, because each bears on the subject
  of the need and so is not definitely irrelevant:
  - DORA 11(6) third subparagraph in N004, the review of a part of the framework;
  - DORA 2(3)(d) in N006, an exclusion from the same scope that the facts rule out (the
    exceptions rule);
  - PSD2 101(3) in N012, a further step of the same complaint.
- **Ambiguities settled:**
  - PSD2 32(1) in N006 became 1 under the necessity rule;
  - DORA 2(1)(f) in N011 became 1 once the testing question was dropped.

All 27 grade 0s now name the provisions read to fix their scope. That includes 11 new hard
negatives, among them the fine regimes of MiCA (111(3), 132(3)), of PSD2 (103(1)) and of the Lead
Overseer (DORA 35(6), 35(8)) in N015.

### 6. Recall@5 by design

No evaluated need has more than 3 relevant chunks at 500 or 300 tokens, so a perfect ranking
reaches Recall@5 = 1 for every one. The two needs that could not, N005 and N007, were replaced.

### 7. Leakage, duplicates and PSD2

| Formulation | Queries | Share of content words, median / max | Longest verbatim run, median / max | Possible copy |
|---|---|---|---|---|
| English formal | 9 | 0.33 / 0.53 | 3 / 5 | 0 |
| English natural | 9 | 0.11 / 0.20 | 1 / 3 | 0 |
| Spanish natural | 9 | 0.02 / 0.05 | 1 / 1 | 0 |

- **Near duplicates:** the highest Jaccard similarity of content words is 0.14 between statements
  (fixtures included) and 0.21 between English queries of different needs; no pair reached 0.35.
- **PSD2:** the qrels use PSD2 Articles 4, 31, 32, 33, 45, 57, 64, 71, 73, 76, 77, 99, 101, 102
  and 103, none of them in `psd2_excluded_articles.json`.

### 8. Cost

It took 32 minutes of wall clock from the first rewritten statement (16:46 UTC) to the judged
commit (17:18 UTC). The 26 model contexts used about 1.46 million tokens:
- the statement writer, 0.06 million;
- the six query writers, 0.22 million;
- judge B, 0.72 million;
- judge C, 0.46 million.

### 9. Critical review

- **The batch now leans on single-provision needs.** Narrowing to one question left 5 direct
  needs and 1 multi_provision. The reference rule also removed the only cross-act need. The next
  batch has to plan questions that are single but need several provisions, or two acts.
- **Judge C did not favour either side.** It took A's label 10 times and B's 7, and the labels it
  saw were random. Ten decisions are too few to rule out a lean towards one style of rationale.
- **A grade 0 now means one of three things:** a parallel regime, a lure whose scope rules it
  out, or a duty about another object, such as DORA 8(2) on risk scenarios. A node that bears on
  the subject of the need became 1. That leaves fewer hard negatives in the kept needs and more
  in the new ones.
- **No definition is graded 2.** Every definition in the batch maps an obvious term.
- **Methodological review (protocol step 7):** done; its two observations are in section 10.

### 10. Methodological review of the person (protocol step 7)

The review raised two observations. Neither changed a label, a query or the seal: DEV stays at
0.5.0 and `adjudications.jsonl` is unchanged.

**N013, PSD2 76(1) fourth subparagraph, grade 3.** The question was whether grade 3 is consistent
with v1.2 when the node applies only to "direct debits as referred to in Article 1 of Regulation
(EU) No 260/2012" and the need states neither the currency nor the scheme.

- The facts of the need do not establish that applicability, and the corpus does not contain that
  regulation. The rationale's wording ("the need does not exclude such a debit") is weaker than
  the label it supports.
- v1.2 anchors on the facts the need states for a grade 0, and gives 2 to an exception that
  "can apply". It has no rule for a positive branch whose applicability is an open fact, and it
  keeps ambiguity for an interpretation, not for a missing fact. So its letter neither requires
  nor forbids a 3 here.
- The item was decided again by a fresh judge C, which saw the rationales of A and B as "Rationale
  A" and "Rationale B" in an order drawn at random, and the open fact stated neutrally. It kept
  grade 3, not ambiguous: the node states the right the need asks about, a missing fact is not an
  interpretation, and for any other debit the first subparagraph and 77(1) carry the answer. The
  session that judged as A did not overrule it.
- Impact: none on the chunk metrics. The node shares its chunk with 76(1), also grade 3, at 500
  tokens (chunk `PSD2:76`) and at 300 (chunk `PSD2:76.1`), so the relevant chunks (2 and 3), the
  grade of each chunk and Recall@5 are the same whatever grade this node has. At node level there
  are 8 relevant nodes; with a grade below 2 there would be 7.
- Gap for a later protocol version, not adopted: how a branch whose applicability depends on a
  fact the need does not state is graded (a grade 3 only if the branch itself answers; whether an
  open applicability makes it ambiguous or grade 1).

**N010, `drivers: []`.** PROTOCOL.md has no formal definition of `drivers`. The nearest text is
the schema comment (properties of the evidence), step 3 and the twins rule of section 2: a twin
has identical own text, the same article title, or belongs to the parallel regime, and it is a
twin of a node that answers the need. N010 is `none`, so it has no node of grade 2 or more to be
the twin of. Its distractors share words or a topic with the need (the terms of a client
agreement, payment information) but are not twins of an answering node. The field stays empty.
With the same reading, `lexical_twin` holds in N004, N006, N008 and N011, where a corpus node is a
twin of the answer; N014 and N015 rest on the parallel-regime clause only.

### 11. Protocol v1.3 (2026-10-12)

The review of section 10 left two gaps of v1.2 open, and the person closed them with two rules.
v1.3 writes them into PROTOCOL.md: applicability and ambiguity in sections 2 and 4, escalation
after a change of protocol in section 12, and the drivers in section 13. The head of PROTOCOL.md
says what changed from v1.2, what did not, and why it came before the next 20 needs. No need was
rewritten and no query changed. DEV is sealed at 0.6.0.

**How batch 1 was checked against v1.3.**

- All 118 labels were screened twice against the three changed clauses, each screen blind to the
  other. The session that judged as A wrote its screen first (18:12 UTC). A fresh context received
  the needs, the nodes, the labels with their rationales and the current rules, and nothing else.
- Both screens flagged PSD2 76(1) fourth subparagraph in N013. The fresh screen also flagged MiCA
  59(5) in N009.
- Each flagged label was an agreement of A and B, so it went to a fresh judge C as an escalated
  agreement (section 12). Each C received the two earlier rationales as "Rationale A" and
  "Rationale B" in random order, the open fact stated neutrally, and the v1.3 rules. Neither C saw
  the screens or the reason for the flag.
- Each transcript shows one file read and no other tool.

| Need | Node | A and B (v1.2) | Judge C (v1.3) | Result |
|---|---|---|---|---|
| N013 | PSD2 76(1), fourth subparagraph | 3 and 3 | 2 | Changed. The node answers directly, but applies only to "direct debits as referred to in Article 1 of Regulation (EU) No 260/2012": a fact the need does not state, and one that could reasonably hold. |
| N009 | MiCA 59(5) | 1 and 1 | 1 | Kept. Even if the content of the emails brought them under it, Article 61(2) and the second subparagraph of 61(1) already answer the email question. |

The judge C of section 10 kept grade 3 under v1.2, which had no rule for this case. v1.3 has one,
and this decision supersedes that one.

The other labels of the changed clauses held:

- The other 13 grade 3 nodes apply on facts the need states.
- DORA 42(6) in N014 stays ambiguous. Its doubt is one of interpretation: whether "must be able
  to end" covers a termination ordered by an authority. The missing fact, whether the vendor is
  critical, is not the only ground.
- The grade 2 nodes that depend on an unstated fact depend on one that could reasonably hold:
  PSD2 101(2) second subparagraph in N012, and 76(3) and its points in N013.

**Impact on the metrics: none.** A grade 2 is still relevant, so N013 keeps 8 relevant nodes and
DEV keeps 27. The node shares its chunk with 76(1), grade 3, at 500 tokens (`PSD2:76`) and at 300
(`PSD2:76.1`). A chunk takes the maximum grade of the nodes it owns, so the chunk grades, and with
them the ideal ranking of nDCG, do not move. N013 keeps 2 relevant chunks at 500 and 3 at 300.
Computed on the nine evaluated needs and both layers: the chunk grades and the relevant nodes
before (0.5.0) and after (0.6.0) are identical. Recall@K, MRR, nDCG and Recall@T are therefore
the same for any run. Only a graded measure at node level would see the change, and Fundamento
computes none.

**Drivers under section 13.** For each need, the nodes that make each driver hold:

| Need | Drivers | Why they hold |
|---|---|---|
| N004 | numeric, lexical_twin | 6(5): "at least once a year". The twin is DORA 16(2): the review of the simplified framework, the same duty for the entities of 16(1). |
| N006 | exception, lexical_twin | 2(1)(b) decides that the payment-institution exemption does not take the business out of DORA. The twin is 2(1)(d): the same inclusion, for exempted e-money institutions. |
| N008 | numeric, lexical_twin | 75(5): "at least once every three months". The twin is MiCA 81(14) with its second subparagraph: the periodic statement of portfolio management, "every three months". |
| N009 | exception, negation, cross_reference (added) | The exception is 61(1) with its second subparagraph, and the negation is 61(2): "shall not entitle". 61(1) names "the requirement for authorisation under Article 59", and 59(1) is graded 2; 61(2) refers back to "paragraph 1". |
| N010 | none | A `none` need has no relevant node, so it has no twin and no reference. Its gap is a Level 1 act, Regulation (EU) 2023/1113, not a figure, a date or a Level 2 act. Its distractors (MiCA 82(1)(a) and (b), PSD2 31(2), 45(1)(a), 57(1)(a)) share words or a topic, not a duty. |
| N011 | lexical_twin (cross_reference removed) | The twin is MiCA 34(9): the same article title, "Governance arrangements", and the same continuity duty for issuers of asset-referenced tokens. 68(7) refers to DORA Articles 11 and 12, but every one of their nodes is graded 1, so the reference only marks where to look. |
| N012 | numeric, exception | 101(2): 15 business days, and 35 in exceptional situations; the derogation is in 101(2), second subparagraph. PSD2 77(2), with its 10 business days, is a different duty: a distractor, not a twin. |
| N013 | numeric, exception, cross_reference | 77(1): 8 weeks. The exception is 76(3) with its points, and the conditions are 76(1)(a) and (b). 77(1) names "the refund referred to in Article 76", and 76(1) fourth subparagraph names paragraph 3 and Article 77. PSD2 73(1) and 71(1) are ruled out by the facts (the debit was authorised): distractors, not twins. |
| N014 | lexical_twin | The twin is MiCA 73(3): the right to terminate outsourcing agreements, for crypto-asset service providers. |
| N015 | numeric, lexical_twin | The `gap_note` names the maximum fine as missing. The twin is MiCA 111(3): maximum administrative fines, for infringements of MiCA. |

Counts after v1.3:

| Driver | Needs |
|---|---|
| lexical_twin | 6 |
| numeric | 5 |
| exception | 4 |
| cross_reference | 2 |
| negation | 1 |
| cross_act, date, level2_gap | 0 each |

**One choice of wording.** `cross_reference` could have meant "a relevant node contains an
explicit reference", and N011 would have kept it. Section 13 also asks that the referred provision
be relevant: v1.2 had already decided that a reference marks where to look, and under the looser
reading N011 would sit in a slice where following the reference finds nothing relevant.

**Record.** `adjudications.jsonl` holds the two escalated agreements, each with its reason.
`validate` reports them apart from the 18 disagreements: one label kept by judge C, one changed.

