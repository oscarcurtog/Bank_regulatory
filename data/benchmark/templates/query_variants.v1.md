# Query variants template, v1 (D-019)

Render it with `ingest.benchmark.render_query_prompt(statement, variants)` and paste the result
into a fresh context that has not read the corpus: a new session, or an agent given no files.
The prompt receives the need statement and the list of formulations wanted, nothing else: no
legal node, no article text, no qrel, no answer.

Record every query it produces with `origin: llm_from_need` and `template: query_variants.v1`,
and review it before it enters the benchmark. Never edit this file in place: the benchmark
manifest seals its SHA-256, so a change needs a new version (`query_variants.v2`).

---PROMPT---
You are helping to build a retrieval benchmark for questions about EU financial regulation.

Below is an information need: what a user wants to know. Write the question this user would
type into a search box, once for each formulation requested.

Information need:
{statement}

Formulations requested:
{variants}

Rules:
- Write from the user's point of view, the way they would actually ask.
- "en formal": English, as a compliance professional writes it. Official terminology is fine.
- "en natural": English, as a non-specialist writes it, with everyday words and no legal jargon.
- "es natural": Spanish, as a Spanish-speaking professional or customer would really ask it. It
  is not a translation of an English question and must not be shaped to resemble legal text.
- Do not quote or paraphrase legal provisions, article numbers or definitions unless the
  information need itself contains them.
- Do not answer the question.
- Every formulation must ask for the whole information need, not for part of it.

Return one line per formulation, in exactly this format:
<lang> <style>: <question>
