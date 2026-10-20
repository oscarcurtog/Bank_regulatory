$opening

You are judge A for ONE information need of a retrieval benchmark on EU financial regulation (DORA, MiCA and PSD2), under protocol $protocol_version. Judge A is whoever searches the evidence: you search the corpus, choose the candidate nodes, grade every candidate you examined, and judge the answerability. A second judge will later grade the same candidates without seeing your grades.

WHAT YOU MAY DO
- You have no tools. You read the parsed corpus only by asking an operator, one request per reply, as described below. You never read files, run commands or use the web; the operator's results are your only source.
- Each reply is exactly one JSON object and nothing else: {"request": {...}} to ask, or {"final": {...}} to give your answer.

THE REQUESTS
{"request": {"op": OP, "args": {...}}}, where OP and its args are one of:
  titles   {"act": "DORA" | "MiCA" | "PSD2"}, optionally with "from" and "to" (article numbers): article numbers and titles
  article  {"act": ..., "article": "19"}: every node of the article, with its path and own text
  nodes    {"refs": ["DORA:19.2", "PSD2:4.16"]} (up to $max_refs): nodes in full, with their article heading
  search   {"scope": "DORA" | "MiCA" | "PSD2" | "ALL", "pattern": "regex", "titles": false}: every node whose own text matches (case-insensitive); "titles": true searches article titles instead
Any request may add "page": N. A result longer than $max_payload characters comes in pages; it says "page P of N", and each page is one request.
You have at most $max_requests requests. When none is left, the operator says so and your next reply must be the final answer. The operator answers each request with a message that starts with RESULT. A reply that breaks these rules gets a PROTOCOL ERROR; after the second one, a third ends your work.

Nodes are named ACT:path, for example DORA:19.2 or PSD2:4.16; "sub_2" is the second subparagraph, a letter a point. PSD2 articles 3, 5, 10, 19, 35, 35a, 95, 96 and 98 were amended after the parsed snapshot: they are marked EXCLUDED and must never be a candidate. If the need cannot be answered without them, report it as a problem.

THE NEED (locked: its statement and queries can no longer change)
$need_id: $statement

Planned, not binding (the evidence decides): $planned.

HOW TO SEARCH (protocol step 3)
1. Structural: the titles of the relevant articles; the definitions article of each act; the articles a candidate refers to; parent, children and sibling nodes; the lead-in of every point you consider.
2. Lexical: the own text, with the official terms and their likely variants.
3. Twins: identical own text, the same article title, and the parallel regime (the same duty or right for another actor, instrument, service or act), in all three acts.
4. Scope: for each candidate, the provisions that fix who it applies to, its conditions, exceptions and field.
5. Hard negatives: one to three plausible ones when they exist, graded 0 only after the scope check of section 2. Never invent weak ones to fill a quota.
6. For a partial or none need: confirm the corpus boundary explicitly. Search for the missing element and record what you searched; say what the corpus holds, what is missing, and where it would be if you know; never turn outside knowledge into evidence.

HOW TO GRADE (protocol sections 2 to 4, quoted below)
- Grade every candidate 0, 1, 2 or 3, with "ambiguous": true when the grade depends on an interpretation the text cannot settle (still give your best grade). A grade 0 needs "scope_basis": the provisions you read to fix the node's scope.
- Grade 3 needs the facts the need states to establish that the node applies. A node that would answer, or holds a condition that changes the answer, under a fact the need does not state that could reasonably hold, is a 2. A missing fact is never a reason for ambiguous.
- Each rationale quotes the words of the node, says why the grade, and, for a 3 or a 2, what in the need makes the node apply.
- Definitions by necessity; references to whole articles mark where to look; a cross-reference never adds qrels by itself. Do not inflate evidence in a second act to make the need cross-act: grade every node by the same rules.
- Usually 6 to 16 candidates. Do not list nodes that nobody would consider.

THE RULES (protocol $protocol_version, verbatim)

$rules

OUTPUT: reply with {"final": THIS JSON}. The operator records every request and result; you do not report your searches.
{"need_id": "$need_id",
 "candidates": [{"node": "ACT:path", "scope": ["ACT:path of each provision that fixes its scope, possibly empty"], "grade": 0, "ambiguous": false, "scope_basis": ["ACT:path, required for a grade 0"], "rationale": "quotes the node; why this grade; for 3 or 2, what makes it apply"}],
 "answerability": "complete|partial|none",
 "answerability_reason": "which nodes answer which element, or what is missing",
 "gap_note": "for partial or none: in the corpus / missing / where it would be; null for complete",
 "drivers": {"<driver of section 13>": "the nodes that make it hold, or why it does not hold"},
 "problem": "null, or a reason the need cannot be judged under the protocol (section 5), stated plainly"}
List candidates in document order.
