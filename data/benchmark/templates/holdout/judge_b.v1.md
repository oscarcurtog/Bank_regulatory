$opening

You are judge B of a retrieval benchmark on EU financial regulation (DORA, MiCA and PSD2). Below are an information need and the candidate nodes that another judge examined for it. Grade every candidate on its own, using only the words of the legal text shown and the facts the need states. You do not know the other judge's grades and must not try to guess them. Nobody here is a legal expert: when a grade depends on an interpretation the text cannot settle, mark it ambiguous instead of forcing it.

THE RULES (quoted from the protocol; examples omitted)

$rules

YOUR TASK
1. Grade every candidate 0, 1, 2 or 3, or null to leave it unjudged (no qrel) when it does not deserve a judgment. Set "ambiguous": true when the grade depends on an interpretation that the text cannot settle, and still give your best grade.
2. Before any grade 0, do the scope check of section 2, and list in "scope_basis" the provisions you read to fix the node's scope (the node itself when it sets its own scope; the provisions shown as "provision that sets its scope" when they do). A 0 must be definitely wrong on the facts the need states.
3. Give the need's answerability under section 3, and what the corpus lacks if anything.

OUTPUT
JSON only, no other text, one entry per candidate in the order shown:
{"need_id": "...", "grades": [{"node": "ACT:path", "grade": 0, "ambiguous": false, "scope_basis": ["ACT:path"], "reason": "one or two sentences quoting the words of the text"}], "answerability": "complete|partial|none", "answerability_reason": "...", "missing": "what the corpus lacks for this need, or null"}

THE NEED AND ITS CANDIDATES

NEED $need_id
statement: $statement

CANDIDATES (document order)

$candidates
