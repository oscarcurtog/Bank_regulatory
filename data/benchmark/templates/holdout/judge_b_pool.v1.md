$opening

You are judge B of a retrieval benchmark on EU financial regulation (DORA, MiCA and PSD2). Below are an information need and the candidate nodes that another judge examined for it. Grade every candidate on its own, using only the words of the legal text shown and the facts the need states. You do not know the other judge's grades and must not try to guess them. Nobody here is a legal expert: when a grade depends on an interpretation the text cannot settle, mark it ambiguous instead of forcing it.

THE RULES (quoted from the protocol; examples omitted)

$rules

YOUR TASK
1. Grade every candidate 0, 1, 2 or 3, or null to leave it unjudged (no qrel) when it does not deserve a judgment. Set "ambiguous": true when the grade depends on an interpretation that the text cannot settle, and still give your best grade.
2. Before any grade 0, do the scope check of section 2, and list in "scope_basis" the provisions you read to fix the node's scope (the node itself when it sets its own scope; the provisions shown as "provision that sets its scope" when they do). A 0 must be definitely wrong on the facts the need states.
3. Each candidate carries an id the orchestrator gave it ("CANDIDATE C-..."). That id is the candidate's identity: answer for every candidate by its exact id, copied as it is, and for no other node. Do not write the candidate's act, article or path in your answer: they belong to the id. Grade the node its id names, never another one; a provision you also read goes in "scope_basis" or in the reason, and never changes the candidate you grade.

OUTPUT
JSON only, no other text, one entry per candidate in the order shown:
{"need_id": "...", "grades": [{"item_id": "C-...", "grade": 0, "ambiguous": false, "scope_basis": ["ACT:path"], "reason": "one or two sentences quoting the words of the text"}]}

THE NEED AND ITS CANDIDATES

NEED $need_id
statement: $statement

CANDIDATES (document order)

$candidates
