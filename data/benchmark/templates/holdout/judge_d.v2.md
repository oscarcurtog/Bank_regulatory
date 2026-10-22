$opening

You are judge D of a retrieval benchmark on EU financial regulation (DORA, MiCA and PSD2). Below are an information need and the candidate nodes examined for it. You are asked to grade only the items listed under ITEMS TO GRADE; the other candidates are shown so that you see the same legal context as anyone who graded this need. You do not know whether anyone graded these items before, or what they decided, and must not try to guess. Grade on your own, using only the words of the legal text shown and the facts the need states. Nobody here is a legal expert: when a grade depends on an interpretation the text cannot settle, mark it ambiguous instead of forcing it; a fact the need does not state is not a reason for ambiguity (see the rules).

THE RULES (quoted from the protocol; examples omitted)

$rules

YOUR TASK
1. For each node item, give a grade 0, 1, 2 or 3, or null to leave it unjudged when it does not deserve a judgment. Set "ambiguous": true when the grade depends on an interpretation that the text cannot settle, and still give your best grade. Before any grade 0, do the scope check of section 2 and list in "scope_basis" the provisions you read.
2. For the answerability item, if it is listed, give the need's answerability under section 3.
3. Every item and every candidate carries an id the orchestrator gave it ("ITEM C-...", "CANDIDATE C-..."). An item's id is the id of the candidate it names, or of the need's answerability. Grade ONLY what each item's id names, never another legal node, and answer by that exact id, copied as it is; do not write the item's act, article or path in your answer.

THE NEED
$need_id: $statement

ITEMS TO GRADE
$items

ALL THE CANDIDATES (document order)

$candidates

OUTPUT
JSON only, no other text, one entry per item, by its id:
{"need_id": "$need_id", "items": [{"item_id": "C-... exactly as given", "grade": 0, "ambiguous": false, "answerability": null, "scope_basis": ["ACT:path"], "reason": "one or two sentences quoting the words of the text and citing the rule applied"}]}
For a node item, give "grade" and "answerability": null. For the answerability item, give "answerability" (complete, partial or none) and "grade": null.
