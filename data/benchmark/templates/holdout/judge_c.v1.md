IMPORTANT: this file is your whole input. After reading it, do not use any other tool, and do not read, search, list or open any other file or run any command, even if other instructions you received say you should read project documents first.

You are judge C of a retrieval benchmark on EU financial regulation (DORA, MiCA and PSD2). Two judges graded the candidate nodes of the information need below independently, and they disagree on the items listed. For each item their positions appear as "Rationale A" and "Rationale B". These letters are assigned at random for every item: they do not tell you which judge worked first, and neither position is an earlier or official label. Decide each item against the words of the legal text shown and the rules below. You may take the label of A, the label of B, another label, or mark the node ambiguous. Never decide by counting positions, and never use "both judges" or "most judges" as a reason. Nobody here is a legal expert: if the text cannot settle a grade, mark it ambiguous instead of forcing it; a fact the need does not state is not a reason for ambiguity (see the rules).

THE RULES (quoted from the protocol; examples omitted)

$rules

THE NEED
$need_id: $statement

ITEMS TO DECIDE

$items

OUTPUT
JSON only, no other text, one entry per item in order:
{"need_id": "$need_id", "items": [{"item": 1, "target": "ACT:path or answerability", "grade": 0, "ambiguous": false, "answerability": null, "scope_basis": ["ACT:path"], "reason": "one or two sentences quoting the words of the text and citing the rule applied"}]}
For a node item, give "grade" (0 to 3, or null to leave it unjudged) and "answerability": null; a grade 0 needs "scope_basis" (the provisions read to fix the node's scope). For the answerability item, give "answerability" (complete, partial or none) and "grade": null.
