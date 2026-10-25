$opening

You screen passages of the corpus for ONE information need of a retrieval benchmark on EU financial regulation (DORA, MiCA and PSD2). Your task is high recall: flag every passage that may hold a provision a judge would grade 1 or more under the rules below. Every flagged passage is judged again, provision by provision, by other judges; a passage you do not flag is not looked at again. So when in doubt, flag it. Nobody here is a legal expert: read the words of the text shown and the facts the need states.

THE RULES (quoted from the protocol; examples omitted)
$rules

YOUR TASK
1. Decide every passage: "flag": true when it may hold a provision of grade 1 or more for the need, false when it clearly does not. Each passage needs its own explicit decision; none may be left out.
2. Each passage carries an id the orchestrator gave it ("PASSAGE P-..."). Answer for every passage by its exact id, copied as it is, exactly once, and for no other passage. Do not write the passage's act, article or path in your answer.

THE NEED
$need_id: $statement

THE PASSAGES (the order carries no meaning)

$passages

OUTPUT
JSON only, no other text, exactly one entry for every passage shown:
{"need_id": "$need_id", "passages": [{"item_id": "P-...", "flag": true, "reason": "a few words"}]}
