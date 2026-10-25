$opening

You decide the answerability of ONE information need of a retrieval benchmark on EU financial regulation (DORA, MiCA and PSD2), after its evidence has been judged. You do not judge the evidence again: the grades below are final. You do not know how the evidence was found and must not try to guess it. Nobody here is a legal expert: decide from the words of the text shown and the facts the need states.

THE RULES (quoted from the protocol; examples omitted)
$rules

YOUR TASK
Give the need's answerability under section 3 of the rules, from the judged evidence below and nothing else, and say what the corpus lacks if anything.

THE NEED
$need_id: $statement

THE JUDGED EVIDENCE (document order; grade 3 answers the question, 2 is needed support, 1 is context; "ambiguous" marks a grade the text could not settle)
$evidence

OUTPUT
JSON only, no other text:
{"need_id": "$need_id", "answerability": "complete|partial|none", "missing": "what the corpus lacks for this need, or null", "reason": "one or two sentences"}
