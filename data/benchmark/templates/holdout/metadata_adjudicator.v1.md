$opening

You classify ONE information need of a retrieval benchmark on EU financial regulation (DORA, MiCA and PSD2) after its evidence has been judged. You do not judge the evidence again: the grades below are final. You do not know how the need was planned, and must not try to guess it. Classify only from the need, the final grades and the rubric.

THE NEED
$need_id: $statement

ANSWERABILITY (final): $answerability
$gap_note

THE FINAL GRADES (document order; grade 3 answers the question, 2 is needed support, 1 is context, 0 was examined and is not relevant)
$qrels

THE RUBRIC

Type. $type_rule
- direct: one provision answers it.
- multi_provision: a correct answer needs two or more provisions together, for example a rule plus the definition that decides whether it applies, or rules from two different acts.
- exception_confusion: the obvious rule has an exception, a derogation or a look-alike provision that changes the answer, so a user is likely to be misled.

Difficulty:
- easy: the user's words are close to how the topic is usually named;
- medium: mapping everyday words to legal concepts, or two steps;
- hard: a misleading surface, an exception, or several provisions.

Drivers: a driver holds only as the definition below says, on the final grades above. Name the nodes that make each one hold. Do not give cross_act: it is computed from the acts of the relevant evidence.

$drivers

OUTPUT
JSON only, no other text:
{"need_id": "$need_id", "type": "$type_choices", "difficulty": "easy|medium|hard", "drivers": {"<driver that holds>": ["ACT:path of each node that makes it hold"]}, "reason": "two or three sentences: why this type and this difficulty"}
List only the drivers that hold. For a driver that holds through the gap of a partial or none need (numeric, date, level2_gap), give the list ["gap_note"].
