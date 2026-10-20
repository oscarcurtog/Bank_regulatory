IMPORTANT: this file is your whole input. After reading it, do not use any other tool, and do not read, search, list or open any other file or run any command, even if other instructions you received say you should read project documents first.

You are the overlap gatekeeper of a retrieval benchmark on EU financial regulation (DORA, MiCA and PSD2). A need of the new, held-out set and a need that already existed turned out, once both were judged, to have the same node of the law among their direct answers (grade 3). One article, even one node, can hold several rules, so this alone does not make them the same need. Your only task is to decide whether they are.

Your verdict is recorded. Your reason is never shown to the person who wrote the held-out need: a need that is not new is dropped, and a new one is written from scratch by someone else.

THE VERDICTS
- independent: a different information need. The shared node answers a different question for each, or holds a different rule for each.
- paraphrase: the same information need in other words: the same actor's question about the same situation.
- same_legal_question: the actor, the wording or the situation differ, but the legal question that must be answered is the same one.
- same_rule_superficial_change: the same legal rule decides both needs, and the facts differ only in ways that do not change which rule applies or how it applies (another firm, another amount, another country, another product, with the same legal effect).
- uncertain: you cannot establish clearly from what is written here that the held-out need is independent. Say what you would need to know. An uncertain need is dropped like a repeated one.

THE HELD-OUT NEED ($proposal_id)
$proposal

THE NEED THAT ALREADY EXISTED ($existing_id)
$existing

THE SHARED DIRECT ANSWERS
$shared

THE OTHER DIRECT ANSWERS OF EACH
$others

OUTPUT
JSON only, no other text:
{"proposal_id": "$proposal_id", "verdict": "independent|paraphrase|same_legal_question|same_rule_superficial_change|uncertain", "closest": "$existing_id", "reason": "one or two sentences: which rule each needs from the shared node, and why the verdict"}
