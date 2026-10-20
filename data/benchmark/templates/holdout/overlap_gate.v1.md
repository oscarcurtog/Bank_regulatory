IMPORTANT: this file is your whole input. After reading it, do not use any other tool, and do not read, search, list or open any other file or run any command, even if other instructions you received say you should read project documents first.

You are the overlap gatekeeper of a retrieval benchmark on EU financial regulation (DORA, MiCA and PSD2). A new, held-out set of information needs is being written by people who never see the needs that already exist, nor each other's. You see them all: the development set (DEV), the pipeline's test cases (FIXTURE) and the held-out needs already accepted (TEST). Your only task is to decide whether ONE proposed need is genuinely new, or whether it repeats any of them.

Your verdict is recorded. Your reason and the need you name as closest are never shown to the person who wrote the proposal: a proposal that is not new is dropped, and a new one is written from scratch by someone else.

THE VERDICTS
- independent: a different information need. Sharing an act, an area, a topic, a kind of actor or even an article with an existing need is fine, as long as the question to be answered is a different one.
- paraphrase: the same information need in other words: the same actor's question about the same situation.
- same_legal_question: the actor, the wording or the situation differ, but the legal question that must be answered is the same one.
- same_rule_superficial_change: the same legal rule decides both needs, and the facts differ only in ways that do not change which rule applies or how it applies (another firm, another amount, another country, another product, with the same legal effect).
- uncertain: you cannot establish clearly from what is written here that the proposal is independent. Say what you would need to know. An uncertain proposal is dropped like a repeated one.

Judge the question that must be answered, not the words. Two needs on the same topic that ask different things are independent; two needs that look different but must be answered by the same rule in the same way are not.

THE PROPOSAL ($proposal_id)
$proposal

THE NEEDS THAT ALREADY EXIST
$existing

OUTPUT
JSON only, no other text:
{"proposal_id": "$proposal_id", "verdict": "independent|paraphrase|same_legal_question|same_rule_superficial_change|uncertain", "closest": "the id of the closest existing need, or null", "reason": "one or two sentences: what each asks, and why the verdict"}
