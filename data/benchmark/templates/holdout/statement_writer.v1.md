$opening

You write ONE information need for a retrieval benchmark on EU financial regulation. An information need is what a real professional or consumer wants to know, written in English as a short statement: who the actor is, the situation, and ONE primary question. Queries will later be written from your statement alone, so the statement must not point at where the answer is in the law, and must not contain the answer.

THE CORPUS
- DORA: Regulation (EU) 2022/2554 on digital operational resilience for the financial sector, as originally adopted.
- MiCA: Regulation (EU) 2023/1114 on markets in crypto-assets, consolidated version of 9 January 2024.
- PSD2: Directive (EU) 2015/2366 on payment services, as first published. Later amendments are not in the corpus, so no need may rely on the PSD2 articles amended since then: Articles 3, 5, 10, 19, 35, 35a, 95, 96 and 98 (broadly: the exclusions from its scope, the application for authorisation, safeguarding of funds, agents, branches and outsourcing, access to payment systems, operational and security risk management, incident reporting, and the technical standards on authentication).
- Only the articles of these three acts. Not in the corpus: recitals, annexes, Level 2 acts (technical standards, delegated and implementing acts), guidelines, national law, and every other EU act.

RULES FOR THE STATEMENT
1. English, two to four sentences, about 40 to 70 words: the actor, the situation, and one primary question.
2. One primary question, asked once. No second question, no follow-up, no "and if so". A question may need several provisions to answer; that is fine, as long as it is one thing the actor wants to know.
3. Plain words that the actor would use. No article, paragraph or point numbers; no titles or numbers of legal acts other than the common names DORA, MiCA and PSD2 (and the common names of EU bodies, such as ESMA, when the actor would use them); no legal wording the actor would not know.
4. State every fact listed in the cell, in plain words: they decide which rules apply. Add no fact beyond them.
5. Do not answer the question, and do not hint at the answer or at where it is in the law.

ATOMICITY CHECK
Before you answer, ask: "Could this statement be split into two questions that a user would reasonably ask separately?" If yes, narrow it until the answer is no.

YOUR CELL ($proposal_id)
- Act(s): $acts
- Area: $area
- Type: $type_name: $type_definition
- Expected answerability: $answerability
- Concept: $concept
- Facts the statement must state:
$facts

OUTPUT
JSON only, no other text:
{"proposal_id": "$proposal_id", "primary": "the statement", "primary_atomicity": "one sentence: why it cannot be split into two questions", "alternative": "a second statement, or null", "alternative_reason": "why you offer an alternative (for instance a doubt about atomicity), or null"}
Offer an alternative only if you doubt that the primary meets every rule; otherwise give null.
