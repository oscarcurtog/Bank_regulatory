$opening Work only from it and your general knowledge.

You design ONE information need for a retrieval benchmark on EU financial regulation. An information need is what a real professional or consumer wants to know: who the actor is, their situation, and ONE primary question. You write its concept: the scenario, the facts that decide which rules apply, and the question. Another writer will turn your concept into the final statement; queries will later be written from that statement alone; someone else will then look for the answer in the articles of the law. You never see the legal text, and you must not answer the question.

THE CORPUS
- DORA: Regulation (EU) 2022/2554 on digital operational resilience for the financial sector, as originally adopted.
- MiCA: Regulation (EU) 2023/1114 on markets in crypto-assets, consolidated version of 9 January 2024.
- PSD2: Directive (EU) 2015/2366 on payment services, as first published. Later amendments are not in the corpus, so no need may rely on the PSD2 articles amended since then: Articles 3, 5, 10, 19, 35, 35a, 95, 96 and 98 (broadly: the exclusions from its scope, the application for authorisation, safeguarding of funds, agents, branches and outsourcing, access to payment systems, operational and security risk management, incident reporting, and the technical standards on authentication).
- Only the articles of these three acts. Not in the corpus: recitals, annexes, Level 2 acts (technical standards, delegated and implementing acts), EBA or ESMA guidelines and Q&A, national law, and every other EU act (for example MiFID II, the AML directives, CRR, GDPR, the E-Money Directive).

YOUR CELL ($cell_id)
- Act(s): $acts
- Area: $area
- Type: $type_name: $type_definition
- Expected answerability: $answerability: $answerability_definition
- Difficulty: $difficulty: $difficulty_definition
- Expected answer: $expected_answer
- Situation: $attempt_frame

RULES FOR THE CONCEPT
1. A realistic actor who needs the answer to take a decision: for example a compliance officer, a lawyer, a product manager, a fintech founder, an ICT service provider, or a payment or crypto customer.
2. One primary question, asked once. It may need several provisions to answer; that is fine, as long as it is one thing the actor wants to know. Never several obligations, several moments of a process, or several independent decisions.
3. Plain words that the actor would use. No article, paragraph or point numbers; no titles or numbers of legal acts other than the common names DORA, MiCA and PSD2 (and the common names of EU bodies, such as ESMA, when the actor would use them); no legal wording the actor would not know.
4. List the facts that decide which rules apply, as facts of the case, in plain words. A fact that the answer depends on must be stated; add no fact that the answer does not need.
5. Do not answer the question, and do not hint at the answer or at where it is in the law.
6. Not navigation ("show me the article on X"), not a request for a summary, not about how the law changed over time or between versions, and not about acts still to be adopted.
7. If the cell cannot hold a realistic need as described, say so in "fit" instead of forcing one.
8. A figure or a date is asked for only when a real actor in this situation would ask for one. Never invent a figure to fit the cell.

ATOMICITY CHECK
Before you answer, ask: "Could this question be split into two questions that a user would reasonably ask separately?" If yes, narrow it until the answer is no.

OUTPUT
JSON only, no other text:
{"cell_id": "$cell_id", "concept": "two to four sentences: the actor, the situation and the question", "facts": ["each fact the statement must state"], "atomicity": "one sentence: why the question cannot be split into two", "figure_or_date": "yes if the answer is a figure, a period or a date; no otherwise", "fit": "fits, or a sentence on why the cell cannot hold a realistic need"}
