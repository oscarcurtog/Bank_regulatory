# Embedding models

Each JSON file in this directory is the contract of one embedding model: exactly what the dense
retrieval baseline runs. `ingest/embeddings.py` validates it (`EmbeddingConfig`, unknown keys
refused) and hashes it. Changing any field makes the embeddings computed before stale.

## multilingual-e5-large

[`multilingual-e5-large.json`](multilingual-e5-large.json). Every value was read from the model's
own files at the pinned revision, not assumed:

| Field | Value | Where it comes from |
|---|---|---|
| `model_id`, `revision` | `intfloat/multilingual-e5-large` at `3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3` | the Hugging Face model API |
| tokenizer | XLM-RoBERTa, SentencePiece, 250,002 tokens; `<s>` opens and `</s>` closes every input | `tokenizer.json`, `tokenizer_config.json` |
| `dimension` | 1024 | `config.json` (`hidden_size`), `1_Pooling/config.json` |
| `max_input_tokens` | 512, prefix and special tokens included | `sentence_bert_config.json` (`max_seq_length`), `tokenizer_config.json`, `config.json` (514 positions, 2 reserved) |
| `query_prefix`, `document_prefix` | `query: `, `passage: ` | model card: every input starts with one of them, also in other languages |
| `pooling` | mean of the token vectors, padding masked out | `1_Pooling/config.json`, `average_pool` in the model card |
| `normalize` | L2 | the `Normalize` module in `modules.json`; `F.normalize` in the model card |
| `similarity` | dot product | with L2-normalized vectors the dot product is the cosine similarity |
| `dtype` | float32 | `config.json` |

`batch_size` and `device` are choices of this project, not of the model. They are part of the
contract because they can change the last bits of a vector: CPU is the reference device because it
is deterministic and runs everywhere.

What is embedded is the prefix plus `chunk.text` for a document, and the prefix plus the query text
for a query. Nothing else is added: no act, path, grade or other metadata. The article heading and
the governing lead-ins are in `chunk.text` already, by the chunker's design (`ingest/chunks.py`).

## Truncation is part of the contract, and it is never silent

The official chunk layer is fixed and counted in `cl100k_base` tokens (D-005). A model counts with
its own tokenizer, so a chunk that fits the layer's budget can exceed the model's input limit.

- Every input is tokenized whole first, with its prefix and special tokens, and counted.
- A document over the limit is cut at the tail, inside the special tokens, and the cut is recorded
  for each chunk: original tokens, encoded tokens, dropped tokens, and how many characters of
  `chunk.text` the model still sees.
- A query is never cut: an over-limit query is an error.
- The encoder receives token ids, not text, so it cannot cut anything by itself, and it refuses an
  input whose length differs from its record.

Libraries such as `sentence-transformers`, or a Hugging Face tokenizer called with
`truncation=True`, cut at the limit without saying so. That is the behaviour this design rules out.

## Model comparison policy

1. The main comparison of embedding models uses one fixed chunk layer: the official one
   (500 `cl100k_base` tokens, chunker 0.1.0, parser 1.4.0), the same chunk ids and the same
   node-to-chunk projection of the qrels.
2. Every model first goes through a tokenizer compatibility check on that layer.
3. A chunk over the model's limit is truncated explicitly, as the model's contract says. The
   metrics are reported together with the truncation exposure: how many chunks, how many tokens,
   which acts, which needs have a truncated relevant chunk, and which judged nodes fall in the
   truncated text.
4. The chunking never changes per model in this comparison. The variable is the model, and the
   input limit belongs to the model.
5. Chunking optimized for one model is a different experiment, reported apart, and never mixed
   into this comparison.

Truncation never changes the benchmark: no grade, relevance, answerability or chunk ownership
moves because part of a chunk went unseen. It is reported as exposure, not corrected.
