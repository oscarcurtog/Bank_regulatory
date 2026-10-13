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

## bge-m3, dense output only

[`bge-m3.json`](bge-m3.json). BGE-M3 produces three representations from one forward pass: a
dense vector, sparse lexical weights and ColBERT multi-vectors. This contract uses the dense vector
only. Every value was read from the model's own files at the pinned revision and from the reference
implementation (FlagEmbedding v1.4.2, `BGEM3FlagModel`), not assumed:

| Field | Value | Where it comes from |
|---|---|---|
| `model_id`, `revision` | `BAAI/bge-m3` at `5617a9f61b028005a4858fdac845db406aefb181` | the Hugging Face model API |
| license | MIT | model card metadata |
| tokenizer | XLM-RoBERTa, SentencePiece, 250,002 tokens; `<s>` opens and `</s>` closes every input | `tokenizer.json`, `tokenizer_config.json` |
| `dimension` | 1024 | `config.json` (`hidden_size`), `1_Pooling/config.json` |
| `max_input_tokens` | 8192, special tokens included | `sentence_bert_config.json` (`max_seq_length`), `tokenizer_config.json` (`model_max_length`), `config.json` (8194 positions, 2 reserved), the model card |
| `query_prefix`, `document_prefix` | none | model card: "no longer requires adding instructions to the queries" |
| `pooling` | the hidden state of the first token (`<s>`, the CLS position) | `1_Pooling/config.json` (`pooling_mode_cls_token`), `_dense_embedding` in FlagEmbedding |
| `normalize` | L2 | the `Normalize` module in `modules.json`; `normalize_embeddings=True` in FlagEmbedding |
| `similarity` | dot product | with L2-normalized vectors the dot product is the cosine similarity |
| `dtype` | float32 | `config.json`; the model card's examples use fp16 for speed, which this project does not |

No wrapper is needed. The dense vector is computed from the plain transformer
(`transformers.AutoModel`), exactly as FlagEmbedding computes it. The sparse and ColBERT heads are
separate files (`sparse_linear.pt`, `colbert_linear.pt`) that the encoder never loads, so neither
lexical weights nor multi-vectors can enter the score. `FlagEmbedding` is not a dependency.

Two differences from the reference library are deliberate and recorded:

- `BGEM3FlagModel` defaults to `query_max_length=512` and `passage_max_length=512` and truncates
  silently at that length. That default belongs to the library, not to the model, whose limit is
  8192. This contract uses the model's limit, as the comparison policy below requires.
- The model card prints its example scores with `use_fp16=True`. This project runs float32 on CPU;
  `tests/integration/test_embedding_model_bge.py` reproduces the card's scores within 0.005 and
  matches the reference dense vector within 1e-5.

The pinned revision ships `pytorch_model.bin` and no safetensors. Loaded plainly, transformers
reads that file and also starts a background download of `model.safetensors` from the Hub's
automatic conversion pull request (`refs/pr/130`, commit `9a0624b8`), a revision nobody pinned.
The encoder turns that download off. The 391 tensors it loads were checked equal, bit for bit, to
the pinned `pytorch_model.bin`.

## qwen3-embedding-0.6b

[`qwen3-embedding-0.6b.json`](qwen3-embedding-0.6b.json), the third model of the retrieval
iteration (experiment 5 of [`config/retrieval/README.md`](../retrieval/README.md)). It differs
from the other two in kind, not only in weights: a decoder (Qwen3, 28 layers) trained for
embeddings in 2025, instruction-aware, with Matryoshka dimensions, where E5 and BGE-M3 are both
XLM-RoBERTa encoders. Every value was read from the model's files at the pinned revision:

| Field | Value | Where it comes from |
|---|---|---|
| `model_id`, `revision` | `Qwen/Qwen3-Embedding-0.6B` at `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | the Hugging Face model API; not gated |
| license | Apache-2.0 | model card metadata |
| tokenizer | byte-level BPE (151,643 tokens); `<\|endoftext\|>` closes every input, nothing opens it | `tokenizer.json` (its post-processor), `tokenizer_config.json` |
| `dimension` | 1024, the full Matryoshka width (the model allows 32 to 1024) | `config.json` (`hidden_size`), `1_Pooling/config.json`, model card |
| `max_input_tokens` | 32768 | `config.json` (`max_position_embeddings`), model card ("32k") |
| `query_prefix` | `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:` | `config_sentence_transformers.json` (the "query" prompt), the model card's `get_detailed_instruct` |
| `document_prefix` | none | `config_sentence_transformers.json` (the "document" prompt is empty) |
| `pooling` | the last token: the closing `<\|endoftext\|>` | `1_Pooling/config.json` (`pooling_mode_lasttoken`), `last_token_pool` in the model card |
| `normalize` | L2 | the `Normalize` module in `modules.json` |
| `similarity` | dot product | the cosine on normalized vectors |
| `dtype` | float32 | the weights are stored in bfloat16, which float32 holds exactly |

The query prompt is the model's own default, not one written for this corpus: a prompt tuned on
DEV would be tuning on the benchmark. The model card pads on the left; the encoder pads on the
right and takes the last real token of each row, which a test shows to give the same vectors as
the card's code (within 4e-7) and the card's scores (within 2e-6). Its config declares no pad
token, so the encoder pads with the end token: padding is masked and, in a causal model, comes
after every real token, so it never reaches a vector.

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

## What makes stored embeddings stale

A stored set (`data/embeddings/<model>/<input set>/`, not versioned) carries in its `meta.json`
the identity it was computed with, and is reused only if the identity of the current run is
equal, field by field. Otherwise the set is reported stale, naming what changed, and computed again.

| Part of the identity | Fields |
|---|---|
| model contract | the SHA-256 of this JSON file's validated content: model id, revision, tokenizer revision, tokenizer file, pooling, prefixes, normalization, similarity, input limit, truncation policies, dimension, batch size, device, dtype |
| tokenizer | the SHA-256 of the tokenizer file actually loaded |
| runtime | Python, torch, transformers, tokenizers, numpy and sentence-transformers versions (`null` when not installed), dtype, device |
| data | for documents: the chunk layer (budget, chunker and parser versions, source hashes, structure and content SHA-256); for queries: benchmark version, `queries.jsonl` SHA-256 and content SHA-256 |
| output | the SHA-256 of the vector matrix, checked on every load |

Recorded but not part of the identity: the creation time, the hardware (platform, CPU, core
count), the Hugging Face cache path and the torch thread count. They describe a run; they do not
change what a vector is, so a new machine or a new day never makes a set stale. A set written with
an older meta schema is stale by definition: its identity cannot be compared field by field.

## What changes between models

The contract is the same shape for every model; the values are the model's own. Pooling (mean of
the token vectors, the first token, or the last real token), the prefixes, the input limit,
normalization and the similarity come from each model's files at its pinned revision. A model
prefixes both queries and documents, only the query (an instruction-aware model), or neither; a
prefix on documents alone is refused, and so are two equal prefixes, because they would erase the
asymmetry the model was trained with.

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
