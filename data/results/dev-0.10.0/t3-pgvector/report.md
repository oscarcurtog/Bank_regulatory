# T3: the E5 baseline in PostgreSQL with pgvector, DEV 0.10.0

An infrastructure migration, not an experiment: the stored E5 vectors and the stored E5 run, moved into the database and compared. Produced by `scripts/pgvector_baseline.py`.

## Same data, same embeddings, same similarity

- 1354 chunks and their vectors (1024 dimensions) loaded; the layer rebuilt from its rows equals the chunk files field by field, and the matrix read back from PostgreSQL has the SHA-256 that meta.json records (`eae637821e3e1fb2`).
- Search: exact: no HNSW or IVFFlat index; ORDER BY embedding <#> query, chunk_id. Operator: <#> (negative inner product); score = -(embedding <#> query). Approximate indexes: none.
- PostgreSQL 18.6 (Debian 18.6-1.pgdg12+2), pgvector 0.8.7, database collation en_US.utf8 (chunk ids are COLLATE "C").

## Same ranking

- Full rankings identical: 0 of 90 queries (every chunk, every position).
- Top 10 identical: 81 of 81 evaluated queries.
- Scores: 7199 of 121860 bit-identical; largest absolute difference 1.907e-06 (32.0 float32 ulps), p99 1.013e-06.
- Ties in the stored run: 56 adjacent pairs with equal scores, 0 of them between bit-identical vectors (0 groups of duplicate vectors in the layer). Smallest gap between neighbours in the top 11 of an evaluated query: 1.431e-06.
- Where the full rankings differ: 518 inverted neighbour pairs, 26 of them exact ties in the stored run; 0 not explained by the two scores' own floating-point differences. No chunk moves more than 2 positions; the first difference is at position 25 at the earliest (median 263.0).
- Fragile top-11 neighbours (stored gap under twice the largest difference): N008-es-1 10/11: gap 1.43e-06 stored, 2.98e-07 moved; N022-en-1 5/6: gap 3.40e-06 stored, 3.81e-06 moved; N023-en-1 2/3: gap 2.56e-06 stored, 1.37e-06 moved. Their order held in this run, but the margins alone do not guarantee it.
- Tolerance τ = 1e-05: every score within τ: True; every top-11 gap above 2τ: False.

## Same metrics

- Official metrics: largest difference 0.0e+00. Every diagnostic block (global, per language, groups, dependent evidence, full evidence) identical: True. Labels were read only for this step, after the rankings had been compared.

## Provenance

- 810 of 810 top-10 results resolved to act, path, text and source document.

## Latency

- one top-10 search, the query vector already computed; 90 queries × 5 passes, warm: one untimed pass first (statement prepared, table in shared buffers).
- Client round trip: p50 5.60 ms, p95 6.82 ms, max 25.04 ms (n = 450, nearest rank).
- Server execution: p50 3.96 ms, max 5.17 ms (n = 90). Plan: Limit > Sort > Bitmap Heap Scan > Bitmap Index Scan.
- In-memory exact search of the baseline (numpy), same vectors: p50 0.052 ms, p95 0.060 ms.
- Machine: Apple M1, macOS-27.0.1-arm64-arm-64bit; PostgreSQL in Docker Desktop on the same machine.
