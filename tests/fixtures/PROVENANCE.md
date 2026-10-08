# Fixture provenance

Every fixture in this directory is a verbatim excerpt of an official EUR-Lex XHTML
document: the `<div id="art_N">` subtree, wrapped in a minimal `<html xmlns=...>`
element so it is a well-formed document on its own. Nothing inside the excerpt was
edited, reformatted or synthesised.

`tests/integration/test_corpus.py::test_fixtures_match_corpus` proves each excerpt
yields exactly the same `Article.text` and node paths as the full document does, so
these files are the real case, not an approximation of it.

Source documents are identified by their CELEX number and by the SHA-256 of the raw
bytes downloaded on 2026-09-05, as recorded in `data/manifest.json`.

Reuse: source [EUR-Lex](https://eur-lex.europa.eu), © European Union, 1998-2026.
Legal documents published in EUR-Lex can be reused under the Commission's reuse
policy (Decision 2011/833/EU). The consolidated texts (`psd2_art9`, `psd2_art69`
and `psd2_art111`) are licensed under CC BY 4.0, which requires acknowledging the
source and indicating changes: the only change is the extraction described above.
`psd2_art9_original` comes from the Official Journal text, like the DORA fixtures.
These excerpts are included solely as test data and are not covered by the
repository's MIT License.

| Fixture | Act | Article | CELEX | Role | Source SHA-256 | Bytes |
|---|---|---|---|---|---|---|
| `dora_art45.xhtml` | DORA | Article 45 | `32022R2554` | original | `cf50d8f023d23b2a...` | 4301 |
| `dora_art60.xhtml` | DORA | Article 60 | `32022R2554` | original | `cf50d8f023d23b2a...` | 23411 |
| `dora_art15.xhtml` | DORA | Article 15 | `32022R2554` | original | `cf50d8f023d23b2a...` | 7800 |
| `dora_art36.xhtml` | DORA | Article 36 | `32022R2554` | original | `cf50d8f023d23b2a...` | 14096 |
| `psd2_art69.xhtml` | PSD2 | Article 69 | `02015L2366-20151223` | consolidated | `e9063e8919bf93bd...` | 2360 |
| `psd2_art111.xhtml` | PSD2 | Article 111 | `02015L2366-20151223` | consolidated | `e9063e8919bf93bd...` | 5766 |
| `psd2_art9.xhtml` | PSD2 | Article 9 | `02015L2366-20151223` | consolidated | `e9063e8919bf93bd...` | 14147 |
| `psd2_art9_original.xhtml` | PSD2 | Article 9 | `32015L2366` | original | `9ca1d7e9a83b9165...` | 24454 |
