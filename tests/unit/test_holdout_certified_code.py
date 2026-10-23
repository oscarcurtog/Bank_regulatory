"""The four files a sealed Gate 1 certification certifies, byte for byte (D-065, D-066): a
construction may rely on that certification only while they are the bytes it certified.
Exposure v2 and the development need exclusions change none of them; changing one needs a new
certification, never an edit of these hashes alone."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ingest import holdout_certification as hcert

# certification gate1-20261023-cc2.1.301-bc0e0e9, sealed (manifest 0b008d7a...): its record of
# the certified code
CERTIFIED_SHA256 = {
    "ingest/holdout_isolation.py": (
        "130f3e80975612a6d8dd03d7516bc6fc5d6a0d0021f61524dc916ac48c963b5f"
    ),
    "ingest/holdout_broker.py": "25fce12e140faf307a4a001f19b2be7145d2d75b9639716ba646dac0845b68c6",
    "ingest/holdout_spool.py": "f015e96c78628fb0cffc3dd8727af90605e9c273589dcf1132c50ba42862f073",
    "ingest/holdout_process.py": "dd87a17b3ada9ab5502468b769020b3b52769c38a2774cbc32caa8c2a2dc9244",
}


def test_the_certified_code_is_the_bytes_the_sealed_certification_certified():
    assert set(CERTIFIED_SHA256) == set(hcert.CERTIFIED_CODE)
    found = {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in CERTIFIED_SHA256}
    assert found == CERTIFIED_SHA256
