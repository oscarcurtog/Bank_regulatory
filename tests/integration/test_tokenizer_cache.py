"""The tokenizer cache (T2): one directory, no network once it holds the file, and
an explicit failure when the tokenizer can be neither read nor downloaded.

Normal operation is the precondition: the file is in the cache or can be
downloaded into it. Every other case cuts the network inside the process (name
resolution fails, as it does offline) and empties tiktoken's in-memory registry,
so that the load really goes to the cache directory.
"""

from __future__ import annotations

import os
import shutil
import socket
from pathlib import Path

import pytest
import tiktoken

from ingest import chunks as ch

SAMPLE = "Regulation (EU) 2022/2554 on digital operational resilience for the financial sector"


def _cut_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def unreachable(*args: object, **kwargs: object) -> None:
        raise socket.gaierror(socket.EAI_NONAME, "network disabled by the test")

    # every connection starts by resolving a name, so failing here cuts them all
    monkeypatch.setattr(socket, "getaddrinfo", unreachable)


def _forget_loaded_encodings(monkeypatch: pytest.MonkeyPatch) -> None:
    # tiktoken keeps each encoding in memory after its first load, and would never
    # read the cache directory again in this process
    monkeypatch.setattr(tiktoken.registry, "ENCODINGS", {})


@pytest.fixture
def offline_copy_of_the_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A copy of the filled cache, with the network cut: what a CI run restores."""
    _forget_loaded_encodings(monkeypatch)
    ch.load_encoding("cl100k_base")  # normal operation: from the cache, or downloaded into it
    copy = tmp_path / "tiktoken"
    shutil.copytree(os.environ["TIKTOKEN_CACHE_DIR"], copy)
    assert any(copy.iterdir())
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(copy))
    _forget_loaded_encodings(monkeypatch)
    _cut_the_network(monkeypatch)
    return copy


def test_unset_the_cache_is_the_project_directory(monkeypatch):
    """The directory CI caches and the Dockerfile names, not the system's temp."""
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", "")  # recorded, so it is restored afterwards
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR")
    _forget_loaded_encodings(monkeypatch)
    ch.load_encoding("cl100k_base")
    assert os.environ["TIKTOKEN_CACHE_DIR"] == str(Path(".cache/tiktoken").absolute())


def test_a_filled_cache_needs_no_network(offline_copy_of_the_cache):
    encoding = ch.load_encoding("cl100k_base")
    assert encoding.name == "cl100k_base"
    assert len(encoding.encode(SAMPLE)) == 19


def test_a_corrupted_cache_is_not_used(offline_copy_of_the_cache):
    """tiktoken checks the SHA-256 of what it reads: a wrong file is discarded and
    fetched again, and with no network that is an error, never a silent count."""
    for cached in offline_copy_of_the_cache.iterdir():
        cached.write_bytes(b"not the tokenizer")
    with pytest.raises(ch.TokenizerUnavailableError):
        ch.load_encoding("cl100k_base")


def test_without_cache_and_without_network_the_failure_is_explicit(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(empty))
    _forget_loaded_encodings(monkeypatch)
    _cut_the_network(monkeypatch)
    # cl100k_counter() is what chunk_document() counts with: no other tokenizer, no estimate
    with pytest.raises(ch.TokenizerUnavailableError, match="cl100k_base") as raised:
        ch.cl100k_counter()
    message = str(raised.value)
    assert str(empty) in message
    assert "TIKTOKEN_CACHE_DIR" in message
    assert isinstance(raised.value.__cause__, OSError)
