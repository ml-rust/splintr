"""`Tokenizer.from_pretrained` must build what `splintr::pretrained` builds.

The Python binding used to reconstruct each bundled vocabulary by hand —
re-listing its vocabulary blob, its pre-tokenizer passes and its special tokens,
then wrapping the result with a default policy and added-token matching left
off. The core Rust loader ends with `with_added_token_matching(true)` and the
vocabulary's own policy, so the same name produced different ids depending on
which side of the FFI boundary asked:

    llama3, Rust:   encode("<|begin_of_text|>hi") -> [128000, 6151]
    llama3, Python: encode("<|begin_of_text|>hi") -> [27, 91, 7413, 3659, ...]

The marker was shattered into ordinary tokens and the shards were in range and
round-tripped, so nothing downstream could tell. The binding now delegates to
the one loader, and these tests pin that it keeps doing so: a second
construction path added here would have to reproduce every id below.
"""

import pytest
from splintr import AnyTokenizer, Tokenizer

#: Every name the bundled loader accepts.
ALL_NAMES = [
    "cl100k_base",
    "o200k_base",
    "llama3",
    "llama3.1",
    "llama3.2",
    "llama3.3",
    "deepseek_v3",
    "deepseek-v3",
    "mistral",
    "mistral_v1",
    "mistral_v2",
    "mistral_v3",
    "whisper",
    "whisper_v1",
    "whisper_v2",
    "whisper_v3",
]

#: `(name, text, ids)` — the ids the Rust loader produces for `encode`, i.e.
#: with the special token spelled out in the text matched as a single id.
MATCHED_IN_TEXT = [
    ("cl100k_base", "Hello<|endoftext|>World", [9906, 100257, 10343]),
    ("llama3", "<|begin_of_text|>hi", [128000, 6151]),
    ("whisper", "<|endoftext|>hi", [50257, 4954]),
]


@pytest.mark.parametrize("name", ALL_NAMES)
def test_returns_the_universal_handle(name):
    """Every bundled vocabulary comes back as the loader's own handle.

    Not a `Tokenizer` for some names and an `AnyTokenizer` for others: the split
    was the visible symptom of the two construction paths.
    """
    assert isinstance(Tokenizer.from_pretrained(name), AnyTokenizer)


@pytest.mark.parametrize("name, text, ids", MATCHED_IN_TEXT)
def test_encode_matches_a_special_token_spelled_in_the_text(name, text, ids):
    tokenizer = Tokenizer.from_pretrained(name)
    assert tokenizer.encode(text) == ids


@pytest.mark.parametrize("name, text, ids", MATCHED_IN_TEXT)
def test_encode_agrees_with_encode_with_special(name, text, ids):
    """Matching is on, so the two mean the same thing for these vocabularies."""
    tokenizer = Tokenizer.from_pretrained(name)
    assert tokenizer.encode(text) == tokenizer.encode_with_special(text)


@pytest.mark.parametrize("name, text, ids", MATCHED_IN_TEXT)
def test_encode_ordinary_still_refuses_to_match(name, text, ids):
    """The opt-out survives: `encode_ordinary` is how you decline the match."""
    tokenizer = Tokenizer.from_pretrained(name)
    ordinary = tokenizer.encode_ordinary(text)
    assert ordinary != ids
    assert tokenizer.decode(ordinary) == text


@pytest.mark.parametrize("name", ALL_NAMES)
def test_roundtrip_of_ordinary_text(name):
    """Text with no special token in it is unaffected by any of the above."""
    tokenizer = Tokenizer.from_pretrained(name)
    text = "The quick brown fox jumps over the lazy dog."
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_unknown_name_raises():
    with pytest.raises(ValueError):
        Tokenizer.from_pretrained("not_a_real_vocabulary")
