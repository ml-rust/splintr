"""
Integration tests for the `SpecialMode` allow-list exposed to Python.

These verify that `encode_ordinary` never promotes a special token's literal
spelling and that `encode_allowed_special` enforces its allow-list, across
every tokenizer wrapper that exposes `encode` (BPE, SentencePiece Unigram,
SentencePiece BPE / SPM, WordPiece).

The BPE and SPM cases use bundled pretrained vocabularies, which already carry
real special/control tokens. The Unigram and WordPiece cases build a minimal
`tokenizer.json` in-memory via `from_json_bytes` — their direct Python
constructors don't wire up an added-token matcher, so a matcher-bearing
tokenizer must come through the json loader to exercise the allow-list at all.
"""

import json

import pytest
from splintr import Tokenizer, from_json_bytes


class TestBpeSpecialMode:
    """cl100k_base (byte-level BPE) has `<|endoftext|>` at id 100257."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("cl100k_base")

    def test_encode_ordinary_does_not_promote_special_token(self, tokenizer):
        ordinary_tokens = tokenizer.encode_ordinary("<|endoftext|>")
        with_special_tokens = tokenizer.encode_with_special("<|endoftext|>")
        assert 100257 not in ordinary_tokens
        assert 100257 in with_special_tokens

    def test_encode_allowed_special_permits_listed_token(self, tokenizer):
        tokens = tokenizer.encode_allowed_special(
            "Hello<|endoftext|>World", ["<|endoftext|>"]
        )
        assert 100257 in tokens

    def test_encode_allowed_special_raises_for_unlisted_token(self, tokenizer):
        with pytest.raises(ValueError):
            tokenizer.encode_allowed_special("Hello<|endoftext|>World", [])


class TestSpmSpecialMode:
    """mistral_v2 (SentencePiece-BPE) has `[INST]`/`[/INST]` at ids 3/4."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("mistral_v2")

    def test_encode_ordinary_does_not_promote_special_token(self, tokenizer):
        ordinary_tokens = tokenizer.encode_ordinary("[INST]")
        assert 3 not in ordinary_tokens

    def test_encode_allowed_special_permits_listed_token(self, tokenizer):
        tokens = tokenizer.encode_allowed_special(
            "[INST]hi[/INST]", ["[INST]", "[/INST]"]
        )
        assert 3 in tokens
        assert 4 in tokens

    def test_encode_allowed_special_raises_for_unlisted_token(self, tokenizer):
        with pytest.raises(ValueError):
            tokenizer.encode_allowed_special("[INST]hi[/INST]", ["[INST]"])


class TestUnigramSpecialMode:
    """A minimal Unigram `tokenizer.json` with `<extra>` as an added token.

    `<extra>` deliberately does NOT also appear as a base vocab piece (unlike
    an earlier version of this fixture, which put `<extra>` in `vocab` at the
    same id as the added token — making Viterbi's *ordinary* segmentation
    legitimately choose it as a single-piece match, since that's just what
    Unigram does with any piece that is literally in its vocabulary). The
    reserved id 3 still needs a base-vocab entry for `decode` to have
    something to render, so it holds an unrelated placeholder piece instead.
    """

    @pytest.fixture
    def tokenizer(self):
        doc = {
            "model": {
                "type": "Unigram",
                "vocab": [
                    ["<unk>", 0.0],
                    ["<s>", 0.0],
                    ["</s>", 0.0],
                    ["<reserved>", 0.0],
                    ["▁Hello", -1.2],
                    ["▁world", -1.5],
                    ["H", -3.0],
                    ["e", -3.0],
                ],
                "unk_id": 0,
            },
            "added_tokens": [{"id": 3, "content": "<extra>", "special": True}],
        }
        return from_json_bytes(json.dumps(doc).encode())

    def test_encode_ordinary_does_not_promote_special_token(self, tokenizer):
        ordinary_tokens = tokenizer.encode_ordinary("<extra>")
        assert 3 not in ordinary_tokens

    def test_encode_allowed_special_permits_listed_token(self, tokenizer):
        tokens = tokenizer.encode_allowed_special("<extra>", ["<extra>"])
        assert 3 in tokens

    def test_encode_allowed_special_raises_for_unlisted_token(self, tokenizer):
        with pytest.raises(ValueError):
            tokenizer.encode_allowed_special("<extra>", [])


class TestWordPieceSpecialMode:
    """A minimal WordPiece `tokenizer.json` (BERT-style) with `[CLS]`/`[SEP]`."""

    @pytest.fixture
    def tokenizer(self):
        doc = {
            "model": {
                "type": "WordPiece",
                "vocab": {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "hello": 3, "world": 4},
                "unk_token": "[UNK]",
            },
            "added_tokens": [
                {"id": 1, "content": "[CLS]", "special": True},
                {"id": 2, "content": "[SEP]", "special": True},
            ],
        }
        return from_json_bytes(json.dumps(doc).encode())

    def test_encode_ordinary_does_not_promote_special_token(self, tokenizer):
        ordinary_tokens = tokenizer.encode_ordinary("[CLS]")
        assert 1 not in ordinary_tokens

    def test_encode_allowed_special_permits_listed_token(self, tokenizer):
        tokens = tokenizer.encode_allowed_special("[CLS]hello[SEP]", ["[CLS]", "[SEP]"])
        assert 1 in tokens
        assert 2 in tokens

    def test_encode_allowed_special_raises_for_unlisted_token(self, tokenizer):
        with pytest.raises(ValueError):
            tokenizer.encode_allowed_special("[CLS]hello[SEP]", ["[CLS]"])
