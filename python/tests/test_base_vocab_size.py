"""
Tests for `splintr.base_vocab_size`.

splintr appends 54 agent tokens above every bundled vocabulary's own ids, so
`Tokenizer.from_pretrained(...).vocab_size` reports the *extended* size.
`base_vocab_size(name)` reports the size the upstream reference implementation
(tiktoken / tokenizers / sentencepiece) reports instead — what a consumer
needs to size an embedding/logit layer or identify a checkpoint's vocabulary
from its tensor shape.

Reference values are derived from the vocabulary construction itself (see
`splintr::core::pretrained` in the Rust crate), not from splintr's own
extended `vocab_size`.
"""

import pytest
from splintr import Tokenizer, base_vocab_size


class TestBaseVocabSize:
    def test_cl100k_base(self):
        # tiktoken: one past `<|endofprompt|>` (100276).
        assert base_vocab_size("cl100k_base") == 100277

    def test_o200k_base(self):
        # tiktoken: one past `<|endofprompt|>` (200018).
        assert base_vocab_size("o200k_base") == 200019

    def test_llama3(self):
        # tokenizers: 128,000 BPE tokens + 256 reserved special-token slots.
        assert base_vocab_size("llama3") == 128256

    def test_deepseek_v3(self):
        # tokenizers: one past `<｜tool▁sep｜>` (128814).
        assert base_vocab_size("deepseek_v3") == 128815

    def test_mistral_v1(self):
        # sentencepiece piece count.
        assert base_vocab_size("mistral_v1") == 32000

    def test_mistral_v2(self):
        # sentencepiece piece count.
        assert base_vocab_size("mistral_v2") == 32768

    def test_mistral_v3(self):
        # tokenizers (Tekken): the base vocabulary file's own piece count.
        assert base_vocab_size("mistral_v3") == 131072

    def test_whisper_v3(self):
        # Whisper carries no agent tokens at all, so its base size is its
        # full generated vocabulary size.
        assert base_vocab_size("whisper_v3") == 51866

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError):
            base_vocab_size("not-a-real-vocabulary")

    def test_never_exceeds_extended_vocab_size(self):
        for name in ["cl100k_base", "o200k_base", "llama3", "deepseek_v3", "mistral_v3"]:
            tok = Tokenizer.from_pretrained(name)
            assert base_vocab_size(name) <= tok.vocab_size
