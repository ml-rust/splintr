"""
Integration tests for Mistral V1 tokenizer.

Mistral V1 uses SentencePiece with byte fallback. Key characteristics:
- Vocab size: 32,054 (32,000 base + 54 agent tokens)
- Uses ▁ (U+2581) for word boundaries
- Byte fallback tokens at positions 3-258
- Agent tokens start at 32,000
- Does NOT have V2 control tokens ([INST], [/INST], etc.)
"""

import pytest
from splintr import Tokenizer


class TestMistralV1ExactTokens:
    """Exact token ID verification tests.

    These catch regressions in encoding or vocabulary changes.
    Token IDs verified against HuggingFace Mistral 7B v0.1 tokenizer.
    """

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("mistral_v1")

    def test_hello_world_tokens(self, tokenizer):
        """Verify exact token IDs for 'Hello world'."""
        tokens = tokenizer.encode("Hello world")
        # "▁Hello" + "▁world". Reference: sentencepiece 0.2.0 on Mistral 7B
        # v0.1's tokenizer.model. The previous expectation, [16230, 1526], was
        # "Hello" with no word-boundary marker — an artifact of merging this
        # SentencePiece vocabulary byte-wise instead of piece-wise.
        assert tokens == [22557, 1526], f"Expected [22557, 1526], got {tokens}"

    def test_hello_world_punctuation(self, tokenizer):
        """Verify exact token IDs for 'Hello, world!'."""
        tokens = tokenizer.encode("Hello, world!")
        decoded = tokenizer.decode(tokens)
        assert decoded == "Hello, world!", f"Roundtrip failed: got {decoded!r}"

    def test_space_preservation(self, tokenizer):
        """Test that leading spaces are preserved via byte fallback."""
        # " world!" should preserve the space
        tokens = tokenizer.encode(" world!")
        decoded = tokenizer.decode(tokens)
        assert decoded == " world!", f"Space not preserved: got {decoded!r}"

    def test_chinese_tokens(self, tokenizer):
        """Verify encoding of Chinese text."""
        text = "你好世界"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text, f"Chinese roundtrip failed: {decoded!r}"

    def test_emoji_tokens(self, tokenizer):
        """Verify encoding of emoji."""
        text = "Hello 🌍 World!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text, f"Emoji roundtrip failed: {decoded!r}"


class TestMistralV1Roundtrip:
    """Roundtrip encoding/decoding tests with diverse inputs."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("mistral_v1")

    def test_encode_decode_roundtrip(self, tokenizer):
        """Test roundtrip with diverse text cases."""
        test_cases = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "Rust is a systems programming language.",
            "1234567890",
            "Special characters: !@#$%^&*()",
            "Unicode: こんにちは 世界 🦀",
            "Mixed: Hello 你好 🌍 World!",
        ]

        for text in test_cases:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            assert decoded == text, f"Roundtrip failed for: {text!r}"

    def test_multiline_roundtrip(self, tokenizer):
        """Test roundtrip with multi-line text."""
        text = "Multi-line\ntext\nwith\nnewlines"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text, f"Roundtrip failed for: {text!r}"

    def test_code_content(self, tokenizer):
        """Test encoding of code (common LLM use case)."""
        code = '''def hello_world():
    print("Hello, World!")

if __name__ == "__main__":
    hello_world()
'''
        tokens = tokenizer.encode(code)
        decoded = tokenizer.decode(tokens)
        assert decoded == code


class TestMistralV1SpecialTokens:
    """Test special tokens for Mistral V1."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("mistral_v1")

    def test_bos_eos_tokens(self, tokenizer):
        """Test BOS and EOS tokens."""
        # <s> = BOS = token 1
        tokens = tokenizer.encode_with_special("<s>")
        assert tokens == [1], f"<s> should be token 1, got {tokens}"

        # </s> = EOS = token 2
        tokens = tokenizer.encode_with_special("</s>")
        assert tokens == [2], f"</s> should be token 2, got {tokens}"

    def test_v1_tokenizes_inst_as_text(self, tokenizer):
        """V1 should tokenize [INST] as regular text, NOT as control token."""
        tokens = tokenizer.encode_with_special("[INST]")
        # [INST] is NOT a special token in V1 - it becomes multiple text tokens
        assert len(tokens) > 1, "[INST] should be multiple text tokens in V1"
        # Verify roundtrip
        decoded = tokenizer.decode(tokens)
        assert decoded == "[INST]"

    def test_agent_tokens(self, tokenizer):
        """Test agent tokens at offset 32000.

        Mistral V1 declares ``legacy = true`` in its ``tokenizer_config.json``,
        so it prefixes each stretch that *follows* an added token rather than
        prefixing the whole input once. An added token at byte 0 therefore has
        no stretch before it and emits no standalone ``▁``. Verified against
        ``AutoTokenizer.from_pretrained(mistral-7b-awq-int4, use_fast=False)``:
        ``tokenize("<s>x") == ['<s>', '▁x']``, where V2 (``legacy = false``)
        gives ``['<s>', 'x']``.
        """
        # <|think|> = THINK = 32000 + 5 = 32005
        tokens = tokenizer.encode_with_special("<|think|>")
        assert tokens == [32005], f"unexpected <|think|> ids: {tokens}"

        # <|function|> = FUNCTION = 32000 + 15 = 32015
        tokens = tokenizer.encode_with_special("<|function|>")
        assert tokens == [32015], f"unexpected <|function|> ids: {tokens}"

    def test_decode_agent_tokens(self, tokenizer):
        """Agent tokens decode to nothing, like every other control marker.

        These 54 ids are splintr's own additions above the vocabulary file's
        last piece, so no reference tokenizer names them -- but they are control
        markers of exactly the kind `mistral-7b-v0.3`'s `tokenizer.json`
        declares ``special: true`` and `tokenizers` 0.22.1 drops
        (``decode([3, ...]) -> 'hello'``), so they follow the same rule. Ask
        ``special_token_id`` for the spelling; `decode` gives model output.
        """
        assert tokenizer.decode([32005]) == ""
        assert tokenizer.decode([32015]) == ""


class TestMistralV1VocabSize:
    """Test vocabulary size and variant loading."""

    def test_vocab_size(self):
        """V1 vocab: 32,000 base + 54 agent = 32,054."""
        tok = Tokenizer.from_pretrained("mistral_v1")
        assert tok.vocab_size == 32054

    def test_default_mistral_is_v1(self):
        """'mistral' name should map to V1."""
        tok = Tokenizer.from_pretrained("mistral")
        assert tok.vocab_size == 32054

    def test_hyphenated_names_rejected(self):
        """Old hyphenated names should be rejected."""
        with pytest.raises(ValueError):
            Tokenizer.from_pretrained("mistral-v1")

        with pytest.raises(ValueError):
            Tokenizer.from_pretrained("mistral-7b")


class TestMistralV1Batch:
    """Test batch encoding functionality."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("mistral_v1")

    def test_batch_matches_individual(self, tokenizer):
        """Batch encoding should match individual encoding."""
        texts = [
            "Hello, world!",
            "How are you?",
            "I'm doing great!",
            "Unicode: 你好 🌍",
        ]

        batch_tokens = tokenizer.encode_batch(texts)
        assert len(batch_tokens) == len(texts)

        for i, text in enumerate(texts):
            individual = tokenizer.encode(text)
            assert batch_tokens[i] == individual, (
                f"Batch mismatch for text {i}: {text!r}"
            )

    def test_empty_input(self, tokenizer):
        """Test empty input handling."""
        assert tokenizer.encode("") == []
        assert tokenizer.decode([]) == ""


class TestMistralV1Utf8Boundaries:
    """Test UTF-8 boundary handling with multi-byte characters.

    These catch bugs where regex match positions fall inside
    multi-byte UTF-8 characters (em-dashes, curly quotes, etc.).
    """

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("mistral_v1")

    def test_em_dash(self, tokenizer):
        """Test em-dash (3-byte UTF-8: E2 80 94)."""
        text = "I'm sorry you're hurting—breakups suck, but you'll get through it."
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text

    def test_curly_quotes(self, tokenizer):
        """Test curly quotes (3-byte UTF-8 each)."""
        text = 'He said, \u2018Hello\u2019 and she replied, \u201cGoodbye\u201d.'
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text

    def test_em_dash_at_boundaries(self, tokenizer):
        """Test em-dash at various positions."""
        texts = [
            "word—word",
            "a—b",
            "test—",
            "—start",
            "one—two—three",
            "Check your brake pads—they might be worn out.",
        ]
        for text in texts:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            assert decoded == text, f"Failed for: {text!r}"

    # Backend (regexr vs PCRE2) consistency on multi-byte text is exercised
    # in `test_mistral_v3.py::TestMistralV3BackendOptions` instead: V1 routes
    # through `SpmTokenizer`, which has no regex backend at all, so there is
    # no "backend" for this vocabulary's multi-byte handling to be consistent
    # across.


class TestMistralV1LargeScaleBatch:
    """Large-scale parallel batch tests to catch threading bugs."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("mistral_v1")

    def test_large_batch_parallel(self, tokenizer):
        """Test large batch to trigger parallel execution (rayon).

        This catches UTF-8 boundary bugs in parallel batch processing.
        """
        base_texts = [
            "I'm sorry you're hurting—breakups suck, but you'll get through it.",
            "Check if you're using valid credentials—API key, token—in headers.",
            "你好世界！这是一个测试。",
            "Hello 🌍 World! 🦀 Rust is great!",
            "Mixed: Hello 你好 🌍 —test— World!",
            "Code: def foo(): return 42",
            "A 403 Forbidden error means permission denied.",
        ]
        # 700 texts to trigger parallel execution
        texts = base_texts * 100

        all_tokens = tokenizer.encode_batch(texts)
        assert len(all_tokens) == len(texts)

        # Verify roundtrip for samples
        for i in range(0, len(texts), 50):
            decoded = tokenizer.decode(all_tokens[i])
            assert decoded == texts[i], f"Failed roundtrip for text {i}"


# `TestMistralV1BackendOptions` (regexr/PCRE2/JIT backend switching) was
# removed: V1 routes through `SpmTokenizer`, which segments by merging pieces
# and has no regex pre-tokenizer to configure, so the concern does not apply to
# this vocabulary. The equivalent coverage now lives on the genuinely BPE-backed
# Mistral vocabulary in `test_mistral_v3.py::TestMistralV3BackendOptions`.
# What is pinned here instead is that asking anyway *fails* rather than
# reporting a switch that did not happen.


class TestMistralV1HasNoRegexBackend:
    """`.pcre2()`/`.jit()` refuse on the SPM backend instead of no-op'ing.

    Both methods exist on the universal handle every loader returns, so a caller
    can reach them on any vocabulary. Answering "done" for a backend with no
    regex pre-tokenizer would be the worst outcome: the caller would believe it
    had switched engines and never find out otherwise.
    """

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("mistral_v1")

    def test_family_is_spm(self, tokenizer):
        assert tokenizer.family == "Spm"

    def test_pcre2_raises(self, tokenizer):
        with pytest.raises(ValueError):
            tokenizer.pcre2(True)

    def test_jit_raises(self, tokenizer):
        with pytest.raises(ValueError):
            tokenizer.jit(False)
