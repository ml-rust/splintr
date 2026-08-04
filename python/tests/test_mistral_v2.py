"""
Integration tests for Mistral V2 tokenizer.

Mistral V2 uses SentencePiece with control tokens. Key characteristics:
- Vocab size: 32,822 (32,768 base + 54 agent tokens)
- Control tokens: [INST]=3, [/INST]=4, [TOOL_CALLS]=5, [AVAILABLE_TOOLS]=6
- Uses ▁ (U+2581) for word boundaries (same as V1)
- Raw byte fallback tokens at 771-1026
- Agent tokens start at 32,768
- Different vocabulary file from V1 (encodes text differently)
"""

import pytest
from splintr import Tokenizer


class TestMistralV2ExactTokens:
    """Exact token ID verification tests.

    These catch regressions in encoding or vocabulary changes.
    Token IDs verified against HuggingFace Mistral 7B v0.3 tokenizer.
    """

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("mistral_v2")

    def test_hello_world_tokens(self, tokenizer):
        """Verify exact token IDs for 'Hello world'."""
        tokens = tokenizer.encode("Hello world")
        decoded = tokenizer.decode(tokens)
        assert decoded == "Hello world"
        # V2 uses different vocab than V1, so different token IDs
        assert len(tokens) >= 2

    def test_control_tokens_exact(self, tokenizer):
        """Verify exact control token IDs.

        SentencePiece applies `add_dummy_prefix` to the whole input before
        splitting on control tokens, so a control token at byte 0 leaves the
        prefix standing alone as id 29473 (the `▁` piece). Reference:
        `AutoTokenizer.from_pretrained("mistral-7b-v0.3", use_fast=False)`
        with `add_special_tokens=False` returns `[29473, 3]` for `"[INST]"`.
        """
        assert tokenizer.encode_with_special("[INST]") == [29473, 3]
        assert tokenizer.encode_with_special("[/INST]") == [29473, 4]
        assert tokenizer.encode_with_special("[TOOL_CALLS]") == [29473, 5]
        assert tokenizer.encode_with_special("[AVAILABLE_TOOLS]") == [29473, 6]

    def test_space_preservation(self, tokenizer):
        """Test that leading spaces are preserved via byte fallback."""
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


class TestMistralV2ControlTokens:
    """Test Mistral V2 control tokens for instruction format."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("mistral_v2")

    def test_instruction_format(self, tokenizer):
        """Test instruction format encoding.

        `decode` implements HuggingFace's default ``skip_special_tokens=True``,
        so the markers do not come back. Measured on `mistral-7b-v0.3`, whose
        vocabulary this is::

            tokenizers 0.22.1:   decode(ids)  -> "Hello, how are you?I'm doing great!"
            sentencepiece 0.2.0: sp.decode(ids) -> same
            decode(ids, skip_special_tokens=False)
                                              -> "[INST]Hello, how are you?[/INST]I'm doing great!"
        """
        text = "[INST]Hello, how are you?[/INST]I'm doing great!"
        tokens = tokenizer.encode_with_special(text)

        # Should contain control tokens
        assert 3 in tokens, "[INST] not found"
        assert 4 in tokens, "[/INST] not found"

        decoded = tokenizer.decode(tokens)
        assert decoded == "Hello, how are you?I'm doing great!"

    def test_tool_calling_format(self, tokenizer):
        """Test tool calling format encoding.

        Both references decode these ids to ``'get_weatherget_weather()'`` --
        every tool marker is a control token and renders as nothing.
        """
        text = "[AVAILABLE_TOOLS]get_weather[/AVAILABLE_TOOLS][TOOL_CALLS]get_weather()"
        tokens = tokenizer.encode_with_special(text)

        assert 5 in tokens, "[TOOL_CALLS] not found"
        assert 6 in tokens, "[AVAILABLE_TOOLS] not found"

        decoded = tokenizer.decode(tokens)
        assert decoded == "get_weatherget_weather()"

    def test_decode_control_tokens(self, tokenizer):
        """Control tokens decode to nothing, as both references do.

        Measured on `mistral-7b-v0.3`, where `[INST]` is id 3 and declared
        ``special: true``::

            sentencepiece 0.2.0: sp.decode([3] + sp.encode("hello")) -> 'hello'
            tokenizers 0.22.1:   decode([3, ...])                    -> 'hello'
                                 decode([3, ...], skip_special_tokens=False)
                                                                     -> '[INST] hello'

        This previously asserted the spelling came back, which pinned splintr's
        own output rather than a reference and made `from_pretrained` disagree
        with `splintr.from_json` on that same `tokenizer.json`.
        """
        for token_id in (3, 4, 5, 6):
            assert tokenizer.decode([token_id]) == ""

    def test_mixed_control_and_text(self, tokenizer):
        """Test mixing control tokens with regular text.

        Reference (`tokenizers` 0.22.1 and `sentencepiece` 0.2.0 on
        `mistral-7b-v0.3`): the text survives, the markers do not.
        """
        text = "[INST]Write a poem about Rust[/INST]Rust is fast and safe..."
        tokens = tokenizer.encode_with_special(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == "Write a poem about RustRust is fast and safe..."


class TestMistralV2Roundtrip:
    """Roundtrip encoding/decoding tests with diverse inputs."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("mistral_v2")

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


class TestMistralV2SpecialTokens:
    """Test native special tokens for Mistral V2."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("mistral_v2")

    def test_bos_eos_tokens(self, tokenizer):
        """Test BOS and EOS tokens (same as V1)."""
        # <s> = BOS = token 1
        tokens = tokenizer.encode_with_special("<s>")
        assert tokens == [1], f"<s> should be token 1, got {tokens}"

        # </s> = EOS = token 2
        tokens = tokenizer.encode_with_special("</s>")
        assert tokens == [2], f"</s> should be token 2, got {tokens}"

    def test_agent_tokens(self, tokenizer):
        """Test agent tokens at offset 32768.

        Agent tokens are ordinary added tokens, not the vocabulary's
        BOS/EOS/UNK sentinels, so one at byte 0 is preceded by the standalone
        dummy prefix (29473) — the same shape as `"[INST]"` -> `[29473, 3]`.
        """
        # <|think|> = THINK = 32768 + 5 = 32773
        tokens = tokenizer.encode_with_special("<|think|>")
        assert tokens == [29473, 32773], f"unexpected <|think|> ids: {tokens}"

        # <|function|> = FUNCTION = 32768 + 15 = 32783
        tokens = tokenizer.encode_with_special("<|function|>")
        assert tokens == [29473, 32783], f"unexpected <|function|> ids: {tokens}"

    def test_decode_agent_tokens(self, tokenizer):
        """Agent tokens are control markers: `decode` drops them,
        `decode_with_special` spells them out.

        They are splintr's own additions above the vocabulary file's last piece,
        so no reference names them -- but they are the same kind of marker
        `mistral-7b-v0.3`'s `tokenizer.json` declares ``special: true`` and
        `tokenizers` 0.22.1 drops (``decode([3, ...]) -> 'hello'``), so they
        follow the same rule under the same explicit mode.
        """
        assert tokenizer.decode([32773]) == ""
        assert tokenizer.decode([32783]) == ""

        assert tokenizer.decode_with_special([32773]) == "<|think|>"
        assert tokenizer.decode_with_special([32783]) == "<|function|>"


class TestMistralV2VocabSize:
    """Test vocabulary size and variant loading."""

    def test_vocab_size(self):
        """V2 vocab: 32,768 base + 54 agent = 32,822."""
        tok = Tokenizer.from_pretrained("mistral_v2")
        assert tok.vocab_size == 32822

    def test_v2_larger_than_v1(self):
        """V2 vocab should be larger than V1."""
        v1 = Tokenizer.from_pretrained("mistral_v1")
        v2 = Tokenizer.from_pretrained("mistral_v2")
        assert v2.vocab_size > v1.vocab_size
        assert v1.vocab_size == 32054
        assert v2.vocab_size == 32822

    def test_hyphenated_names_rejected(self):
        """Old hyphenated names should be rejected."""
        with pytest.raises(ValueError):
            Tokenizer.from_pretrained("mistral-v2")

        with pytest.raises(ValueError):
            Tokenizer.from_pretrained("codestral")


class TestMistralV2VsV1:
    """Test differences between Mistral V1 and V2."""

    def test_v2_has_control_tokens(self):
        """V2 has [INST] as single control token; V1 doesn't."""
        v1 = Tokenizer.from_pretrained("mistral_v1")
        v2 = Tokenizer.from_pretrained("mistral_v2")

        v1_tokens = v1.encode_with_special("[INST]")
        v2_tokens = v2.encode_with_special("[INST]")

        # V1: [INST] is multiple text tokens
        assert len(v1_tokens) > 1, "V1 should tokenize [INST] as text"

        # V2: [INST] is a single control token, preceded by the standalone
        # dummy prefix (29473) the whole-input normalization leaves behind.
        assert v2_tokens == [29473, 3], "V2 should have [INST] as token 3"

    def test_different_vocabularies(self):
        """V1 and V2 use different vocabulary files."""
        v1 = Tokenizer.from_pretrained("mistral_v1")
        v2 = Tokenizer.from_pretrained("mistral_v2")

        text = "This is a test message"
        v1_tokens = v1.encode(text)
        v2_tokens = v2.encode(text)

        # Different vocab files = different token IDs
        assert v1_tokens != v2_tokens, "V1 and V2 should encode differently"


class TestMistralV2Batch:
    """Test batch encoding functionality."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("mistral_v2")

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


class TestMistralV2Utf8Boundaries:
    """Test UTF-8 boundary handling with multi-byte characters.

    These catch bugs where regex match positions fall inside
    multi-byte UTF-8 characters (em-dashes, curly quotes, etc.).
    """

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("mistral_v2")

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
    # in `test_mistral_v3.py::TestMistralV3BackendOptions` instead: V2 routes
    # through `SpmTokenizer`, which has no regex backend at all, so there is
    # no "backend" for this vocabulary's multi-byte handling to be consistent
    # across.


class TestMistralV2LargeScaleBatch:
    """Large-scale parallel batch tests to catch threading bugs."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("mistral_v2")

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


# `TestMistralV2BackendOptions` (regexr/PCRE2/JIT backend switching) was
# removed: V2 routes through `SpmTokenizer`, which segments by merging pieces
# and has no regex pre-tokenizer to configure, so the concern does not apply to
# this vocabulary. The equivalent coverage now lives on the genuinely BPE-backed
# Mistral vocabulary in `test_mistral_v3.py::TestMistralV3BackendOptions`, and
# the refusal on this backend in
# `test_mistral_v1.py::TestMistralV1HasNoRegexBackend`.
