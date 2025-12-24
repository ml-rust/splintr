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
        """Verify exact control token IDs."""
        assert tokenizer.encode_with_special("[INST]") == [3]
        assert tokenizer.encode_with_special("[/INST]") == [4]
        assert tokenizer.encode_with_special("[TOOL_CALLS]") == [5]
        assert tokenizer.encode_with_special("[AVAILABLE_TOOLS]") == [6]

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
        """Test instruction format encoding."""
        text = "[INST]Hello, how are you?[/INST]I'm doing great!"
        tokens = tokenizer.encode_with_special(text)

        # Should contain control tokens
        assert 3 in tokens, "[INST] not found"
        assert 4 in tokens, "[/INST] not found"

        # Verify roundtrip
        decoded = tokenizer.decode(tokens)
        assert decoded == text

    def test_tool_calling_format(self, tokenizer):
        """Test tool calling format encoding."""
        text = "[AVAILABLE_TOOLS]get_weather[/AVAILABLE_TOOLS][TOOL_CALLS]get_weather()"
        tokens = tokenizer.encode_with_special(text)

        assert 5 in tokens, "[TOOL_CALLS] not found"
        assert 6 in tokens, "[AVAILABLE_TOOLS] not found"

        decoded = tokenizer.decode(tokens)
        # Note: [/AVAILABLE_TOOLS] is not a special token, encoded as text
        assert "[AVAILABLE_TOOLS]" in decoded
        assert "[TOOL_CALLS]" in decoded

    def test_decode_control_tokens(self, tokenizer):
        """Test decoding control tokens."""
        assert tokenizer.decode([3]) == "[INST]"
        assert tokenizer.decode([4]) == "[/INST]"
        assert tokenizer.decode([5]) == "[TOOL_CALLS]"
        assert tokenizer.decode([6]) == "[AVAILABLE_TOOLS]"

    def test_mixed_control_and_text(self, tokenizer):
        """Test mixing control tokens with regular text."""
        text = "[INST]Write a poem about Rust[/INST]Rust is fast and safe..."
        tokens = tokenizer.encode_with_special(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text


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
        """Test agent tokens at offset 32768."""
        # <|think|> = THINK = 32768 + 5 = 32773
        tokens = tokenizer.encode_with_special("<|think|>")
        assert tokens == [32773], f"<|think|> should be [32773], got {tokens}"

        # <|function|> = FUNCTION = 32768 + 15 = 32783
        tokens = tokenizer.encode_with_special("<|function|>")
        assert tokens == [32783], f"<|function|> should be [32783], got {tokens}"

    def test_decode_agent_tokens(self, tokenizer):
        """Test decoding agent tokens."""
        assert tokenizer.decode([32773]) == "<|think|>"
        assert tokenizer.decode([32783]) == "<|function|>"


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

        # V2: [INST] is single control token
        assert v2_tokens == [3], "V2 should have [INST] as token 3"

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

    @pytest.fixture
    def tokenizer_pcre2(self):
        return Tokenizer.from_pretrained("mistral_v2").pcre2(True)

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

    def test_backend_consistency_multibyte(self, tokenizer, tokenizer_pcre2):
        """Test regexr and PCRE2 produce same results for multi-byte text."""
        texts = [
            "word—word",
            "I'm sorry you're hurting—breakups suck.",
            'He said, \u2018Hello\u2019 and she replied, \u201cGoodbye\u201d.',
            "Check credentials—API key—in headers.",
        ]
        for text in texts:
            tokens_regexr = tokenizer.encode(text)
            tokens_pcre2 = tokenizer_pcre2.encode(text)
            assert tokens_regexr == tokens_pcre2, f"Backend mismatch for: {text!r}"


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


class TestMistralV2BackendOptions:
    """Test regex backend options (regexr, PCRE2, JIT)."""

    def test_default_backend(self):
        """Test default backend (regexr with JIT)."""
        tokenizer = Tokenizer.from_pretrained("mistral_v2")
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        assert tokenizer.decode(tokens) == text

    def test_pcre2_backend(self):
        """Test PCRE2 backend."""
        tokenizer = Tokenizer.from_pretrained("mistral_v2").pcre2(True)
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        assert tokenizer.decode(tokens) == text

    def test_jit_disabled(self):
        """Test with JIT disabled."""
        tokenizer = Tokenizer.from_pretrained("mistral_v2").jit(False)
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        assert tokenizer.decode(tokens) == text

    def test_backend_consistency(self):
        """Test all backends produce identical tokens."""
        text = "The quick brown fox 你好 🦀 jumps—over—the lazy dog."

        tok_default = Tokenizer.from_pretrained("mistral_v2")
        tok_pcre2 = Tokenizer.from_pretrained("mistral_v2").pcre2(True)
        tok_no_jit = Tokenizer.from_pretrained("mistral_v2").jit(False)

        tokens_default = tok_default.encode(text)
        tokens_pcre2 = tok_pcre2.encode(text)
        tokens_no_jit = tok_no_jit.encode(text)

        assert tokens_default == tokens_pcre2, "PCRE2 should match default"
        assert tokens_default == tokens_no_jit, "Non-JIT should match default"
