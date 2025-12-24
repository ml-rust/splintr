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
        # SentencePiece: "Hello" + "▁world"
        assert tokens == [16230, 1526], f"Expected [16230, 1526], got {tokens}"

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
        """Test agent tokens at offset 32000."""
        # <|think|> = THINK = 32000 + 5 = 32005
        tokens = tokenizer.encode_with_special("<|think|>")
        assert tokens == [32005], f"<|think|> should be [32005], got {tokens}"

        # <|function|> = FUNCTION = 32000 + 15 = 32015
        tokens = tokenizer.encode_with_special("<|function|>")
        assert tokens == [32015], f"<|function|> should be [32015], got {tokens}"

    def test_decode_agent_tokens(self, tokenizer):
        """Test decoding agent tokens."""
        assert tokenizer.decode([32005]) == "<|think|>"
        assert tokenizer.decode([32015]) == "<|function|>"


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

    @pytest.fixture
    def tokenizer_pcre2(self):
        return Tokenizer.from_pretrained("mistral_v1").pcre2(True)

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


class TestMistralV1BackendOptions:
    """Test regex backend options (regexr, PCRE2, JIT)."""

    def test_default_backend(self):
        """Test default backend (regexr with JIT)."""
        tokenizer = Tokenizer.from_pretrained("mistral_v1")
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        assert tokenizer.decode(tokens) == text

    def test_pcre2_backend(self):
        """Test PCRE2 backend."""
        tokenizer = Tokenizer.from_pretrained("mistral_v1").pcre2(True)
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        assert tokenizer.decode(tokens) == text

    def test_jit_disabled(self):
        """Test with JIT disabled."""
        tokenizer = Tokenizer.from_pretrained("mistral_v1").jit(False)
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        assert tokenizer.decode(tokens) == text

    def test_backend_consistency(self):
        """Test all backends produce identical tokens."""
        text = "The quick brown fox 你好 🦀 jumps—over—the lazy dog."

        tok_default = Tokenizer.from_pretrained("mistral_v1")
        tok_pcre2 = Tokenizer.from_pretrained("mistral_v1").pcre2(True)
        tok_no_jit = Tokenizer.from_pretrained("mistral_v1").jit(False)

        tokens_default = tok_default.encode(text)
        tokens_pcre2 = tok_pcre2.encode(text)
        tokens_no_jit = tok_no_jit.encode(text)

        assert tokens_default == tokens_pcre2, "PCRE2 should match default"
        assert tokens_default == tokens_no_jit, "Non-JIT should match default"
