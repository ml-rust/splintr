"""
Integration tests for cl100k_base tokenizer (GPT-4, GPT-3.5-turbo).

These tests verify that the cl100k_base tokenizer correctly encodes and decodes text,
handles special tokens, and produces consistent results.
"""

import pytest
from splintr import Tokenizer, CL100K_AGENT_TOKENS, CL100K_BASE_PATTERN


class TestCl100kExactTokens:
    """Exact token ID verification tests.

    These tests verify specific token IDs to catch any regression in
    encoding or vocabulary changes.
    """

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("cl100k_base")

    def test_hello_world_tokens(self, tokenizer):
        """Verify exact token IDs for 'Hello world'."""
        tokens = tokenizer.encode("Hello world")
        assert tokens == [9906, 1917], f"Expected [9906, 1917], got {tokens}"

    def test_hello_world_punctuation_tokens(self, tokenizer):
        """Verify exact token IDs for 'Hello, world!'."""
        tokens = tokenizer.encode("Hello, world!")
        assert tokens == [9906, 11, 1917, 0], f"Expected [9906, 11, 1917, 0], got {tokens}"

    def test_chinese_tokens(self, tokenizer):
        """Verify exact token IDs for '你好世界'."""
        tokens = tokenizer.encode("你好世界")
        assert tokens == [57668, 53901, 3574, 244, 98220], (
            f"Expected [57668, 53901, 3574, 244, 98220], got {tokens}"
        )

    def test_emoji_tokens(self, tokenizer):
        """Verify exact token IDs for 'Hello 🌍 World!'."""
        tokens = tokenizer.encode("Hello 🌍 World!")
        assert tokens == [9906, 11410, 234, 235, 4435, 0], (
            f"Expected [9906, 11410, 234, 235, 4435, 0], got {tokens}"
        )


class TestCl100kTokenizer:
    """Test suite for cl100k_base tokenizer."""

    @pytest.fixture
    def tokenizer(self):
        """Create a cl100k_base tokenizer for testing."""
        return Tokenizer.from_pretrained("cl100k_base")

    def test_encode_decode_roundtrip(self, tokenizer):
        """Test basic encoding and decoding roundtrip."""
        test_cases = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "Rust is a systems programming language.",
            "1234567890",
            "Special characters: !@#$%^&*()",
            "Multi-line\ntext\nwith\nnewlines",
            "Unicode: こんにちは 世界 🦀",
        ]

        for text in test_cases:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            assert decoded == text, f"Roundtrip failed for: {text!r}"

    def test_vocab_size(self, tokenizer):
        """Test that vocab size is correct (100,256 BPE tokens)."""
        # cl100k_base has 100,256 BPE tokens plus special tokens
        assert tokenizer.vocab_size >= 100256, (
            f"Vocab size should be at least 100,256, got {tokenizer.vocab_size}"
        )


class TestCl100kOpenAISpecialTokens:
    """Test OpenAI standard special tokens."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("cl100k_base")

    def test_endoftext(self, tokenizer):
        """Test endoftext token."""
        tokens = tokenizer.encode_with_special("Hello<|endoftext|>World")
        assert 100257 in tokens, "Should contain endoftext (100257)"

    def test_fim_tokens(self, tokenizer):
        """Test FIM (Fill-in-the-Middle) tokens."""
        tokens = tokenizer.encode_with_special("<|fim_prefix|>code<|fim_middle|>")
        assert 100258 in tokens, "Should contain fim_prefix (100258)"
        assert 100259 in tokens, "Should contain fim_middle (100259)"

    def test_fim_suffix(self, tokenizer):
        """Test fim_suffix token."""
        tokens = tokenizer.encode_with_special("<|fim_suffix|>")
        assert 100260 in tokens, "Should contain fim_suffix (100260)"

    def test_endofprompt(self, tokenizer):
        """Test endofprompt token."""
        tokens = tokenizer.encode_with_special("<|endofprompt|>")
        assert 100276 in tokens, "Should contain endofprompt (100276)"


class TestCl100kAgentTokens:
    """Test splintr agent tokens for cl100k."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("cl100k_base")

    def test_conversation_tokens(self, tokenizer):
        """Test conversation tokens."""
        tokens = tokenizer.encode_with_special(
            "<|system|>You are helpful.<|user|>Hi<|assistant|>"
        )
        assert 100277 in tokens, "Should contain system (100277)"
        assert 100278 in tokens, "Should contain user (100278)"
        assert 100279 in tokens, "Should contain assistant (100279)"

    def test_thinking_tokens(self, tokenizer):
        """Test thinking tokens."""
        tokens = tokenizer.encode_with_special("<|think|>Let me reason...<|/think|>")
        assert 100282 in tokens, "Should contain think (100282)"
        assert 100283 in tokens, "Should contain think_end (100283)"

    def test_function_calling_tokens(self, tokenizer):
        """Test function calling tokens."""
        tokens = tokenizer.encode_with_special("<|function|>get_weather<|/function|>")
        assert 100292 in tokens, "Should contain function (100292)"
        assert 100293 in tokens, "Should contain function_end (100293)"


class TestCl100kAgentTokensClass:
    """Test CL100K_AGENT_TOKENS class constants."""

    def test_conversation_tokens(self):
        """Test conversation token IDs."""
        assert CL100K_AGENT_TOKENS.SYSTEM == 100277
        assert CL100K_AGENT_TOKENS.USER == 100278
        assert CL100K_AGENT_TOKENS.ASSISTANT == 100279
        assert CL100K_AGENT_TOKENS.IM_START == 100280
        assert CL100K_AGENT_TOKENS.IM_END == 100281

    def test_thinking_tokens(self):
        """Test thinking token IDs."""
        assert CL100K_AGENT_TOKENS.THINK == 100282
        assert CL100K_AGENT_TOKENS.THINK_END == 100283

    def test_react_tokens(self):
        """Test ReAct agent loop token IDs."""
        assert CL100K_AGENT_TOKENS.PLAN == 100284
        assert CL100K_AGENT_TOKENS.PLAN_END == 100285
        assert CL100K_AGENT_TOKENS.STEP == 100286
        assert CL100K_AGENT_TOKENS.STEP_END == 100287
        assert CL100K_AGENT_TOKENS.ACT == 100288
        assert CL100K_AGENT_TOKENS.ACT_END == 100289
        assert CL100K_AGENT_TOKENS.OBSERVE == 100290
        assert CL100K_AGENT_TOKENS.OBSERVE_END == 100291

    def test_function_tokens(self):
        """Test function calling token IDs."""
        assert CL100K_AGENT_TOKENS.FUNCTION == 100292
        assert CL100K_AGENT_TOKENS.FUNCTION_END == 100293
        assert CL100K_AGENT_TOKENS.RESULT == 100294
        assert CL100K_AGENT_TOKENS.RESULT_END == 100295
        assert CL100K_AGENT_TOKENS.ERROR == 100296
        assert CL100K_AGENT_TOKENS.ERROR_END == 100297


class TestCl100kChatMLFormat:
    """Test ChatML format commonly used with GPT models."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("cl100k_base")

    def test_chatml_format(self, tokenizer):
        """Test ChatML format encoding/decoding."""
        chat = (
            "<|im_start|>system\n"
            "You are a helpful assistant."
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "Hello!"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        tokens = tokenizer.encode_with_special(chat)

        # Verify special tokens are present
        assert 100280 in tokens  # im_start
        assert 100281 in tokens  # im_end

        # Verify roundtrip
        decoded = tokenizer.decode(tokens)
        assert decoded == chat


class TestCl100kBatchEncoding:
    """Test batch encoding functionality."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("cl100k_base")

    def test_batch_encode(self, tokenizer):
        """Test batch encoding."""
        texts = [
            "Hello, world!",
            "How are you?",
            "I'm doing great!",
        ]

        batch_tokens = tokenizer.encode_batch(texts)

        assert len(batch_tokens) == 3

        # Verify each batch result matches individual encoding
        for i, text in enumerate(texts):
            individual = tokenizer.encode(text)
            assert batch_tokens[i] == individual, (
                f"Batch encoding should match individual encoding for text {i}: {text!r}"
            )


class TestCl100kSpecialTokenDecode:
    """Test that special tokens decode correctly."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("cl100k_base")

    def test_decode_endoftext(self, tokenizer):
        """Test decoding endoftext token."""
        decoded = tokenizer.decode([100257])
        assert decoded == "<|endoftext|>"

    def test_decode_fim_prefix(self, tokenizer):
        """Test decoding fim_prefix token."""
        decoded = tokenizer.decode([100258])
        assert decoded == "<|fim_prefix|>"

    def test_decode_endofprompt(self, tokenizer):
        """Test decoding endofprompt token."""
        decoded = tokenizer.decode([100276])
        assert decoded == "<|endofprompt|>"


class TestCl100kEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("cl100k_base")

    def test_empty_input(self, tokenizer):
        """Test empty input handling."""
        tokens = tokenizer.encode("")
        assert tokens == [], "Empty input should produce empty tokens"

        decoded = tokenizer.decode([])
        assert decoded == "", "Empty tokens should decode to empty string"

    def test_whitespace_only(self, tokenizer):
        """Test whitespace-only input."""
        text = "   \n\t  "
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text


class TestCl100kCodeContent:
    """Test code-related content (GPT-4 is commonly used for code)."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("cl100k_base")

    def test_python_code(self, tokenizer):
        """Test Python code encoding/decoding."""
        code = '''
def hello_world():
    print("Hello, World!")

if __name__ == "__main__":
    hello_world()
'''
        tokens = tokenizer.encode(code)
        decoded = tokenizer.decode(tokens)
        assert decoded == code

    def test_fim_format(self, tokenizer):
        """Test FIM (Fill-in-the-Middle) format used for code completion."""
        fim = "<|fim_prefix|>def hello():\n    <|fim_suffix|>\n    return result<|fim_middle|>"

        tokens = tokenizer.encode_with_special(fim)

        # Verify FIM tokens are present
        assert 100258 in tokens  # fim_prefix
        assert 100259 in tokens  # fim_middle
        assert 100260 in tokens  # fim_suffix

        # Verify roundtrip
        decoded = tokenizer.decode(tokens)
        assert decoded == fim


class TestCl100kPattern:
    """Test CL100K_BASE_PATTERN constant."""

    def test_pattern_is_string(self):
        """Test that CL100K_BASE_PATTERN is a string."""
        assert isinstance(CL100K_BASE_PATTERN, str)

    def test_pattern_is_non_empty(self):
        """Test that CL100K_BASE_PATTERN is non-empty."""
        assert len(CL100K_BASE_PATTERN) > 0


class TestCl100kStreamingDecoder:
    """Test streaming decoder with cl100k."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("cl100k_base")

    def test_streaming_decoder(self, tokenizer):
        """Test streaming decoder produces correct output."""
        text = "Hello, world!"
        tokens = tokenizer.encode(text)

        decoder = tokenizer.streaming_decoder()
        result = ""
        for token in tokens:
            chunk = decoder.add_token(token)
            if chunk:
                result += chunk
        result += decoder.flush()

        assert result == text

    def test_streaming_decoder_with_special_tokens(self, tokenizer):
        """Test streaming decoder with special tokens."""
        text = "<|endoftext|>Hello<|endofprompt|>"
        tokens = tokenizer.encode_with_special(text)

        decoder = tokenizer.streaming_decoder()
        result = ""
        for token in tokens:
            chunk = decoder.add_token(token)
            if chunk:
                result += chunk
        result += decoder.flush()

        assert result == text


class TestCl100kBackendOptions:
    """Test regex backend options (pcre2, jit)."""

    def test_default_backend(self):
        """Test default backend (regexr with JIT)."""
        tokenizer = Tokenizer.from_pretrained("cl100k_base")
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text

    def test_pcre2_backend(self):
        """Test switching to PCRE2 backend."""
        tokenizer = Tokenizer.from_pretrained("cl100k_base").pcre2(True)
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text

    def test_pcre2_switch_back_to_regexr(self):
        """Test switching from PCRE2 back to regexr."""
        tokenizer = (
            Tokenizer.from_pretrained("cl100k_base")
            .pcre2(True)
            .pcre2(False)
        )
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text

    def test_jit_disabled(self):
        """Test with JIT disabled."""
        tokenizer = Tokenizer.from_pretrained("cl100k_base").jit(False)
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text

    def test_jit_enabled(self):
        """Test with JIT explicitly enabled."""
        tokenizer = Tokenizer.from_pretrained("cl100k_base").jit(True)
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text

    def test_pcre2_with_jit_disabled(self):
        """Test PCRE2 backend with JIT disabled."""
        tokenizer = (
            Tokenizer.from_pretrained("cl100k_base")
            .pcre2(True)
            .jit(False)
        )
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text

    def test_pcre2_with_jit_enabled(self):
        """Test PCRE2 backend with JIT enabled."""
        tokenizer = (
            Tokenizer.from_pretrained("cl100k_base")
            .pcre2(True)
            .jit(True)
        )
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text

    def test_backend_consistency(self):
        """Test that different backends produce same tokens."""
        text = "The quick brown fox jumps over the lazy dog. 你好世界 🦀"

        # Default (regexr + JIT)
        tok_default = Tokenizer.from_pretrained("cl100k_base")
        tokens_default = tok_default.encode(text)

        # PCRE2 + JIT
        tok_pcre2 = Tokenizer.from_pretrained("cl100k_base").pcre2(True)
        tokens_pcre2 = tok_pcre2.encode(text)

        # regexr without JIT
        tok_no_jit = Tokenizer.from_pretrained("cl100k_base").jit(False)
        tokens_no_jit = tok_no_jit.encode(text)

        # All backends should produce identical tokens
        assert tokens_default == tokens_pcre2, "PCRE2 should produce same tokens"
        assert tokens_default == tokens_no_jit, "Non-JIT should produce same tokens"


class TestCl100kUtf8Boundaries:
    """Test UTF-8 boundary handling with multi-byte characters.

    These tests catch bugs where regex match positions fall inside
    multi-byte UTF-8 characters (em-dashes, curly quotes, etc.).
    """

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("cl100k_base")

    @pytest.fixture
    def tokenizer_pcre2(self):
        return Tokenizer.from_pretrained("cl100k_base").pcre2(True)

    def test_em_dash(self, tokenizer):
        """Test encoding text with em-dash (3-byte UTF-8 character)."""
        text = "I'm sorry you're hurting—breakups suck, but you'll get through it."
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text

    def test_curly_quotes(self, tokenizer):
        """Test encoding text with curly quotes (3-byte UTF-8 characters)."""
        text = 'He said, \u2018Hello\u2019 and she replied, \u201cGoodbye\u201d.'
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text

    def test_mixed_multibyte(self, tokenizer):
        """Test encoding text with mixed multi-byte characters."""
        text = "Check if you're using valid credentials—API key, token—in headers."
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text

    def test_em_dash_at_boundaries(self, tokenizer):
        """Test em-dash at various positions that may cause boundary issues."""
        texts = [
            "word—word",
            "a—b",
            "test—",
            "—start",
            "one—two—three",
            "Check your brake pads or rotors—they might be worn out.",
        ]
        for text in texts:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            assert decoded == text, f"Failed for: {text!r}"

    def test_batch_encode_multibyte(self, tokenizer):
        """Test batch encoding with multi-byte characters."""
        texts = [
            "I'm sorry you're hurting—breakups suck.",
            "Check if you're using valid credentials.",
            "That weird noise could hint at a few things!",
            "Grinding while braking? Check your brake pads—they might be worn.",
        ]
        all_tokens = tokenizer.encode_batch(texts)
        for i, (text, tokens) in enumerate(zip(texts, all_tokens)):
            decoded = tokenizer.decode(tokens)
            assert decoded == text, f"Failed for text {i}: {text!r}"

    def test_batch_encode_with_special_multibyte(self, tokenizer):
        """Test batch encoding with special tokens and multi-byte characters."""
        texts = [
            "<|user|>I'm hurting—help me<|assistant|>Here's how—step by step:",
            "<|system|>You're a helpful assistant<|user|>What's this—a bug?",
        ]
        all_tokens = tokenizer.encode_batch_with_special(texts)
        for i, (text, tokens) in enumerate(zip(texts, all_tokens)):
            decoded = tokenizer.decode(tokens)
            assert decoded == text, f"Failed for text {i}: {text!r}"

    def test_backend_consistency_multibyte(self, tokenizer, tokenizer_pcre2):
        """Test that regexr and PCRE2 produce same results for multi-byte text."""
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

    def test_large_batch_multibyte_parallel(self, tokenizer):
        """Test large batch encoding with multi-byte chars to trigger parallel execution.

        This test catches UTF-8 boundary bugs that only manifest in parallel
        batch processing (rayon threads).
        """
        # Create many texts with em-dashes and curly quotes at various positions
        base_texts = [
            "I'm sorry you're hurting—breakups suck, but you'll get through it.",
            "Check if you're using valid credentials—API key, token—in headers.",
            "That weird noise could hint at a few things—grinding, rattling, knocking.",
            "Grinding while braking? Check your brake pads or rotors—they might be worn out.",
            'He said, \u2018Hello\u2019 and she replied, \u201cGoodbye\u201d.',
            "word—word—word—word—word",
            "A 403 Forbidden error means your API request is authenticated but lacks permission.",
        ]
        # Repeat to trigger parallel execution
        texts = base_texts * 100  # 700 texts

        all_tokens = tokenizer.encode_batch(texts)
        assert len(all_tokens) == len(texts)

        # Verify roundtrip for a sample
        for i in range(0, len(texts), 50):
            decoded = tokenizer.decode(all_tokens[i])
            assert decoded == texts[i], f"Failed roundtrip for text {i}"
