"""
Integration tests for cl100k_base tokenizer (GPT-4, GPT-3.5-turbo).

These tests verify that the cl100k_base tokenizer correctly encodes and decodes text,
handles special tokens, and produces consistent results.
"""

import pytest
from splintr import Tokenizer, CL100K_AGENT_TOKENS, CL100K_BASE_PATTERN


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
