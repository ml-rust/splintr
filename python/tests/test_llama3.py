"""
Integration tests for Llama 3/3.1/3.2/3.3 tokenizer.

These tests verify that the Llama 3 tokenizer correctly encodes and decodes text,
handles special tokens, and produces consistent results.
"""

import pytest
from splintr import Tokenizer, LLAMA3_AGENT_TOKENS, LLAMA3_PATTERN


class TestLlama3Tokenizer:
    """Test suite for Llama 3 tokenizer."""

    @pytest.fixture
    def tokenizer(self):
        """Create a Llama 3 tokenizer for testing."""
        return Tokenizer.from_pretrained("llama3")

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
        """Test that vocab size is correct (128,000 BPE tokens)."""
        # Llama 3 has 128,000 BPE tokens plus special tokens
        assert tokenizer.vocab_size >= 128000, (
            f"Vocab size should be at least 128,000, got {tokenizer.vocab_size}"
        )


class TestLlama3MetaSpecialTokens:
    """Test official Meta special tokens from Llama 3.3."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("llama3")

    def test_begin_end_of_text(self, tokenizer):
        """Test begin/end of text tokens."""
        tokens = tokenizer.encode_with_special("<|begin_of_text|>Hello<|end_of_text|>")
        assert 128000 in tokens, "Should contain begin_of_text (128000)"
        assert 128001 in tokens, "Should contain end_of_text (128001)"

    def test_header_markers(self, tokenizer):
        """Test header markers."""
        tokens = tokenizer.encode_with_special(
            "<|start_header_id|>system<|end_header_id|>"
        )
        assert 128006 in tokens, "Should contain start_header_id (128006)"
        assert 128007 in tokens, "Should contain end_header_id (128007)"

    def test_end_of_turn(self, tokenizer):
        """Test end of turn token."""
        tokens = tokenizer.encode_with_special("<|eot_id|>")
        assert 128009 in tokens, "Should contain eot_id (128009)"


class TestLlama31SpecialTokens:
    """Test Llama 3.1+ specific tokens."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("llama3.1")

    def test_finetune_right_pad_id(self, tokenizer):
        """Test finetune_right_pad_id (added in 3.1)."""
        tokens = tokenizer.encode_with_special("<|finetune_right_pad_id|>")
        assert 128004 in tokens, "Should contain finetune_right_pad_id (128004)"

    def test_eom_id(self, tokenizer):
        """Test eom_id - end of message for tool use (added in 3.1)."""
        tokens = tokenizer.encode_with_special("<|eom_id|>")
        assert 128008 in tokens, "Should contain eom_id (128008)"

    def test_python_tag(self, tokenizer):
        """Test python_tag for code interpreter (added in 3.1)."""
        tokens = tokenizer.encode_with_special("<|python_tag|>")
        assert 128010 in tokens, "Should contain python_tag (128010)"


class TestLlama32VisionTokens:
    """Test Llama 3.2-Vision specific tokens."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("llama3.2")

    def test_step_id(self, tokenizer):
        """Test step_id token (added in 3.2-Vision)."""
        tokens = tokenizer.encode_with_special("<|step_id|>")
        assert 128005 in tokens, "Should contain step_id (128005)"

    def test_image_token(self, tokenizer):
        """Test official Meta image token (added in 3.2-Vision)."""
        tokens = tokenizer.encode_with_special("<|image|>content<|/image|>")
        assert 128256 in tokens, "Should contain image (128256)"
        assert 128257 in tokens, "Should contain image_end (128257)"

    def test_image_decode(self, tokenizer):
        """Test decoding image token."""
        decoded = tokenizer.decode([128256])
        assert decoded == "<|image|>"

        decoded = tokenizer.decode([128005])
        assert decoded == "<|step_id|>"


class TestLlama3AgentTokens:
    """Test splintr agent tokens for Llama 3."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("llama3")

    def test_conversation_tokens(self, tokenizer):
        """Test conversation tokens."""
        tokens = tokenizer.encode_with_special(
            "<|system|>You are helpful.<|user|>Hi<|assistant|>"
        )
        assert 128300 in tokens, "Should contain system (128300)"
        assert 128301 in tokens, "Should contain user (128301)"
        assert 128302 in tokens, "Should contain assistant (128302)"

    def test_thinking_tokens(self, tokenizer):
        """Test thinking tokens."""
        tokens = tokenizer.encode_with_special("<|think|>Let me reason...<|/think|>")
        assert 128305 in tokens, "Should contain think (128305)"
        assert 128306 in tokens, "Should contain think_end (128306)"

    def test_function_calling_tokens(self, tokenizer):
        """Test function calling tokens."""
        tokens = tokenizer.encode_with_special("<|function|>get_weather<|/function|>")
        assert 128315 in tokens, "Should contain function (128315)"
        assert 128316 in tokens, "Should contain function_end (128316)"


class TestLlama3AgentTokensClass:
    """Test LLAMA3_AGENT_TOKENS class constants."""

    def test_meta_tokens(self):
        """Test official Meta token IDs."""
        assert LLAMA3_AGENT_TOKENS.BEGIN_OF_TEXT == 128000
        assert LLAMA3_AGENT_TOKENS.END_OF_TEXT == 128001
        assert LLAMA3_AGENT_TOKENS.FINETUNE_RIGHT_PAD_ID == 128004
        assert LLAMA3_AGENT_TOKENS.STEP_ID == 128005  # Llama 3.2-Vision
        assert LLAMA3_AGENT_TOKENS.START_HEADER_ID == 128006
        assert LLAMA3_AGENT_TOKENS.END_HEADER_ID == 128007
        assert LLAMA3_AGENT_TOKENS.EOM_ID == 128008
        assert LLAMA3_AGENT_TOKENS.EOT_ID == 128009
        assert LLAMA3_AGENT_TOKENS.PYTHON_TAG == 128010

    def test_vision_tokens(self):
        """Test Llama 3.2-Vision multimodal token IDs."""
        # IMAGE is aligned with official Meta <|image|> token
        assert LLAMA3_AGENT_TOKENS.IMAGE == 128256
        assert LLAMA3_AGENT_TOKENS.IMAGE_END == 128257

    def test_conversation_tokens(self):
        """Test conversation token IDs."""
        assert LLAMA3_AGENT_TOKENS.SYSTEM == 128300
        assert LLAMA3_AGENT_TOKENS.USER == 128301
        assert LLAMA3_AGENT_TOKENS.ASSISTANT == 128302
        assert LLAMA3_AGENT_TOKENS.IM_START == 128303
        assert LLAMA3_AGENT_TOKENS.IM_END == 128304

    def test_thinking_tokens(self):
        """Test thinking token IDs."""
        assert LLAMA3_AGENT_TOKENS.THINK == 128305
        assert LLAMA3_AGENT_TOKENS.THINK_END == 128306

    def test_react_tokens(self):
        """Test ReAct agent loop token IDs."""
        assert LLAMA3_AGENT_TOKENS.PLAN == 128307
        assert LLAMA3_AGENT_TOKENS.PLAN_END == 128308
        assert LLAMA3_AGENT_TOKENS.STEP == 128309
        assert LLAMA3_AGENT_TOKENS.STEP_END == 128310
        assert LLAMA3_AGENT_TOKENS.ACT == 128311
        assert LLAMA3_AGENT_TOKENS.ACT_END == 128312
        assert LLAMA3_AGENT_TOKENS.OBSERVE == 128313
        assert LLAMA3_AGENT_TOKENS.OBSERVE_END == 128314

    def test_function_tokens(self):
        """Test function calling token IDs."""
        assert LLAMA3_AGENT_TOKENS.FUNCTION == 128315
        assert LLAMA3_AGENT_TOKENS.FUNCTION_END == 128316
        assert LLAMA3_AGENT_TOKENS.RESULT == 128317
        assert LLAMA3_AGENT_TOKENS.RESULT_END == 128318
        assert LLAMA3_AGENT_TOKENS.ERROR == 128319
        assert LLAMA3_AGENT_TOKENS.ERROR_END == 128320


class TestLlama3ChatFormat:
    """Test Llama 3 chat template format."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("llama3")

    def test_chat_format(self, tokenizer):
        """Test Llama 3 chat format encoding/decoding."""
        chat = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful assistant."
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "Hello!"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        tokens = tokenizer.encode_with_special(chat)

        # Verify special tokens are present
        assert 128000 in tokens  # begin_of_text
        assert 128006 in tokens  # start_header_id
        assert 128007 in tokens  # end_header_id
        assert 128009 in tokens  # eot_id

        # Verify roundtrip
        decoded = tokenizer.decode(tokens)
        assert decoded == chat


class TestLlama3BatchEncoding:
    """Test batch encoding functionality."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("llama3")

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


class TestLlama3SpecialTokenDecode:
    """Test that special tokens decode correctly."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("llama3")

    def test_decode_begin_of_text(self, tokenizer):
        """Test decoding begin_of_text token."""
        decoded = tokenizer.decode([128000])
        assert decoded == "<|begin_of_text|>"

    def test_decode_eot_id(self, tokenizer):
        """Test decoding eot_id token."""
        decoded = tokenizer.decode([128009])
        assert decoded == "<|eot_id|>"

    def test_decode_eom_id(self, tokenizer):
        """Test decoding eom_id token."""
        decoded = tokenizer.decode([128008])
        assert decoded == "<|eom_id|>"

    def test_decode_python_tag(self, tokenizer):
        """Test decoding python_tag token."""
        decoded = tokenizer.decode([128010])
        assert decoded == "<|python_tag|>"


class TestLlama3EdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("llama3")

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


class TestLlama3Variants:
    """Test that all from_pretrained variants work."""

    @pytest.mark.parametrize(
        "variant",
        ["llama3", "llama3.1", "llama3.2", "llama3.3"],
    )
    def test_from_pretrained_variants(self, variant):
        """Test that all Llama 3 variants can be loaded."""
        tokenizer = Tokenizer.from_pretrained(variant)
        assert tokenizer is not None

    def test_all_variants_same_encoding(self):
        """Test that all variants produce the same encoding."""
        text = "Hello, world!"

        t1 = Tokenizer.from_pretrained("llama3")
        t2 = Tokenizer.from_pretrained("llama3.1")
        t3 = Tokenizer.from_pretrained("llama3.2")
        t4 = Tokenizer.from_pretrained("llama3.3")

        tokens1 = t1.encode(text)
        tokens2 = t2.encode(text)
        tokens3 = t3.encode(text)
        tokens4 = t4.encode(text)

        assert tokens1 == tokens2 == tokens3 == tokens4, (
            "All Llama 3 variants should produce same encoding"
        )


class TestLlama3Pattern:
    """Test LLAMA3_PATTERN constant."""

    def test_pattern_is_string(self):
        """Test that LLAMA3_PATTERN is a string."""
        assert isinstance(LLAMA3_PATTERN, str)

    def test_pattern_is_non_empty(self):
        """Test that LLAMA3_PATTERN is non-empty."""
        # Note: LLAMA3_PATTERN uses PCRE2 syntax (e.g., \p{L}) which is not
        # compatible with Python's re module. The pattern is used by the
        # Rust PCRE2 engine internally.
        assert len(LLAMA3_PATTERN) > 0


class TestLlama3StreamingDecoder:
    """Test streaming decoder with Llama 3."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("llama3")

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
        text = "<|begin_of_text|>Hello<|eot_id|>"
        tokens = tokenizer.encode_with_special(text)

        decoder = tokenizer.streaming_decoder()
        result = ""
        for token in tokens:
            chunk = decoder.add_token(token)
            if chunk:
                result += chunk
        result += decoder.flush()

        assert result == text
