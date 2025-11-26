"""
Integration tests for DeepSeek V3 tokenizer.

These tests verify that the DeepSeek V3 tokenizer correctly encodes and decodes text,
handles ByteLevel BPE encoding, special tokens, and produces consistent results.
"""

import pytest
from splintr import Tokenizer, DEEPSEEK_V3_AGENT_TOKENS


class TestDeepSeekV3Tokenizer:
    """Test suite for DeepSeek V3 tokenizer."""

    @pytest.fixture
    def tokenizer(self):
        """Create a DeepSeek V3 tokenizer for testing."""
        return Tokenizer.from_pretrained("deepseek_v3")

    def test_encode_decode_roundtrip(self, tokenizer):
        """Test basic encoding and decoding roundtrip."""
        test_cases = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "Rust is a systems programming language.",
            "1234567890",
            "Special characters: !@#$%^&*()",
            "Multi-line\ntext\nwith\nnewlines",
        ]

        for text in test_cases:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            assert decoded == text, f"Roundtrip failed for: {text!r}"

    def test_vocab_size(self, tokenizer):
        """Test that vocab size is correct (128,000 BPE tokens)."""
        # DeepSeek V3 has 128,000 BPE tokens plus special tokens
        assert tokenizer.vocab_size >= 128000, (
            f"Vocab size should be at least 128,000, got {tokenizer.vocab_size}"
        )


class TestDeepSeekV3ExactTokens:
    """Exact token ID verification tests.

    These tests verify specific token IDs to catch any regression in
    ByteLevel encoding or vocabulary changes.
    """

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("deepseek_v3")

    def test_hello_world_tokens(self, tokenizer):
        """Verify exact token IDs for 'Hello world'."""
        tokens = tokenizer.encode("Hello world")
        assert tokens == [19923, 2058], f"Expected [19923, 2058], got {tokens}"

    def test_space_prefix_tokens(self, tokenizer):
        """Verify exact token IDs for ' hello world '."""
        tokens = tokenizer.encode(" hello world ")
        assert tokens == [44388, 2058, 223], f"Expected [44388, 2058, 223], got {tokens}"

    def test_chinese_tokens(self, tokenizer):
        """Verify exact token IDs for '你好世界'."""
        tokens = tokenizer.encode("你好世界")
        assert tokens == [30594, 3427], f"Expected [30594, 3427], got {tokens}"

    def test_mixed_chinese_english_tokens(self, tokenizer):
        """Verify exact token IDs for 'Hello 你好 World 世界!'."""
        tokens = tokenizer.encode("Hello 你好 World 世界!")
        assert tokens == [19923, 223, 30594, 4495, 223, 3427, 3], (
            f"Expected [19923, 223, 30594, 4495, 223, 3427, 3], got {tokens}"
        )

    def test_emoji_tokens(self, tokenizer):
        """Verify exact token IDs for 'Hello 🌍 World!'."""
        tokens = tokenizer.encode("Hello 🌍 World!")
        assert tokens == [19923, 73369, 238, 4495, 3], (
            f"Expected [19923, 73369, 238, 4495, 3], got {tokens}"
        )


class TestDeepSeekV3ByteLevel:
    """Test ByteLevel BPE encoding specific features."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("deepseek_v3")

    def test_chinese_text(self, tokenizer):
        """Test ByteLevel encoding handles Chinese text correctly."""
        test_cases = [
            "你好",
            "你好世界",
            "中文测试",
            "Hello 你好 World 世界!",
            "混合文本 mixed text 测试",
        ]

        for text in test_cases:
            tokens = tokenizer.encode(text)
            assert len(tokens) > 0, f"Chinese text should produce tokens: {text!r}"
            decoded = tokenizer.decode(tokens)
            assert decoded == text, f"Chinese roundtrip failed for: {text!r}"

    def test_emoji(self, tokenizer):
        """Test ByteLevel encoding handles emoji correctly."""
        test_cases = [
            "Hello 🌍 World!",
            "🦀 Rust is awesome! 🚀",
            "Emoji test: 😀😎🎉",
        ]

        for text in test_cases:
            tokens = tokenizer.encode(text)
            assert len(tokens) > 0, f"Emoji text should produce tokens: {text!r}"
            decoded = tokenizer.decode(tokens)
            assert decoded == text, f"Emoji roundtrip failed for: {text!r}"

    def test_space_handling(self, tokenizer):
        """Test that spaces are preserved correctly (ByteLevel maps space to Ġ)."""
        test_cases = [
            " hello",
            "hello ",
            " hello world ",
            "  double  spaces  ",
            "   leading spaces",
        ]

        for text in test_cases:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            assert decoded == text, f"Space handling failed for: {text!r}"


class TestDeepSeekV3NativeSpecialTokens:
    """Test official DeepSeek native special tokens."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("deepseek_v3")

    def test_begin_end_of_sentence(self, tokenizer):
        """Test begin/end of sentence tokens."""
        tokens = tokenizer.encode_with_special(
            "<｜begin▁of▁sentence｜>Hello<｜end▁of▁sentence｜>"
        )
        assert 0 in tokens, "Should contain begin_of_sentence (0)"
        assert 1 in tokens, "Should contain end_of_sentence (1)"

    def test_thinking_tokens(self, tokenizer):
        """Test thinking tokens (DeepSeek R1 style)."""
        tokens = tokenizer.encode_with_special("<think>Let me think...</think>")
        assert 128798 in tokens, "Should contain think (128798)"
        assert 128799 in tokens, "Should contain think_end (128799)"

    def test_user_assistant_tokens(self, tokenizer):
        """Test user/assistant tokens."""
        tokens = tokenizer.encode_with_special("<｜User｜>Hi<｜Assistant｜>")
        assert 128803 in tokens, "Should contain User (128803)"
        assert 128804 in tokens, "Should contain Assistant (128804)"

    def test_eot_token(self, tokenizer):
        """Test EOT (end of turn) token."""
        tokens = tokenizer.encode_with_special("<|EOT|>")
        assert 128805 in tokens, "Should contain EOT (128805)"


class TestDeepSeekV3FIMTokens:
    """Test DeepSeek FIM (Fill-in-the-Middle) tokens."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("deepseek_v3")

    def test_fim_tokens(self, tokenizer):
        """Test FIM tokens for code completion."""
        tokens = tokenizer.encode_with_special(
            "<｜fim▁begin｜>prefix<｜fim▁hole｜>suffix<｜fim▁end｜>"
        )
        assert 128800 in tokens, "Should contain fim_hole (128800)"
        assert 128801 in tokens, "Should contain fim_begin (128801)"
        assert 128802 in tokens, "Should contain fim_end (128802)"


class TestDeepSeekV3ToolTokens:
    """Test DeepSeek tool calling tokens."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("deepseek_v3")

    def test_tool_calls_tokens(self, tokenizer):
        """Test tool calls structure tokens."""
        tokens = tokenizer.encode_with_special(
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        )
        assert 128806 in tokens, "Should contain tool_calls_begin (128806)"
        assert 128807 in tokens, "Should contain tool_calls_end (128807)"
        assert 128808 in tokens, "Should contain tool_call_begin (128808)"
        assert 128809 in tokens, "Should contain tool_call_end (128809)"

    def test_tool_outputs_tokens(self, tokenizer):
        """Test tool outputs structure tokens."""
        tokens = tokenizer.encode_with_special(
            "<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>result<｜tool▁output▁end｜><｜tool▁outputs▁end｜>"
        )
        assert 128810 in tokens, "Should contain tool_outputs_begin (128810)"
        assert 128811 in tokens, "Should contain tool_outputs_end (128811)"
        assert 128812 in tokens, "Should contain tool_output_begin (128812)"
        assert 128813 in tokens, "Should contain tool_output_end (128813)"

    def test_tool_sep_token(self, tokenizer):
        """Test tool separator token."""
        tokens = tokenizer.encode_with_special("<｜tool▁sep｜>")
        assert 128814 in tokens, "Should contain tool_sep (128814)"


class TestDeepSeekV3AgentTokens:
    """Test splintr agent tokens for DeepSeek V3."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("deepseek_v3")

    def test_conversation_tokens(self, tokenizer):
        """Test conversation tokens."""
        tokens = tokenizer.encode_with_special(
            "<|system|>You are helpful.<|user|>Hi<|assistant|>"
        )
        assert 128900 in tokens, "Should contain system (128900)"
        assert 128901 in tokens, "Should contain user (128901)"
        assert 128902 in tokens, "Should contain assistant (128902)"

    def test_thinking_tokens(self, tokenizer):
        """Test thinking tokens (splintr style)."""
        tokens = tokenizer.encode_with_special("<|think|>Let me reason...<|/think|>")
        assert 128905 in tokens, "Should contain think (128905)"
        assert 128906 in tokens, "Should contain think_end (128906)"

    def test_function_calling_tokens(self, tokenizer):
        """Test function calling tokens."""
        tokens = tokenizer.encode_with_special("<|function|>get_weather<|/function|>")
        assert 128915 in tokens, "Should contain function (128915)"
        assert 128916 in tokens, "Should contain function_end (128916)"


class TestDeepSeekV3AgentTokensClass:
    """Test DEEPSEEK_V3_AGENT_TOKENS class constants."""

    def test_native_tokens(self):
        """Test official DeepSeek native token IDs."""
        assert DEEPSEEK_V3_AGENT_TOKENS.BEGIN_OF_SENTENCE == 0
        assert DEEPSEEK_V3_AGENT_TOKENS.END_OF_SENTENCE == 1
        assert DEEPSEEK_V3_AGENT_TOKENS.PAD_NATIVE == 2

    def test_thinking_tokens(self):
        """Test DeepSeek thinking token IDs."""
        assert DEEPSEEK_V3_AGENT_TOKENS.THINK_NATIVE == 128798
        assert DEEPSEEK_V3_AGENT_TOKENS.THINK_END_NATIVE == 128799

    def test_fim_tokens(self):
        """Test FIM token IDs."""
        assert DEEPSEEK_V3_AGENT_TOKENS.FIM_HOLE == 128800
        assert DEEPSEEK_V3_AGENT_TOKENS.FIM_BEGIN == 128801
        assert DEEPSEEK_V3_AGENT_TOKENS.FIM_END == 128802

    def test_chat_tokens(self):
        """Test chat token IDs."""
        assert DEEPSEEK_V3_AGENT_TOKENS.USER_NATIVE == 128803
        assert DEEPSEEK_V3_AGENT_TOKENS.ASSISTANT_NATIVE == 128804
        assert DEEPSEEK_V3_AGENT_TOKENS.EOT == 128805

    def test_tool_tokens(self):
        """Test tool calling token IDs."""
        assert DEEPSEEK_V3_AGENT_TOKENS.TOOL_CALLS_BEGIN == 128806
        assert DEEPSEEK_V3_AGENT_TOKENS.TOOL_CALLS_END == 128807
        assert DEEPSEEK_V3_AGENT_TOKENS.TOOL_CALL_BEGIN == 128808
        assert DEEPSEEK_V3_AGENT_TOKENS.TOOL_CALL_END == 128809
        assert DEEPSEEK_V3_AGENT_TOKENS.TOOL_OUTPUTS_BEGIN == 128810
        assert DEEPSEEK_V3_AGENT_TOKENS.TOOL_OUTPUTS_END == 128811
        assert DEEPSEEK_V3_AGENT_TOKENS.TOOL_OUTPUT_BEGIN == 128812
        assert DEEPSEEK_V3_AGENT_TOKENS.TOOL_OUTPUT_END == 128813
        assert DEEPSEEK_V3_AGENT_TOKENS.TOOL_SEP == 128814

    def test_agent_conversation_tokens(self):
        """Test agent conversation token IDs."""
        assert DEEPSEEK_V3_AGENT_TOKENS.SYSTEM == 128900
        assert DEEPSEEK_V3_AGENT_TOKENS.USER == 128901
        assert DEEPSEEK_V3_AGENT_TOKENS.ASSISTANT == 128902
        assert DEEPSEEK_V3_AGENT_TOKENS.IM_START == 128903
        assert DEEPSEEK_V3_AGENT_TOKENS.IM_END == 128904

    def test_agent_thinking_tokens(self):
        """Test agent thinking token IDs."""
        assert DEEPSEEK_V3_AGENT_TOKENS.THINK == 128905
        assert DEEPSEEK_V3_AGENT_TOKENS.THINK_END == 128906

    def test_agent_react_tokens(self):
        """Test ReAct agent loop token IDs."""
        assert DEEPSEEK_V3_AGENT_TOKENS.PLAN == 128907
        assert DEEPSEEK_V3_AGENT_TOKENS.PLAN_END == 128908
        assert DEEPSEEK_V3_AGENT_TOKENS.STEP == 128909
        assert DEEPSEEK_V3_AGENT_TOKENS.STEP_END == 128910
        assert DEEPSEEK_V3_AGENT_TOKENS.ACT == 128911
        assert DEEPSEEK_V3_AGENT_TOKENS.ACT_END == 128912
        assert DEEPSEEK_V3_AGENT_TOKENS.OBSERVE == 128913
        assert DEEPSEEK_V3_AGENT_TOKENS.OBSERVE_END == 128914

    def test_agent_function_tokens(self):
        """Test agent function calling token IDs."""
        assert DEEPSEEK_V3_AGENT_TOKENS.FUNCTION == 128915
        assert DEEPSEEK_V3_AGENT_TOKENS.FUNCTION_END == 128916
        assert DEEPSEEK_V3_AGENT_TOKENS.RESULT == 128917
        assert DEEPSEEK_V3_AGENT_TOKENS.RESULT_END == 128918
        assert DEEPSEEK_V3_AGENT_TOKENS.ERROR == 128919
        assert DEEPSEEK_V3_AGENT_TOKENS.ERROR_END == 128920


class TestDeepSeekV3ChatFormat:
    """Test DeepSeek V3 chat template format."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("deepseek_v3")

    def test_chat_format(self, tokenizer):
        """Test DeepSeek V3 chat format encoding/decoding."""
        chat = "<｜begin▁of▁sentence｜><｜User｜>Hello!<｜Assistant｜>Hi there!<|EOT|>"

        tokens = tokenizer.encode_with_special(chat)

        # Verify special tokens are present
        assert 0 in tokens  # begin_of_sentence
        assert 128803 in tokens  # User
        assert 128804 in tokens  # Assistant
        assert 128805 in tokens  # EOT

        # Verify roundtrip
        decoded = tokenizer.decode(tokens)
        assert decoded == chat

    def test_thinking_format(self, tokenizer):
        """Test DeepSeek V3 thinking format (R1-style reasoning)."""
        chat = (
            "<｜User｜>What is 2+2?"
            "<｜Assistant｜><think>Let me calculate: 2+2=4</think>"
            "The answer is 4.<|EOT|>"
        )

        tokens = tokenizer.encode_with_special(chat)

        # Verify special tokens
        assert 128803 in tokens  # User
        assert 128804 in tokens  # Assistant
        assert 128798 in tokens  # think
        assert 128799 in tokens  # /think
        assert 128805 in tokens  # EOT

        # Verify roundtrip
        decoded = tokenizer.decode(tokens)
        assert decoded == chat


class TestDeepSeekV3BatchEncoding:
    """Test batch encoding functionality."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("deepseek_v3")

    def test_batch_encode(self, tokenizer):
        """Test batch encoding."""
        texts = [
            "Hello, world!",
            "你好世界",
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


class TestDeepSeekV3SpecialTokenDecode:
    """Test that special tokens decode correctly."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("deepseek_v3")

    def test_decode_begin_of_sentence(self, tokenizer):
        """Test decoding begin_of_sentence token."""
        decoded = tokenizer.decode([0])
        assert decoded == "<｜begin▁of▁sentence｜>"

    def test_decode_end_of_sentence(self, tokenizer):
        """Test decoding end_of_sentence token."""
        decoded = tokenizer.decode([1])
        assert decoded == "<｜end▁of▁sentence｜>"

    def test_decode_think(self, tokenizer):
        """Test decoding think token."""
        decoded = tokenizer.decode([128798])
        assert decoded == "<think>"

    def test_decode_think_end(self, tokenizer):
        """Test decoding think_end token."""
        decoded = tokenizer.decode([128799])
        assert decoded == "</think>"

    def test_decode_user(self, tokenizer):
        """Test decoding User token."""
        decoded = tokenizer.decode([128803])
        assert decoded == "<｜User｜>"

    def test_decode_assistant(self, tokenizer):
        """Test decoding Assistant token."""
        decoded = tokenizer.decode([128804])
        assert decoded == "<｜Assistant｜>"

    def test_decode_eot(self, tokenizer):
        """Test decoding EOT token."""
        decoded = tokenizer.decode([128805])
        assert decoded == "<|EOT|>"


class TestDeepSeekV3EdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("deepseek_v3")

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


class TestDeepSeekV3Variants:
    """Test that all from_pretrained variants work."""

    @pytest.mark.parametrize(
        "variant",
        ["deepseek_v3", "deepseek-v3"],
    )
    def test_from_pretrained_variants(self, variant):
        """Test that all DeepSeek V3 variants can be loaded."""
        tokenizer = Tokenizer.from_pretrained(variant)
        assert tokenizer is not None

    def test_all_variants_same_encoding(self):
        """Test that all variants produce the same encoding."""
        text = "Hello, world!"

        t1 = Tokenizer.from_pretrained("deepseek_v3")
        t2 = Tokenizer.from_pretrained("deepseek-v3")

        tokens1 = t1.encode(text)
        tokens2 = t2.encode(text)

        assert tokens1 == tokens2, "Both DeepSeek V3 variants should produce same encoding"


class TestDeepSeekV3MixedSpecialTokens:
    """Test mixed special tokens from different sources."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("deepseek_v3")

    def test_mixed_native_and_agent_tokens(self, tokenizer):
        """Test mixing native DeepSeek tokens with splintr agent tokens."""
        chat = (
            "<｜User｜>Tell me about Rust."
            "<|think|>User wants info about Rust programming language.<|/think|>"
            "<｜Assistant｜>Rust is a systems programming language."
        )

        tokens = tokenizer.encode_with_special(chat)

        # Native tokens
        assert 128803 in tokens  # User (native)
        assert 128804 in tokens  # Assistant (native)

        # Agent tokens
        assert 128905 in tokens  # think (agent)
        assert 128906 in tokens  # /think (agent)

        # Verify roundtrip
        decoded = tokenizer.decode(tokens)
        assert decoded == chat


class TestDeepSeekV3ByteLevelStreamingDecoder:
    """Test ByteLevel streaming decoder with DeepSeek V3.

    The ByteLevelStreamingDecoder properly handles ByteLevel BPE encoding
    by first decoding token bytes from ByteLevel representation to raw bytes,
    then assembling into valid UTF-8 strings.
    """

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("deepseek_v3")

    def test_byte_level_streaming_decoder_ascii(self, tokenizer):
        """Test ByteLevel streaming decoder with ASCII text."""
        text = "Hello, world!"
        tokens = tokenizer.encode(text)

        decoder = tokenizer.byte_level_streaming_decoder()
        result = ""
        for token in tokens:
            chunk = decoder.add_token(token)
            if chunk:
                result += chunk
        result += decoder.flush()

        assert result == text

    def test_byte_level_streaming_decoder_chinese(self, tokenizer):
        """Test ByteLevel streaming decoder with Chinese text."""
        text = "你好世界"
        tokens = tokenizer.encode(text)

        decoder = tokenizer.byte_level_streaming_decoder()
        result = ""
        for token in tokens:
            chunk = decoder.add_token(token)
            if chunk:
                result += chunk
        result += decoder.flush()

        assert result == text

    def test_byte_level_streaming_decoder_mixed(self, tokenizer):
        """Test ByteLevel streaming decoder with mixed content."""
        text = "Hello 你好 World 世界!"
        tokens = tokenizer.encode(text)

        decoder = tokenizer.byte_level_streaming_decoder()
        result = ""
        for token in tokens:
            chunk = decoder.add_token(token)
            if chunk:
                result += chunk
        result += decoder.flush()

        assert result == text

    def test_byte_level_streaming_decoder_emoji(self, tokenizer):
        """Test ByteLevel streaming decoder with emoji."""
        text = "Hello 🌍 World!"
        tokens = tokenizer.encode(text)

        decoder = tokenizer.byte_level_streaming_decoder()
        result = ""
        for token in tokens:
            chunk = decoder.add_token(token)
            if chunk:
                result += chunk
        result += decoder.flush()

        assert result == text

    def test_byte_level_streaming_decoder_spaces(self, tokenizer):
        """Test ByteLevel streaming decoder with spaces."""
        text = " hello world "
        tokens = tokenizer.encode(text)

        decoder = tokenizer.byte_level_streaming_decoder()
        result = ""
        for token in tokens:
            chunk = decoder.add_token(token)
            if chunk:
                result += chunk
        result += decoder.flush()

        assert result == text

    def test_byte_level_streaming_decoder_special_tokens(self, tokenizer):
        """Test ByteLevel streaming decoder with special tokens."""
        text = "<｜begin▁of▁sentence｜>Hello<|EOT|>"
        tokens = tokenizer.encode_with_special(text)

        decoder = tokenizer.byte_level_streaming_decoder()
        result = ""
        for token in tokens:
            chunk = decoder.add_token(token)
            if chunk:
                result += chunk
        result += decoder.flush()

        assert result == text

    def test_byte_level_streaming_decoder_mixed_special(self, tokenizer):
        """Test ByteLevel streaming decoder with mixed content and special tokens."""
        text = "<｜User｜>你好!<|think|>Let me think...<|/think|><｜Assistant｜>Hello!"
        tokens = tokenizer.encode_with_special(text)

        decoder = tokenizer.byte_level_streaming_decoder()
        result = ""
        for token in tokens:
            chunk = decoder.add_token(token)
            if chunk:
                result += chunk
        result += decoder.flush()

        assert result == text

    def test_byte_level_streaming_decoder_add_tokens(self, tokenizer):
        """Test ByteLevel streaming decoder add_tokens method."""
        text = "Hello, world!"
        tokens = tokenizer.encode(text)

        decoder = tokenizer.byte_level_streaming_decoder()
        result = decoder.add_tokens(tokens) or ""
        result += decoder.flush()

        assert result == text

    def test_byte_level_streaming_decoder_reset(self, tokenizer):
        """Test ByteLevel streaming decoder reset method."""
        text = "Hello"
        tokens = tokenizer.encode(text)

        decoder = tokenizer.byte_level_streaming_decoder()
        # Add first token, should have pending
        decoder.add_token(tokens[0])
        assert decoder.pending_bytes >= 0  # May or may not have pending

        # Reset clears everything
        decoder.reset()
        assert not decoder.has_pending
        assert decoder.pending_bytes == 0

    def test_byte_level_streaming_decoder_repr(self, tokenizer):
        """Test ByteLevel streaming decoder __repr__."""
        decoder = tokenizer.byte_level_streaming_decoder()
        repr_str = repr(decoder)
        assert "ByteLevelStreamingDecoder" in repr_str
        assert "pending_bytes" in repr_str
