"""
Integration tests for o200k_base tokenizer (GPT-4o).

These tests verify that the o200k_base tokenizer correctly encodes and decodes text,
handles special tokens, and produces consistent results.
"""

import pytest
from splintr import Tokenizer, O200K_AGENT_TOKENS, O200K_BASE_PATTERN


class TestO200kExactTokens:
    """Exact token ID verification tests.

    These tests verify specific token IDs to catch any regression in
    encoding or vocabulary changes.
    """

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("o200k_base")

    def test_hello_world_tokens(self, tokenizer):
        """Verify exact token IDs for 'Hello world'."""
        tokens = tokenizer.encode("Hello world")
        assert tokens == [13225, 2375], f"Expected [13225, 2375], got {tokens}"

    def test_hello_world_punctuation_tokens(self, tokenizer):
        """Verify exact token IDs for 'Hello, world!'."""
        tokens = tokenizer.encode("Hello, world!")
        assert tokens == [13225, 11, 2375, 0], f"Expected [13225, 11, 2375, 0], got {tokens}"

    def test_chinese_tokens(self, tokenizer):
        """Verify exact token IDs for '你好世界'."""
        tokens = tokenizer.encode("你好世界")
        assert tokens == [177519, 28428], f"Expected [177519, 28428], got {tokens}"

    def test_emoji_tokens(self, tokenizer):
        """Verify exact token IDs for 'Hello 🌍 World!'."""
        tokens = tokenizer.encode("Hello 🌍 World!")
        assert tokens == [13225, 130321, 235, 5922, 0], (
            f"Expected [13225, 130321, 235, 5922, 0], got {tokens}"
        )


class TestO200kTokenizer:
    """Test suite for o200k_base tokenizer."""

    @pytest.fixture
    def tokenizer(self):
        """Create an o200k_base tokenizer for testing."""
        return Tokenizer.from_pretrained("o200k_base")

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
        """Test that vocab size is correct (199,998 BPE tokens)."""
        # o200k_base has 199,998 BPE tokens plus special tokens
        assert tokenizer.vocab_size >= 199998, (
            f"Vocab size should be at least 199,998, got {tokenizer.vocab_size}"
        )


class TestO200kOpenAISpecialTokens:
    """Test OpenAI standard special tokens."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("o200k_base")

    def test_endoftext(self, tokenizer):
        """Test endoftext token."""
        tokens = tokenizer.encode_with_special("Hello<|endoftext|>World")
        assert 199999 in tokens, "Should contain endoftext (199999)"

    def test_endofprompt(self, tokenizer):
        """Test endofprompt token."""
        tokens = tokenizer.encode_with_special("<|endofprompt|>")
        assert 200018 in tokens, "Should contain endofprompt (200018)"


class TestO200kAgentTokens:
    """Test splintr agent tokens for o200k."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("o200k_base")

    def test_conversation_tokens(self, tokenizer):
        """Test conversation tokens."""
        tokens = tokenizer.encode_with_special(
            "<|system|>You are helpful.<|user|>Hi<|assistant|>"
        )
        assert 200019 in tokens, "Should contain system (200019)"
        assert 200020 in tokens, "Should contain user (200020)"
        assert 200021 in tokens, "Should contain assistant (200021)"

    def test_thinking_tokens(self, tokenizer):
        """Test thinking tokens."""
        tokens = tokenizer.encode_with_special("<|think|>Let me reason...<|/think|>")
        assert 200024 in tokens, "Should contain think (200024)"
        assert 200025 in tokens, "Should contain think_end (200025)"

    def test_function_calling_tokens(self, tokenizer):
        """Test function calling tokens."""
        tokens = tokenizer.encode_with_special("<|function|>get_weather<|/function|>")
        assert 200034 in tokens, "Should contain function (200034)"
        assert 200035 in tokens, "Should contain function_end (200035)"


class TestO200kAgentTokensClass:
    """Test O200K_AGENT_TOKENS class constants."""

    def test_conversation_tokens(self):
        """Test conversation token IDs."""
        assert O200K_AGENT_TOKENS.SYSTEM == 200019
        assert O200K_AGENT_TOKENS.USER == 200020
        assert O200K_AGENT_TOKENS.ASSISTANT == 200021
        assert O200K_AGENT_TOKENS.IM_START == 200022
        assert O200K_AGENT_TOKENS.IM_END == 200023

    def test_thinking_tokens(self):
        """Test thinking token IDs."""
        assert O200K_AGENT_TOKENS.THINK == 200024
        assert O200K_AGENT_TOKENS.THINK_END == 200025

    def test_react_tokens(self):
        """Test ReAct agent loop token IDs."""
        assert O200K_AGENT_TOKENS.PLAN == 200026
        assert O200K_AGENT_TOKENS.PLAN_END == 200027
        assert O200K_AGENT_TOKENS.STEP == 200028
        assert O200K_AGENT_TOKENS.STEP_END == 200029
        assert O200K_AGENT_TOKENS.ACT == 200030
        assert O200K_AGENT_TOKENS.ACT_END == 200031
        assert O200K_AGENT_TOKENS.OBSERVE == 200032
        assert O200K_AGENT_TOKENS.OBSERVE_END == 200033

    def test_function_tokens(self):
        """Test function calling token IDs."""
        assert O200K_AGENT_TOKENS.FUNCTION == 200034
        assert O200K_AGENT_TOKENS.FUNCTION_END == 200035
        assert O200K_AGENT_TOKENS.RESULT == 200036
        assert O200K_AGENT_TOKENS.RESULT_END == 200037
        assert O200K_AGENT_TOKENS.ERROR == 200038
        assert O200K_AGENT_TOKENS.ERROR_END == 200039

    def test_multimodal_tokens(self):
        """Test multimodal token IDs (GPT-4o supports vision)."""
        assert O200K_AGENT_TOKENS.IMAGE == 200061
        assert O200K_AGENT_TOKENS.IMAGE_END == 200062
        assert O200K_AGENT_TOKENS.AUDIO == 200063
        assert O200K_AGENT_TOKENS.AUDIO_END == 200064
        assert O200K_AGENT_TOKENS.VIDEO == 200065
        assert O200K_AGENT_TOKENS.VIDEO_END == 200066


class TestO200kChatMLFormat:
    """Test ChatML format commonly used with GPT models."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("o200k_base")

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
        assert 200022 in tokens  # im_start
        assert 200023 in tokens  # im_end

        # Verify roundtrip
        decoded = tokenizer.decode(tokens)
        assert decoded == chat


class TestO200kBatchEncoding:
    """Test batch encoding functionality."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("o200k_base")

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


class TestO200kSpecialTokenDecode:
    """Test that special tokens decode correctly."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("o200k_base")

    def test_decode_endoftext(self, tokenizer):
        """Test decoding endoftext token."""
        decoded = tokenizer.decode([199999])
        assert decoded == "<|endoftext|>"

    def test_decode_endofprompt(self, tokenizer):
        """Test decoding endofprompt token."""
        decoded = tokenizer.decode([200018])
        assert decoded == "<|endofprompt|>"


class TestO200kEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("o200k_base")

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


class TestO200kCodeContent:
    """Test code-related content (GPT-4o is commonly used for code)."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("o200k_base")

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


class TestO200kMultimodalTokens:
    """Test multimodal placeholder tokens (GPT-4o supports vision)."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("o200k_base")

    def test_image_tokens(self, tokenizer):
        """Test image tokens."""
        tokens = tokenizer.encode_with_special("<|image|>image data<|/image|>")
        assert O200K_AGENT_TOKENS.IMAGE in tokens, f"Should contain image ({O200K_AGENT_TOKENS.IMAGE})"
        assert O200K_AGENT_TOKENS.IMAGE_END in tokens, f"Should contain image_end ({O200K_AGENT_TOKENS.IMAGE_END})"

    def test_audio_tokens(self, tokenizer):
        """Test audio tokens."""
        tokens = tokenizer.encode_with_special("<|audio|>audio data<|/audio|>")
        assert O200K_AGENT_TOKENS.AUDIO in tokens, f"Should contain audio ({O200K_AGENT_TOKENS.AUDIO})"
        assert O200K_AGENT_TOKENS.AUDIO_END in tokens, f"Should contain audio_end ({O200K_AGENT_TOKENS.AUDIO_END})"

    def test_video_tokens(self, tokenizer):
        """Test video tokens."""
        tokens = tokenizer.encode_with_special("<|video|>video data<|/video|>")
        assert O200K_AGENT_TOKENS.VIDEO in tokens, f"Should contain video ({O200K_AGENT_TOKENS.VIDEO})"
        assert O200K_AGENT_TOKENS.VIDEO_END in tokens, f"Should contain video_end ({O200K_AGENT_TOKENS.VIDEO_END})"


class TestO200kVsCl100k:
    """Test that o200k has larger vocab than cl100k."""

    def test_o200k_larger_vocab(self):
        """Test that o200k has a larger vocabulary."""
        o200k = Tokenizer.from_pretrained("o200k_base")
        cl100k = Tokenizer.from_pretrained("cl100k_base")

        assert o200k.vocab_size > cl100k.vocab_size, (
            "o200k should have larger vocab than cl100k"
        )

    def test_o200k_more_efficient(self):
        """Test that o200k is generally more efficient (fewer tokens for same text)."""
        o200k = Tokenizer.from_pretrained("o200k_base")
        cl100k = Tokenizer.from_pretrained("cl100k_base")

        # For most text, o200k should produce fewer or equal tokens
        text = "The quick brown fox jumps over the lazy dog."

        o200k_tokens = o200k.encode(text)
        cl100k_tokens = cl100k.encode(text)

        # o200k typically produces fewer tokens due to larger vocab
        # This is a soft assertion - not always true for all text
        assert len(o200k_tokens) <= len(cl100k_tokens) + 2, (
            "o200k should generally produce similar or fewer tokens than cl100k"
        )


class TestO200kPattern:
    """Test O200K_BASE_PATTERN constant."""

    def test_pattern_is_string(self):
        """Test that O200K_BASE_PATTERN is a string."""
        assert isinstance(O200K_BASE_PATTERN, str)

    def test_pattern_is_non_empty(self):
        """Test that O200K_BASE_PATTERN is non-empty."""
        assert len(O200K_BASE_PATTERN) > 0


class TestO200kStreamingDecoder:
    """Test streaming decoder with o200k."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("o200k_base")

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


class TestO200kUtf8Boundaries:
    """Test UTF-8 boundary handling with multi-byte characters.

    These tests catch bugs where regex match positions fall inside
    multi-byte UTF-8 characters (em-dashes, curly quotes, etc.).
    This is especially important for o200k_base which uses a complex pattern.
    """

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("o200k_base")

    @pytest.fixture
    def tokenizer_pcre2(self):
        return Tokenizer.from_pretrained("o200k_base").pcre2(True)

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
        base_texts = [
            "I'm sorry you're hurting—breakups suck, but you'll get through it.",
            "Check if you're using valid credentials—API key, token—in headers.",
            "That weird noise could hint at a few things—grinding, rattling, knocking.",
            "Grinding while braking? Check your brake pads or rotors—they might be worn out.",
            'He said, \u2018Hello\u2019 and she replied, \u201cGoodbye\u201d.',
            "word—word—word—word—word",
            "A 403 Forbidden error means your API request is authenticated but lacks permission.",
        ]
        texts = base_texts * 100  # 700 texts

        all_tokens = tokenizer.encode_batch(texts)
        assert len(all_tokens) == len(texts)

        for i in range(0, len(texts), 50):
            decoded = tokenizer.decode(all_tokens[i])
            assert decoded == texts[i], f"Failed roundtrip for text {i}"
