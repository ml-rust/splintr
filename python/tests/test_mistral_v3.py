"""
Integration tests for Mistral V3/Tekken tokenizer.

Mistral V3 (Tekken) uses Tiktoken-style BPE encoding (NOT SentencePiece).
Key characteristics:
- Vocab size: ~131,126 (131,072 base + 54 agent tokens)
- Uses Tiktoken-style encoding with its own MISTRAL_V3_PATTERN (O200K-like, but
  with no contraction branches and single-digit \\p{N})
- Much larger vocabulary than V1/V2 (4x larger)
- Used by: Mistral NeMo, Mistral Large 2, Pixtral
"""

import pytest
from splintr import Tokenizer, MISTRAL_V3_AGENT_TOKENS

from .conftest import requires_pcre2


class TestMistralV3Loading:
    """Test loading Mistral V3 tokenizer."""

    def test_load_mistral_v3(self):
        """Test loading mistral_v3."""
        tok = Tokenizer.from_pretrained("mistral_v3")
        assert tok is not None
        assert tok.vocab_size > 130000


class TestMistralV3VocabSize:
    """Test vocabulary size."""

    def test_vocab_size(self):
        """V3 vocab: 131,072 base + 54 agent = 131,126."""
        tok = Tokenizer.from_pretrained("mistral_v3")
        assert tok.vocab_size == 131126

    def test_v3_much_larger_than_v2(self):
        """V3 vocab should be ~4x larger than V2."""
        v2 = Tokenizer.from_pretrained("mistral_v2")
        v3 = Tokenizer.from_pretrained("mistral_v3")
        assert v3.vocab_size > v2.vocab_size * 3
        assert v2.vocab_size == 32822
        assert v3.vocab_size == 131126


class TestMistralV3NativeSpecialTokens:
    """Test native special tokens (BOS/EOS/UNK)."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("mistral_v3")

    def test_bos_token(self, tokenizer):
        """Test <s> = BOS = token 1."""
        tokens = tokenizer.encode_with_special("<s>")
        assert tokens == [1], f"<s> should be token 1, got {tokens}"

    def test_eos_token(self, tokenizer):
        """Test </s> = EOS = token 2."""
        tokens = tokenizer.encode_with_special("</s>")
        assert tokens == [2], f"</s> should be token 2, got {tokens}"

    def test_unk_token(self, tokenizer):
        """Test <unk> = UNK = token 0."""
        tokens = tokenizer.encode_with_special("<unk>")
        assert tokens == [0], f"<unk> should be token 0, got {tokens}"

    def test_decode_bos_eos_unk(self, tokenizer):
        """BOS/EOS/UNK decode to nothing, and render on request.

        `decode` implements HuggingFace's default ``skip_special_tokens=True``,
        which V1/V2 were measured to follow; V3 follows it by family
        consistency (no Tekken reference is installed to measure it against --
        see `special_decode_ids` in `src/core/pretrained.rs`).
        """
        assert tokenizer.decode([0]) == ""
        assert tokenizer.decode([1]) == ""
        assert tokenizer.decode([2]) == ""
        assert tokenizer.decode_with_special([0]) == "<unk>"
        assert tokenizer.decode_with_special([1]) == "<s>"
        assert tokenizer.decode_with_special([2]) == "</s>"


class TestMistralV3AgentTokens:
    """Test splintr agent tokens for Mistral V3."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("mistral_v3")

    def test_conversation_tokens(self, tokenizer):
        """Test conversation tokens."""
        # Agent tokens start at 131072 for V3
        tokens = tokenizer.encode_with_special("<|system|>")
        assert tokens == [131072]

        tokens = tokenizer.encode_with_special("<|user|>")
        assert tokens == [131073]

        tokens = tokenizer.encode_with_special("<|assistant|>")
        assert tokens == [131074]

    def test_thinking_tokens(self, tokenizer):
        """Test thinking tokens."""
        tokens = tokenizer.encode_with_special("<|think|>")
        assert tokens == [131077]

        tokens = tokenizer.encode_with_special("<|/think|>")
        assert tokens == [131078]

    def test_function_calling_tokens(self, tokenizer):
        """Test function calling tokens."""
        tokens = tokenizer.encode_with_special("<|function|>")
        assert tokens == [131087]

        tokens = tokenizer.encode_with_special("<|/function|>")
        assert tokens == [131088]


class TestMistralV3AgentTokensClass:
    """Test MISTRAL_V3_AGENT_TOKENS class constants."""

    def test_conversation_tokens(self):
        """Test conversation token IDs."""
        assert MISTRAL_V3_AGENT_TOKENS.SYSTEM == 131072
        assert MISTRAL_V3_AGENT_TOKENS.USER == 131073
        assert MISTRAL_V3_AGENT_TOKENS.ASSISTANT == 131074
        assert MISTRAL_V3_AGENT_TOKENS.IM_START == 131075
        assert MISTRAL_V3_AGENT_TOKENS.IM_END == 131076

    def test_thinking_tokens(self):
        """Test thinking token IDs."""
        assert MISTRAL_V3_AGENT_TOKENS.THINK == 131077
        assert MISTRAL_V3_AGENT_TOKENS.THINK_END == 131078

    def test_react_tokens(self):
        """Test ReAct agent loop token IDs."""
        assert MISTRAL_V3_AGENT_TOKENS.PLAN == 131079
        assert MISTRAL_V3_AGENT_TOKENS.PLAN_END == 131080
        assert MISTRAL_V3_AGENT_TOKENS.STEP == 131081
        assert MISTRAL_V3_AGENT_TOKENS.STEP_END == 131082
        assert MISTRAL_V3_AGENT_TOKENS.ACT == 131083
        assert MISTRAL_V3_AGENT_TOKENS.ACT_END == 131084
        assert MISTRAL_V3_AGENT_TOKENS.OBSERVE == 131085
        assert MISTRAL_V3_AGENT_TOKENS.OBSERVE_END == 131086

    def test_function_tokens(self):
        """Test function calling token IDs."""
        assert MISTRAL_V3_AGENT_TOKENS.FUNCTION == 131087
        assert MISTRAL_V3_AGENT_TOKENS.FUNCTION_END == 131088
        assert MISTRAL_V3_AGENT_TOKENS.RESULT == 131089
        assert MISTRAL_V3_AGENT_TOKENS.RESULT_END == 131090
        assert MISTRAL_V3_AGENT_TOKENS.ERROR == 131091
        assert MISTRAL_V3_AGENT_TOKENS.ERROR_END == 131092

    def test_code_tokens(self):
        """Test code-related token IDs."""
        assert MISTRAL_V3_AGENT_TOKENS.CODE == 131093
        assert MISTRAL_V3_AGENT_TOKENS.CODE_END == 131094
        assert MISTRAL_V3_AGENT_TOKENS.OUTPUT == 131095
        assert MISTRAL_V3_AGENT_TOKENS.OUTPUT_END == 131096
        assert MISTRAL_V3_AGENT_TOKENS.LANG == 131097
        assert MISTRAL_V3_AGENT_TOKENS.LANG_END == 131098

    def test_rag_tokens(self):
        """Test RAG-related token IDs."""
        assert MISTRAL_V3_AGENT_TOKENS.CONTEXT == 131099
        assert MISTRAL_V3_AGENT_TOKENS.CONTEXT_END == 131100
        assert MISTRAL_V3_AGENT_TOKENS.QUOTE == 131101
        assert MISTRAL_V3_AGENT_TOKENS.QUOTE_END == 131102


class TestMistralV3DecodeAgentTokens:
    """Decoding agent tokens.

    `decode` implements HuggingFace's default ``skip_special_tokens=True``, so
    the declared markers do not come back -- V3 drops them exactly as V1/V2 do
    (`sentencepiece` 0.2.0 and `tokenizers` 0.22.1 both decode `[3, ...]` on
    `mistral-7b-v0.3` as plain text). No Tekken reference exists on this
    machine to measure V3 itself against; see `special_decode_ids` in
    `src/core/pretrained.rs` for that caveat stated in full.

    `decode_with_special` is splintr's ``skip_special_tokens=False`` and still
    spells every one of them.
    """

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("mistral_v3")

    @pytest.mark.parametrize(
        ("token_id", "spelling"),
        [
            (131072, "<|system|>"),
            (131073, "<|user|>"),
            (131074, "<|assistant|>"),
            (131077, "<|think|>"),
            (131078, "<|/think|>"),
        ],
    )
    def test_agent_token_drops_but_stays_reachable(self, tokenizer, token_id, spelling):
        """Each agent token decodes to nothing, and renders on request."""
        assert tokenizer.decode([token_id]) == ""
        assert tokenizer.decode_with_special([token_id]) == spelling

    def test_control_tokens_drop(self, tokenizer):
        """The native BOS/EOS/UNK markers follow the same rule."""
        for token_id, spelling in ((0, "<unk>"), (1, "<s>"), (2, "</s>")):
            assert tokenizer.decode([token_id]) == ""
            assert tokenizer.decode_with_special([token_id]) == spelling


class TestMistralV3SpecialTokensMixed:
    """Test special tokens mixed with text."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("mistral_v3")

    def test_special_tokens_in_mixed_text(self, tokenizer):
        """Test that special tokens are recognized in mixed content."""
        tokens = tokenizer.encode_with_special("<|system|>Hi<|user|>Hello<|assistant|>World")

        # Verify special tokens are present
        assert 131072 in tokens  # system
        assert 131073 in tokens  # user
        assert 131074 in tokens  # assistant

        # The markers are dropped by the default decode, and recovered by
        # `decode_with_special` (splintr's `skip_special_tokens=False`).
        decoded = tokenizer.decode(tokens)
        assert "<|system|>" not in decoded
        assert "<|user|>" not in decoded
        assert "<|assistant|>" not in decoded
        assert "Hi" in decoded and "Hello" in decoded and "World" in decoded
        assert (
            tokenizer.decode_with_special(tokens)
            == "<|system|>Hi<|user|>Hello<|assistant|>World"
        )

    def test_thinking_tokens_mixed(self, tokenizer):
        """Test thinking tokens mixed with text."""
        tokens = tokenizer.encode_with_special("<|think|>reasoning<|/think|>")

        # Verify thinking tokens are present
        assert 131077 in tokens  # think
        assert 131078 in tokens  # /think

        decoded = tokenizer.decode(tokens)
        assert "<|think|>" not in decoded
        assert "<|/think|>" not in decoded
        assert "reasoning" in decoded
        assert tokenizer.decode_with_special(tokens) == "<|think|>reasoning<|/think|>"


class TestMistralV3VsOthers:
    """Test differences between Mistral versions."""

    def test_different_from_v1(self):
        """V3 should encode differently than V1."""
        v1 = Tokenizer.from_pretrained("mistral_v1")
        v3 = Tokenizer.from_pretrained("mistral_v3")

        text = "Hello"
        v1_tokens = v1.encode(text)
        v3_tokens = v3.encode(text)

        assert v1_tokens != v3_tokens

    def test_different_from_v2(self):
        """V3 should encode differently than V2."""
        v2 = Tokenizer.from_pretrained("mistral_v2")
        v3 = Tokenizer.from_pretrained("mistral_v3")

        text = "Test"
        v2_tokens = v2.encode(text)
        v3_tokens = v3.encode(text)

        assert v2_tokens != v3_tokens


class TestMistralV3BasicEncoding:
    """Test basic encoding functionality."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("mistral_v3")

    def test_encodes_text(self, tokenizer):
        """V3 should be able to encode basic text."""
        tokens = tokenizer.encode("Hello")
        assert len(tokens) > 0

    def test_empty_input(self, tokenizer):
        """Test empty input handling."""
        tokens = tokenizer.encode("")
        assert tokens == []

        decoded = tokenizer.decode([])
        assert decoded == ""

    def test_batch_encoding(self, tokenizer):
        """Test batch encoding."""
        texts = ["Hello", "World", "Test"]
        batch_tokens = tokenizer.encode_batch(texts)

        assert len(batch_tokens) == 3
        for i, text in enumerate(texts):
            individual = tokenizer.encode(text)
            assert batch_tokens[i] == individual


class TestMistralV3Roundtrip:
    """Test encode -> decode roundtrip preserves text."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("mistral_v3")

    def test_roundtrip_hello_world(self, tokenizer):
        """Test roundtrip for 'Hello world'."""
        text = "Hello world"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text, "Spaces should be preserved"

    def test_roundtrip_with_punctuation(self, tokenizer):
        """Test roundtrip for 'Hello, world!'."""
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text

    def test_roundtrip_leading_space(self, tokenizer):
        """Test roundtrip for ' hello world '."""
        text = " hello world "
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text, "Leading/trailing spaces should be preserved"

    def test_roundtrip_multiple_spaces(self, tokenizer):
        """Test roundtrip for 'hello  world'."""
        text = "hello  world"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text, "Multiple spaces should be preserved"

    def test_roundtrip_chinese(self, tokenizer):
        """Test roundtrip for Chinese text."""
        text = "你好世界"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text

    def test_roundtrip_emoji(self, tokenizer):
        """Test roundtrip for 'Hello 🌍 World!'."""
        text = "Hello 🌍 World!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text

    def test_roundtrip_multiline(self, tokenizer):
        """Test roundtrip for multiline text."""
        text = "Multi-line\ntext\nwith\nnewlines"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text

    def test_roundtrip_code(self, tokenizer):
        """Test roundtrip for code."""
        text = "def hello():\n    print('Hello')"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text


class TestMistralV3BackendOptions:
    """Test regex backend options (regexr, PCRE2, JIT).

    This coverage used to live on the V1/V2 test files, but V1/V2 route
    through `SpmTokenizer` (SentencePiece merges), which has no regex
    pre-tokenizer to configure — `.pcre2()`/`.jit()` refuse on it rather
    than report a switch that did not happen. V3/Tekken
    is the genuinely BPE-backed Mistral vocabulary (byte-level BPE, like
    cl100k/o200k), so it is the one that can actually exercise backend
    switching while still being a Mistral tokenizer.
    """

    def test_default_backend(self):
        """Test default backend (regexr with JIT)."""
        tokenizer = Tokenizer.from_pretrained("mistral_v3")
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        assert tokenizer.decode(tokens) == text

    @requires_pcre2
    def test_pcre2_backend(self):
        """Test switching to PCRE2 backend."""
        tokenizer = Tokenizer.from_pretrained("mistral_v3").pcre2(True)
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        assert tokenizer.decode(tokens) == text

    @requires_pcre2
    def test_pcre2_switch_back_to_regexr(self):
        """Test switching from PCRE2 back to regexr."""
        tokenizer = Tokenizer.from_pretrained("mistral_v3").pcre2(True).pcre2(False)
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        assert tokenizer.decode(tokens) == text

    def test_jit_disabled(self):
        """Test with JIT disabled."""
        tokenizer = Tokenizer.from_pretrained("mistral_v3").jit(False)
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        assert tokenizer.decode(tokens) == text

    def test_jit_enabled(self):
        """Test with JIT explicitly enabled."""
        tokenizer = Tokenizer.from_pretrained("mistral_v3").jit(True)
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        assert tokenizer.decode(tokens) == text

    @requires_pcre2
    def test_pcre2_with_jit_disabled(self):
        """Test PCRE2 backend with JIT disabled."""
        tokenizer = Tokenizer.from_pretrained("mistral_v3").pcre2(True).jit(False)
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        assert tokenizer.decode(tokens) == text

    @requires_pcre2
    def test_pcre2_with_jit_enabled(self):
        """Test PCRE2 backend with JIT enabled."""
        tokenizer = Tokenizer.from_pretrained("mistral_v3").pcre2(True).jit(True)
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        assert tokenizer.decode(tokens) == text

    @requires_pcre2
    def test_backend_consistency(self):
        """Test that different backends produce the same tokens."""
        text = "The quick brown fox jumps over the lazy dog. 你好世界 🦀"

        tok_default = Tokenizer.from_pretrained("mistral_v3")
        tok_pcre2 = Tokenizer.from_pretrained("mistral_v3").pcre2(True)
        tok_no_jit = Tokenizer.from_pretrained("mistral_v3").jit(False)

        tokens_default = tok_default.encode(text)
        tokens_pcre2 = tok_pcre2.encode(text)
        tokens_no_jit = tok_no_jit.encode(text)

        assert tokens_default == tokens_pcre2, "PCRE2 should produce same tokens"
        assert tokens_default == tokens_no_jit, "Non-JIT should produce same tokens"

    @requires_pcre2
    def test_backend_consistency_multibyte(self):
        """Test regexr and PCRE2 produce same results for multi-byte text."""
        texts = [
            "word—word",
            "I'm sorry you're hurting—breakups suck.",
            'He said, ‘Hello’ and she replied, “Goodbye”.',
            "Check credentials—API key—in headers.",
        ]
        tokenizer = Tokenizer.from_pretrained("mistral_v3")
        tokenizer_pcre2 = Tokenizer.from_pretrained("mistral_v3").pcre2(True)
        for text in texts:
            tokens_regexr = tokenizer.encode(text)
            tokens_pcre2 = tokenizer_pcre2.encode(text)
            assert tokens_regexr == tokens_pcre2, f"Backend mismatch for: {text!r}"
