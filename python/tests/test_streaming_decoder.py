"""
Regression tests for the one streaming decoder, on every tokenizer class.

Python used to expose two decoder classes — `StreamingDecoder` and
`ByteLevelStreamingDecoder` — each built from cloned vocabulary maps rather than
from the tokenizer's decode configuration. The caller picked, and picking wrong
produced mojibake silently. Neither honoured the `special=true` ids `decode`
drops, the `▁` metaspace substitution, or `<0xNN>` byte fallback, so
`"".join(stream)` disagreed with `decode` for every tokenizer configured with
any of those. And only the BPE class had a factory at all.

There is now one `StreamingDecoder`, obtainable only from a tokenizer's own
`streaming_decoder()`, on every class. These tests pin the property that makes
that worth doing:

    "".join(chunks) + flush() == decode(ids)

asserted against each backend's own `decode` rather than against a literal, so
they test the agreement rather than re-deriving one side of it.
"""

import base64
import json

import pytest
from splintr import (
    CL100K_BASE_PATTERN,
    SentencePieceTokenizer,
    SpmTokenizer,
    Tokenizer,
    WordPieceTokenizer,
    from_json_bytes,
)


def stream_one_by_one(tokenizer, ids):
    """Drive `ids` through the tokenizer's own decoder, one token at a time."""
    decoder = tokenizer.streaming_decoder()
    out = ""
    for token_id in ids:
        if chunk := decoder.add_token(token_id):
            out += chunk
    return out + decoder.flush()


def stream_in_one_call(tokenizer, ids):
    """The same ids through `add_tokens` — grouping must not change the text."""
    decoder = tokenizer.streaming_decoder()
    return (decoder.add_tokens(list(ids)) or "") + decoder.flush()


def assert_stream_matches_decode(tokenizer, ids):
    """The property, under both groupings."""
    expected = tokenizer.decode(list(ids))
    assert stream_one_by_one(tokenizer, ids) == expected
    assert stream_in_one_call(tokenizer, ids) == expected
    return expected


def tiktoken_vocab_bytes():
    """A tiktoken-format vocabulary: all 256 single bytes, plus a few words.

    Every byte is a token, so any text encodes; the multi-byte words exist so a
    character can also arrive whole rather than byte by byte.
    """
    lines = []
    for rank, b in enumerate(range(256)):
        lines.append(f"{base64.b64encode(bytes([b])).decode()} {rank}")
    for offset, word in enumerate([b"Hello", b" world", "世界".encode()]):
        lines.append(f"{base64.b64encode(word).decode()} {256 + offset}")
    return "\n".join(lines).encode()


AGREEMENT_TEXTS = [
    "Hello world",
    "Hello world 世界",
    "🎉 emoji and a combining é",
    "mixed 텍스트 with\ttabs\nand  spaces   ",
]


class TestBpeTokenizerClass:
    """`Tokenizer` — the raw (non-ByteLevel) BPE class, built from a vocabulary."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_bytes(
            tiktoken_vocab_bytes(),
            CL100K_BASE_PATTERN,
            {"<|endoftext|>": 259},
        )

    @pytest.mark.parametrize("text", AGREEMENT_TEXTS)
    def test_stream_matches_decode(self, tokenizer, text):
        ids = tokenizer.encode(text)
        assert assert_stream_matches_decode(tokenizer, ids) == text

    def test_stream_matches_decode_with_special_tokens(self, tokenizer):
        """A special token spelled out in the text streams as one unit."""
        ids = tokenizer.encode_with_special("<|endoftext|>Hello world")
        assert 259 in ids
        assert assert_stream_matches_decode(tokenizer, ids) == "<|endoftext|>Hello world"


class TestByteLevelVocabulary:
    """DeepSeek V3 — a ByteLevel BPE vocabulary.

    This is the case the deleted second class existed for: its tokens spell raw
    bytes as printable characters (`Ġ` for space), so a decoder that skipped the
    unmapping produced mojibake. The rule now comes from the tokenizer, so the
    single factory serves it.
    """

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("deepseek_v3")

    @pytest.mark.parametrize("text", AGREEMENT_TEXTS)
    def test_stream_matches_decode(self, tokenizer, text):
        assert assert_stream_matches_decode(tokenizer, tokenizer.encode(text)) == text

    def test_stream_matches_decode_with_special_tokens(self, tokenizer):
        text = "<｜User｜>你好!<|think|>reasoning<|/think|>"
        ids = tokenizer.encode_with_special(text)
        assert assert_stream_matches_decode(tokenizer, ids) == text


class TestRawVocabulary:
    """cl100k_base — a raw vocabulary, for contrast with the ByteLevel one."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("cl100k_base")

    @pytest.mark.parametrize("text", AGREEMENT_TEXTS)
    def test_stream_matches_decode(self, tokenizer, text):
        assert assert_stream_matches_decode(tokenizer, tokenizer.encode(text)) == text


class TestSentencePieceTokenizerClass:
    """`SentencePieceTokenizer` (Unigram) — a metaspace vocabulary.

    `decode` turns `▁` into a space and drops the dummy prefix. The old Python
    decoder did neither, so it emitted `▁Hello▁world` where `decode` said
    `Hello world`.
    """

    @pytest.fixture
    def tokenizer(self):
        return SentencePieceTokenizer(
            tokens=["<unk>", "<s>", "</s>", "▁Hello", "▁world", "▁", "!"],
            scores=[0.0, 0.0, 0.0, -1.0, -1.5, -3.0, -2.0],
            eos_token_id=2,
        )

    def test_stream_matches_decode(self, tokenizer):
        ids = tokenizer.encode("Hello world")
        assert "Hello world" in assert_stream_matches_decode(tokenizer, ids)

    def test_metaspace_substitution_reaches_the_stream(self, tokenizer):
        """The substitution is the thing that used to be lost: no `▁` survives
        into the streamed text, exactly as none survives `decode`."""
        streamed = assert_stream_matches_decode(tokenizer, [3, 4, 6])
        assert "▁" not in streamed
        assert "Hello world!" in streamed


class TestSpmTokenizerClass:
    """`SpmTokenizer` (SentencePiece BPE) — metaspace plus byte fallback."""

    @pytest.fixture
    def tokenizer(self):
        tokens = ["<unk>", "<s>", "</s>"]
        tokens += [f"<0x{b:02X}>" for b in range(256)]
        tokens += ["▁Hello", "▁world"]
        return SpmTokenizer(tokens=tokens, scores=[], bos_token_id=1, eos_token_id=2)

    def test_stream_matches_decode(self, tokenizer):
        # The `▁Hello`/`▁world` pieces themselves: `scores=[]` means merge order
        # is id order, so `encode` never forms them — it spells the text out in
        # byte-fallback ids instead (asserted separately below).
        assert assert_stream_matches_decode(tokenizer, [259, 260]) == "Hello world"

    def test_byte_fallback_metaspace_stays_literal(self, tokenizer):
        """A `▁` spelled through byte-fallback ids is NOT substituted for a space.

        Ground truth is the `sentencepiece` package 0.2.0, which decodes the ids
        for `<0xE2> <0x96> <0x81>` to the literal `▁` — the substitution applies
        to piece surfaces, never to bytes byte-fallback produced. HuggingFace's
        own declared chain agrees: `Replace(▁→" ")` runs before `ByteFallback`.
        """
        ids = tokenizer.encode("Hello world")
        assert assert_stream_matches_decode(tokenizer, ids) == "▁Hello▁world"

    def test_byte_fallback_char_reassembles_across_add_token_calls(self, tokenizer):
        """`𐍈` (U+10348) is in no piece, so it encodes as four `<0xNN>` tokens.

        The old Python decoder could not resolve those at all — it buffered the
        literal text `<0xF0><0x90><0x8D><0x88>`. The bytes must now reassemble
        across `add_token` calls, with nothing emitted until the character is
        complete.
        """
        ids = tokenizer.encode("Hello 𐍈")
        expected = assert_stream_matches_decode(tokenizer, ids)
        assert "𐍈" in expected
        # The literal `<0xNN>` spelling is what an unresolved fallback looks like.
        assert "<0x" not in expected


class TestWordPieceTokenizerClass:
    """`WordPieceTokenizer` — `##` continuations and a `[CLS]`/`[SEP]` skip set."""

    @pytest.fixture
    def tokenizer(self):
        return WordPieceTokenizer(
            vocab=["[UNK]", "[CLS]", "[SEP]", "hello", "world", "test", "##ing"],
            unk_token_id=0,
        )

    def test_stream_matches_decode(self, tokenizer):
        ids = tokenizer.encode("hello world testing")
        assert_stream_matches_decode(tokenizer, ids)

    def test_stream_matches_decode_with_skipped_specials(self, tokenizer):
        """`[CLS]`/`[SEP]` are dropped by `decode`; the stream must drop them too."""
        assert_stream_matches_decode(tokenizer, [1, 3, 4, 2])


class TestSpecialTokensDecodeSkips:
    """A `tokenizer.json` whose added tokens are `special: true`.

    HuggingFace's default `skip_special_tokens=True` means `decode` drops them.
    The old Python decoder emitted their surfaces instead, so `"".join(stream)`
    contained `<s>`/`</s>` that `decode` did not.
    """

    @pytest.fixture
    def tokenizer(self):
        doc = {
            "added_tokens": [
                {"id": 0, "content": "<unk>", "special": True},
                {"id": 1, "content": "<s>", "special": True},
                {"id": 2, "content": "</s>", "special": True},
            ],
            "model": {
                "type": "BPE",
                "unk_token": "<unk>",
                "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2, "hello": 3, "world": 4},
                "merges": [],
            },
        }
        return from_json_bytes(json.dumps(doc).encode())

    def test_special_ids_are_skipped_by_the_stream_as_by_decode(self, tokenizer):
        assert assert_stream_matches_decode(tokenizer, [1, 3, 4, 2]) == "helloworld"


class TestDeclaredDecoderPipelineStreams:
    """Mistral's declared chain, driven one token at a time.

    `Replace ▁→" "` → `ByteFallback` → `Fuse` → `Strip` is incrementally
    computable, so `AnyTokenizer.streaming_decoder()` answers with it rather
    than with the backend's own decode — which would render the raw pieces
    (`▁hello▁world`) the pipeline exists to turn into text.
    """

    @pytest.fixture
    def tokenizer(self):
        doc = {
            "added_tokens": [
                {"id": 0, "content": "<unk>", "special": True},
                {"id": 1, "content": "<s>", "special": True},
                {"id": 2, "content": "</s>", "special": True},
            ],
            "decoder": {
                "type": "Sequence",
                "decoders": [
                    {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                    {"type": "ByteFallback"},
                    {"type": "Fuse"},
                    {"type": "Strip", "content": " ", "start": 1, "stop": 0},
                ],
            },
            "model": {
                "type": "BPE",
                "unk_token": "<unk>",
                "byte_fallback": True,
                "vocab": {
                    "<unk>": 0,
                    "<s>": 1,
                    "</s>": 2,
                    "▁hello": 3,
                    "▁world": 4,
                    "<0xF0>": 5,
                    "<0x90>": 6,
                    "<0x8D>": 7,
                    "<0x88>": 8,
                },
                "merges": [],
            },
        }
        return from_json_bytes(json.dumps(doc).encode())

    def test_stream_matches_decode(self, tokenizer):
        assert assert_stream_matches_decode(tokenizer, [3, 4]) == "hello world"

    def test_stream_skips_the_special_ids_decode_skips(self, tokenizer):
        assert assert_stream_matches_decode(tokenizer, [1, 3, 4, 2]) == "hello world"

    def test_byte_fallback_char_reassembles_across_add_token_calls(self, tokenizer):
        """`𐍈` arrives as four `<0xNN>` tokens; nothing is emitted until it is
        whole, and the result matches `decode`."""
        expected = assert_stream_matches_decode(tokenizer, [3, 5, 6, 7, 8, 4])
        assert expected == "hello𐍈 world"

        decoder = tokenizer.streaming_decoder()
        assert decoder.add_token(3) == "hello"
        for token_id in (5, 6, 7):
            assert decoder.add_token(token_id) is None
            assert decoder.has_pending
        # A declared `ByteFallback` decodes a *run* of byte tokens as a unit —
        # HuggingFace emits one U+FFFD per byte when the whole run is invalid,
        # so the run cannot be closed until a non-byte token arrives or the
        # stream is flushed. The character therefore lands on `flush`.
        assert decoder.add_token(8) is None
        assert decoder.has_pending
        assert decoder.flush() == "𐍈"
        assert not decoder.has_pending


class TestUnstreamableDeclaredDecoderRefuses:
    """A declared pipeline that cannot be evaluated one chunk at a time.

    `BPEDecoder` branches on the *last* token, which a stream cannot know.
    Answering with the backend's own decode would silently render something
    other than what `decode` renders, so the factory raises instead.
    """

    @pytest.fixture
    def tokenizer(self):
        doc = {
            "decoder": {"type": "BPEDecoder", "suffix": "</w>"},
            "model": {
                "type": "BPE",
                "vocab": {"hello</w>": 0, "world</w>": 1},
                "merges": [],
            },
        }
        return from_json_bytes(json.dumps(doc).encode())

    def test_streaming_decoder_raises_instead_of_returning_a_wrong_decoder(
        self, tokenizer
    ):
        with pytest.raises(ValueError) as excinfo:
            tokenizer.streaming_decoder()
        # The message names the step that was refused, so a caller can act on it.
        assert "BPEDecoder" in str(excinfo.value)

    def test_whole_sequence_decode_still_works(self, tokenizer):
        """The refusal is precise: only streaming is unavailable.

        `BPEDecoder` turns the word suffix into a space on every token but the
        last, which is exactly why it needs the whole sequence.
        """
        assert tokenizer.decode([0, 1]) == "hello world"


class TestDecoderSurface:
    """The Python-visible surface of the surviving class."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("cl100k_base")

    def test_repr_names_the_class_and_the_buffer(self, tokenizer):
        decoder = tokenizer.streaming_decoder()
        assert "StreamingDecoder" in repr(decoder)
        assert "pending_bytes" in repr(decoder)

    def test_reset_clears_the_buffer(self, tokenizer):
        decoder = tokenizer.streaming_decoder()
        lead = {tokenizer.decode_bytes([tid]): tid for tid in range(256)}[b"\xe4"]

        assert decoder.add_token(lead) is None
        assert decoder.has_pending
        assert decoder.pending_bytes == 1

        decoder.reset()
        assert not decoder.has_pending
        assert decoder.pending_bytes == 0
        assert decoder.flush() == ""

    def test_an_unknown_id_is_skipped_rather_than_raised(self, tokenizer):
        """`add_token` is the lenient form, matching `decode_lossy`.

        A stream is fed by a running model; one stray id must not abort the
        generation. Callers who want the strict reading decode the whole
        sequence.
        """
        decoder = tokenizer.streaming_decoder()
        assert decoder.add_token(9_999_999) is None
        assert not decoder.has_pending
        assert decoder.add_token(tokenizer.encode("A")[0]) == "A"

    def test_the_class_cannot_be_constructed_directly(self):
        """There is no way to pair a decoder with a vocabulary it did not come
        from — the mistake the deleted second class made expressible."""
        from splintr import StreamingDecoder

        with pytest.raises(TypeError):
            StreamingDecoder()
