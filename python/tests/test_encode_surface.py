"""The encoding surface is the same on every Python tokenizer class.

Splintr exposes five tokenizer classes. Before 0.11 they disagreed: `encode`
applied the boundary template on `AnyTokenizer` but not on `Tokenizer`, and a
method named `encode_with_special_tokens` (post-processor template) sat one word
away from `encode_with_special` (special-token *strings* found in the text)
meaning something entirely different. A caller reading either name guessed wrong
about half the time, and the guess was silent — the ids just differed.

So the surface is now uniform, and these tests pin that:

    encode(text)                            model-ready, boundary template applied
    encode_raw(text)                        content tokens only, no template
    encode_ordinary(text)                   never match a special token spelled in the text
    encode_with_special(text)               match every special token spelled in the text
    encode_allowed_special(text, allowed)   match only the listed ones
    encode_batch(texts)                     batch form of `encode`

`encode_with_special_tokens` is gone; its meaning is exactly `encode`.
"""

import base64
import json

import pytest
from splintr import (
    SentencePieceTokenizer,
    SpmTokenizer,
    Tokenizer,
    WordPieceTokenizer,
    from_json_bytes,
)

UNIFORM_METHODS = (
    "encode",
    "encode_raw",
    "encode_ordinary",
    "encode_with_special",
    "encode_allowed_special",
    "encode_batch",
)

#: The decode-side counterpart. `decode` is HuggingFace's default
#: `skip_special_tokens=True`; `decode_with_special` and
#: `streaming_decoder_with_special` are its `False`, and a class that offered
#: only the default would leave a caller with no way to see the markers at all.
#: Both must therefore exist on every class, exactly as the encode modes do.
UNIFORM_DECODE_METHODS = (
    "decode",
    "decode_with_special",
    "streaming_decoder",
    "streaming_decoder_with_special",
    "decode_token_bytes",
    "decode_token",
)

#: A BERT-style WordPiece vocabulary whose `post_processor` really does wrap the
#: content in `[CLS]`/`[SEP]` — the only way to tell `encode` and `encode_raw`
#: apart, since a tokenizer with no template makes them identical by definition.
WORDPIECE_WITH_TEMPLATE = {
    "model": {
        "type": "WordPiece",
        "vocab": {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "hello": 3, "world": 4},
        "unk_token": "[UNK]",
    },
    "added_tokens": [
        {"id": 1, "content": "[CLS]", "special": True},
        {"id": 2, "content": "[SEP]", "special": True},
    ],
    "post_processor": {
        "type": "TemplateProcessing",
        "single": [
            {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
        ],
        "pair": [
            {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
            {"Sequence": {"id": "B", "type_id": 1}},
            {"SpecialToken": {"id": "[SEP]", "type_id": 1}},
        ],
        "special_tokens": {
            "[CLS]": {"id": "[CLS]", "ids": [1], "tokens": ["[CLS]"]},
            "[SEP]": {"id": "[SEP]", "ids": [2], "tokens": ["[SEP]"]},
        },
    },
}


#: A minimal tiktoken-format vocabulary (base64 token, space, rank per line).
#:
#: `Tokenizer` is built from this rather than from `Tokenizer.from_pretrained`,
#: which now returns the loader's `AnyTokenizer` for every bundled vocabulary —
#: using it here would test that handle twice and leave the directly-constructed
#: `Tokenizer` class, the one this suite exists to keep in step, uncovered.
TINY_BPE_VOCAB = b"\n".join(
    base64.b64encode(token) + b" " + str(rank).encode()
    for rank, token in enumerate(
        [b"h", b"e", b"l", b"o", b"w", b"r", b"d", b" ", b"hello", b" world"]
    )
)


def every_tokenizer_class():
    """One live instance of each of the five tokenizer classes.

    Built through each class's own entry point rather than a single loader, so a
    method that exists only on the loader-produced handle does not pass for the
    directly-constructed ones.
    """
    return {
        "Tokenizer": Tokenizer.from_bytes(
            TINY_BPE_VOCAB, r"\S+|\s+", {"<|endoftext|>": 100}
        ),
        "AnyTokenizer": from_json_bytes(json.dumps(WORDPIECE_WITH_TEMPLATE).encode()),
        "SpmTokenizer": SpmTokenizer(
            ["<unk>", "<s>", "</s>", "▁hello", "▁world"], [], 1, 2
        ),
        "SentencePieceTokenizer": SentencePieceTokenizer(
            tokens=["<unk>", "<s>", "</s>", "▁Hello", "▁world"],
            scores=[0.0, 0.0, 0.0, -1.2, -1.5],
            eos_token_id=2,
            bos_token_id=1,
        ),
        "WordPieceTokenizer": WordPieceTokenizer(
            ["[UNK]", "[CLS]", "[SEP]", "hello", "world"], 0, 100, False
        ),
    }


@pytest.mark.parametrize("name", sorted(every_tokenizer_class()))
@pytest.mark.parametrize("method", UNIFORM_METHODS)
def test_every_class_exposes_the_uniform_method(name, method):
    tokenizer = every_tokenizer_class()[name]
    assert callable(getattr(tokenizer, method, None)), (
        f"{name} is missing {method}; the six encoding methods must mean the "
        "same thing on every tokenizer class"
    )


@pytest.mark.parametrize("name", sorted(every_tokenizer_class()))
@pytest.mark.parametrize("method", UNIFORM_DECODE_METHODS)
def test_every_class_exposes_the_uniform_decode_method(name, method):
    tokenizer = every_tokenizer_class()[name]
    assert callable(getattr(tokenizer, method, None)), (
        f"{name} is missing {method}; special-token handling on decode must be "
        "spellable on every tokenizer class, not only on some of them"
    )


@pytest.mark.parametrize("name", sorted(every_tokenizer_class()))
def test_decode_with_special_only_adds_what_decode_drops(name):
    """The two modes differ on exactly the declared-special ids, nowhere else.

    Asserted as a relationship rather than as five per-class literals, because
    the five fixtures are deliberately different shapes (`[CLS]`/`[SEP]` for the
    BERT-family ones, `<s>`/`</s>`/`<unk>` for the SentencePiece-shaped ones, no
    declared specials at all for the raw `Tokenizer`). Two clauses, and together
    they pin the whole semantics:

    * every id renders *something* under the explicit mode -- that is what makes
      it a usable `skip_special_tokens=False`, since an id that vanished in both
      modes would leave the caller no way to see it;
    * where `decode` renders an id at all, the explicit mode renders it
      identically -- so the mode only ever restores dropped markers and never
      quietly changes ordinary text.
    """
    tokenizer = every_tokenizer_class()[name]
    for token_id in range(5):
        default = tokenizer.decode([token_id])
        rendered = tokenizer.decode_with_special([token_id])
        assert rendered != "", (
            f"{name}: id {token_id} renders nothing even with specials on"
        )
        if default != "":
            assert rendered == default, (
                f"{name}: id {token_id} is not a declared special, so the two "
                "modes must agree on it"
            )


@pytest.mark.parametrize("name", sorted(every_tokenizer_class()))
def test_encode_with_special_tokens_is_gone(name):
    """The confusable name is deleted, not aliased.

    Leaving it as a shim would preserve exactly the trap the rename removes: two
    method names one word apart with unrelated meanings.
    """
    tokenizer = every_tokenizer_class()[name]
    assert not hasattr(tokenizer, "encode_with_special_tokens")


class TestTemplateVsContent:
    """`encode` is `encode_raw` plus the boundary template — nothing else."""

    @pytest.fixture
    def tokenizer(self):
        return from_json_bytes(json.dumps(WORDPIECE_WITH_TEMPLATE).encode())

    def test_encode_applies_the_template(self, tokenizer):
        assert tokenizer.encode("hello world") == [1, 3, 4, 2]

    def test_encode_raw_omits_the_template(self, tokenizer):
        assert tokenizer.encode_raw("hello world") == [3, 4]

    def test_encode_is_encode_raw_wrapped(self, tokenizer):
        raw = tokenizer.encode_raw("hello world")
        assert tokenizer.encode("hello world") == [1, *raw, 2]

    def test_batch_is_the_batch_form_of_encode(self, tokenizer):
        texts = ["hello world", "world hello"]
        assert tokenizer.encode_batch(texts) == [tokenizer.encode(t) for t in texts]

    def test_ordinary_keeps_the_boundary_template(self, tokenizer):
        """Refusing to match `[CLS]` *in the text* must not drop the model's own.

        The two are independent: boundary tokens come from the template, so a
        caller locking down special-token matching for safety would otherwise
        silently lose the wrapper the model was trained with.
        """
        ids = tokenizer.encode_ordinary("[CLS]hello")
        assert ids[0] == 1 and ids[-1] == 2
        # The `[CLS]` typed in the text became ordinary content, not id 1.
        assert ids[1:-1].count(1) == 0


class TestNoTemplateMeansIdentical:
    """A bundled vocabulary declares no template, so the two agree there.

    `from_pretrained` hands back the loader's `AnyTokenizer`; the point holds
    for it because a bundled vocabulary's policy carries an EOS id and its named
    specials but no boundary template, so `apply_single` is a passthrough.
    """

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_pretrained("cl100k_base")

    def test_encode_equals_encode_raw(self, tokenizer):
        text = "The quick brown fox"
        assert tokenizer.encode(text) == tokenizer.encode_raw(text)

    def test_encode_raw_is_the_reference_ids(self, tokenizer):
        assert tokenizer.encode_raw("Hello, world!") == [9906, 11, 1917, 0]

    def test_batch_is_the_batch_form_of_encode(self, tokenizer):
        texts = ["Hello, world!", "How are you?"]
        assert tokenizer.encode_batch(texts) == [tokenizer.encode(t) for t in texts]


#: Per-class fixture data for `decode_token`/`decode_token_bytes`: an id that
#: renders ordinary content, its expected text, an id that carries no surface
#: (a special every fixture's `decode` already drops — see
#: `test_decode_with_special_only_adds_what_decode_drops` above), and an id
#: nowhere in the vocabulary.
#:
#: `normal_text` is not always the token's own spelling: `SpmTokenizer` and
#: `SentencePieceTokenizer` render their `▁` word-start marker as a leading
#: space per id (that substitution is part of a single id's rendering, not a
#: sequence-level post-op), while the single-leading-space *strip* `decode`
#: applies is a sequence-level post-op `decode_token` never runs — which is
#: exactly what `test_decode_token_concatenation_is_not_decode` below pins.
DECODE_TOKEN_CASES = {
    "Tokenizer": (8, "hello", 100, 1_000_000),
    "AnyTokenizer": (3, "hello", 1, 99),
    "SpmTokenizer": (3, " hello", 1, 99),
    "SentencePieceTokenizer": (3, " Hello", 1, 99),
    "WordPieceTokenizer": (3, "hello", 1, 99),
}


@pytest.mark.parametrize("name", sorted(DECODE_TOKEN_CASES))
def test_decode_token_bytes_returns_bytes(name):
    """`decode_token_bytes` is `bytes`, not a list of ints."""
    tokenizer = every_tokenizer_class()[name]
    normal_id, normal_text, _empty_id, _oor_id = DECODE_TOKEN_CASES[name]
    result = tokenizer.decode_token_bytes(normal_id)
    assert isinstance(result, bytes)
    assert result == normal_text.encode()


@pytest.mark.parametrize("name", sorted(DECODE_TOKEN_CASES))
def test_decode_token_normal_id(name):
    """An ordinary content id renders its text through `decode_token` too."""
    tokenizer = every_tokenizer_class()[name]
    normal_id, normal_text, _empty_id, _oor_id = DECODE_TOKEN_CASES[name]
    assert tokenizer.decode_token(normal_id) == normal_text


#: The classes whose fixture actually has an id that renders to nothing — one
#: its `decode` drops. The plain `Tokenizer` fixture is excluded because it has
#: no such id: `from_bytes` declares `<|endoftext|>` as a special but nothing as
#: decode-skipped, and a special that is not skipped renders its own spelling
#: (`b"<|endoftext|>"`), which is the same answer `decode` gives. That mirrors
#: the bundled OpenAI vocabularies, whose skip set is empty by measurement —
#: `tiktoken` has no `skip_special_tokens` mode and renders every special.
EMPTY_RENDERING_IDS = {
    name: case[2] for name, case in DECODE_TOKEN_CASES.items() if name != "Tokenizer"
}


@pytest.mark.parametrize("name", sorted(EMPTY_RENDERING_IDS))
def test_decode_token_empty_id_is_not_an_error(name):
    """An id in the vocabulary with no surface renders empty, not a raise."""
    tokenizer = every_tokenizer_class()[name]
    empty_id = EMPTY_RENDERING_IDS[name]
    assert tokenizer.decode_token_bytes(empty_id) == b""
    assert tokenizer.decode_token(empty_id) == ""


def test_decode_token_renders_an_unskipped_special_verbatim():
    """The other side of the rule above, pinned rather than left implicit."""
    tokenizer = every_tokenizer_class()["Tokenizer"]
    assert tokenizer.decode_token_bytes(100) == b"<|endoftext|>"
    assert tokenizer.decode_token(100) == "<|endoftext|>"


@pytest.mark.parametrize("name", sorted(DECODE_TOKEN_CASES))
def test_decode_token_out_of_range_raises(name):
    """An id outside the vocabulary altogether raises on both methods."""
    tokenizer = every_tokenizer_class()[name]
    _normal_id, _normal_text, _empty_id, oor_id = DECODE_TOKEN_CASES[name]
    with pytest.raises(ValueError):
        tokenizer.decode_token_bytes(oor_id)
    with pytest.raises(ValueError):
        tokenizer.decode_token(oor_id)


#: Classes/vocabularies where concatenating `decode_token` over a real
#: sequence is *known* to differ from `decode` on the same ids: `SpmTokenizer`
#: and `SentencePieceTokenizer` strip a single leading space (the
#: `add_prefix_space` dummy prefix) only in `decode`'s sequence-level post-op,
#: and `WordPieceTokenizer`/`AnyTokenizer` (WordPiece here) insert the
#: word-separator space between tokens only when assembling the whole
#: sequence. The plain `Tokenizer` fixture is excluded on purpose: its
#: ByteLevel vocabulary bakes spaces directly into token surfaces (` world`
#: is its own token), so there is no separator or byte-fallback step for this
#: particular vocabulary to disagree over — asserting a difference there would
#: assert something that is not true of this fixture.
NON_REASSEMBLY_CASES = {
    "AnyTokenizer": [3, 4],
    "SpmTokenizer": [3, 4],
    "SentencePieceTokenizer": [3, 4],
    "WordPieceTokenizer": [3, 4],
}


@pytest.mark.parametrize("name", sorted(NON_REASSEMBLY_CASES))
def test_decode_token_concatenation_is_not_decode(name):
    """`decode_token` is deliberately not composable into `decode`'s output.

    `Tokenize::decode_token_bytes` documents this: no leading-space strip, no
    first-token rule, no word separator runs per id, so concatenating it over
    a sequence gives the pre-post-processing bytes `decode` starts from, not
    `decode`'s own output. `streaming_decoder` is the way to render a
    sequence; this test exists to make sure nobody reaches for `decode_token`
    instead and gets a silently wrong string.
    """
    tokenizer = every_tokenizer_class()[name]
    ids = NON_REASSEMBLY_CASES[name]
    reassembled = "".join(tokenizer.decode_token(i) for i in ids)
    assert reassembled != tokenizer.decode(ids), (
        f"{name}: decode_token concatenation unexpectedly matches decode; "
        "the non-reassembly property this test exists to pin does not hold "
        "for this fixture"
    )
