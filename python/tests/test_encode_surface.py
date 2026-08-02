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
