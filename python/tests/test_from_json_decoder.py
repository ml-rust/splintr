"""
Regression tests for the declared `decoder` pipeline surviving the FFI boundary.

A HuggingFace `tokenizer.json` may declare a `decoder` — an ordered chain such
as Mistral's `Replace ▁→" "` → `ByteFallback` → `Fuse` → `Strip` — and for those
files decoding *is* that chain: the backend's own decode renders raw pieces
(`▁hello▁world`, `<0x0A>`) instead of text.

`from_json` returns the universal `AnyTokenizer` handle, which carries four
things: the backend, the special-token policy, the declared `decoder`, and the
ids to skip on decode. An earlier binding unpacked that handle into a
family-specific Python wrapper and kept only the first two, so every
`from_json` tokenizer silently lost its decode pipeline the moment it crossed
into Python — Rust decoded correctly, Python did not. These tests fail loudly
if that ever happens again.

Expected values are ground truth from `tokenizers` 0.22.1 (HuggingFace's own
implementation), measured on the same inputs.
"""

import json
from pathlib import Path

import pytest
from splintr import from_json, from_json_bytes

# Mistral's own decoder chain, verbatim from its `tokenizer.json`.
MISTRAL_DECODER = {
    "type": "Sequence",
    "decoders": [
        {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
        {"type": "ByteFallback"},
        {"type": "Fuse"},
        {"type": "Strip", "content": " ", "start": 1, "stop": 0},
    ],
}

MISTRAL_TOKENIZER_JSON = Path(
    "/home/farhan/Projects/models/mistral-7b-v0.3/tokenizer.json"
)

requires_mistral_checkpoint = pytest.mark.skipif(
    not MISTRAL_TOKENIZER_JSON.is_file(),
    reason=f"reference checkpoint not present at {MISTRAL_TOKENIZER_JSON}",
)


class TestDeclaredDecoderPipeline:
    """A self-contained `tokenizer.json` whose decoding is its declared chain.

    Deliberately in-memory rather than reading a checkpoint: this is the
    regression test for the binding, so it must run everywhere, on every
    machine, with nothing downloaded.
    """

    @pytest.fixture
    def tokenizer(self):
        doc = {
            "added_tokens": [
                {"id": 0, "content": "<unk>", "special": True},
                {"id": 1, "content": "<s>", "special": True},
                {"id": 2, "content": "</s>", "special": True},
            ],
            "decoder": MISTRAL_DECODER,
            "model": {
                "type": "BPE",
                "unk_token": "<unk>",
                "byte_fallback": True,
                "vocab": {
                    "<unk>": 0,
                    "<s>": 1,
                    "</s>": 2,
                    "<0x0A>": 3,
                    "▁": 4,
                    "▁hello": 5,
                    "▁world": 6,
                },
                "merges": [],
            },
        }
        return from_json_bytes(json.dumps(doc).encode())

    def test_replace_fuse_and_strip_run(self, tokenizer):
        """`▁hello` + `▁world` must decode as text, not as raw pieces.

        `'▁hello▁world'` is the exact symptom of a dropped pipeline: `Replace`
        never converted `▁` to a space and `Strip` never removed the leading one.
        """
        assert tokenizer.decode([5, 6]) == "hello world"

    def test_special_tokens_are_skipped(self, tokenizer):
        """`special=true` ids drop out before the chain runs (HF's default
        `skip_special_tokens=True`), so BOS/EOS leave no trace in the text."""
        assert tokenizer.decode([1, 5, 6, 2]) == "hello world"

    def test_strip_removes_exactly_one_leading_space(self, tokenizer):
        """`Strip{start: 1}` removes one leading space, not every one — a bare
        `▁` before the content must survive as a real space."""
        assert tokenizer.decode([1, 4, 5, 6]) == " hello world"

    def test_byte_fallback_reassembles_hex_tokens(self, tokenizer):
        """`<0x0A>` is a byte-fallback token; `ByteFallback` turns it into the
        newline byte instead of leaving the literal `<0x0A>` spelling."""
        assert tokenizer.decode([5, 3, 6]) == "hello\n world"

    def test_handle_reports_its_backend_family(self, tokenizer):
        """The universal handle stays queryable rather than being flattened
        into a family-specific wrapper."""
        assert tokenizer.family == "BPE"


@requires_mistral_checkpoint
class TestMistralFromJsonDecode:
    """The real Mistral 7B v0.3 `tokenizer.json` — the file the bug was found on.

    Token ids and expected strings are from `tokenizers` 0.22.1:
    `hf.decode(hf.encode(text, add_special_tokens=False).ids)`.
    """

    @pytest.fixture
    def tokenizer(self):
        return from_json(str(MISTRAL_TOKENIZER_JSON))

    @pytest.mark.parametrize(
        "ids,expected",
        [
            ([7080, 29477, 2294], "hello world"),
            ([29113, 12078, 1151, 29565], "café résumé"),
            (
                [29473, 29910, 29887, 31089, 29761, 30378, 30521, 29877, 29891],
                "日本語のテキスト",
            ),
            ([1032, 1055], "a b"),
            ([29473, 1436, 3469, 29473, 1343, 1027], " spaced  out  "),
            (
                [1569, 1053, 29500, 29512, 2097, 781, 3055, 1372, 2086],
                "def f(x):\n    return x",
            ),
        ],
    )
    def test_decode_matches_huggingface(self, tokenizer, ids, expected):
        assert tokenizer.decode(ids) == expected

    def test_decode_skips_the_bos_the_template_added(self, tokenizer):
        """`encode` prepends `<s>` (id 1) via the file's post-processor; decoding
        that sequence must come back as the original text, not `<s>hello world`."""
        assert tokenizer.decode([1, 7080, 29477, 2294]) == "hello world"

    def test_policy_survives_alongside_the_decoder(self, tokenizer):
        """The handle carries the special-token policy too, so `encode` applies
        the file's `TemplateProcessing` single template (`<s>` + A) while
        `encode_raw` gives the bare content tokens.

        Asserted as a relationship rather than a literal id list: this is a
        claim about the policy travelling with the handle, not about the BPE
        backend's segmentation, which the vocabulary-specific suites cover.
        """
        raw = tokenizer.encode_raw("hello world")
        assert tokenizer.encode("hello world") == [1] + raw
        assert tokenizer.special_token_id("<s>") == 1
