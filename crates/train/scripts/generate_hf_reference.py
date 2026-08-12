#!/usr/bin/env python3
"""Regenerate the HuggingFace reference vocabulary the parity test checks against.

    pip install tokenizers
    python crates/train/scripts/generate_hf_reference.py

Writes `crates/train/tests/fixtures/hf_bpe_reference.json` from
`crates/train/tests/fixtures/parity_corpus.txt`.

The fixture is committed so the test needs no Python and no network. Run this
only to refresh it against a newer `tokenizers`, and treat a resulting diff as a
finding rather than something to paper over: it means either they changed their
merge selection or we changed ours.

The configuration below is chosen so that any difference is the *merge loop*
rather than the setup around it:

* `WhitespaceSplit` is the one pre-tokenization both sides reproduce exactly,
  and the corpus is free of punctuation so there is nothing for the two
  punctuation policies to disagree about.
* `min_frequency=1` and no special tokens remove every other knob.
* The corresponding splintr-train configuration is `PreTok::Whitespace` with
  `Seeding::Chars` — character seeding because HuggingFace seeds from the
  characters the corpus contains, where byte seeding would add all 256 bytes.
"""

import json
import pathlib
import sys

try:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
except ImportError:
    sys.exit("needs `pip install tokenizers`")

VOCAB_SIZE = 500

fixtures = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures"
corpus = fixtures / "parity_corpus.txt"
output = fixtures / "hf_bpe_reference.json"

lines = corpus.read_text().splitlines()

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
tokenizer.train_from_iterator(
    lines,
    trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=1,
        special_tokens=[],
        show_progress=False,
        initial_alphabet=[],
    ),
)

model = json.loads(tokenizer.to_str())["model"]
output.write_text(
    json.dumps({"vocab": model["vocab"], "merges": model["merges"]}, indent=4, sort_keys=True)
    + "\n"
)
print(f"wrote {output}: {len(model['vocab'])} pieces, {len(model['merges'])} merges")
