"""The fixed test corpus shared by every reference-tokenizer fixture extractor.

`scripts/extract_gguf_vocab.py` (`--reference-spm`/`--reference-hf`, for
GGUF-loaded vocabularies) and `scripts/extract_reference_cases.py` (for
splintr's bundled pretrained vocabularies) both need the exact same corpus:
whichever tokenizer end is under test, the ground truth is "run this fixed
set of strings through the real reference tokenizer and record its ids".
Splitting the literal out here means the two scripts can never drift apart
on what "the reference corpus" means.

Committed so every run of either script against the same reference model
reproduces the same fixture. Picked to hit what actually breaks
SentencePiece- and BPE-style tokenizers, not to dodge awkward cases:
whitespace edge cases (empty, runs, leading/trailing, tabs, newlines),
ordinary words, punctuation, contractions, CJK, emoji including a ZWJ
sequence, accented Latin, digits/digit runs, a code snippet, and a
mixed-script line.
"""

from __future__ import annotations

REFERENCE_CORPUS: list[str] = [
    "",
    " ",
    "  ",
    "   ",
    "a",
    " a",
    "a ",
    " a ",
    "hello world",
    "Hello World!",
    "the quick brown fox jumps over the lazy dog",
    "\tindented\twith\ttabs",
    "line one\nline two\nline three",
    "\n\n\n",
    "trailing whitespace   ",
    "   leading whitespace",
    "multiple   internal    spaces",
    "I've got it, don't worry.",
    "it's a test — isn't it?",
    "punctuation: ,.!?;:'\"()[]{}",
    "hyphenated-word and em—dash",
    "こんにちは世界",
    "日本語のテキストです。",
    "你好，世界！",
    "안녕하세요 세계",
    "emoji test 😀🎉🚀",
    "family: 👨‍👩‍👧‍👦",
    "flag: 🏳️‍🌈",
    "café résumé naïve",
    "Zürich Ångström Curaçao",
    "0123456789",
    "the year 2024 had 365 days",
    "price: $19.99, quantity: 42",
    "def add(a, b):\n    return a + b\n",
    "if (x == 42) { print(\"hi\"); }",
    "Mixed 混合 текст with 日本語 and English.",
    "русский текст с числами 123",
    "a\tb\nc d",
    "   \t\n   ",
    "The quick fox",  # U+00A0 non-breaking spaces
    # ------------------------------------------------------------------------
    # Case/script-discriminating block, added after the DeepSeek V3 vocabulary
    # was found loaded with the wrong pre-tokenizer pattern (o200k's, not its
    # own three-pass split) while every case above still passed. None of the
    # entries above exercise a letter-run case boundary, a punctuation-then-
    # letters split, a long unbroken digit run, or an unbroken kana/CJK run --
    # exactly the branches where BPE pre-tokenizer patterns diverge from one
    # another. Each entry below is a distinct phenomenon, not a variation on
    # an existing one. Append only -- do not reorder or edit entries above,
    # fixtures generated from this corpus are positional.
    # ------------------------------------------------------------------------
    # camelCase / PascalCase identifiers (letter-run case-boundary splitting)
    "getUserName",
    "myVariableName",
    "HTTPRequestHandler",
    "XMLHttpRequest",
    "isFooBarBaz123",
    "aB1cD2eF3",
    "camelCaseVariable_with_snake",
    "toString",
    # punctuation immediately followed by a letter run, no whitespace
    ".isValidEmail",
    "config.getValue()",
    "self.assertEqual(x, 42)",
    "user_id.toString()",
    "(parenthesizedWord)",
    "#hashtagWord",
    "@mentionUser",
    # long/irregularly-grouped digit runs
    "12345678901234567890",
    "9999999999",
    "3.14159265358979",
    "phone: 1234567890123",
    "id_000111222333444",
    "binary 1010101010101010",
    # unbroken CJK ideograph / kana runs
    "北京市海淀区",
    "こんにちはカタカナ",
    "東京タワーは高い",
    "한국어와中文混合",
    "ひらがなカタカナ漢字",
    "台北101大樓",
    # mixed-script text with no whitespace at the script boundary
    "Mixed混合Text文字",
    "Löwe löwe Léopard 狮子",
    "user123名前test",
    "GitHub仓库clone",
    "日本語123English",
    "emoji😀letterRun日本語",
    "PascalCase北京市camelCase",
]
