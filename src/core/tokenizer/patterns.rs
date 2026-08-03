/// Default regex pattern for cl100k_base (GPT-4, GPT-3.5-turbo).
///
/// Transcribed verbatim from tiktoken's cl100k pattern, with possessive
/// quantifiers (`?+`/`++`/`*+`) lowered to greedy — proven split-equivalent over
/// 40k+ random strings, and greedy compiles on the regexr backend (which has no
/// possessive support). Note the `\s+$` end-anchored branch and trailing bare
/// `\s`, which an earlier hand-simplified approximation omitted.
pub const CL100K_BASE_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s+$|\s*[\r\n]|\s+(?!\S)|\s";

/// Default regex pattern for o200k_base (GPT-4o). Transcribed from tiktoken's
/// o200k pattern (already greedy upstream).
pub const O200K_BASE_PATTERN: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Pre-tokenizer pattern for Llama 3/3.1/3.2/3.3.
///
/// Transcribed verbatim from the `Split` pre-tokenizer's `Regex` in Meta's
/// `llama-3.2-1b/tokenizer.json` — the pattern the model was actually trained
/// with. llama.cpp records the same string byte-for-byte as the
/// "original regex from tokenizer.json" for `LLAMA_VOCAB_PRE_TYPE_LLAMA3`
/// (`llama-vocab.cpp:286`); the expression it feeds its own engine
/// (`llama-vocab.cpp:289`) differs only by expanding `(?i:'s|'t|…)` into
/// `(?:'[sS]|'[tT]|…)`, which is the same language.
///
/// This is NOT [`O200K_BASE_PATTERN`] and must never be re-aliased to it.
/// o200k's two leading `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}…]+` branches
/// split letter runs on upper/lower case boundaries — an OpenAI convention
/// Llama 3 does not share, since its pre-tokenizer takes whole letter runs with
/// a plain `\p{L}+`. Aliasing the two breaks every camelCase merge: with the
/// o200k split `XMLHttpRequest` encodes as `[10833, 2977, 1939]` instead of the
/// correct `[10833, 27459]`.
///
/// Identical to [`QWEN2_PATTERN`] apart from `\p{N}{1,3}` (digit runs of up to
/// three) versus Qwen's single-digit `\p{N}`, so the two are not
/// interchangeable either.
pub const LLAMA3_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Regex pattern for SentencePiece-based tokenizers (Mistral V1/V2, Llama 2, Gemma).
///
/// SentencePiece tokenizers use a simple word boundary approach:
/// - `[^\s]+` = Match one or more non-whitespace characters (words)
/// - `|\s+` = OR match one or more whitespace characters
///
/// This differs from GPT-style tokenizers which use complex patterns for contractions,
/// unicode categories, and punctuation handling. SentencePiece relies on the BPE
/// vocabulary itself to handle these cases during encoding.
pub const SENTENCEPIECE_PATTERN: &str = r"[^\s]+|\s+";

/// Regex pattern for Mistral V3/Tekken tokenizer.
///
/// This pattern is specifically from the Mistral NeMo tokenizer and differs from O200K:
/// - No English contraction handling (`'s`, `'t`, `'re`, etc.)
/// - Single digit numbers `\p{N}` instead of `\p{N}{1,3}`
/// - Otherwise similar Unicode category handling
pub const MISTRAL_V3_PATTERN: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// GPT-2 style pre-tokenizer pattern used by Whisper.
///
/// Whisper's `tokenizer.json` declares a `ByteLevel` pre-tokenizer which applies
/// this regex to split text before BPE merging.
pub const GPT2_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// Pre-tokenizer pattern for Qwen2 / Qwen3 (llama.cpp's `qwen2` pre-tokenizer).
///
/// Identical to [`LLAMA3_PATTERN`] except that digits split one at a time
/// (`\p{N}`) rather than in runs of up to three (`\p{N}{1,3}`). That single
/// difference changes the resulting tokens, so the two are not interchangeable
/// and must stay separate constants.
pub const QWEN2_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Pre-tokenizer expression list for DeepSeek V3/R1.
///
/// Transcribed verbatim from the three `Split` pre-tokenizers in DeepSeek's own
/// `tokenizer.json`, which llama.cpp records byte-for-byte as
/// `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM` (`llama-vocab.cpp:318-325`). The passes
/// run in order, each subdividing the pieces the previous one produced — see
/// [`Tokenizer::new_byte_level_chain`](super::Tokenizer::new_byte_level_chain).
/// They are **not** collapsible into one
/// alternation: pass 1 cuts digit runs into groups of three *before* pass 3's
/// letter/punctuation split ever sees them, so a single regex would have to
/// resolve both at once and would pick different boundaries.
///
/// This is NOT [`O200K_BASE_PATTERN`], which this vocabulary was previously and
/// wrongly loaded with. o200k splits letter runs on case boundaries and has no
/// dedicated CJK/kana branch at all; DeepSeek isolates CJK and kana runs in
/// pass 2 and takes whole letter runs via `[\p{L}\p{M}]+`.
pub const DEEPSEEK_V3_PATTERNS: &[&str] = &[
    r"\p{N}{1,3}",
    // U+4E00-U+9FA5 CJK, U+3040-U+309F hiragana, U+30A0-U+30FF katakana.
    // Written as escapes so no editor or transcription step can silently
    // renormalize the literal characters into a different codepoint.
    r"[\u{4E00}-\u{9FA5}\u{3040}-\u{309F}\u{30A0}-\u{30FF}]+",
    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
];
