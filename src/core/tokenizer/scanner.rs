//! Direct scanners for the bundled pre-tokenizer patterns.
//!
//! These patterns are fixed, known alternations, so running them on a general
//! regex engine pays for machinery they never need: the engine tries branches in
//! order at every position, and each trial is a call. Profiling single-text
//! encode put ~55% of the time inside the tagged-NFA interpreter for exactly
//! that reason. Here the branch is chosen from the first byte through a
//! 256-entry class table, and runs are scanned eight bytes at a time.
//!
//! Every scanner must agree with its expression byte-for-byte — same language,
//! different recogniser — so the tests below diff each one against the compiled
//! regex rather than against hand-written expectations. The regex is the
//! definition of correctness: `tests/fixtures/pretrained/` pins *it* against the
//! reference tokenizers.

use unicode_general_category::{get_general_category, GeneralCategory};

/// First-byte class, used only to pick a branch. ASCII only; every byte with the
/// high bit set is `Lead` and resolved by decoding the character.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum Class {
    /// Not whitespace, letter or digit.
    Punct = 0,
    Letter = 1,
    Digit = 2,
    /// Whitespace other than `\r`/`\n`.
    Space = 3,
    /// `\r` or `\n`, which several branches treat specially.
    Newline = 4,
    /// Start of a multi-byte character.
    Lead = 5,
}

const CLASS: [Class; 256] = {
    let mut table = [Class::Punct; 256];
    let mut b = 0usize;
    while b < 256 {
        table[b] = if b >= 0x80 {
            Class::Lead
        } else if (b >= b'a' as usize && b <= b'z' as usize)
            || (b >= b'A' as usize && b <= b'Z' as usize)
        {
            Class::Letter
        } else if b >= b'0' as usize && b <= b'9' as usize {
            Class::Digit
        } else if b == b'\r' as usize || b == b'\n' as usize {
            Class::Newline
        } else if b == b' ' as usize || b == b'\t' as usize || b == 0x0b || b == 0x0c {
            Class::Space
        } else {
            Class::Punct
        };
        b += 1;
    }
    table
};

// --- SWAR ------------------------------------------------------------------
//
// Eight bytes are loaded as a `u64` and tested together with ordinary
// arithmetic, which needs no target-specific intrinsics and so behaves the same
// on x86-64 and aarch64. Every helper below assumes the word holds only ASCII;
// callers check that first, because the range tricks rely on the high bit of
// each byte being clear before the subtraction.

const ONES: u64 = 0x0101_0101_0101_0101;
const HIGH: u64 = 0x8080_8080_8080_8080;

/// High bit set in each byte lane whose value is `< n`.
///
/// Each lane's high bit is set first, as a guard, so the subtraction borrows
/// into that bit and stops there instead of propagating into the next lane. A
/// lane that ends with its guard cleared is one that borrowed, i.e. one below
/// `n`.
///
/// The guard is what makes this per-lane. The bare
/// `(x - n) & !x & HIGH` form — Bit Twiddling Hacks' `hasless` — answers only
/// "is *any* lane below n", because a borrow out of a low lane corrupts the
/// lanes above it. Using it to find *which* lane made `scan_letters` run past
/// the end of a letter run and take the following character with it, splitting
/// `hello[]!?` as `hello[` + `]!?`.
///
/// Valid for `n <= 128` and lanes `< 128`: the guarded lane is then at least
/// `0x80` and `n` at most `0x80`, so a lane can never underflow.
#[inline(always)]
fn lanes_below(x: u64, n: u8) -> u64 {
    let guarded = x | HIGH;
    !guarded.wrapping_sub(ONES.wrapping_mul(n as u64)) & HIGH
}

/// High bit set in each byte lane holding an ASCII letter.
#[inline(always)]
fn ascii_letter_lanes(word: u64) -> u64 {
    // Case-folding first means one range test rather than two.
    let lowered = word | 0x2020_2020_2020_2020;
    let below_a = lanes_below(lowered, b'a');
    let at_most_z = lanes_below(lowered, b'z' + 1);
    at_most_z & !below_a & HIGH
}

/// Reads eight bytes at `pos` as a little-endian word, or `None` near the end.
#[inline(always)]
fn word_at(bytes: &[u8], pos: usize) -> Option<u64> {
    bytes
        .get(pos..pos + 8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().expect("slice of exactly eight bytes")))
}

/// Advances over ASCII letters eight at a time, stopping at the first byte that
/// is not one (including any non-ASCII lead byte, left to the caller).
#[inline]
fn swar_skip_ascii_letters(bytes: &[u8], mut pos: usize) -> usize {
    // Decline in one byte. Without this the whole probe — an eight-byte load, a
    // high-bit test and the scalar tail below — runs before concluding there is
    // nothing to skip, and a script written entirely in non-ASCII letters
    // (Greek, Cyrillic, Thai) pays it once per character. Equivalent to what
    // follows: neither loop can advance from a byte that is not an ASCII letter.
    if bytes
        .get(pos)
        .is_none_or(|&b| (b | 0x20).wrapping_sub(b'a') >= 26)
    {
        return pos;
    }
    while let Some(word) = word_at(bytes, pos) {
        // A non-ASCII byte anywhere in the word invalidates the range trick, so
        // hand the whole word back to the scalar loop.
        if word & HIGH != 0 {
            break;
        }
        let not_letters = !ascii_letter_lanes(word) & HIGH;
        if not_letters != 0 {
            // Lane index of the first non-letter; byte 0 is the low lane.
            return pos + (not_letters.trailing_zeros() / 8) as usize;
        }
        pos += 8;
    }
    while pos < bytes.len() && (bytes[pos] | 0x20).wrapping_sub(b'a') < 26 {
        pos += 1;
    }
    pos
}

/// Advances over ASCII uppercase letters eight at a time.
///
/// The case-split branches (o200k, gpt-oss, Kimi) scan an uppercase run and
/// then a lowercase run, and did so one character per indirect call through
/// [`RunFn`] — a call per byte, where cl100k's single `\p{L}+` run uses
/// [`swar_skip_ascii_letters`] and covers eight. That per-call overhead, not
/// the class tests, is why those vocabularies measured ~30% below cl100k.
#[inline]
fn swar_skip_ascii_upper(bytes: &[u8], mut pos: usize) -> usize {
    while let Some(word) = word_at(bytes, pos) {
        if word & HIGH != 0 {
            break;
        }
        let in_run = lanes_below(word, b'Z' + 1) & !lanes_below(word, b'A') & HIGH;
        let out = !in_run & HIGH;
        if out != 0 {
            return pos + (out.trailing_zeros() / 8) as usize;
        }
        pos += 8;
    }
    while pos < bytes.len() && bytes[pos].is_ascii_uppercase() {
        pos += 1;
    }
    pos
}

/// Advances over ASCII lowercase letters eight at a time. The lowercase half of
/// [`swar_skip_ascii_upper`].
#[inline]
fn swar_skip_ascii_lower(bytes: &[u8], mut pos: usize) -> usize {
    while let Some(word) = word_at(bytes, pos) {
        if word & HIGH != 0 {
            break;
        }
        let in_run = lanes_below(word, b'z' + 1) & !lanes_below(word, b'a') & HIGH;
        let out = !in_run & HIGH;
        if out != 0 {
            return pos + (out.trailing_zeros() / 8) as usize;
        }
        pos += 8;
    }
    while pos < bytes.len() && bytes[pos].is_ascii_lowercase() {
        pos += 1;
    }
    pos
}

/// Advances to the first byte that is not plain ASCII.
///
/// The whole point of DeepSeek's CJK/kana pass: every character it can match
/// lives in U+3040..U+30FF or U+4E00..U+9FA5, so no ASCII byte can ever start a
/// match and prose in a Latin script is skipped eight bytes at a time.
#[inline]
fn swar_skip_ascii(bytes: &[u8], mut pos: usize) -> usize {
    while let Some(word) = word_at(bytes, pos) {
        let high = word & HIGH;
        if high != 0 {
            return pos + (high.trailing_zeros() / 8) as usize;
        }
        pos += 8;
    }
    while pos < bytes.len() && bytes[pos] < 0x80 {
        pos += 1;
    }
    pos
}

/// Advances to the first byte that is an ASCII digit or starts a multi-byte
/// character — the two ways `\p{N}` can begin.
#[inline]
fn swar_skip_to_number(bytes: &[u8], mut pos: usize) -> usize {
    while let Some(word) = word_at(bytes, pos) {
        // Any non-ASCII byte could begin a non-ASCII `\p{N}`, and it also
        // invalidates the range trick, so stop the word there either way.
        if word & HIGH != 0 {
            break;
        }
        let digits = lanes_below(word, b'9' + 1) & !lanes_below(word, b'0') & HIGH;
        if digits != 0 {
            return pos + (digits.trailing_zeros() / 8) as usize;
        }
        pos += 8;
    }
    while pos < bytes.len() && bytes[pos] < 0x80 && !bytes[pos].is_ascii_digit() {
        pos += 1;
    }
    pos
}

// --- character predicates ---------------------------------------------------

/// `\p{L}`.
#[inline]
fn is_letter_char(c: char) -> bool {
    matches!(
        get_general_category(c),
        GeneralCategory::UppercaseLetter
            | GeneralCategory::LowercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
    )
}

/// `\p{N}`.
#[inline]
fn is_number_char(c: char) -> bool {
    matches!(
        get_general_category(c),
        GeneralCategory::DecimalNumber
            | GeneralCategory::LetterNumber
            | GeneralCategory::OtherNumber
    )
}

/// `\p{M}`.
#[inline]
fn is_mark_char(c: char) -> bool {
    matches!(
        get_general_category(c),
        GeneralCategory::NonspacingMark
            | GeneralCategory::SpacingMark
            | GeneralCategory::EnclosingMark
    )
}

/// `\p{P}` or `\p{S}`, the classes DeepSeek's third pass runs on.
#[inline]
fn is_punct_or_symbol_char(c: char) -> bool {
    matches!(
        get_general_category(c),
        GeneralCategory::ConnectorPunctuation
            | GeneralCategory::DashPunctuation
            | GeneralCategory::OpenPunctuation
            | GeneralCategory::ClosePunctuation
            | GeneralCategory::InitialPunctuation
            | GeneralCategory::FinalPunctuation
            | GeneralCategory::OtherPunctuation
            | GeneralCategory::MathSymbol
            | GeneralCategory::CurrencySymbol
            | GeneralCategory::ModifierSymbol
            | GeneralCategory::OtherSymbol
    )
}

/// Uppercase half of o200k's case-split letter run:
/// `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]`.
#[inline]
fn is_upper_run_char(c: char) -> bool {
    matches!(
        get_general_category(c),
        GeneralCategory::UppercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
    ) || is_mark_char(c)
}

/// Lowercase half: `[\p{Ll}\p{Lm}\p{Lo}\p{M}]`.
#[inline]
fn is_lower_run_char(c: char) -> bool {
    matches!(
        get_general_category(c),
        GeneralCategory::LowercaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
    ) || is_mark_char(c)
}

/// Character at `pos`, which must be a char boundary, and its UTF-8 length.
#[inline]
fn char_at(text: &str, pos: usize) -> (char, usize) {
    let c = text[pos..]
        .chars()
        .next()
        .expect("pos is a char boundary inside the string");
    (c, c.len_utf8())
}

/// Length of the character at `pos` when it satisfies `pred`.
///
/// The ASCII shortcut is the point: `pred` is only reached for a multi-byte
/// character, so the common path never decodes or looks up a category.
#[inline]
fn char_len_if(
    text: &str,
    bytes: &[u8],
    pos: usize,
    ascii: impl Fn(Class) -> bool,
    pred: impl Fn(char) -> bool,
) -> Option<usize> {
    let class = CLASS[bytes[pos] as usize];
    if class == Class::Lead {
        let (c, len) = char_at(text, pos);
        pred(c).then_some(len)
    } else {
        ascii(class).then_some(1)
    }
}

#[inline]
fn letter_at(text: &str, bytes: &[u8], pos: usize) -> Option<usize> {
    char_len_if(text, bytes, pos, |c| c == Class::Letter, is_letter_char)
}

#[inline]
fn number_at(text: &str, bytes: &[u8], pos: usize) -> Option<usize> {
    char_len_if(text, bytes, pos, |c| c == Class::Digit, is_number_char)
}

#[inline]
fn space_at(text: &str, bytes: &[u8], pos: usize) -> Option<usize> {
    char_len_if(
        text,
        bytes,
        pos,
        |c| matches!(c, Class::Space | Class::Newline),
        char::is_whitespace,
    )
}

/// `[^\s\p{L}\p{N}]`.
#[inline]
fn punct_at(text: &str, bytes: &[u8], pos: usize) -> Option<usize> {
    char_len_if(
        text,
        bytes,
        pos,
        |c| c == Class::Punct,
        |c| !c.is_whitespace() && !is_letter_char(c) && !is_number_char(c),
    )
}

/// Scan `[\p{L}\p{M}]` from `pos`, whose first character is `n` bytes long.
///
/// Both scanners accept the same class and differ only in whether they try to
/// skip ASCII in bulk, so choosing between them is a pure question of which is
/// cheaper here — and the answer is decided by the first character. A run that
/// opens with a non-ASCII letter is usually written entirely in one, and Greek,
/// Cyrillic and Thai would otherwise offer the bulk skip a character at a time
/// and have it decline every time.
#[inline]
fn scan_letter_or_mark_run(text: &str, bytes: &[u8], pos: usize, n: usize) -> usize {
    if n == 1 {
        scan_run_of(text, bytes, pos + n, LETTER_OR_MARK)
    } else {
        scan_run(text, bytes, pos + n, letter_or_mark_at)
    }
}

/// `[\p{L}\p{M}]` as a run: ASCII letters eight bytes at a time, then one
/// character wherever the bulk skip stops.
///
/// The class is overwhelmingly ASCII letters on the text these models see, and
/// walking them one byte at a time — a table lookup, a compare and a branch
/// each — was a quarter of a deepseek encode.
const LETTER_OR_MARK: Run = Run {
    at: letter_or_mark_at,
    skip_ascii: swar_skip_ascii_letters,
};

/// `[\p{L}\p{M}]`, DeepSeek's letter class.
#[inline]
fn letter_or_mark_at(text: &str, bytes: &[u8], pos: usize) -> Option<usize> {
    char_len_if(
        text,
        bytes,
        pos,
        |c| c == Class::Letter,
        |c| is_letter_char(c) || is_mark_char(c),
    )
}

/// `[\p{P}\p{S}]`, DeepSeek's punctuation class. Wider than `punct_at`: it
/// excludes digits and whitespace by category rather than by exclusion, so a
/// character in neither class (a control code, say) belongs to neither run.
#[inline]
fn punct_or_symbol_at(text: &str, bytes: &[u8], pos: usize) -> Option<usize> {
    let class = CLASS[bytes[pos] as usize];
    if class == Class::Lead {
        let (c, len) = char_at(text, pos);
        is_punct_or_symbol_char(c).then_some(len)
    } else {
        // ASCII punctuation is exactly the printable non-alphanumeric set.
        let b = bytes[pos];
        (class == Class::Punct && b.is_ascii_graphic()).then_some(1)
    }
}

// --- run scanners -----------------------------------------------------------

/// End of the maximal `\p{L}+` run starting at `pos`.
#[inline]
fn scan_letters(text: &str, bytes: &[u8], mut pos: usize) -> usize {
    loop {
        pos = swar_skip_ascii_letters(bytes, pos);
        if pos >= bytes.len() || bytes[pos] < 0x80 {
            return pos;
        }
        match letter_at(text, bytes, pos) {
            Some(n) => pos += n,
            None => return pos,
        }
    }
}

/// A character-class predicate: the byte length of the character at `pos` when
/// it is in the class, `None` otherwise.
type RunFn = fn(&str, &[u8], usize) -> Option<usize>;

/// One character class the case-split branches scan runs of.
///
/// `at` answers for a single character and is the definition; `skip_ascii`
/// covers the ASCII members of the same class eight bytes at a time. Both are
/// needed because the classes contain non-ASCII characters that only `at` can
/// judge, while on Latin prose almost every character is ASCII and `at`'s
/// per-call cost dominates.
///
/// The two must agree on ASCII, or runs would end in different places. The
/// scanner-vs-regex tests are what enforce that.
#[derive(Clone, Copy)]
struct Run {
    at: RunFn,
    skip_ascii: fn(&[u8], usize) -> usize,
}

/// Lead bytes of the three-byte sequences encoding `U+5000..=U+9FFF`.
///
/// That span sits inside CJK Unified Ideographs, which is uniformly `\p{Lo}`,
/// so a class containing `\p{Lo}` can accept such a character from its lead
/// byte alone — no decode, no category lookup. It is the bulk of Chinese and of
/// Japanese kanji. The neighbouring blocks are deliberately absent: Hiragana,
/// Katakana and the compatibility ideographs carry marks, punctuation and
/// unassigned code points, so no such shortcut is sound for them.
const IDEOGRAPH_LEAD: std::ops::RangeInclusive<u8> = 0xE5..=0xE9;

/// Skip a run of ideographs, three bytes at a time.
///
/// The counterpart to the SWAR ASCII skip, for the population on the other side
/// of it. ASCII is classified from its byte through [`CLASS`]; this classifies
/// from the lead byte the same way, where the general path decodes the
/// character and reads the Unicode tables once per character.
#[inline]
fn skip_ideographs(bytes: &[u8], mut pos: usize) -> usize {
    while pos + 3 <= bytes.len() && IDEOGRAPH_LEAD.contains(&bytes[pos]) {
        pos += 3;
    }
    pos
}

/// Both bulk skips, alternating until neither advances.
///
/// Folded into the run's existing `skip_ascii` slot rather than added beside
/// it: the scan loop runs per character, and a second indirect call there costs
/// more than the skip saves. A class that excludes `\p{Lo}` — Kimi's, which
/// subtracts `\p{Han}` — keeps the plain ASCII skip.
///
/// Which runs are wrapped also moves codegen, in a direction that does not
/// follow from the source: wrapping every run measured worse than wrapping only
/// these two. Treat the asymmetry as deliberate and re-measure before tidying
/// it. The tests below pin the behaviour; nothing pins the codegen.
macro_rules! ascii_or_ideograph_skip {
    ($name:ident, $ascii:ident) => {
        fn $name(bytes: &[u8], mut pos: usize) -> usize {
            loop {
                let advanced = skip_ideographs(bytes, $ascii(bytes, pos));
                if advanced == pos {
                    return pos;
                }
                pos = advanced;
            }
        }
    };
}

ascii_or_ideograph_skip!(skip_upper_or_ideograph, swar_skip_ascii_upper);
ascii_or_ideograph_skip!(skip_lower_or_ideograph, swar_skip_ascii_lower);

/// End of the maximal run of `run`'s class, ASCII spans taken eight bytes at a
/// time and everything else one character at a time.
#[inline]
fn scan_run_of(text: &str, bytes: &[u8], mut pos: usize, run: Run) -> usize {
    loop {
        // The ASCII members in bulk...
        pos = (run.skip_ascii)(bytes, pos);
        if pos >= bytes.len() {
            return pos;
        }
        // ...then one character, which is either a non-ASCII member of the
        // class or the byte that ends the run.
        match (run.at)(text, bytes, pos) {
            Some(n) => pos += n,
            None => return pos,
        }
    }
}

/// End of the maximal run of characters satisfying `at`, starting at `pos`.
#[inline]
fn scan_run(
    text: &str,
    bytes: &[u8],
    mut pos: usize,
    at: impl Fn(&str, &[u8], usize) -> Option<usize>,
) -> usize {
    while pos < bytes.len() {
        match at(text, bytes, pos) {
            Some(n) => pos += n,
            None => break,
        }
    }
    pos
}

/// Case-insensitive `'s|'t|'re|'ve|'m|'ll|'d` at `pos`, including the quote.
#[inline]
fn contraction_len(bytes: &[u8], pos: usize) -> Option<usize> {
    if bytes.get(pos) != Some(&b'\'') {
        return None;
    }
    let lower = |i: usize| bytes.get(i).map(|b| b | 0x20);
    match lower(pos + 1)? {
        b's' | b'd' | b'm' | b't' => Some(2),
        b'l' if lower(pos + 2) == Some(b'l') => Some(3),
        b'v' | b'r' if lower(pos + 2) == Some(b'e') => Some(3),
        _ => None,
    }
}

/// Which whitespace branch an alternation reaches first.
#[derive(Clone, Copy, PartialEq, Eq)]
enum WhitespaceOrder {
    /// cl100k: `\s+$` precedes `\s*[\r\n]`, so a run that both contains a
    /// newline and reaches the end of the text is taken whole.
    EndOfTextFirst,
    /// llama3/o200k/deepseek: `\s*[\r\n]+` precedes `\s+(?!\S)`, so the same run
    /// is cut after its last newline.
    NewlineFirst,
}

/// Resolves a whitespace run to the span its alternation would match.
///
/// All four patterns end in the same three or four whitespace branches, and the
/// only thing that varies is which comes first.
#[inline]
fn whitespace_span(text: &str, bytes: &[u8], pos: usize, order: WhitespaceOrder) -> (usize, usize) {
    let len = bytes.len();
    let run_end = scan_run(text, bytes, pos, space_at);

    let last_newline = bytes[pos..run_end]
        .iter()
        .rposition(|&b| b == b'\r' || b == b'\n')
        .map(|offset| pos + offset + 1);

    match order {
        // `\s+$` — the whole run, when it reaches the end of the text.
        WhitespaceOrder::EndOfTextFirst if run_end == len => return (pos, run_end),
        // `\s*[\r\n]` / `\s*[\r\n]+` — greedy `\s*` backtracks to the last
        // newline, so the token ends just past it.
        _ => {
            if let Some(end) = last_newline {
                return (pos, end);
            }
        }
    }

    // `\s+(?!\S)` — at the end of the text the lookahead holds with nothing
    // given back, so the run is taken whole; otherwise one character is given
    // back so the lookahead sees whitespace rather than the character that ends
    // the run. That needs at least two characters.
    if run_end == len {
        return (pos, run_end);
    }
    let last_char_len = text[pos..run_end]
        .chars()
        .next_back()
        .map(char::len_utf8)
        .unwrap_or(1);
    if run_end - pos > last_char_len {
        return (pos, run_end - last_char_len);
    }

    // `\s` / `\s+` — a single character, which is the whole run here.
    (pos, run_end)
}

// --- the cl100k family ------------------------------------------------------

/// Shape shared by cl100k_base, Llama 3 and Qwen 2.
#[derive(Clone, Copy)]
struct Family {
    /// `\p{N}{1,3}` versus Qwen's single `\p{N}`.
    max_digits: u32,
    whitespace: WhitespaceOrder,
}

const CL100K: Family = Family {
    max_digits: 3,
    whitespace: WhitespaceOrder::EndOfTextFirst,
};
const LLAMA3: Family = Family {
    max_digits: 3,
    whitespace: WhitespaceOrder::NewlineFirst,
};
const QWEN2: Family = Family {
    max_digits: 1,
    whitespace: WhitespaceOrder::NewlineFirst,
};

/// Splits by a cl100k-shaped pattern:
/// `contraction | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}{1,n} | ?[^\s\p{L}\p{N}]+[\r\n]* | whitespace`
fn family_spans(text: &str, out: &mut Vec<(usize, usize)>, scheme: Family) {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut pos = 0usize;

    while pos < len {
        let start = pos;

        // Contractions. cl100k writes this as `'(?i:[sdmt]|ll|ve|re)` and
        // Llama 3 as `(?i:'s|'t|…)`; the accepted set is the same.
        if let Some(n) = contraction_len(bytes, pos) {
            pos += n;
            out.push((start, pos));
            continue;
        }

        // `[^\r\n\p{L}\p{N}]?\p{L}+` — the optional prefix is any single
        // character that is not CR, LF, a letter or a digit, so a space and a
        // punctuation mark both qualify: " word" and "!word" are each one token.
        // It is only taken when a letter run actually follows.
        if let Some(n) = letter_at(text, bytes, pos) {
            pos = scan_letters(text, bytes, pos + n);
            out.push((start, pos));
            continue;
        }
        if let Some(prefix) = prefix_len(text, bytes, pos) {
            if let Some(n) = pos
                .checked_add(prefix)
                .filter(|&p| p < len)
                .and_then(|p| letter_at(text, bytes, p))
            {
                pos = scan_letters(text, bytes, pos + prefix + n);
                out.push((start, pos));
                continue;
            }
        }

        // `\p{N}{1,n}` — no leading space, unlike the letter branch.
        if let Some(n) = number_at(text, bytes, pos) {
            pos += n;
            for _ in 1..scheme.max_digits {
                match (pos < len).then(|| number_at(text, bytes, pos)).flatten() {
                    Some(n) => pos += n,
                    None => break,
                }
            }
            out.push((start, pos));
            continue;
        }

        // ` ?[^\s\p{L}\p{N}]+[\r\n]*`
        let after_space = if bytes[pos] == b' ' { pos + 1 } else { pos };
        if after_space < len && punct_at(text, bytes, after_space).is_some() {
            pos = scan_run(text, bytes, after_space, punct_at);
            while pos < len && (bytes[pos] == b'\r' || bytes[pos] == b'\n') {
                pos += 1;
            }
            out.push((start, pos));
            continue;
        }

        if space_at(text, bytes, pos).is_some() {
            let (s, e) = whitespace_span(text, bytes, pos, scheme.whitespace);
            pos = e;
            out.push((s, e));
            continue;
        }

        // No branch matches here, which the engine handles by trying the next
        // position. Nothing is emitted.
        let (_, l) = char_at(text, pos);
        pos += l;
    }
}

/// `[^\r\n\p{L}\p{N}]?` — length of the optional prefix character at `pos`.
///
/// Letters are excluded as well as digits and the two newline characters. The
/// cl100k family only reaches this after `letter_at` has already failed, so the
/// distinction is invisible there, but o200k asks the question directly: without
/// it the prefix would swallow the `i` of `iPhone` and the branch would go on to
/// match `Phone`, giving one token where the expression gives two.
#[inline]
fn prefix_len(text: &str, bytes: &[u8], pos: usize) -> Option<usize> {
    match CLASS[bytes[pos] as usize] {
        Class::Newline | Class::Digit | Class::Letter => None,
        Class::Lead => {
            let (c, l) = char_at(text, pos);
            (!is_letter_char(c) && !is_number_char(c)).then_some(l)
        }
        _ => Some(1),
    }
}

pub(super) fn cl100k_spans(text: &str, out: &mut Vec<(usize, usize)>) {
    family_spans(text, out, CL100K)
}

pub(super) fn llama3_spans(text: &str, out: &mut Vec<(usize, usize)>) {
    family_spans(text, out, LLAMA3)
}

pub(super) fn qwen2_spans(text: &str, out: &mut Vec<(usize, usize)>) {
    family_spans(text, out, QWEN2)
}

// --- o200k ------------------------------------------------------------------

/// Splits by the o200k_base pattern.
///
/// Its two leading branches split a letter run on the case boundary — an
/// uppercase head followed by a lowercase tail — which is why `XMLHttpRequest`
/// becomes `XMLHttp` + `Request` here but stays whole under Llama 3.
pub(super) fn o200k_spans(text: &str, out: &mut Vec<(usize, usize)>) {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut pos = 0usize;

    while pos < len {
        let start = pos;

        if let Some(end) = o200k_letter_branches(text, bytes, pos) {
            pos = end;
            out.push((start, pos));
            continue;
        }

        if let Some(n) = number_at(text, bytes, pos) {
            pos += n;
            for _ in 1..3 {
                match (pos < len).then(|| number_at(text, bytes, pos)).flatten() {
                    Some(n) => pos += n,
                    None => break,
                }
            }
            out.push((start, pos));
            continue;
        }

        // ` ?[^\s\p{L}\p{N}]+[\r\n/]*` — note the `/`, which o200k includes in
        // the trailing class and the other patterns do not.
        let after_space = if bytes[pos] == b' ' { pos + 1 } else { pos };
        if after_space < len && punct_at(text, bytes, after_space).is_some() {
            pos = scan_run(text, bytes, after_space, punct_at);
            while pos < len && matches!(bytes[pos], b'\r' | b'\n' | b'/') {
                pos += 1;
            }
            out.push((start, pos));
            continue;
        }

        if space_at(text, bytes, pos).is_some() {
            let (s, e) = whitespace_span(text, bytes, pos, WhitespaceOrder::NewlineFirst);
            pos = e;
            out.push((s, e));
            continue;
        }

        let (_, l) = char_at(text, pos);
        pos += l;
    }
}

/// The two case-split letter branches, in the alternation's order.
///
/// A: `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|…)?`
/// B: `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|…)?`
///
/// Each optional prefix is greedy, so a branch is tried with the prefix before
/// without it.
fn o200k_letter_branches(text: &str, bytes: &[u8], pos: usize) -> Option<usize> {
    case_split_branches(
        text,
        bytes,
        pos,
        Run {
            at: upper_run_at,
            skip_ascii: skip_upper_or_ideograph,
        },
        Run {
            at: lower_run_at,
            skip_ascii: skip_lower_or_ideograph,
        },
    )
}

/// The case-split branches over caller-supplied class predicates.
///
/// o200k and Kimi share this shape and differ only in the two classes: Kimi
/// subtracts `\p{Han}` from both. Passing them in keeps one implementation of
/// the backtracking, which is the part that is easy to get subtly wrong.
fn case_split_branches(
    text: &str,
    bytes: &[u8],
    pos: usize,
    upper: Run,
    lower: Run,
) -> Option<usize> {
    match ascii_case_split(bytes, pos) {
        AsciiCase::Match(end) => return Some(end),
        AsciiCase::NoMatch => return None,
        AsciiCase::Undecided => {}
    }

    let with_prefix = prefix_len(text, bytes, pos).map(|p| pos + p);

    // Branch A, prefix first then without.
    for &q in [with_prefix, Some(pos)].iter().flatten() {
        if q < bytes.len() {
            if let Some(end) = case_split_branch_a(text, bytes, q, upper, lower) {
                return Some(end);
            }
        }
    }
    // Branch B, likewise.
    for &q in [with_prefix, Some(pos)].iter().flatten() {
        if q < bytes.len() {
            if let Some(end) = case_split_branch_b(text, bytes, q, upper, lower) {
                return Some(end);
            }
        }
    }
    None
}

/// Both branches at once, for the all-ASCII word that most prose is made of.
///
/// The general form is four attempts — two prefix positions by two branches —
/// because the uppercase and lowercase classes overlap on `\p{Lm}`, `\p{Lo}`
/// and `\p{M}`, so a greedy `U*` may have to give characters back. **No ASCII
/// character is in both classes**, which collapses all of it: `A-Z` then `a-z`
/// is the longest match either branch can produce, and whichever branch claims
/// it, the end is the same offset.
///
/// [`ascii_case_split`]'s verdict. `NoMatch` and `Undecided` are distinct
/// because a chunk of punctuation or whitespace is *provably* not a letter
/// branch, and re-deriving that through the general form is wasted work — it is
/// the shape a quarter of prose chunks take.
enum AsciiCase {
    Match(usize),
    NoMatch,
    Undecided,
}

#[inline]
fn ascii_case_split(bytes: &[u8], pos: usize) -> AsciiCase {
    let Some(&first) = bytes.get(pos) else {
        return AsciiCase::Undecided;
    };
    if first >= 0x80 {
        return AsciiCase::Undecided;
    }

    // `[^\r\n\p{L}\p{N}]?`, greedy, so it is taken whenever it can be.
    let start =
        pos + usize::from(!first.is_ascii_alphanumeric() && first != b'\r' && first != b'\n');

    let mut i = start;
    while i < bytes.len() && bytes[i].is_ascii_uppercase() {
        i += 1;
    }
    while i < bytes.len() && bytes[i].is_ascii_lowercase() {
        i += 1;
    }

    if i == start {
        // No letters here. Only an ASCII byte proves it: a non-ASCII one could
        // be a letter of either class and start a run this cannot see.
        return match bytes.get(start) {
            Some(&b) if b < 0x80 => AsciiCase::NoMatch,
            None => AsciiCase::NoMatch,
            Some(_) => AsciiCase::Undecided,
        };
    }
    // A non-ASCII byte could extend the run under either class.
    if bytes.get(i).is_some_and(|&b| b >= 0x80) {
        return AsciiCase::Undecided;
    }
    AsciiCase::Match(i + trailing_contraction(bytes, i))
}

/// `U* L+` — greedy `U*` gives characters back until a lowercase-class run can
/// start, because the two classes overlap on `\p{Lm}`, `\p{Lo}` and `\p{M}`.
fn case_split_branch_a(
    text: &str,
    bytes: &[u8],
    start: usize,
    upper: Run,
    lower: Run,
) -> Option<usize> {
    let upper_end = scan_run_of(text, bytes, start, upper);

    // Try the longest `U*` first, then shorter ones, as the engine does.
    let mut boundary = upper_end;
    loop {
        if boundary < bytes.len() {
            if let Some(n) = (lower.at)(text, bytes, boundary) {
                let lower_end = scan_run_of(text, bytes, boundary + n, lower);
                return Some(lower_end + trailing_contraction(bytes, lower_end));
            }
        }
        if boundary <= start {
            return None;
        }
        boundary = prev_char_boundary(text, start, boundary);
    }
}

/// `U+ L*` — no backtracking needed: the greedy `U+` succeeds or the branch
/// fails, and `L*` may be empty.
fn case_split_branch_b(
    text: &str,
    bytes: &[u8],
    start: usize,
    upper: Run,
    lower: Run,
) -> Option<usize> {
    let upper_end = scan_run_of(text, bytes, start, upper);
    if upper_end == start {
        return None;
    }
    let lower_end = scan_run_of(text, bytes, upper_end, lower);
    Some(lower_end + trailing_contraction(bytes, lower_end))
}

#[inline]
fn upper_run_at(text: &str, bytes: &[u8], pos: usize) -> Option<usize> {
    if bytes[pos] < 0x80 {
        bytes[pos].is_ascii_uppercase().then_some(1)
    } else {
        let (c, l) = char_at(text, pos);
        is_upper_run_char(c).then_some(l)
    }
}

#[inline]
fn lower_run_at(text: &str, bytes: &[u8], pos: usize) -> Option<usize> {
    if bytes[pos] < 0x80 {
        bytes[pos].is_ascii_lowercase().then_some(1)
    } else {
        let (c, l) = char_at(text, pos);
        is_lower_run_char(c).then_some(l)
    }
}

#[inline]
fn trailing_contraction(bytes: &[u8], pos: usize) -> usize {
    contraction_len(bytes, pos).unwrap_or(0)
}

/// Start of the character preceding `pos`, never going below `floor`.
#[inline]
fn prev_char_boundary(text: &str, floor: usize, pos: usize) -> usize {
    let mut p = pos.saturating_sub(1);
    while p > floor && !text.is_char_boundary(p) {
        p -= 1;
    }
    p
}

// --- Kimi -------------------------------------------------------------------

/// `\p{Han}`, as 21 sorted, non-overlapping ranges.
///
/// Han is a Unicode *script*, not a general category, so the
/// `unicode_general_category` table this file uses elsewhere cannot answer it.
/// Rather than transcribe ranges from a standards document — where a stale or
/// mistyped bound would show up only on whichever character a test happened to
/// miss — this table was **derived from regexr itself**, by asking `^\p{Han}$`
/// about every scalar value. That makes agreement with the expression a
/// property of how the table was built, and
/// `han_table_matches_the_regex_over_every_scalar_value` re-derives it on every
/// test run so a regexr upgrade that moved a boundary fails loudly here instead
/// of silently changing token ids.
const HAN_RANGES: [(char, char); 21] = [
    ('\u{2E80}', '\u{2E99}'),
    ('\u{2E9B}', '\u{2EF3}'),
    ('\u{2F00}', '\u{2FD5}'),
    ('\u{3005}', '\u{3005}'),
    ('\u{3007}', '\u{3007}'),
    ('\u{3021}', '\u{3029}'),
    ('\u{3038}', '\u{303B}'),
    ('\u{3400}', '\u{4DBF}'),
    ('\u{4E00}', '\u{9FFF}'),
    ('\u{F900}', '\u{FA6D}'),
    ('\u{FA70}', '\u{FAD9}'),
    ('\u{16FE2}', '\u{16FE3}'),
    ('\u{16FF0}', '\u{16FF6}'),
    ('\u{20000}', '\u{2A6DF}'),
    ('\u{2A700}', '\u{2B81D}'),
    ('\u{2B820}', '\u{2CEAD}'),
    ('\u{2CEB0}', '\u{2EBE0}'),
    ('\u{2EBF0}', '\u{2EE5D}'),
    ('\u{2F800}', '\u{2FA1D}'),
    ('\u{30000}', '\u{3134A}'),
    ('\u{31350}', '\u{33479}'),
];

/// `\p{Han}` — binary search over [`HAN_RANGES`].
#[inline]
fn is_han_char(c: char) -> bool {
    HAN_RANGES
        .binary_search_by(|&(lo, hi)| {
            if c < lo {
                core::cmp::Ordering::Greater
            } else if c > hi {
                core::cmp::Ordering::Less
            } else {
                core::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// One `\p{Han}` character at `pos`, or `None`.
///
/// No ASCII byte is Han, so the common case costs one comparison.
#[inline]
fn han_run_at(text: &str, bytes: &[u8], pos: usize) -> Option<usize> {
    if bytes[pos] < 0x80 {
        return None;
    }
    let (c, l) = char_at(text, pos);
    is_han_char(c).then_some(l)
}

/// `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]` — o200k's uppercase half with
/// Han subtracted.
#[inline]
fn kimi_upper_run_at(text: &str, bytes: &[u8], pos: usize) -> Option<usize> {
    let n = upper_run_at(text, bytes, pos)?;
    let (c, _) = char_at(text, pos);
    (!is_han_char(c)).then_some(n)
}

/// `[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]` — likewise for the lowercase half.
#[inline]
fn kimi_lower_run_at(text: &str, bytes: &[u8], pos: usize) -> Option<usize> {
    let n = lower_run_at(text, bytes, pos)?;
    let (c, _) = char_at(text, pos);
    (!is_han_char(c)).then_some(n)
}

/// Splits by the Kimi pattern (Moonshot AI).
///
/// o200k's shape with two changes, both about Han: a leading `[\p{Han}]+`
/// branch, and Han subtracted from the two letter classes so that branch is
/// reachable — without the subtraction the case-split branches would consume a
/// Han run first and the leading branch would never fire. The punctuation tail
/// is `[\r\n]*` here, not o200k's `[\r\n/]*`.
pub(super) fn kimi_spans(text: &str, out: &mut Vec<(usize, usize)>) {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut pos = 0usize;

    while pos < len {
        let start = pos;

        // `[\p{Han}]+`, first in the alternation. It cannot overlap the letter
        // branches (they exclude Han) so trying it first needs no backtracking.
        if han_run_at(text, bytes, pos).is_some() {
            pos = scan_run(text, bytes, pos, han_run_at);
            out.push((start, pos));
            continue;
        }

        if let Some(end) =
            // Kimi subtracts `\p{Han}` from both classes. No Han character is
            // ASCII, so the ASCII skip is the same one o200k's runs build on —
            // but not the ideograph skip layered over it there, which is
            // exactly what Kimi excludes.
            case_split_branches(
                text,
                bytes,
                pos,
                Run {
                    at: kimi_upper_run_at,
                    skip_ascii: swar_skip_ascii_upper,
                },
                Run {
                    at: kimi_lower_run_at,
                    skip_ascii: swar_skip_ascii_lower,
                },
            )
        {
            pos = end;
            out.push((start, pos));
            continue;
        }

        if let Some(n) = number_at(text, bytes, pos) {
            pos += n;
            for _ in 1..3 {
                match (pos < len).then(|| number_at(text, bytes, pos)).flatten() {
                    Some(n) => pos += n,
                    None => break,
                }
            }
            out.push((start, pos));
            continue;
        }

        // ` ?[^\s\p{L}\p{N}]+[\r\n]*` — no `/` in the tail, unlike o200k.
        let after_space = if bytes[pos] == b' ' { pos + 1 } else { pos };
        if after_space < len && punct_at(text, bytes, after_space).is_some() {
            pos = scan_run(text, bytes, after_space, punct_at);
            while pos < len && matches!(bytes[pos], b'\r' | b'\n') {
                pos += 1;
            }
            out.push((start, pos));
            continue;
        }

        if space_at(text, bytes, pos).is_some() {
            let (s, e) = whitespace_span(text, bytes, pos, WhitespaceOrder::NewlineFirst);
            pos = e;
            out.push((s, e));
            continue;
        }

        let (_, l) = char_at(text, pos);
        pos += l;
    }
}

// --- DeepSeek V3, third pass ------------------------------------------------

/// Splits by the third DeepSeek V3 expression.
///
/// The first two passes (`\p{N}{1,3}` and a CJK/kana range) stay on the engine:
/// they are single runs rather than alternations, so they cost the engine little
/// and gain little here. This pass is the alternation, and it is the same shape
/// as the cl100k family over different classes — `[\p{L}\p{M}]` for letters and
/// `[\p{P}\p{S}]` for punctuation, with an extra leading branch pairing one
/// ASCII punctuation mark with an ASCII letter run.
pub(super) fn deepseek_v3_pass3_spans(text: &str, out: &mut Vec<(usize, usize)>) {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut pos = 0usize;
    while pos < len {
        match deepseek_pass3_match(text, bytes, pos, false) {
            Some(end) => {
                out.push((pos, end));
                pos = end;
            }
            // A digit not followed by letters reaches no branch, so the engine
            // moves on without emitting. Passes 1 and 2 are what claim those
            // characters.
            None => pos += char_at(text, pos).1,
        }
    }
}

/// The two shapes ordinary text is mostly made of: a word, and a space then a
/// word. `None` if `pos` starts neither.
///
/// Split out because the fused walk needs it *before* it tests for digits and
/// CJK, and the staged pass needs it first among its branches. Neither shape can
/// be a digit or a CJK character, so running it ahead of those tests cannot
/// take a character the earlier passes had claimed — and it saves every word in
/// ordinary text two failed checks.
///
/// A letter cannot open the punctuation branch, and it makes the optional prefix
/// of the letter branch empty — so the letter run starts where it does. A space
/// cannot open the punctuation branch either, and is exactly what that prefix
/// accepts: not a newline, not a letter, not punctuation or a symbol. Both
/// therefore reach the same span the alternation reaches by trying and failing.
///
/// The letter after a space is matched as a character, not as an ASCII byte:
/// scripts written entirely in multi-byte letters still separate their words
/// with ASCII spaces, and testing only for ASCII would hand them the cost of
/// this route without the benefit.
#[inline]
fn deepseek_word_match(text: &str, bytes: &[u8], pos: usize, stop_at_cjk: bool) -> Option<usize> {
    let run = |at: usize, n: usize| {
        if stop_at_cjk {
            scan_run_of(text, bytes, at + n, LETTER_OR_MARK_NOT_CJK)
        } else {
            scan_letter_or_mark_run(text, bytes, at, n)
        }
    };
    match CLASS[bytes[pos] as usize] {
        Class::Letter => Some(run(pos, 1)),
        Class::Space if bytes[pos] == b' ' && pos + 1 < bytes.len() => {
            let next = bytes[pos + 1];
            if CLASS[next as usize] == Class::Letter {
                Some(run(pos + 1, 1))
            } else if next >= 0x80 {
                let at = if stop_at_cjk {
                    letter_or_mark_not_cjk_at(text, bytes, pos + 1)
                } else {
                    letter_or_mark_at(text, bytes, pos + 1)
                };
                at.map(|n| run(pos + 1, n))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// One match of DeepSeek's third-pass alternation starting at `pos`, or `None`
/// where no branch matches there.
///
/// `stop_at_cjk` is what lets [`deepseek_v3_spans`] fuse the three passes: CJK
/// is `\p{Lo}`, so a letter run would otherwise swallow the characters the
/// second pass had already cut out of the piece this one used to see.
#[inline]
fn deepseek_pass3_match(text: &str, bytes: &[u8], pos: usize, stop_at_cjk: bool) -> Option<usize> {
    let len = bytes.len();
    let run = |text: &str, bytes: &[u8], at: usize, n: usize| {
        if stop_at_cjk {
            scan_run_of(text, bytes, at + n, LETTER_OR_MARK_NOT_CJK)
        } else {
            scan_letter_or_mark_run(text, bytes, at, n)
        }
    };
    let letter_at = |text: &str, bytes: &[u8], at: usize| {
        if stop_at_cjk {
            letter_or_mark_not_cjk_at(text, bytes, at)
        } else {
            letter_or_mark_at(text, bytes, at)
        }
    };

    if let Some(end) = deepseek_word_match(text, bytes, pos, stop_at_cjk) {
        return Some(end);
    }

    // `[!-/:-@\[-`{-~][A-Za-z]+` — one ASCII punctuation mark followed by ASCII
    // letters.
    if bytes[pos].is_ascii_graphic()
        && CLASS[bytes[pos] as usize] == Class::Punct
        && pos + 1 < len
        && bytes[pos + 1].is_ascii_alphabetic()
    {
        return Some(swar_skip_ascii_letters(bytes, pos + 1));
    }

    // `[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+` — the prefix class differs from the
    // cl100k family's: it excludes punctuation and symbols rather than digits,
    // so a digit *can* introduce a letter run here.
    if let Some(n) = letter_at(text, bytes, pos) {
        return Some(run(text, bytes, pos, n));
    }
    if let Some(prefix) = deepseek_prefix_len(text, bytes, pos) {
        if let Some(n) = (pos + prefix < len)
            .then(|| letter_at(text, bytes, pos + prefix))
            .flatten()
        {
            return Some(run(text, bytes, pos + prefix, n));
        }
    }

    // ` ?[\p{P}\p{S}]+[\r\n]*`
    let after_space = if bytes[pos] == b' ' { pos + 1 } else { pos };
    if after_space < len && punct_or_symbol_at(text, bytes, after_space).is_some() {
        let mut end = scan_run(text, bytes, after_space, punct_or_symbol_at);
        while end < len && (bytes[end] == b'\r' || bytes[end] == b'\n') {
            end += 1;
        }
        return Some(end);
    }

    if space_at(text, bytes, pos).is_some() {
        // `\s+(?!\S)` gives a character back so the lookahead sees whitespace
        // rather than what ends the run — but only when something follows the
        // run *in the piece this pass was given*. Passes 1 and 2 cut that piece
        // at the next digit or CJK character, so a run ending at one ends the
        // piece, the lookahead holds, and the run is taken whole. Newlines are
        // an earlier branch and unaffected by what follows.
        if stop_at_cjk {
            let run_end = scan_run(text, bytes, pos, space_at);
            if run_end < len
                && !bytes[pos..run_end]
                    .iter()
                    .any(|&b| b == b'\r' || b == b'\n')
                && earlier_pass_starts_at(text, bytes, run_end)
            {
                return Some(run_end);
            }
        }
        let (_, e) = whitespace_span(text, bytes, pos, WhitespaceOrder::NewlineFirst);
        return Some(e);
    }

    None
}

/// Whether pass 1 or pass 2 claims the character at `pos` — that is, whether the
/// piece pass 3 used to be given ends here.
#[inline]
fn earlier_pass_starts_at(text: &str, bytes: &[u8], pos: usize) -> bool {
    if bytes[pos] < 0x80 {
        return bytes[pos].is_ascii_digit();
    }
    let (c, _) = char_at(text, pos);
    is_deepseek_cjk(c) || is_number_char(c)
}

/// `[\p{L}\p{M}]` minus the characters DeepSeek's second pass claims.
#[inline]
fn letter_or_mark_not_cjk_at(text: &str, bytes: &[u8], pos: usize) -> Option<usize> {
    let class = CLASS[bytes[pos] as usize];
    if class != Class::Lead {
        return (class == Class::Letter).then_some(1);
    }
    let (c, len) = char_at(text, pos);
    if is_deepseek_cjk(c) {
        return None;
    }
    (is_letter_char(c) || is_mark_char(c)).then_some(len)
}

const LETTER_OR_MARK_NOT_CJK: Run = Run {
    at: letter_or_mark_not_cjk_at,
    skip_ascii: swar_skip_ascii_letters,
};

/// DeepSeek's three `Split` passes as one traversal, emitting the pieces the
/// staged form produces rather than the matches of a single pass.
///
/// The three patterns partition the text: `\p{N}{1,3}` takes digits, the second
/// pass takes its CJK ranges, and the third splits what is left. Because the
/// classes are disjoint, what the staged form computes by cutting the text into
/// spans and re-splitting each one can be decided in a single left-to-right
/// walk — which is what this does, at one third of the passes over the text and
/// none of the intermediate span buffers.
///
/// Two things preserve the staged meaning. A letter run stops at CJK, since the
/// second pass had already removed those characters from the piece the third
/// pass saw. And a digit run is capped at three characters, as `{1,3}` caps it.
/// Streams its pieces rather than collecting them: the staged form never held
/// more than one piece's spans at a time, because each stage fed the next one
/// piece at a time, and collecting a whole document's spans first would trade
/// that for a buffer proportional to the text.
pub(crate) fn deepseek_v3_for_each<'p>(text: &'p str, out: &mut dyn FnMut(&'p str)) {
    deepseek_v3_walk(text, |s, e| {
        if let Some(piece) = text.get(s..e) {
            out(piece);
        }
    });
}

fn deepseek_v3_walk(text: &str, mut emit: impl FnMut(usize, usize)) {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut pos = 0usize;
    // Characters no pass claims become one piece with their neighbours, exactly
    // as the unmatched gap between two matches did — which is a cursor, not a
    // flag: whatever lies between the last match and this one is that gap.
    let mut last = 0usize;

    while pos < len {
        // Ahead of the two passes below because neither a digit nor a CJK
        // character can start a word, so this cannot take what they claim.
        if let Some(end) = deepseek_word_match(text, bytes, pos, true) {
            if pos > last {
                emit(last, pos);
            }
            emit(pos, end);
            last = end;
            pos = end;
            continue;
        }

        // `\p{N}{1,3}` — the first pass, capped at three characters.
        if let Some(n) = number_at(text, bytes, pos) {
            let start = pos;
            pos += n;
            for _ in 1..3 {
                match (pos < len).then(|| number_at(text, bytes, pos)).flatten() {
                    Some(n) => pos += n,
                    None => break,
                }
            }
            if start > last {
                emit(last, start);
            }
            emit(start, pos);
            last = pos;
            continue;
        }

        // The second pass's CJK ranges.
        if bytes[pos] >= 0x80 {
            let (c, l) = char_at(text, pos);
            if is_deepseek_cjk(c) {
                let start = pos;
                pos += l;
                while pos < len && bytes[pos] >= 0x80 {
                    let (c, l) = char_at(text, pos);
                    if !is_deepseek_cjk(c) {
                        break;
                    }
                    pos += l;
                }
                if start > last {
                    emit(last, start);
                }
                emit(start, pos);
                last = pos;
                continue;
            }
        }

        match deepseek_pass3_match(text, bytes, pos, true) {
            Some(end) => {
                if pos > last {
                    emit(last, pos);
                }
                emit(pos, end);
                last = end;
                pos = end;
            }
            None => pos += char_at(text, pos).1,
        }
    }
    if last < len {
        emit(last, len);
    }
}

/// `[^\r\n\p{L}\p{P}\p{S}]?`.
#[inline]
fn deepseek_prefix_len(text: &str, bytes: &[u8], pos: usize) -> Option<usize> {
    if CLASS[bytes[pos] as usize] == Class::Newline {
        return None;
    }
    let (c, l) = if bytes[pos] < 0x80 {
        (bytes[pos] as char, 1)
    } else {
        char_at(text, pos)
    };
    (!is_letter_char(c) && !is_punct_or_symbol_char(c)).then_some(l)
}

/// Splits by DeepSeek V3's first pass, `\p{N}{1,3}`.
///
/// Not an alternation, so the engine has less to lose here than on the third
/// pass — but the pass still walks the whole text to find sparse digit runs, and
/// that walk is what `swar_skip_to_number` removes.
pub(super) fn deepseek_v3_pass1_spans(text: &str, out: &mut Vec<(usize, usize)>) {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut pos = 0usize;

    while pos < len {
        pos = swar_skip_to_number(bytes, pos);
        if pos >= len {
            break;
        }
        let Some(n) = number_at(text, bytes, pos) else {
            // A non-ASCII character that is not `\p{N}`; step over it whole.
            let (_, l) = char_at(text, pos);
            pos += l;
            continue;
        };
        let start = pos;
        pos += n;
        for _ in 1..3 {
            match (pos < len).then(|| number_at(text, bytes, pos)).flatten() {
                Some(n) => pos += n,
                None => break,
            }
        }
        out.push((start, pos));
    }
}

/// Splits by DeepSeek V3's second pass, the CJK/hiragana/katakana run.
pub(super) fn deepseek_v3_pass2_spans(text: &str, out: &mut Vec<(usize, usize)>) {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut pos = 0usize;

    while pos < len {
        pos = swar_skip_ascii(bytes, pos);
        if pos >= len {
            break;
        }
        let (c, l) = char_at(text, pos);
        if !is_deepseek_cjk(c) {
            pos += l;
            continue;
        }
        let start = pos;
        pos += l;
        while pos < len && bytes[pos] >= 0x80 {
            let (c, l) = char_at(text, pos);
            if !is_deepseek_cjk(c) {
                break;
            }
            pos += l;
        }
        out.push((start, pos));
    }
}

/// `[\u{4E00}-\u{9FA5}\u{3040}-\u{309F}\u{30A0}-\u{30FF}]`.
#[inline]
fn is_deepseek_cjk(c: char) -> bool {
    matches!(c, '\u{4E00}'..='\u{9FA5}' | '\u{3040}'..='\u{30FF}')
}

/// Appends the pre-token spans of `text` as `(start, end)` byte offsets.
pub(crate) type SpanScanner = fn(&str, &mut Vec<(usize, usize)>);

/// The scanner equivalent to `pattern`, if one has been proven against it.
///
/// Keyed on the exact expression text rather than on a vocabulary name, so a
/// tokenizer built from a `tokenizer.json` that happens to carry one of these
/// expressions gets the scanner too, and one carrying a near-miss does not.
pub(crate) fn for_pattern(pattern: &str) -> Option<SpanScanner> {
    use crate::core::tokenizer::patterns as p;

    // `match` cannot bind against non-literal constants, hence the chain.
    if pattern == p::CL100K_BASE_PATTERN {
        Some(cl100k_spans)
    } else if pattern == p::LLAMA3_PATTERN {
        Some(llama3_spans)
    } else if pattern == p::QWEN2_PATTERN {
        Some(qwen2_spans)
    } else if pattern == p::O200K_BASE_PATTERN {
        Some(o200k_spans)
    } else if pattern == p::KIMI_PATTERN {
        Some(kimi_spans)
    } else if pattern == p::DEEPSEEK_V3_PATTERNS[0] {
        Some(deepseek_v3_pass1_spans)
    } else if pattern == p::DEEPSEEK_V3_PATTERNS[1] || pattern == p::DEEPSEEK_V3_PASS2_LITERAL {
        Some(deepseek_v3_pass2_spans)
    } else if pattern == p::DEEPSEEK_V3_PATTERNS[2] {
        Some(deepseek_v3_pass3_spans)
    } else {
        None
    }
}

#[cfg(test)]
mod fused_tests {
    use super::*;

    /// Apply one `Split{Isolated}` pass: the matches, and the gaps between them.
    fn isolate(text: &str, scan: fn(&str, &mut Vec<(usize, usize)>)) -> Vec<(usize, usize)> {
        let mut matches = Vec::new();
        scan(text, &mut matches);
        let mut out = Vec::new();
        let mut last = 0;
        for &(s, e) in &matches {
            if s > last {
                out.push((last, s));
            }
            if e > s {
                out.push((s, e));
            }
            last = e;
        }
        if last < text.len() {
            out.push((last, text.len()));
        }
        out
    }

    /// The three passes composed the way the pipeline composes them: each pass
    /// re-splits every piece the one before it produced.
    fn staged(text: &str) -> Vec<String> {
        let mut level = vec![text.to_string()];
        for scan in [
            deepseek_v3_pass1_spans as fn(&str, &mut Vec<(usize, usize)>),
            deepseek_v3_pass2_spans,
            deepseek_v3_pass3_spans,
        ] {
            let mut next = Vec::new();
            for piece in &level {
                for (s, e) in isolate(piece, scan) {
                    next.push(piece[s..e].to_string());
                }
            }
            level = next;
        }
        level
    }

    fn fused(text: &str) -> Vec<String> {
        let mut pieces = Vec::new();
        deepseek_v3_for_each(text, &mut |piece| pieces.push(piece.to_string()));
        pieces
    }

    /// The whole point of the fused walk: it must be the three passes, exactly.
    #[test]
    fn the_fused_walk_matches_the_three_passes() {
        let cases = [
            "",
            "hello world",
            " the quick brown fox",
            "abc123def",
            "12345",
            "1 2 3",
            "a1b2c3",
            "中文字",
            "abc中def",
            "中123文",
            "ひらがなカタカナ",
            "  spaced   out  ",
            "line\nbreak\r\nhere",
            "!!!shout",
            "a.b,c;d",
            " ?!@#",
            "café naïve",
            "Привет мир",
            "ελληνικά",
            "ไทยไม่มีช่องว่าง",
            "\u{0}control\u{7f}",
            "mixed 中文 and 123 and abc!",
            "\u{200d}zwj\u{200c}",
            "emoji 🎉 here",
            "tab\tsep",
            "trailing   ",
            "   leading",
            "1234567890",
            "a\u{301}combining",
            "ＦＵＬＬＷＩＤＴＨ",
            // A whitespace run ending where pass 1 or pass 2 cut the piece:
            // `\s+(?!\S)` reaches the end of the piece there, so the run is
            // taken whole rather than giving its last character back.
            "  0",
            "  中",
            "\u{a0} 中",
            "\u{3000}\t5b",
            "a \u{a0}\u{a0}文0",
            "x  \r\n  1",
            " \u{a0}\u{3000} ひ",
        ];
        for text in cases {
            assert_eq!(fused(text), staged(text), "fused walk diverged on {text:?}");
        }
    }

    /// Hand-picked cases are what let a regression through last time, so the
    /// equivalence is also asserted over generated text drawn from every class
    /// the three passes disagree about.
    #[test]
    fn the_fused_walk_matches_the_three_passes_on_generated_text() {
        use proptest::prelude::*;

        // One character from each class that decides a branch: digits (pass 1),
        // CJK and kana (pass 2), letters and marks, punctuation, symbols,
        // whitespace, newlines, and characters no pass claims.
        let alphabet: Vec<char> =
            "ab_Zé中文ひカ0159 \t\n\r\u{a0}\u{3000}.,!?#$@'\u{200d}\u{301}\u{0}\u{7f}\u{ad}αДไ🎉²"
                .chars()
                .collect();
        let mut runner = proptest::test_runner::TestRunner::deterministic();
        let strategy = proptest::collection::vec(0usize..alphabet.len(), 0..48);
        runner
            .run(&strategy, |picks| {
                let text: String = picks.iter().map(|&i| alphabet[i]).collect();
                prop_assert_eq!(fused(&text), staged(&text), "diverged on {:?}", text);
                Ok(())
            })
            .expect("fused walk must equal the three passes on every generated string");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tokenizer::patterns::{
        CL100K_BASE_PATTERN, DEEPSEEK_V3_PATTERNS, KIMI_PATTERN, LLAMA3_PATTERN,
        O200K_BASE_PATTERN, QWEN2_PATTERN,
    };

    type Scanner = fn(&str, &mut Vec<(usize, usize)>);

    /// Every scanner against the expression it replaces.
    fn all_scanners() -> Vec<(&'static str, &'static str, Scanner)> {
        vec![
            ("cl100k", CL100K_BASE_PATTERN, cl100k_spans as Scanner),
            ("llama3", LLAMA3_PATTERN, llama3_spans as Scanner),
            ("qwen2", QWEN2_PATTERN, qwen2_spans as Scanner),
            ("o200k", O200K_BASE_PATTERN, o200k_spans as Scanner),
            ("kimi", KIMI_PATTERN, kimi_spans as Scanner),
            (
                "deepseek-pass1",
                DEEPSEEK_V3_PATTERNS[0],
                deepseek_v3_pass1_spans as Scanner,
            ),
            (
                "deepseek-pass2",
                DEEPSEEK_V3_PATTERNS[1],
                deepseek_v3_pass2_spans as Scanner,
            ),
            (
                "deepseek-pass3",
                DEEPSEEK_V3_PATTERNS[2],
                deepseek_v3_pass3_spans as Scanner,
            ),
        ]
    }

    /// The two spellings of the DeepSeek CJK pass must be the same expression,
    /// and both must reach the scanner.
    ///
    /// The lookup compares expression text exactly, so accepting a second
    /// spelling is only safe if it really is the same class — asserted here by
    /// running the compiled engine over both and against the scanner, rather
    /// than by looking at them.
    #[test]
    fn both_spellings_of_the_deepseek_cjk_pass_agree() {
        use crate::core::tokenizer::patterns::DEEPSEEK_V3_PASS2_LITERAL;

        assert!(
            for_pattern(DEEPSEEK_V3_PASS2_LITERAL).is_some(),
            "the spelling `tokenizer.json` uses must reach the scanner"
        );

        let escaped = regexr::RegexBuilder::new(DEEPSEEK_V3_PATTERNS[1])
            .jit(true)
            .build()
            .expect("escaped spelling compiles");
        let literal = regexr::RegexBuilder::new(DEEPSEEK_V3_PASS2_LITERAL)
            .jit(true)
            .build()
            .expect("literal spelling compiles");

        // Inside the class and just outside it on both sides of every range,
        // mixed with text the class must not claim.
        for text in [
            "abc",
            "中文字",
            "ひらがな",
            "カタカナ",
            "a中b",
            "\u{4DFF}\u{4E00}\u{9FA5}\u{9FA6}",
            "\u{303F}\u{3040}\u{309F}\u{30A0}\u{30FF}\u{3100}",
            "1 2 3",
            "",
        ] {
            let want: Vec<(usize, usize)> = escaped
                .find_iter(text)
                .map(|m| (m.start(), m.end()))
                .collect();
            let got: Vec<(usize, usize)> = literal
                .find_iter(text)
                .map(|m| (m.start(), m.end()))
                .collect();
            assert_eq!(want, got, "spellings disagree on {text:?}");

            let mut scanned = Vec::new();
            deepseek_v3_pass2_spans(text, &mut scanned);
            assert_eq!(want, scanned, "scanner disagrees on {text:?}");
        }
    }

    /// The scanners exist to make the bundled vocabularies fast, so a new one
    /// arriving with a pattern nothing here recognises should be visible
    /// immediately rather than as an unexplained encode-speed cliff.
    ///
    /// Mistral V3 and Whisper are the stated exceptions: Tekken's pattern and
    /// GPT-2's are shapes no scanner covers yet, and they fall back to the
    /// regex engine. Mistral V1/V2 report no pattern at all.
    #[test]
    fn every_bundled_vocabulary_that_should_have_a_scanner_has_one() {
        use crate::core::pretrained::{patterns, PretrainedVocab::*};

        let no_scanner = [MistralV3, WhisperV1, WhisperV2, WhisperV3];
        for vocab in [
            Cl100kBase, O200kBase, GptOss, Llama3, DeepseekV3, Qwen3, Glm4, KimiK2, KimiK3,
            MistralV1, MistralV2, MistralV3, WhisperV1, WhisperV2, WhisperV3,
        ] {
            let Some(pats) = patterns(vocab) else {
                continue;
            };
            let expected = !no_scanner.contains(&vocab);
            for pattern in pats {
                assert_eq!(
                    for_pattern(pattern).is_some(),
                    expected,
                    "{vocab:?} scanner coverage changed for {pattern}"
                );
            }
        }
    }

    /// Re-derive `\p{Han}` from the engine and compare it to the embedded table,
    /// scalar value by scalar value.
    ///
    /// The table exists because `unicode_general_category` cannot answer a
    /// *script* query, and it was generated by asking regexr this exact
    /// question. Asking again on every test run is what keeps it honest: a
    /// regexr upgrade that moves a Han boundary would otherwise change token ids
    /// for Kimi silently, and only for inputs containing that one character.
    #[test]
    fn han_table_matches_the_regex_over_every_scalar_value() {
        let re = regexr::Regex::new(r"^\p{Han}$").expect("regexr knows \\p{Han}");
        let mut buf = [0u8; 4];
        let mut disagreements = 0usize;
        let mut first: Option<char> = None;
        for cp in 0u32..=0x10FFFF {
            let Some(c) = char::from_u32(cp) else {
                continue;
            };
            if re.is_match(c.encode_utf8(&mut buf)) != is_han_char(c) {
                disagreements += 1;
                first.get_or_insert(c);
            }
        }
        assert_eq!(
            disagreements,
            0,
            "HAN_RANGES disagrees with the engine on {disagreements} code points, \
             first U+{:04X} — regenerate the table",
            first.map(u32::from).unwrap_or(0)
        );
    }

    /// Inputs that turn on Kimi's Han branch specifically, diffed against the
    /// expression like every other scanner case.
    ///
    /// The random corpus reaches these shapes eventually; naming them makes a
    /// regression say which rule broke instead of printing a random string.
    #[test]
    fn kimi_scanner_agrees_with_its_regex_on_han_boundaries() {
        for input in [
            "中文English混合",
            "汉字abc",
            "北京市 Pascal",
            " 中文",
            "中文 ",
            "中文123",
            "123中文",
            "中\u{0301}文",
            "あ中ア文",
            "\u{3005}\u{3006}\u{3007}",
            "\u{9FFF}\u{A000}",
            "\u{20000}\u{20001}x",
            "XMLHttp中文Request",
            "中文'sX",
            "中!文",
            "中\n文",
        ] {
            assert_agrees("kimi", KIMI_PATTERN, kimi_spans as Scanner, input);
        }
    }

    fn assert_agrees(name: &str, pattern: &str, scan: Scanner, input: &str) {
        let re = regexr::RegexBuilder::new(pattern)
            .jit(true)
            .build()
            .expect("pattern compiles");
        let expected: Vec<(usize, usize)> =
            re.find_iter(input).map(|m| (m.start(), m.end())).collect();
        let mut got = Vec::new();
        scan(input, &mut got);
        assert_eq!(
            got,
            expected,
            "{name} scanner disagrees with its regex on {input:?}\n  scanner: {:?}\n  regex:   {:?}",
            got.iter().map(|&(s, e)| &input[s..e]).collect::<Vec<_>>(),
            expected
                .iter()
                .map(|&(s, e)| &input[s..e])
                .collect::<Vec<_>>()
        );
    }

    const SHAPED: &[&str] = &[
        "",
        " ",
        "  ",
        "   ",
        "\n",
        "\n\n",
        " \n",
        " \n ",
        "\t\n\t",
        "hello world",
        "hello[]!?",
        "[]!?",
        "The[]getUserName",
        "[]get",
        "hello[]",
        "a{}b",
        "Zürich{}",
        " hello",
        "!hello",
        "hello!",
        "don't stop",
        "DON'T STOP",
        "it's o'clock 'tis",
        "'s 'S 'll 'LL 've 're 'd 'm 't",
        "'x 'zz '",
        "123",
        "1234",
        "12345678",
        " 123",
        "abc123def",
        "a1b2c3",
        "!!!",
        " !!!",
        "!!!\n\n",
        " ...\r\n",
        "( )",
        "{\"key\": \"value\", \"n\": 42}",
        "def f(x):\n    return x + 1\n",
        "trailing   ",
        "trailing\n",
        "trailing \n ",
        "  leading",
        "a  b",
        "a   b",
        "a \n b",
        "中文测试",
        " 中文",
        "中文 abc",
        "café naïve",
        "emoji 🚀 here",
        "🚀🚀",
        " 🚀",
        "Ⅷ Ⅸ",
        "½ ¾",
        "\u{3000}abc",
        "a\u{00a0}b",
        "x\u{2028}y",
        "\r\n\r\n",
        "a\r\nb",
        "  \r\n  ",
        "word\u{0301} mark",
        // Case splits, for o200k's two leading branches.
        "XMLHttpRequest",
        "camelCase",
        "PascalCase",
        "ALLCAPS",
        "ALLCAPSthenLower",
        "aB",
        "Ab",
        "A",
        "AB",
        "iPhone",
        "McDonald's",
        "HTTPServer's",
        "ÉCOLE école",
        // Slashes, for o200k's trailing class.
        "path/to/file",
        "a//b",
        "!/",
        " //\n",
        // Digit runs and CJK/kana, for DeepSeek's first two passes. The digit
        // cases straddle the 1..3 grouping and the SWAR skip; the script cases
        // sit on the edges of the three ranges the second pass accepts.
        "1",
        "12",
        "1234567",
        "abc 1234567 def",
        "a1234567890b",
        "\u{4e00}\u{9fa5}",
        "\u{4dff}\u{4e00}",
        "\u{9fa5}\u{9fa6}",
        "\u{303f}\u{3040}",
        "\u{30ff}\u{3100}",
        "\u{3040}\u{309f}\u{30a0}\u{30ff}",
        "ひらがな カタカナ 漢字",
        "long ascii prefix before 漢字 appears",
        "漢字123ひらがな",
        // Long ASCII letter runs, to exercise the SWAR path and its tail.
        "abcdefg",
        "abcdefgh",
        "abcdefghi",
        "abcdefghijklmnopqrstuvwxyz",
        "abcdefghijklmnopqrstuvwxyz0",
        "Supercalifragilisticexpialidocious and more words here",
        "aaaaaaaa\u{4e2d}",
        "aaaaaaaaaaaaaaaa\u{4e2d}bbbbbbbb",
    ];

    #[test]
    fn scanners_agree_with_their_regex_on_shaped_inputs() {
        for (name, pattern, scan) in all_scanners() {
            for input in SHAPED {
                assert_agrees(name, pattern, scan, input);
            }
        }
    }

    /// The all-ASCII shortcut in [`ascii_case_split`], at its edges.
    ///
    /// It collapses four branch attempts into one scan by relying on no ASCII
    /// character being in both letter classes. Each case below is a way that
    /// could be wrong: an uppercase run with no lowercase tail (branch B, not
    /// A), a run the optional prefix must or must not absorb, a contraction
    /// after either branch, and a run that a non-ASCII byte extends — which
    /// must fall back to the general form rather than stopping short.
    #[test]
    fn scanners_agree_with_their_regex_on_case_split_shapes() {
        const CASES: &[&str] = &[
            "XMLHttpRequest",
            " XMLHttpRequest",
            "ABC",
            " ABC",
            "ABC def",
            "aB",
            "Ab",
            "A",
            " A",
            "(the",
            "(The",
            "(ABC",
            "\r\nthe",
            "\nThe",
            "don't",
            "DON'T",
            "Don's",
            "ABC'S",
            "it's Sam's",
            // Non-ASCII must extend the run, so the shortcut has to decline.
            "caFé",
            "ABCé",
            "Ünicode",
            "aÜb",
            "abcé def",
            "ABC\u{0301}",
            "the\u{00a0}end",
            "A\u{4e2d}B",
            // Prefix that is itself non-ASCII.
            "\u{00a0}the",
            "\u{2028}The",
        ];
        for (name, pattern, scan) in all_scanners() {
            for input in CASES {
                assert_agrees(name, pattern, scan, input);
            }
        }
    }

    /// Long inputs, where the SWAR run scanners do most of the work.
    ///
    /// The cases above are short enough that a run rarely fills a single
    /// eight-byte word, so they exercise the scalar tails far more than the
    /// vectorised bodies. These are long enough for the reverse, and they place
    /// script and digit changes mid-document so a run ends inside a word rather
    /// than on its edge.
    #[test]
    fn scanners_agree_with_their_regex_on_long_inputs() {
        let filler = "The quick brown fox jumps over the lazy dog. ";
        let cases: Vec<String> = vec![
            filler.repeat(400),
            format!("{}\n{}", filler.repeat(200), filler.repeat(200)),
            format!("{} \n \n {}", filler.repeat(200), filler.repeat(200)),
            format!(
                "{}\n/path/to/file {}",
                filler.repeat(200),
                filler.repeat(200)
            ),
            format!("{}\n中文测试 {}", filler.repeat(200), filler.repeat(200)),
            format!("{}\n\n\nx{}", filler.repeat(200), filler.repeat(200)),
            format!("{}\n漢字ひらがな{}", filler.repeat(200), "漢字".repeat(600)),
            format!("{}\n1234567890 {}", filler.repeat(200), "42 ".repeat(600)),
            // Trailing whitespace at the end of a long text, for `\s+$`.
            format!("{}\nx{}   ", filler.repeat(200), filler.repeat(200)),
        ];
        for (name, pattern, scan) in all_scanners() {
            for input in &cases {
                assert_agrees(name, pattern, scan, input);
            }
        }
    }

    /// A letter run ending on every possible byte, at every offset in a SWAR word.
    ///
    /// The bug this exists for: `lanes_below` used the bare `hasless` form,
    /// whose borrow escapes a lane and corrupts the ones above it, so the letter
    /// scan reported the wrong stopping lane and swallowed the character after
    /// the run — `hello[]!?` split as `hello[` + `]!?`. It needed a run ending
    /// at a particular offset inside the eight-byte window followed by a
    /// particular byte, which random sampling had not happened to produce.
    /// Walking both axes exhaustively is a few thousand cheap cases.
    #[test]
    fn scanners_agree_at_every_run_boundary_in_a_swar_word() {
        let cl100k = regexr::RegexBuilder::new(CL100K_BASE_PATTERN)
            .jit(true)
            .build()
            .expect("cl100k pattern compiles");

        for run in 0..20usize {
            for byte in 0u8..=127 {
                let c = byte as char;
                // Control characters are legal input but make failure messages
                // unreadable, and they exercise no distinct lane behaviour.
                if c.is_control() && c != '\n' && c != '\r' && c != '\t' {
                    continue;
                }
                let input = format!("{}{}{}", "a".repeat(run), c, "b".repeat(3));
                let expected: Vec<(usize, usize)> = cl100k
                    .find_iter(&input)
                    .map(|m| (m.start(), m.end()))
                    .collect();
                let mut got = Vec::new();
                cl100k_spans(&input, &mut got);
                assert_eq!(
                    got,
                    expected,
                    "run={run} byte={byte:#04x} input={input:?}\n  scanner: {:?}",
                    got.iter().map(|&(s, e)| &input[s..e]).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn scanners_agree_with_their_regex_on_random_inputs() {
        // Assembled from the character kinds the branches turn on, rather than
        // from prose: disagreements live at run boundaries.
        const ALPHABET: &[&str] = &[
            "a",
            "z",
            "A",
            "Z",
            "1",
            "9",
            " ",
            "  ",
            "\t",
            "\n",
            "\r",
            "\r\n",
            "'",
            "!",
            ".",
            ",",
            "-",
            "_",
            "/",
            "中",
            "é",
            "É",
            "🚀",
            "\u{00a0}",
            "\u{3000}",
            "Ⅷ",
            "½",
            "\u{0301}",
            "«",
            "€",
            "\u{05d0}",
            // Han boundaries, for Kimi's leading `[\p{Han}]+` branch and the
            // subtraction that makes it reachable. Each pair straddles an edge
            // of HAN_RANGES, so a table off by one code point shows up here
            // rather than on whichever input a user happens to tokenize:
            // U+2E99/U+2E9A (gap), U+3005-3007 (3006 is not Han), U+9FFF/U+A000
            // (end of the URO), plus the compat block and a 4-byte extension.
            "\u{2E99}",
            "\u{2E9A}",
            "\u{3005}",
            "\u{3006}",
            "\u{3007}",
            "\u{9FFF}",
            "\u{A000}",
            "\u{F900}",
            "\u{20000}",
            // Kana: CJK but *not* Han, so Kimi's letter branches must still take
            // them while DeepSeek's own CJK pass does not distinguish.
            "\u{3042}",
            "\u{30A2}",
        ];

        // xorshift64*, so a failure is reproducible from the seed alone.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };

        let scanners = all_scanners();
        for _ in 0..3000 {
            let pieces = 1 + (next() % 14) as usize;
            let mut input = String::new();
            for _ in 0..pieces {
                input.push_str(ALPHABET[(next() % ALPHABET.len() as u64) as usize]);
            }
            for &(name, pattern, scan) in &scanners {
                assert_agrees(name, pattern, scan, &input);
            }
        }
    }

    /// The lead-byte range must cover only code points that are certainly
    /// `\p{Lo}`, or a run would swallow a character its class excludes.
    #[test]
    fn the_ideograph_lead_range_is_uniformly_other_letter() {
        for lead in IDEOGRAPH_LEAD {
            for (lo, hi) in [(0x80u32, 0x80u32), (0xBF, 0xBF)] {
                let cp = ((lead as u32 & 0x0F) << 12) | ((lo & 0x3F) << 6) | (hi & 0x3F);
                let c = char::from_u32(cp).expect("valid scalar");
                assert_eq!(
                    get_general_category(c),
                    GeneralCategory::OtherLetter,
                    "U+{cp:04X} (lead {lead:#04X}) is not OtherLetter"
                );
            }
        }
    }

    /// Every code point the range can encode, not just its corners.
    #[test]
    fn every_code_point_the_lead_range_encodes_is_other_letter() {
        for cp in 0x5000u32..=0x9FFF {
            let c = char::from_u32(cp).expect("valid scalar");
            assert_eq!(
                get_general_category(c),
                GeneralCategory::OtherLetter,
                "U+{cp:04X} is not OtherLetter"
            );
        }
    }

    /// The bulk skip must stop exactly where the run does, and must not read
    /// past a truncated trailing sequence.
    #[test]
    fn ideographs_are_skipped_in_bulk() {
        let text = "漢字漢字a";
        assert_eq!(skip_ideographs(text.as_bytes(), 0), 12, "four ideographs");
        assert_eq!(skip_ideographs(text.as_bytes(), 12), 12, "stops at ASCII");
        assert_eq!(skip_ideographs(b"", 0), 0);
        // Hiragana is outside the range and must not be skipped.
        assert_eq!(skip_ideographs("ひらがな".as_bytes(), 0), 0);
    }

    /// The wiring, not just the helper: o200k's case-split runs must take the
    /// skip and Kimi's must not, because Kimi subtracts `\p{Han}` from both
    /// classes. Reverting either would be silent without this.
    #[test]
    fn only_the_runs_containing_other_letter_take_the_skip() {
        let han = "漢字漢字";
        let upper = Run {
            at: upper_run_at,
            skip_ascii: skip_upper_or_ideograph,
        };
        assert_eq!(scan_run_of(han, han.as_bytes(), 0, upper), han.len());

        let kimi_upper = Run {
            at: kimi_upper_run_at,
            skip_ascii: swar_skip_ascii_upper,
        };
        assert_eq!(
            scan_run_of(han, han.as_bytes(), 0, kimi_upper),
            0,
            "Kimi excludes Han from its case-split classes"
        );
    }
}
