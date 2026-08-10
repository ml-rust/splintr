//! SentencePiece `Precompiled` charsmap normalizer.
//!
//! SentencePiece ships its normalization rules as a precompiled blob: a
//! darts-clone double-array trie mapping input byte sequences to offsets into a
//! concatenated, null-terminated table of normalized replacement strings. This
//! is what HuggingFace stores as `normalizer.precompiled_charsmap`.
//!
//! Matching it exactly (rather than approximating with NFKC) is required for
//! byte-identical parity on the long tail of Unicode (CJK-compatibility,
//! astral-plane, and other characters where the charsmap and NFKC disagree).

/// Which implementation's reading of the charsmap to reproduce.
///
/// The blob is the same; the two references walk it differently and disagree.
/// Measured on `t5-base`, `"¹\u{fe0f}"` normalizes to `"1"` under HuggingFace
/// and `"1\u{fe0f}"` under sentencepiece — a different id either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CharsmapDialect {
    /// sentencepiece's `Normalizer::NormalizePrefix`: at each byte position take
    /// the **longest** matching rule and consume the bytes it matched.
    SentencePiece,
    /// HuggingFace `tokenizers`, via `spm_precompiled`: per grapheme cluster,
    /// take the **shortest** rule matching any prefix of a cluster under 6 bytes
    /// and replace the *whole cluster* with it; only a cluster with no rule at
    /// all falls back to a per-character pass. Replacing on a prefix match is
    /// deliberate upstream, and is why the variation selector above disappears.
    HuggingFace,
}

/// A parsed SentencePiece precompiled charsmap.
#[derive(Debug, Clone)]
pub struct Precompiled {
    /// darts-clone double-array units.
    trie: Vec<u32>,
    /// Null-terminated normalized replacement strings, concatenated.
    normalized: Vec<u8>,
    /// Which reference implementation [`normalize`](Self::normalize) reproduces.
    dialect: CharsmapDialect,
}

// darts-clone unit accessors.
#[inline]
fn has_leaf(unit: u32) -> bool {
    ((unit >> 8) & 1) == 1
}
#[inline]
fn value(unit: u32) -> u32 {
    unit & 0x7fff_ffff
}
#[inline]
fn label(unit: u32) -> u32 {
    unit & 0x8000_00ff
}
#[inline]
fn offset(unit: u32) -> u32 {
    (unit >> 10) << ((unit & (1 << 9)) >> 6)
}

impl Precompiled {
    /// Parse a charsmap blob (the raw bytes of `precompiled_charsmap`) as
    /// sentencepiece itself reads it.
    ///
    /// Layout: `u32 LE trie_size` · `trie_size` bytes of `u32 LE` trie units ·
    /// remaining bytes = the normalized-strings table.
    ///
    /// A charsmap taken out of a `tokenizer.json` wants
    /// [`CharsmapDialect::HuggingFace`] instead — see [`with_dialect`](Self::with_dialect).
    pub fn from_bytes(blob: &[u8]) -> Option<Self> {
        if blob.len() < 4 {
            return None;
        }
        let trie_size = u32::from_le_bytes([blob[0], blob[1], blob[2], blob[3]]) as usize;
        let trie_end = 4 + trie_size;
        // `trie_size == 0` is rejected alongside the malformed cases: the
        // matcher starts from the root unit, so a table with no root has no
        // usable entry point. Accepting it would build a `Precompiled` that
        // panics on the first non-empty input instead of one that normalizes
        // nothing.
        if trie_end > blob.len() || trie_size == 0 || !trie_size.is_multiple_of(4) {
            return None;
        }
        let trie = blob[4..trie_end]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let normalized = blob[trie_end..].to_vec();
        Some(Self {
            trie,
            normalized,
            dialect: CharsmapDialect::SentencePiece,
        })
    }

    /// Choose which reference implementation [`normalize`](Self::normalize)
    /// reproduces. Defaults to [`CharsmapDialect::SentencePiece`].
    pub fn with_dialect(mut self, dialect: CharsmapDialect) -> Self {
        self.dialect = dialect;
        self
    }

    /// darts-clone's `commonPrefixSearch`: call `hit` with the matched byte
    /// length and value at every prefix carrying a rule, shortest first, until
    /// it returns false. The two dialects want different entries out of it.
    fn common_prefix_search(&self, key: &[u8], mut hit: impl FnMut(usize, u32) -> bool) {
        let mut node_pos = 0usize;
        let mut unit = self.trie[0];
        node_pos ^= offset(unit) as usize;
        for (i, &b) in key.iter().enumerate() {
            // A NUL byte terminates an entry in the normalized table, so the
            // reference walk stops there rather than following it into the trie.
            if b == 0 {
                return;
            }
            node_pos ^= b as usize;
            if node_pos >= self.trie.len() {
                return;
            }
            unit = self.trie[node_pos];
            if label(unit) != b as u32 {
                return;
            }
            node_pos ^= offset(unit) as usize;
            if has_leaf(unit) {
                if let Some(&v) = self.trie.get(node_pos) {
                    if !hit(i + 1, value(v)) {
                        return;
                    }
                }
            }
        }
    }

    /// Longest common-prefix match of `key`, returning the matched byte length
    /// and the value — sentencepiece's `NormalizePrefix`.
    fn longest_prefix(&self, key: &[u8]) -> Option<(usize, u32)> {
        let mut best = None;
        self.common_prefix_search(key, |len, v| {
            best = Some((len, v));
            true
        });
        best
    }

    /// `spm_precompiled`'s `transform`: the replacement under the **shortest**
    /// rule matching a prefix of `chunk`. The match may be shorter than `chunk`;
    /// the caller replaces `chunk` entirely regardless.
    fn transform(&self, chunk: &str) -> Option<&[u8]> {
        let mut first = None;
        self.common_prefix_search(chunk.as_bytes(), |_, v| {
            first = Some(v);
            false
        });
        first.map(|v| self.normalized_at(v))
    }

    /// The null-terminated normalized string at `offset`.
    fn normalized_at(&self, offset: u32) -> &[u8] {
        let start = offset as usize;
        let mut end = start;
        while end < self.normalized.len() && self.normalized[end] != 0 {
            end += 1;
        }
        &self.normalized[start..end]
    }

    /// Normalize `text` under this charsmap's [`CharsmapDialect`].
    pub fn normalize(&self, text: &str) -> String {
        match self.dialect {
            CharsmapDialect::SentencePiece => self.normalize_spm(text),
            CharsmapDialect::HuggingFace => self.normalize_hf(text),
        }
    }

    /// `spm_precompiled::Precompiled::normalize_string`.
    fn normalize_hf(&self, text: &str) -> String {
        use unicode_segmentation::UnicodeSegmentation;
        let mut out = String::with_capacity(text.len());
        for grapheme in text.graphemes(true) {
            // The `< 6` bound is upstream's: a longer cluster goes straight to
            // the per-character pass even when a rule for it exists.
            if grapheme.len() < 6 {
                if let Some(norm) = self.transform(grapheme) {
                    out.push_str(&String::from_utf8_lossy(norm));
                    continue;
                }
            }
            for c in grapheme.chars() {
                let mut buf = [0u8; 4];
                match self.transform(c.encode_utf8(&mut buf)) {
                    Some(norm) => out.push_str(&String::from_utf8_lossy(norm)),
                    None => out.push(c),
                }
            }
        }
        out
    }

    /// sentencepiece's `Normalizer::Normalize`: repeatedly replace the longest
    /// matching prefix at the current byte position.
    fn normalize_spm(&self, text: &str) -> String {
        let bytes = text.as_bytes();
        let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            match self.longest_prefix(&bytes[i..]) {
                Some((len, off)) if len > 0 => {
                    out.extend_from_slice(self.normalized_at(off));
                    i += len;
                }
                _ => {
                    // No rule: copy one whole UTF-8 character verbatim.
                    let ch_len = utf8_len(bytes[i]);
                    let end = (i + ch_len).min(bytes.len());
                    out.extend_from_slice(&bytes[i..end]);
                    i = end;
                }
            }
        }
        // The table is valid UTF-8 by construction; fall back lossily if not.
        String::from_utf8(out)
            .unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
    }
}

/// Byte length of the UTF-8 character starting with `first`.
///
/// Shared with the BPE byte-fallback path, which walks an unresolved span a
/// character at a time (see `Tokenizer::bpe`).
#[inline]
pub(super) fn utf8_len(first: u8) -> usize {
    if first < 0x80 {
        1
    } else if first >> 5 == 0b110 {
        2
    } else if first >> 4 == 0b1110 {
        3
    } else if first >> 3 == 0b11110 {
        4
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal darts-clone charsmap carrying one rule: `A` → `1`.
    ///
    /// Same construction as `gguf::tests::tab_to_space_charsmap` — root at
    /// offset 1, one labelled unit reached as `1 ^ b`, and a zero leaf pointing
    /// at offset 0 of the replacement table.
    fn a_to_one() -> Precompiled {
        let mut trie = [0u32; 80];
        trie[0] = 1 << 10;
        trie[(1 ^ b'A') as usize] = (1 << 10) | 0x100 | b'A' as u32;

        let mut blob = Vec::new();
        blob.extend_from_slice(&((trie.len() * 4) as u32).to_le_bytes());
        for unit in trie {
            blob.extend_from_slice(&unit.to_le_bytes());
        }
        blob.extend_from_slice(b"1\0");
        Precompiled::from_bytes(&blob).expect("hand-built charsmap parses")
    }

    #[test]
    fn both_dialects_agree_on_a_bare_rule() {
        let spm = a_to_one();
        let hf = a_to_one().with_dialect(CharsmapDialect::HuggingFace);
        assert_eq!(spm.normalize("A"), "1");
        assert_eq!(hf.normalize("A"), "1");
        assert_eq!(spm.normalize("xAy"), "x1y");
        assert_eq!(hf.normalize("xAy"), "x1y");
    }

    /// The divergence this dialect exists for: a rule matching a *prefix* of a
    /// grapheme cluster replaces the whole cluster under HuggingFace, taking the
    /// combining mark with it. sentencepiece consumes only the matched bytes.
    ///
    /// This is the synthetic form of `"¹\u{fe0f}"` on `t5-base`, which
    /// `tokenizers` 0.22.1 normalizes to `"1"`.
    #[test]
    fn huggingface_replaces_a_whole_grapheme_cluster() {
        assert_eq!(a_to_one().normalize("A\u{301}"), "1\u{301}");
        assert_eq!(
            a_to_one()
                .with_dialect(CharsmapDialect::HuggingFace)
                .normalize("A\u{301}"),
            "1"
        );
    }

    /// A cluster of 6 bytes or more skips the cluster pass entirely, so the mark
    /// survives even though the rule still fires per character.
    #[test]
    fn huggingface_falls_back_per_character_on_a_long_cluster() {
        let hf = a_to_one().with_dialect(CharsmapDialect::HuggingFace);
        assert_eq!(hf.normalize("A\u{301}\u{301}"), "1");
        assert_eq!(
            hf.normalize("A\u{301}\u{301}\u{301}"),
            "1\u{301}\u{301}\u{301}"
        );
    }
}
