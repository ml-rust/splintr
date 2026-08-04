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

/// A parsed SentencePiece precompiled charsmap.
#[derive(Debug, Clone)]
pub struct Precompiled {
    /// darts-clone double-array units.
    trie: Vec<u32>,
    /// Null-terminated normalized replacement strings, concatenated.
    normalized: Vec<u8>,
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
    /// Parse a charsmap blob (the raw bytes of `precompiled_charsmap`).
    ///
    /// Layout: `u32 LE trie_size` · `trie_size` bytes of `u32 LE` trie units ·
    /// remaining bytes = the normalized-strings table.
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
        Some(Self { trie, normalized })
    }

    /// Longest common-prefix match of `key` in the trie, returning the matched
    /// byte length and the value (offset into `normalized`).
    fn longest_prefix(&self, key: &[u8]) -> Option<(usize, u32)> {
        let mut node_pos = 0usize;
        let mut unit = self.trie[0];
        node_pos ^= offset(unit) as usize;
        let mut best: Option<(usize, u32)> = None;
        for (i, &b) in key.iter().enumerate() {
            node_pos ^= b as usize;
            if node_pos >= self.trie.len() {
                break;
            }
            unit = self.trie[node_pos];
            if label(unit) != b as u32 {
                break;
            }
            node_pos ^= offset(unit) as usize;
            if has_leaf(unit) {
                if let Some(&v) = self.trie.get(node_pos) {
                    best = Some((i + 1, value(v)));
                }
            }
        }
        best
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

    /// Normalize `text` by repeatedly replacing the longest matching prefix with
    /// its normalized form (SentencePiece's algorithm).
    pub fn normalize(&self, text: &str) -> String {
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
