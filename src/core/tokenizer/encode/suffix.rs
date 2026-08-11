//! `model.end_of_word_suffix`: BPE over words that carry a word-final marker.

use super::super::types::Tokenizer;
use crate::core::bpe::{byte_pair_encode_pieces_presegmented, Piece, Seed};

impl Tokenizer {
    /// Encode one chunk of a vocabulary that declares `end_of_word_suffix`,
    /// where the unit the vocabulary is keyed in is the word **plus** that
    /// marker.
    ///
    /// HuggingFace's `BPE::merge_word` builds a word one character at a time and
    /// appends the suffix to the last of them *before* any merge runs, so the
    /// word-final symbol is a different vocabulary entry from the same character
    /// mid-word (`o` vs `o</w>`) and every merge above it inherits the
    /// distinction. CLIP's `hello` is therefore `hello</w>` (3306); `hello`
    /// (12887) is a mid-word spelling that no complete word can produce.
    ///
    /// That is why this runs where it does — ahead of the whole-chunk vocabulary
    /// probe rather than inside the merge. The probe answers first for any chunk
    /// the vocabulary contains, and for a suffixed model its answer would be
    /// exactly that unreachable mid-word spelling.
    ///
    /// `bytes` must already be in the space the vocabulary is keyed in (see
    /// [`Tokenizer::encode_chunk_into`]), and the suffix is appended in that same
    /// space: it is ASCII in every file that declares one, and ByteLevel maps
    /// ASCII to itself, so the marker's spelling is its own bytes either way.
    ///
    /// Input that is not valid UTF-8 has no last *character* to mark, so it
    /// takes the unsuffixed path — the same answer, since a merge list keyed by
    /// suffixed spellings can name none of its symbols anyway.
    pub(super) fn encode_suffixed_bytes_into(
        &self,
        bytes: &[u8],
        suffix: &str,
        out: &mut Vec<u32>,
    ) {
        let Ok(text) = std::str::from_utf8(bytes) else {
            return self.encode_unsuffixed_bytes_into(bytes, out);
        };
        if text.is_empty() {
            return;
        }

        crate::core::scratch::with_text(|buf| {
            buf.push_str(text);
            buf.push_str(suffix);
            let marked = buf.as_bytes();

            // The same cache protocol every other chunk takes — whole-chunk
            // vocabulary hit, then cache, then merge, then record — asked of the
            // marked form throughout, which is the only form this vocabulary
            // answers about. Keying the cache on it too keeps the two questions
            // about one chunk keyed alike.
            let hash = crate::core::encoder::Encoder::hash_of(marked);
            if let Some(id) = self.chunk_encoder().get_with_hash(marked, hash) {
                out.push(id);
                return;
            }
            if self.chunk_cache.extend_into(hash, marked, out) {
                return;
            }

            let start = out.len();
            self.bpe_suffixed_into(text, marked, suffix.len(), out);
            self.chunk_cache.put(hash, marked, &out[start..]);
        })
    }

    /// Merge a marked word, seeded one symbol per character with the marker
    /// carried by the last of them.
    ///
    /// The segmentation is the caller's, not one the merge could derive: `o</w>`
    /// is one symbol of six bytes, and nothing about the buffer says where that
    /// symbol begins — hence [`byte_pair_encode_pieces_presegmented`], the same
    /// entry point HuggingFace's byte-fallback order uses for the same reason.
    ///
    /// The seeds carry no ids, so every symbol resolves through the vocabulary
    /// by its own surface. A symbol the vocabulary does not have is dropped,
    /// which is this crate's contract for every vocabulary that declares no byte
    /// fallback — and a ByteLevel model that declares a suffix has an entry for
    /// each of its 256 alphabet characters both bare and marked, so there is
    /// nothing to drop.
    fn bpe_suffixed_into(&self, text: &str, marked: &[u8], suffix_len: usize, out: &mut Vec<u32>) {
        let mut seeds: Vec<Seed> = text
            .char_indices()
            .map(|(start, c)| Seed {
                start,
                len: c.len_utf8(),
                id: None,
            })
            .collect();
        // The marker belongs to the last character, not beside it: HuggingFace
        // looks the two up as one string.
        if let Some(last) = seeds.last_mut() {
            last.len += suffix_len;
        }

        let pieces =
            byte_pair_encode_pieces_presegmented(marked, &seeds, self.rank_lookup(), &self.encoder);
        out.reserve(pieces.len());
        out.extend(pieces.into_iter().filter_map(|piece| match piece {
            Piece::Token(id) => Some(id),
            Piece::Unresolved { .. } => None,
        }));
    }
}
