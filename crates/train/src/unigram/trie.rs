//! The candidate set as a trie, so a lattice is built by walking rather than by
//! hashing every substring.
//!
//! Building a lattice asks, for each start position, which pieces begin there.
//! Answered with a hash lookup per `(start, end)` pair that is `O(length)` in
//! the substring hashed and cannot stop early, the whole build is
//! `O(n · L²)` bytes hashed for a word of `n` characters and a maximum piece
//! length `L` — and it pays the full `L` lookups at every position even where
//! no piece extends past the first character.
//!
//! Walking a trie makes each step a single hash of `(node, char)`, which is
//! eight bytes whatever the piece length, and stops at the first character no
//! piece continues. That is `O(n · L)` steps with a small constant, and on text
//! with long pre-tokens — Chinese and Thai, where a "word" runs to tens of
//! characters because the script has no spaces — the early exit is most of the
//! win.
//!
//! Keyed by `(node, char)` in one map rather than a child map per node: a
//! candidate pool runs to a million pieces, and a `HashMap` per node would cost
//! more in headers than in edges.

use rustc_hash::FxHashMap;

/// The node every walk starts from.
pub(crate) const ROOT: u32 = 0;

/// Candidate pieces, indexed for prefix walking.
pub(crate) struct PieceTrie {
    edges: FxHashMap<(u32, char), u32>,
    /// Nodes that complete a piece, and which piece.
    terminal: FxHashMap<u32, u32>,
    nodes: u32,
}

impl PieceTrie {
    /// Index `pieces`, where each entry's position is its id.
    pub(crate) fn build<S: AsRef<str>>(pieces: &[S]) -> Self {
        let mut trie = Self {
            edges: FxHashMap::default(),
            terminal: FxHashMap::default(),
            nodes: 1,
        };
        for (id, piece) in pieces.iter().enumerate() {
            let mut node = ROOT;
            for ch in piece.as_ref().chars() {
                node = match trie.edges.get(&(node, ch)) {
                    Some(&next) => next,
                    None => {
                        let next = trie.nodes;
                        trie.nodes += 1;
                        trie.edges.insert((node, ch), next);
                        next
                    }
                };
            }
            // The root is only terminal for an empty piece, which is not a
            // candidate; every other node belongs to exactly one spelling.
            if node != ROOT {
                trie.terminal.insert(node, id as u32);
            }
        }
        trie
    }

    /// Follow one character, or `None` when no piece continues this way.
    #[inline]
    pub(crate) fn step(&self, node: u32, ch: char) -> Option<u32> {
        self.edges.get(&(node, ch)).copied()
    }

    /// The piece ending at `node`, if one does.
    #[inline]
    pub(crate) fn piece(&self, node: u32) -> Option<u32> {
        self.terminal.get(&node).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn walk(trie: &PieceTrie, text: &str) -> Option<u32> {
        let mut node = ROOT;
        for ch in text.chars() {
            node = trie.step(node, ch)?;
        }
        trie.piece(node)
    }

    #[test]
    fn finds_every_piece_it_was_built_from() {
        let pieces = ["a", "ab", "abc", "b", "日本", "日本語"];
        let trie = PieceTrie::build(&pieces);
        for (id, piece) in pieces.iter().enumerate() {
            assert_eq!(walk(&trie, piece), Some(id as u32), "{piece:?}");
        }
    }

    #[test]
    fn a_prefix_that_is_not_a_piece_is_not_terminal() {
        let trie = PieceTrie::build(&["abc"]);
        assert!(walk(&trie, "a").is_none());
        assert!(walk(&trie, "ab").is_none());
        assert_eq!(walk(&trie, "abc"), Some(0));
    }

    /// The early exit the whole structure exists for: a character no piece
    /// continues with stops the walk instead of costing another lookup per
    /// remaining length.
    #[test]
    fn an_unknown_continuation_stops_the_walk() {
        let trie = PieceTrie::build(&["abc"]);
        let a = trie.step(ROOT, 'a').expect("a is a prefix");
        assert!(trie.step(a, 'z').is_none());
    }

    /// A multi-byte character is one step, not one per byte — the property that
    /// keeps CJK from being three times the depth.
    #[test]
    fn a_multibyte_character_is_a_single_step() {
        let trie = PieceTrie::build(&["語"]);
        let node = trie.step(ROOT, '語').expect("one step");
        assert_eq!(trie.piece(node), Some(0));
    }

    #[test]
    fn an_empty_trie_matches_nothing() {
        let empty: [&str; 0] = [];
        let trie = PieceTrie::build(&empty);
        assert!(trie.step(ROOT, 'a').is_none());
        assert!(trie.piece(ROOT).is_none());
    }
}
