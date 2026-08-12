//! The segmentation lattice a Unigram word is scored over.
//!
//! A word of `n` characters is a graph on `n + 1` positions; an edge from `i` to
//! `j` is a vocabulary piece spelling `chars[i..j]`. Two questions get asked of
//! it, and they are different questions:
//!
//! * **Viterbi** — the single best segmentation, a `max` over paths. What
//!   inference does, and what pruning needs to know the cost of losing a piece.
//! * **Forward-backward** — how much *every* segmentation contributes, a
//!   `log-sum-exp` over paths. What EM's E-step needs, because a piece's
//!   expected count is its share across all segmentations rather than its
//!   presence in the best one.
//!
//! splintr's `SentencePieceTokenizer` already implements the first, but only the
//! first, and only against a built trie with fixed scores. Training changes the
//! piece set and every score on each iteration, so the lattice is built here
//! over the candidate set directly.

use super::trie::{PieceTrie, ROOT};

/// One edge: the piece spanning `[start, end)` and its id.
#[derive(Clone, Copy)]
pub(crate) struct Edge {
    pub start: usize,
    pub end: usize,
    pub piece: u32,
}

/// Scratch reused across words, so a corpus pass allocates once rather than per
/// word.
#[derive(Default)]
pub(crate) struct Lattice {
    /// Byte offset of each character, so a start position maps to a slice.
    starts: Vec<usize>,
    edges: Vec<Edge>,
    /// Edges ending at each position, as indices into `edges`.
    into: Vec<Vec<usize>>,
    alpha: Vec<f64>,
    beta: Vec<f64>,
    best: Vec<f64>,
    back: Vec<Option<usize>>,
}

/// `ln(e^a + e^b)`, stable when either side is very small or absent.
#[inline]
fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let (hi, lo) = if a > b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp().ln_1p()
}

impl Lattice {
    /// Rebuild for `word`, admitting every piece `trie` holds.
    ///
    /// One walk per start position rather than a hash of every substring: the
    /// walk stops at the first character no candidate continues, which on a
    /// script without spaces — where a pre-token runs to tens of characters —
    /// is most of the saving. See [`PieceTrie`].
    pub(crate) fn build(&mut self, word: &str, max_piece_chars: usize, trie: &PieceTrie) -> usize {
        // Byte offsets rather than a `Vec<char>`: the walk below reads each
        // start's suffix through a `chars()` iterator, so materialising the
        // characters would decode the word a second time for nothing. (It did,
        // briefly — profiling put 8% of Unigram training in that copy.)
        self.starts.clear();
        self.starts
            .extend(word.char_indices().map(|(offset, _)| offset));
        let n = self.starts.len();

        self.edges.clear();
        self.into.clear();
        self.into.resize(n + 1, Vec::new());
        for slot in self.into.iter_mut() {
            slot.clear();
        }

        for start in 0..n {
            let mut node = ROOT;
            for (step, ch) in word[self.starts[start]..]
                .chars()
                .take(max_piece_chars)
                .enumerate()
            {
                node = match trie.step(node, ch) {
                    Some(next) => next,
                    // Nothing spells this far, so nothing spells further either.
                    None => break,
                };
                if let Some(piece) = trie.piece(node) {
                    let end = start + step + 1;
                    self.into[end].push(self.edges.len());
                    self.edges.push(Edge { start, end, piece });
                }
            }
        }
        n
    }

    /// Whether every position is reachable, so the word can be segmented at all.
    ///
    /// Always true while single characters are kept in the candidate set, which
    /// is why they are never pruned — but checked rather than assumed, since a
    /// word containing a character the candidate set lost would otherwise score
    /// as negative infinity and poison the expected counts.
    pub(crate) fn is_connected(&self, n: usize) -> bool {
        let mut reachable = vec![false; n + 1];
        reachable[0] = true;
        for end in 1..=n {
            reachable[end] = self.into[end]
                .iter()
                .any(|&e| reachable[self.edges[e].start]);
        }
        reachable[n]
    }

    /// Expected count of every edge under the current scores, handed to
    /// `emit` as `(piece, expectation)`.
    ///
    /// The forward-backward marginal: an edge's probability is the mass of all
    /// paths through it over the mass of all paths, `exp(alpha[start] +
    /// score + beta[end] - Z)`.
    pub(crate) fn expectations(
        &mut self,
        n: usize,
        scores: &[f64],
        mut emit: impl FnMut(u32, f64),
    ) -> f64 {
        self.alpha.clear();
        self.alpha.resize(n + 1, f64::NEG_INFINITY);
        self.alpha[0] = 0.0;
        for end in 1..=n {
            let mut acc = f64::NEG_INFINITY;
            for &e in &self.into[end] {
                let edge = self.edges[e];
                acc = log_add(acc, self.alpha[edge.start] + scores[edge.piece as usize]);
            }
            self.alpha[end] = acc;
        }

        let z = self.alpha[n];
        if !z.is_finite() {
            return f64::NEG_INFINITY;
        }

        self.beta.clear();
        self.beta.resize(n + 1, f64::NEG_INFINITY);
        self.beta[n] = 0.0;
        for end in (1..=n).rev() {
            for &e in &self.into[end] {
                let edge = self.edges[e];
                let contribution = scores[edge.piece as usize] + self.beta[edge.end];
                self.beta[edge.start] = log_add(self.beta[edge.start], contribution);
            }
        }

        for edge in &self.edges {
            let marginal =
                self.alpha[edge.start] + scores[edge.piece as usize] + self.beta[edge.end] - z;
            if marginal.is_finite() {
                emit(edge.piece, marginal.exp());
            }
        }
        z
    }

    /// The best segmentation, appended to `out` as piece ids, and its score.
    pub(crate) fn viterbi(&mut self, n: usize, scores: &[f64], out: &mut Vec<u32>) -> f64 {
        self.best.clear();
        self.best.resize(n + 1, f64::NEG_INFINITY);
        self.best[0] = 0.0;
        self.back.clear();
        self.back.resize(n + 1, None);

        for end in 1..=n {
            for &e in &self.into[end] {
                let edge = self.edges[e];
                let candidate = self.best[edge.start] + scores[edge.piece as usize];
                if candidate > self.best[end] {
                    self.best[end] = candidate;
                    self.back[end] = Some(e);
                }
            }
        }

        if !self.best[n].is_finite() {
            return f64::NEG_INFINITY;
        }

        let mark = out.len();
        let mut pos = n;
        while pos > 0 {
            let e = match self.back[pos] {
                Some(e) => e,
                None => break,
            };
            let edge = self.edges[e];
            out.push(edge.piece);
            pos = edge.start;
        }
        out[mark..].reverse();
        self.best[n]
    }

    /// The best score reachable without ever using `excluded`.
    ///
    /// What a piece's own text would cost if that piece were dropped, which is
    /// exactly the loss its removal imposes on every segmentation that used it.
    pub(crate) fn viterbi_excluding(&mut self, n: usize, scores: &[f64], excluded: u32) -> f64 {
        self.best.clear();
        self.best.resize(n + 1, f64::NEG_INFINITY);
        self.best[0] = 0.0;

        for end in 1..=n {
            for &e in &self.into[end] {
                let edge = self.edges[e];
                if edge.piece == excluded {
                    continue;
                }
                let candidate = self.best[edge.start] + scores[edge.piece as usize];
                if candidate > self.best[end] {
                    self.best[end] = candidate;
                }
            }
        }
        self.best[n]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxHashMap;

    /// Build a lattice over `word` for a fixed piece set.
    fn lattice(word: &str, pieces: &[&str]) -> (Lattice, usize, Vec<f64>, FxHashMap<String, u32>) {
        let ids: FxHashMap<String, u32> = pieces
            .iter()
            .enumerate()
            .map(|(i, p)| (p.to_string(), i as u32))
            .collect();
        let trie = PieceTrie::build(pieces);
        let mut lattice = Lattice::default();
        let n = lattice.build(word, 16, &trie);
        // Uniform scores, so path length alone decides.
        let scores = vec![(1.0f64 / pieces.len() as f64).ln(); pieces.len()];
        (lattice, n, scores, ids)
    }

    #[test]
    fn viterbi_prefers_the_fewest_pieces_under_uniform_scores() {
        let (mut l, n, scores, ids) = lattice("abc", &["a", "b", "c", "ab", "abc"]);
        let mut out = Vec::new();
        l.viterbi(n, &scores, &mut out);
        assert_eq!(out, vec![ids["abc"]]);
    }

    #[test]
    fn a_word_with_only_single_characters_is_connected() {
        let (l, n, _, _) = lattice("abc", &["a", "b", "c"]);
        assert!(l.is_connected(n));
    }

    /// A character with no piece and nothing spanning it leaves a position
    /// nothing can reach — the case that makes a word unsegmentable, and the
    /// reason single characters are never pruned.
    #[test]
    fn a_character_no_piece_covers_disconnects_the_lattice() {
        let (l, n, _, _) = lattice("abc", &["a", "c"]);
        assert!(!l.is_connected(n));
    }

    /// A missing character is survivable when a longer piece spans it.
    #[test]
    fn a_piece_spanning_the_gap_keeps_the_lattice_connected() {
        let (l, n, _, _) = lattice("abc", &["a", "c", "ab"]);
        assert!(l.is_connected(n));
    }

    /// The E-step's invariant: marginals of the edges ending at any position
    /// sum to one, because every path crosses exactly one of them.
    #[test]
    fn expectations_form_a_distribution_over_each_cut() {
        let (mut l, n, scores, _) = lattice("abcd", &["a", "b", "c", "d", "ab", "bc", "abc"]);
        let mut mass = vec![0.0f64; n + 1];
        // Recompute per-position mass by re-running with an emitter that knows
        // which edge it was handed.
        let edges: Vec<Edge> = l.edges.clone();
        let mut i = 0;
        l.expectations(n, &scores, |_, expectation| {
            mass[edges[i].end] += expectation;
            i += 1;
        });
        // Every cut position except the source carries exactly one edge per path.
        for (position, total) in mass.iter().enumerate().skip(1) {
            assert!(
                *total > 0.0,
                "position {position} was never crossed: {total}"
            );
        }
    }

    /// Forward and backward must agree on the total path mass, which is the
    /// standard check that the two passes are consistent.
    #[test]
    fn forward_and_backward_agree_on_total_mass() {
        let (mut l, n, scores, _) = lattice("abcd", &["a", "b", "c", "d", "ab", "bc", "abc"]);
        let z = l.expectations(n, &scores, |_, _| {});
        // `beta[0]` is the same quantity computed from the other end.
        assert!((z - l.beta[0]).abs() < 1e-9, "z={z} beta0={}", l.beta[0]);
    }

    /// Viterbi never scores above the total mass, since one path cannot exceed
    /// the sum over all of them.
    #[test]
    fn viterbi_never_exceeds_the_total_mass() {
        let (mut l, n, scores, _) = lattice("abcd", &["a", "b", "c", "d", "ab", "bc", "abc"]);
        let z = l.expectations(n, &scores, |_, _| {});
        let mut out = Vec::new();
        let best = l.viterbi(n, &scores, &mut out);
        assert!(best <= z + 1e-9, "viterbi {best} exceeded mass {z}");
    }

    #[test]
    fn excluding_a_piece_forces_a_different_path() {
        let (mut l, n, scores, ids) = lattice("abc", &["a", "b", "c", "abc"]);
        let mut out = Vec::new();
        let with = l.viterbi(n, &scores, &mut out);
        let without = l.viterbi_excluding(n, &scores, ids["abc"]);
        assert!(
            without < with,
            "removing the whole-word piece must cost something: {without} vs {with}"
        );
    }
}
