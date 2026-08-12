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
//!
//! # Layout
//!
//! Both questions walk the edges *ending at* each position, so the edges are
//! stored sorted by their end with an offset per position — the compressed
//! sparse row layout. The obvious alternative, a `Vec` of edge lists indexed by
//! position, was measured and is worse: it allocates per position per word, and
//! since a corpus pass visits hundreds of thousands of words several times per
//! pruning round, the allocator dominated what should be a scan.
//!
//! That layout is also what makes [`LatticeCache`] worth having. Edges depend
//! only on the word and the candidate set, so within a pruning round they can be
//! built once and replayed; with CSR a replay is two slices, with edge lists it
//! was a rebuild.

use super::trie::{PieceTrie, ROOT};

/// One edge: the piece spanning `[start, end)` and its id.
///
/// `u32` positions rather than `usize`: these are character offsets inside one
/// pre-token, which never approaches four billion, and the cache holds millions
/// of them.
#[derive(Clone, Copy)]
pub(crate) struct Edge {
    pub start: u32,
    pub end: u32,
    pub piece: u32,
}

/// One word's edges, borrowed: the CSR pair plus the character count.
#[derive(Clone, Copy)]
pub(crate) struct LatticeView<'a> {
    edges: &'a [Edge],
    /// `n + 2` offsets; edges ending at `p` are `edges[ends[p]..ends[p + 1]]`.
    ends: &'a [u32],
    pub n: usize,
}

impl<'a> LatticeView<'a> {
    /// Edges ending at `position`.
    #[inline]
    fn ending_at(&self, position: usize) -> &'a [Edge] {
        let from = self.ends[position] as usize;
        let to = self.ends[position + 1] as usize;
        &self.edges[from..to]
    }

    /// Every edge, in no particular order.
    #[inline]
    fn all(&self) -> &'a [Edge] {
        self.edges
    }

    /// Whether every position is reachable, so the word can be segmented at all.
    ///
    /// Always true while single characters are kept in the candidate set, which
    /// is why they are never pruned — but checked rather than assumed, since a
    /// word containing a character the candidate set lost would otherwise score
    /// as negative infinity and poison the expected counts.
    pub(crate) fn is_connected(&self) -> bool {
        let mut reachable = vec![false; self.n + 1];
        reachable[0] = true;
        for end in 1..=self.n {
            reachable[end] = self
                .ending_at(end)
                .iter()
                .any(|edge| reachable[edge.start as usize]);
        }
        reachable[self.n]
    }
}

/// Every word's edges, built once and replayed for as long as the candidate set
/// is unchanged.
///
/// Within a pruning round the piece set is fixed and only the scores move, yet
/// each round runs the corpus three times — once per EM iteration and once for
/// the loss pass. Building once and replaying is the same work done a third as
/// often.
///
/// One flat arena with a range per word, not a `Vec` per word: a corpus has
/// hundreds of thousands of words, and a per-word allocation each would cost
/// more in headers and indirection than in edges.
#[derive(Default)]
pub(crate) struct LatticeCache {
    edges: Vec<Edge>,
    ends: Vec<u32>,
    /// Per word: where its edges and its offsets start, and its length.
    spans: Vec<Span>,
}

#[derive(Clone, Copy)]
struct Span {
    edges_from: u32,
    edges_to: u32,
    ends_from: u32,
    n: u32,
}

impl LatticeCache {
    pub(crate) fn clear(&mut self) {
        self.edges.clear();
        self.ends.clear();
        self.spans.clear();
    }

    /// Record what [`Lattice::build`] just produced, in word order.
    pub(crate) fn push(&mut self, lattice: &LatticeBuilder) {
        let edges_from = self.edges.len() as u32;
        self.edges.extend_from_slice(&lattice.edges);
        let ends_from = self.ends.len() as u32;
        self.ends.extend_from_slice(&lattice.ends);
        self.spans.push(Span {
            edges_from,
            edges_to: self.edges.len() as u32,
            ends_from,
            n: lattice.n as u32,
        });
    }

    /// The lattice recorded for word `index`.
    #[inline]
    pub(crate) fn view(&self, index: usize) -> LatticeView<'_> {
        let span = self.spans[index];
        let n = span.n as usize;
        LatticeView {
            edges: &self.edges[span.edges_from as usize..span.edges_to as usize],
            ends: &self.ends[span.ends_from as usize..span.ends_from as usize + n + 2],
            n,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.spans.len()
    }

    /// Drop the edges of pruned pieces and renumber the rest.
    ///
    /// `remap` maps each old piece id to its new one, or [`u32::MAX`] for a
    /// piece that was pruned.
    ///
    /// A pruning round only ever *removes* candidates, so the next round's
    /// lattices are a subset of this round's — filtering them is a linear pass
    /// over the edges where rebuilding walks the trie again for every start
    /// position of every word, which profiling put at a fifth of a training run.
    /// The result is identical either way: an edge exists exactly when its piece
    /// is still a candidate, and both paths keep the edges grouped by end in
    /// their original order.
    ///
    /// The offsets array is untouched in length — a word contributes `n + 2`
    /// entries whatever its edges do — so only the edges compact, and they
    /// compact leftwards into space this same pass has already freed.
    pub(crate) fn retain(&mut self, remap: &[u32]) {
        let mut write = 0usize;
        for span in &mut self.spans {
            let n = span.n as usize;
            let ends_from = span.ends_from as usize;
            let edges_from = span.edges_from as usize;
            span.edges_from = write as u32;

            let mut kept = 0u32;
            for position in 0..=n {
                let from = self.ends[ends_from + position] as usize;
                let to = self.ends[ends_from + position + 1] as usize;
                // Rewrite this position's offset only after reading it; the next
                // position's is still needed and is rewritten on its own turn.
                self.ends[ends_from + position] = kept;
                for offset in from..to {
                    let edge = self.edges[edges_from + offset];
                    let piece = remap[edge.piece as usize];
                    if piece != u32::MAX {
                        self.edges[write] = Edge { piece, ..edge };
                        write += 1;
                        kept += 1;
                    }
                }
            }
            self.ends[ends_from + n + 1] = kept;
            span.edges_to = write as u32;
        }
        self.edges.truncate(write);
    }
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

/// Builds a word's CSR edge arrays; reused across words so a corpus pass
/// allocates once rather than per word.
///
/// Deliberately separate from [`Lattice`], which owns the *algorithm* scratch.
/// A [`LatticeView`] borrows the builder, and the algorithms need their own
/// scratch mutably — one struct holding both would make every call site fight
/// the borrow checker for no reason.
#[derive(Default)]
pub(crate) struct LatticeBuilder {
    /// Byte offset of each character, so a start position maps to a slice.
    starts: Vec<usize>,
    /// Build output in start order, before the sort by end.
    scratch: Vec<Edge>,
    edges: Vec<Edge>,
    ends: Vec<u32>,
    /// Placement cursors for the counting sort.
    cursor: Vec<u32>,
    n: usize,
}

/// Scratch for the two algorithms, reused across words.
#[derive(Default)]
pub(crate) struct Lattice {
    alpha: Vec<f64>,
    beta: Vec<f64>,
    best: Vec<f64>,
    back: Vec<Option<Edge>>,
}

impl LatticeBuilder {
    /// Rebuild for `word`, admitting every piece `trie` holds.
    ///
    /// One walk per start position rather than a hash of every substring: the
    /// walk stops at the first character no candidate continues, which on a
    /// script without spaces — where a pre-token runs to tens of characters —
    /// is most of the saving. See [`PieceTrie`].
    pub(crate) fn build(&mut self, word: &str, max_piece_chars: usize, trie: &PieceTrie) {
        // Byte offsets rather than a `Vec<char>`: the walk below reads each
        // start's suffix through a `chars()` iterator, so materialising the
        // characters would decode the word a second time for nothing.
        self.starts.clear();
        self.starts
            .extend(word.char_indices().map(|(offset, _)| offset));
        let n = self.starts.len();
        self.n = n;

        self.scratch.clear();
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
                    self.scratch.push(Edge {
                        start: start as u32,
                        end: (start + step + 1) as u32,
                        piece,
                    });
                }
            }
        }

        // Counting sort by end into the CSR arrays.
        self.ends.clear();
        self.ends.resize(n + 2, 0);
        for edge in &self.scratch {
            self.ends[edge.end as usize + 1] += 1;
        }
        for position in 1..self.ends.len() {
            self.ends[position] += self.ends[position - 1];
        }
        self.cursor.clear();
        self.cursor.extend_from_slice(&self.ends);
        self.edges.clear();
        self.edges.resize(
            self.scratch.len(),
            Edge {
                start: 0,
                end: 0,
                piece: 0,
            },
        );
        for edge in &self.scratch {
            let slot = &mut self.cursor[edge.end as usize];
            self.edges[*slot as usize] = *edge;
            *slot += 1;
        }
    }

    /// A view over what [`build`](Self::build) just produced.
    #[inline]
    pub(crate) fn view(&self) -> LatticeView<'_> {
        LatticeView {
            edges: &self.edges,
            ends: &self.ends,
            n: self.n,
        }
    }
}

impl Lattice {
    /// Expected count of every edge under the current scores, handed to `emit`
    /// as `(piece, expectation)`.
    ///
    /// The forward-backward marginal: an edge's probability is the mass of all
    /// paths through it over the mass of all paths, `exp(alpha[start] + score +
    /// beta[end] - Z)`.
    pub(crate) fn expectations(
        &mut self,
        view: LatticeView<'_>,
        scores: &[f64],
        mut emit: impl FnMut(u32, f64),
    ) -> f64 {
        let n = view.n;
        self.alpha.clear();
        self.alpha.resize(n + 1, f64::NEG_INFINITY);
        self.alpha[0] = 0.0;
        // Pairwise rather than the textbook max-then-sum. Measured: with a
        // shifted maximum this pass costs 2% *more* instructions, because these
        // lattices are sparse — a handful of edges per position, where
        // `k exp + 1 ln` beats `(k-1) exp + (k-1) ln_1p` only once k is large,
        // and the edge list has to be read twice to find the maximum first.
        for end in 1..=n {
            let mut acc = f64::NEG_INFINITY;
            for edge in view.ending_at(end) {
                acc = log_add(
                    acc,
                    self.alpha[edge.start as usize] + scores[edge.piece as usize],
                );
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
            for edge in view.ending_at(end) {
                let contribution = scores[edge.piece as usize] + self.beta[edge.end as usize];
                self.beta[edge.start as usize] =
                    log_add(self.beta[edge.start as usize], contribution);
            }
        }

        for edge in view.all() {
            let marginal = self.alpha[edge.start as usize]
                + scores[edge.piece as usize]
                + self.beta[edge.end as usize]
                - z;
            if marginal.is_finite() {
                emit(edge.piece, marginal.exp());
            }
        }
        z
    }

    /// The best segmentation, appended to `out` as piece ids, and its score.
    pub(crate) fn viterbi(
        &mut self,
        view: LatticeView<'_>,
        scores: &[f64],
        out: &mut Vec<u32>,
    ) -> f64 {
        let n = view.n;
        self.best.clear();
        self.best.resize(n + 1, f64::NEG_INFINITY);
        self.best[0] = 0.0;
        self.back.clear();
        self.back.resize(n + 1, None);

        for end in 1..=n {
            for edge in view.ending_at(end) {
                let candidate = self.best[edge.start as usize] + scores[edge.piece as usize];
                if candidate > self.best[end] {
                    self.best[end] = candidate;
                    self.back[end] = Some(*edge);
                }
            }
        }

        if !self.best[n].is_finite() {
            return f64::NEG_INFINITY;
        }

        let mark = out.len();
        let mut pos = n;
        while pos > 0 {
            let edge = match self.back[pos] {
                Some(edge) => edge,
                None => break,
            };
            out.push(edge.piece);
            pos = edge.start as usize;
        }
        out[mark..].reverse();
        self.best[n]
    }

    /// The best score reachable without ever using `excluded`.
    ///
    /// What a piece's own text would cost if that piece were dropped, which is
    /// exactly the loss its removal imposes on every segmentation that used it.
    pub(crate) fn viterbi_excluding(
        &mut self,
        view: LatticeView<'_>,
        scores: &[f64],
        excluded: u32,
    ) -> f64 {
        let n = view.n;
        self.best.clear();
        self.best.resize(n + 1, f64::NEG_INFINITY);
        self.best[0] = 0.0;

        for end in 1..=n {
            for edge in view.ending_at(end) {
                if edge.piece == excluded {
                    continue;
                }
                let candidate = self.best[edge.start as usize] + scores[edge.piece as usize];
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
    fn lattice(word: &str, pieces: &[&str]) -> (LatticeBuilder, Vec<f64>, FxHashMap<String, u32>) {
        let ids: FxHashMap<String, u32> = pieces
            .iter()
            .enumerate()
            .map(|(i, p)| (p.to_string(), i as u32))
            .collect();
        let trie = PieceTrie::build(pieces);
        let mut builder = LatticeBuilder::default();
        builder.build(word, 16, &trie);
        // Uniform scores, so path length alone decides.
        let scores = vec![(1.0f64 / pieces.len() as f64).ln(); pieces.len()];
        (builder, scores, ids)
    }

    /// Every edge must land in the slot its end names — the invariant the whole
    /// CSR layout rests on.
    #[test]
    fn edges_are_indexed_by_the_position_they_end_at() {
        let (b, _, _) = lattice("abcd", &["a", "b", "c", "d", "ab", "bc", "abc"]);
        let view = b.view();
        for end in 1..=view.n {
            for edge in view.ending_at(end) {
                assert_eq!(edge.end as usize, end, "edge filed under the wrong end");
            }
        }
        let indexed: usize = (1..=view.n).map(|e| view.ending_at(e).len()).sum();
        assert_eq!(
            indexed,
            view.all().len(),
            "every edge is indexed exactly once"
        );
    }

    #[test]
    fn viterbi_prefers_the_fewest_pieces_under_uniform_scores() {
        let (b, scores, ids) = lattice("abc", &["a", "b", "c", "ab", "abc"]);
        let mut l = Lattice::default();
        let mut out = Vec::new();
        l.viterbi(b.view(), &scores, &mut out);
        assert_eq!(out, vec![ids["abc"]]);
    }

    #[test]
    fn a_word_with_only_single_characters_is_connected() {
        let (b, _, _) = lattice("abc", &["a", "b", "c"]);
        assert!(b.view().is_connected());
    }

    /// A character with no piece and nothing spanning it leaves a position
    /// nothing can reach — the case that makes a word unsegmentable, and the
    /// reason single characters are never pruned.
    #[test]
    fn a_character_no_piece_covers_disconnects_the_lattice() {
        let (b, _, _) = lattice("abc", &["a", "c"]);
        assert!(!b.view().is_connected());
    }

    /// A missing character is survivable when a longer piece spans it.
    #[test]
    fn a_piece_spanning_the_gap_keeps_the_lattice_connected() {
        let (b, _, _) = lattice("abc", &["a", "c", "ab"]);
        assert!(b.view().is_connected());
    }

    /// Forward and backward must agree on the total path mass, which is the
    /// standard check that the two passes are consistent.
    #[test]
    fn forward_and_backward_agree_on_total_mass() {
        let (b, scores, _) = lattice("abcd", &["a", "b", "c", "d", "ab", "bc", "abc"]);
        let mut l = Lattice::default();
        let z = l.expectations(b.view(), &scores, |_, _| {});
        assert!((z - l.beta[0]).abs() < 1e-9, "z={z} beta0={}", l.beta[0]);
    }

    /// Viterbi never scores above the total mass, since one path cannot exceed
    /// the sum over all of them.
    #[test]
    fn viterbi_never_exceeds_the_total_mass() {
        let (b, scores, _) = lattice("abcd", &["a", "b", "c", "d", "ab", "bc", "abc"]);
        let mut l = Lattice::default();
        let z = l.expectations(b.view(), &scores, |_, _| {});
        let mut out = Vec::new();
        let best = l.viterbi(b.view(), &scores, &mut out);
        assert!(best <= z + 1e-9, "viterbi {best} exceeded mass {z}");
    }

    #[test]
    fn excluding_a_piece_forces_a_different_path() {
        let (b, scores, ids) = lattice("abc", &["a", "b", "c", "abc"]);
        let mut l = Lattice::default();
        let mut out = Vec::new();
        let with = l.viterbi(b.view(), &scores, &mut out);
        let without = l.viterbi_excluding(b.view(), &scores, ids["abc"]);
        assert!(
            without < with,
            "removing the whole-word piece must cost something: {without} vs {with}"
        );
    }
}
