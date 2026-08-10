//! A byte trie over a vocabulary, laid out for prefix search.
//!
//! Both backends that match surfaces against a position in the text — WordPiece's
//! greedy longest match and Unigram's lattice — otherwise ask a hash map "is
//! `text[start..end]` a token?" once per candidate `end`, re-hashing a longer and
//! longer prefix each time. That is quadratic in the length of the longest piece,
//! for an answer a single forward walk gives in linear time.
//!
//! # Why a double array
//!
//! The obvious layout stores each node's edges contiguously and searches them,
//! which makes a transition cost a binary search over the node's fan-out. That
//! search is the walk: Unigram restarts it at every character of every word it
//! has not already cached, so it is where the lattice's time goes.
//!
//! A double array makes the transition arithmetic instead. Every state owns a
//! `base`, and its child on byte `b` lives at `base ^ b` — so all of a state's
//! children fall in one 256-slot block and the transition is an XOR, a load and
//! a compare, whatever the fan-out. `check` says which state a slot belongs to,
//! which is what distinguishes a real edge from a slot some other state happens
//! to own; without it the XOR would answer for edges that do not exist.
//!
//! The cost is construction: the slots a state needs must all be free at the
//! same offset, so building searches for a `base` that fits. That is paid once
//! per vocabulary, against a walk paid once per character of every document.

use std::collections::VecDeque;

/// Marks a node that is not itself a token.
pub(crate) const NO_TOKEN: u32 = u32::MAX;

/// The state a walk starts from.
pub(crate) const ROOT: u32 = 0;

/// Marks a slot no state owns. No real state can be this, so it can never be
/// mistaken for a parent.
const FREE: u32 = u32::MAX;

/// Marks the root's own slot as taken. Distinct from [`FREE`] so allocation
/// skips it, and distinct from every state index so no transition can conclude
/// the root is its own child.
const ROOT_OWNED: u32 = u32::MAX - 1;

pub(crate) struct ByteTrie {
    /// Where each state's children start, in the XOR sense: the child on byte
    /// `b` is at `base[state] ^ b`.
    base: Vec<u32>,
    /// Which state owns each slot, or [`FREE`].
    check: Vec<u32>,
    /// Token id at each slot, or [`NO_TOKEN`] where the state spells no token.
    values: Vec<u32>,
}

impl ByteTrie {
    /// Build from `(surface, id)` pairs, where a surface is the bytes to match
    /// with any continuation prefix already removed.
    pub(crate) fn build<'a>(entries: impl Iterator<Item = (&'a str, u32)>) -> Self {
        // Grown as an adjacency list first: the double array needs a node's
        // whole label set at once to place it, and that is not known until its
        // last surface has been inserted.
        let mut children: Vec<Vec<(u8, u32)>> = vec![Vec::new()];
        let mut node_value = vec![NO_TOKEN];

        for (surface, id) in entries {
            if surface.is_empty() {
                continue;
            }
            let mut node = 0usize;
            for &byte in surface.as_bytes() {
                node = match children[node].iter().find(|(label, _)| *label == byte) {
                    Some(&(_, next)) => next as usize,
                    None => {
                        children.push(Vec::new());
                        node_value.push(NO_TOKEN);
                        let next = children.len() - 1;
                        children[node].push((byte, next as u32));
                        next
                    }
                };
            }
            // Last writer wins, matching the `token_to_id` maps these tries
            // stand in for: those are filled with `insert` in ascending id
            // order, so a surface appearing twice resolves to its *higher* id.
            node_value[node] = id;
        }

        let mut trie = Self {
            base: vec![0; 256],
            check: vec![FREE; 256],
            values: vec![NO_TOKEN; 256],
        };
        trie.check[ROOT as usize] = ROOT_OWNED;
        trie.values[ROOT as usize] = node_value[0];

        // Breadth-first, so a state is placed before the children it will own —
        // and so the sweep of free slots below stays near the front of the
        // array instead of restarting from it for every node.
        let mut slot_of = vec![0u32; children.len()];
        let mut queue = VecDeque::from([0usize]);
        let mut cursor = 1usize;
        while let Some(node) = queue.pop_front() {
            let edges = std::mem::take(&mut children[node]);
            if edges.is_empty() {
                continue;
            }
            let base = trie.find_base(&edges, &mut cursor);
            let state = slot_of[node];
            trie.base[state as usize] = base;
            for &(label, child) in &edges {
                let slot = (base ^ label as u32) as usize;
                trie.reserve(slot + 1);
                trie.check[slot] = state;
                trie.values[slot] = node_value[child as usize];
                slot_of[child as usize] = slot as u32;
                queue.push_back(child as usize);
            }
        }

        trie
    }

    /// Grow the arrays so slot `len - 1` exists.
    fn reserve(&mut self, len: usize) {
        if len <= self.check.len() {
            return;
        }
        let len = len.next_power_of_two();
        self.base.resize(len, 0);
        self.check.resize(len, FREE);
        self.values.resize(len, NO_TOKEN);
    }

    /// Whether `slot` is available to a state's children.
    ///
    /// Past the end counts as free — the array grows to fit whatever is chosen,
    /// so a state placed near the end is not forced to search back through the
    /// crowded front.
    #[inline]
    fn is_free(&self, slot: usize) -> bool {
        slot != ROOT as usize && self.check.get(slot).is_none_or(|&owner| owner == FREE)
    }

    /// A `base` at which every one of `edges`' labels lands on a free slot.
    ///
    /// `cursor` walks forward over slots that are already taken and is carried
    /// between calls, so the search does not rescan the packed front of the
    /// array for every state. It is a hint, not a bound: the candidate below
    /// runs ahead of it and the cursor never passes a slot that is still free.
    fn find_base(&self, edges: &[(u8, u32)], cursor: &mut usize) -> u32 {
        while !self.is_free(*cursor) {
            *cursor += 1;
        }
        // Anchoring on the first label means the candidate `base` always places
        // that label on a slot known to be free, so only the rest can fail.
        let first = edges[0].0 as u32;
        let mut candidate = *cursor;
        loop {
            let base = candidate as u32 ^ first;
            if edges
                .iter()
                .all(|&(label, _)| self.is_free((base ^ label as u32) as usize))
            {
                return base;
            }
            candidate += 1;
            while !self.is_free(candidate) {
                candidate += 1;
            }
        }
    }

    /// The state reached from `state` by `byte`, or `None` if no edge matches.
    ///
    /// The caller keeps the state, which is what lets a search advance one byte
    /// at a time — a lattice needs every prefix that is a token, not only the
    /// longest, and cannot restart the walk per candidate.
    #[inline]
    pub(crate) fn step(&self, state: u32, byte: u8) -> Option<u32> {
        let slot = (self.base[state as usize] ^ byte as u32) as usize;
        // The `check` test is the whole correctness of the layout: the XOR
        // always lands somewhere, and only this says the landing is an edge of
        // *this* state rather than a slot belonging to another.
        (self.check.get(slot) == Some(&state)).then_some(slot as u32)
    }

    /// The token id `state` spells, or [`NO_TOKEN`].
    #[inline]
    pub(crate) fn value(&self, state: u32) -> u32 {
        self.values[state as usize]
    }

    /// Longest prefix of `bytes` that spells a token: its byte length and id.
    ///
    /// Byte-wise rather than character-wise, which needs no boundary table: a
    /// vocabulary surface is valid UTF-8, so a node carrying a token always
    /// sits on a character boundary and a partial character can never come
    /// back.
    #[inline]
    pub(crate) fn longest_prefix(&self, bytes: &[u8]) -> Option<(usize, u32)> {
        let mut state = ROOT;
        let mut best = None;
        for (i, &byte) in bytes.iter().enumerate() {
            // `break`, not `?` — a walk that stops short still reports the
            // deepest token it passed through.
            let Some(next) = self.step(state, byte) else {
                break;
            };
            state = next;
            let value = self.value(state);
            if value != NO_TOKEN {
                best = Some((i + 1, value));
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trie(entries: &[(&str, u32)]) -> ByteTrie {
        ByteTrie::build(entries.iter().copied())
    }

    #[test]
    fn longest_prefix_finds_the_deepest_token_it_passes() {
        let t = trie(&[("a", 1), ("ab", 2), ("abcd", 3)]);
        assert_eq!(t.longest_prefix(b"a"), Some((1, 1)));
        assert_eq!(t.longest_prefix(b"abc"), Some((2, 2)));
        assert_eq!(t.longest_prefix(b"abcde"), Some((4, 3)));
        assert_eq!(t.longest_prefix(b"z"), None);
        assert_eq!(t.longest_prefix(b""), None);
    }

    /// A walk must not fall into another state's slots. The `check` array is
    /// the only thing preventing it, so this pins that a byte with no edge is
    /// reported as no edge even when the XOR lands on an occupied slot.
    #[test]
    fn a_missing_edge_is_not_answered_by_another_states_slot() {
        // Enough branching that the allocator has to interleave states. ASCII
        // only, so a surface is exactly the bytes asserted on below.
        let leads: Vec<u8> = (b'a'..=b'z')
            .chain(b'A'..=b'Z')
            .chain(b'0'..=b'9')
            .collect();
        let entries: Vec<(String, u32)> = leads
            .iter()
            .map(|&b| (format!("{}x", b as char), b as u32))
            .collect();
        let t = ByteTrie::build(entries.iter().map(|(s, id)| (s.as_str(), *id)));
        for &b in &leads {
            assert_eq!(t.longest_prefix(&[b, b'x']), Some((2, b as u32)), "{b}");
            // `y` is never an edge anywhere, so no state may claim it.
            assert_eq!(t.longest_prefix(&[b, b'y']), None, "{b} followed by y");
        }
    }

    /// Every prefix that is a token has to be reachable one byte at a time,
    /// which is what the lattice relies on.
    #[test]
    fn stepping_reports_every_token_prefix_in_order() {
        let t = trie(&[("ab", 7), ("abc", 8), ("abcd", 9)]);
        let mut state = ROOT;
        let mut seen = Vec::new();
        for (i, &byte) in b"abcd".iter().enumerate() {
            state = t.step(state, byte).expect("edge exists");
            if t.value(state) != NO_TOKEN {
                seen.push((i + 1, t.value(state)));
            }
        }
        assert_eq!(seen, vec![(2, 7), (3, 8), (4, 9)]);
    }

    /// A surface appearing twice keeps the higher id, matching the maps these
    /// tries stand in for.
    #[test]
    fn a_repeated_surface_keeps_the_later_id() {
        let t = trie(&[("a", 1), ("a", 5)]);
        assert_eq!(t.longest_prefix(b"a"), Some((1, 5)));
    }
}
