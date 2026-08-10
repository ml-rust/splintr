//! A byte trie over a vocabulary, laid out for prefix search.
//!
//! Both backends that match surfaces against a position in the text — WordPiece's
//! greedy longest match and Unigram's lattice — otherwise ask a hash map "is
//! `text[start..end]` a token?" once per candidate `end`, re-hashing a longer and
//! longer prefix each time. That is quadratic in the length of the longest piece,
//! for an answer a single forward walk gives in linear time.
//!
//! Children are stored compressed-sparse-row: node `i` owns
//! `labels[starts[i]..starts[i + 1]]` and the matching entries of `targets`,
//! sorted by label. Contiguous, so a walk touches one cache line per level rather
//! than chasing a map. The root's children are indexed directly by byte instead,
//! because the root is the one node whose fan-out is the whole alphabet: a binary
//! search there costs about as much as the entire rest of the walk, and every
//! position in the text pays it.

/// Marks a node that is not itself a token.
pub(crate) const NO_TOKEN: u32 = u32::MAX;

/// Marks an absent edge.
const NO_NODE: u32 = u32::MAX;

/// The node a walk starts from.
pub(crate) const ROOT: u32 = 0;

pub(crate) struct ByteTrie {
    root: Box<[u32; 256]>,
    starts: Vec<u32>,
    labels: Vec<u8>,
    targets: Vec<u32>,
    /// Token id at each node, or [`NO_TOKEN`] where the node spells no token.
    values: Vec<u32>,
}

impl ByteTrie {
    /// Build from `(surface, id)` pairs, where a surface is the bytes to match
    /// with any continuation prefix already removed.
    pub(crate) fn build<'a>(entries: impl Iterator<Item = (&'a str, u32)>) -> Self {
        // Grown as an adjacency list, then flattened — building CSR directly
        // would need each node's child count before its children are known.
        let mut children: Vec<Vec<(u8, u32)>> = vec![Vec::new()];
        let mut values = vec![NO_TOKEN];

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
                        values.push(NO_TOKEN);
                        let next = children.len() - 1;
                        children[node].push((byte, next as u32));
                        next
                    }
                };
            }
            // Last writer wins, matching the `token_to_id` maps these tries
            // stand in for: those are filled with `insert` in ascending id
            // order, so a surface appearing twice resolves to its *higher* id.
            values[node] = id;
        }

        let mut starts = Vec::with_capacity(children.len() + 1);
        let mut labels = Vec::new();
        let mut targets = Vec::new();
        for edges in &mut children {
            starts.push(labels.len() as u32);
            edges.sort_unstable_by_key(|(label, _)| *label);
            for &(label, target) in edges.iter() {
                labels.push(label);
                targets.push(target);
            }
        }
        starts.push(labels.len() as u32);

        let mut root = Box::new([NO_NODE; 256]);
        for &(label, target) in &children[0] {
            root[label as usize] = target;
        }

        Self {
            root,
            starts,
            labels,
            targets,
            values,
        }
    }

    /// The node reached from `node` by `byte`, or `None` if no edge matches.
    ///
    /// The caller keeps the node, which is what lets a search advance one byte
    /// at a time — a lattice needs every prefix that is a token, not only the
    /// longest, and cannot restart the walk per candidate.
    #[inline]
    pub(crate) fn step(&self, node: u32, byte: u8) -> Option<u32> {
        if node == ROOT {
            let next = self.root[byte as usize];
            return (next != NO_NODE).then_some(next);
        }
        let node = node as usize;
        let (from, to) = (self.starts[node] as usize, self.starts[node + 1] as usize);
        let at = self.labels.get(from..to)?.binary_search(&byte).ok()?;
        self.targets.get(from + at).copied()
    }

    /// The token id `node` spells, or [`NO_TOKEN`].
    #[inline]
    pub(crate) fn value(&self, node: u32) -> u32 {
        self.values[node as usize]
    }

    /// Longest prefix of `bytes` that spells a token: its byte length and id.
    ///
    /// Byte-wise rather than character-wise, which needs no boundary table: a
    /// vocabulary surface is valid UTF-8, so a node carrying a token always
    /// sits on a character boundary and a partial character can never come
    /// back.
    #[inline]
    pub(crate) fn longest_prefix(&self, bytes: &[u8]) -> Option<(usize, u32)> {
        let mut node = ROOT;
        let mut best = None;
        for (i, &byte) in bytes.iter().enumerate() {
            // `break`, not `?` — a walk that stops short still reports the
            // deepest token it passed through.
            let Some(next) = self.step(node, byte) else {
                break;
            };
            node = next;
            let value = self.value(node);
            if value != NO_TOKEN {
                best = Some((i + 1, value));
            }
        }
        best
    }
}
