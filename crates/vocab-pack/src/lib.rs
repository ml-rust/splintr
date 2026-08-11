//! Turn a `.tiktoken` rank file into the binary form splintr's loader reads.
//!
//! # Why this exists at all
//!
//! splintr loads a bundled vocabulary from a packed binary rather than from the
//! `base64(token) rank` text, because the text costs roughly twice as long to
//! load: every line needs base64-decoding and a decimal parse, and there are
//! 100k–200k of them.
//!
//! That binary used to be committed next to the text, which meant two files
//! carrying the same ranks, a script to regenerate one from the other, and a
//! test to catch anyone who regenerated only one of them. It also meant every
//! published vocabulary crate was an opaque blob: nothing in a diff, nothing to
//! grep, and no way for a reader to check what they were about to compile in.
//!
//! Building it instead removes all of that. The crates ship the text — which is
//! reviewable, diffable, and the same format `Tokenizer::from_file` accepts —
//! and the packed form is derived at build time, so it cannot disagree with the
//! text it came from.
//!
//! # Format
//!
//! ```text
//! "SPLNTRV1"        magic, 8 bytes
//! count             u32, little-endian, number of entries
//! entry*            count entries, in file order
//!
//! entry = varint(rank) varint(len) token[len]
//! ```
//!
//! `varint` is LEB128: seven bits per byte, low byte first, high bit set on
//! every byte but the last.
//!
//! Ranks are stored, not implied by position. Every vocabulary bundled today is
//! contiguous, so position would encode them correctly and save a couple of
//! bytes per token — but nothing in the tiktoken format promises contiguity, and
//! a vocabulary with a hole in its rank space would be silently renumbered
//! rather than rejected. That is wrong ids with no error, which is the one
//! failure mode a tokenizer must not have.

use base64::engine::general_purpose::STANDARD;
use base64::Engine;

/// Magic at the head of every packed vocabulary.
pub const MAGIC: &[u8; 8] = b"SPLNTRV1";

/// Magic at the head of every packed merge list.
///
/// ```text
/// "SPLNTRM1"                 magic, 8 bytes
/// count                      u32, little-endian
/// (varint(id) varint(split))* count rules, in merge-priority order
/// ```
///
/// `id` is the vocabulary id of what the merge produces; `split` is the byte
/// length of its left half, so the rule's two operands are `result[..split]`
/// and `result[split..]`.
///
/// Storing the split rather than only the result is what lets a reader classify
/// every vocabulary entry:
///
/// - a **merge result** — BPE can produce it;
/// - an **atom** — never a result, but an operand of some merge, so BPE builds
///   from it (the byte alphabet, `<0xNN>` byte-fallback pieces, CLIP's `x</w>`
///   end-of-word forms);
/// - an **orphan** — neither. BPE can never produce it and nothing is built
///   from it, yet it holds an id. Gemma 4 has 20,522, of which 6,298 are longer
///   than one character.
///
/// That last class is why the split is worth its bytes. A tokenizer that
/// answers a whole chunk from the vocabulary before merging — the standard fast
/// path — will emit an orphan that no correct BPE can reach, which is a wrong
/// id rather than a slow one. Only the operand set separates orphans from
/// atoms, and only the rules carry it.
pub const MERGES_MAGIC: &[u8; 8] = b"SPLNTRM1";

/// Pack `.tiktoken` text into the binary form.
///
/// Blank lines are skipped. A line is `base64(token) rank`, split on the LAST
/// space: cl100k's rank 50256 is an empty token, which is a line that begins
/// with its separator, and splitting on the first space would read the rank as
/// the token.
pub fn pack(text: &[u8]) -> Result<Vec<u8>, String> {
    let mut entries = Vec::with_capacity(text.len() / 2);
    let mut count: u32 = 0;

    for (line_no, line) in text.split(|&b| b == b'\n').enumerate() {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if line.is_empty() {
            continue;
        }
        let at = line
            .iter()
            .rposition(|&b| b == b' ')
            .ok_or_else(|| format!("line {}: no space between token and rank", line_no + 1))?;
        let token = STANDARD
            .decode(&line[..at])
            .map_err(|e| format!("line {}: {e}", line_no + 1))?;
        let rank: u32 = std::str::from_utf8(&line[at + 1..])
            .map_err(|e| format!("line {}: {e}", line_no + 1))?
            .trim()
            .parse()
            .map_err(|e| format!("line {}: {e}", line_no + 1))?;

        write_varint(&mut entries, rank as u64);
        write_varint(&mut entries, token.len() as u64);
        entries.extend_from_slice(&token);
        count += 1;
    }

    if count == 0 {
        return Err("no entries".to_string());
    }
    let mut out = Vec::with_capacity(MAGIC.len() + 4 + entries.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&count.to_le_bytes());
    out.extend_from_slice(&entries);
    Ok(out)
}

/// Pack `<manifest>/vocabs/<stem>.tiktoken` into `$OUT_DIR/<stem>.splv`, and
/// tell cargo to rerun when the text changes.
///
/// The whole of a vocabulary crate's build script.
pub fn pack_into_out_dir(stem: &str) {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let out = std::env::var("OUT_DIR").expect("OUT_DIR");
    let src = format!("{manifest}/vocabs/{stem}.tiktoken");
    println!("cargo::rerun-if-changed={src}");

    let text = std::fs::read(&src).unwrap_or_else(|e| panic!("{src}: {e}"));
    let packed = pack(&text).unwrap_or_else(|e| panic!("{src}: {e}"));
    let dst = format!("{out}/{stem}.splv");
    std::fs::write(&dst, packed).unwrap_or_else(|e| panic!("{dst}: {e}"));
}

/// Pack a HuggingFace `tokenizer.json` into `(vocabulary, merges)`.
///
/// # Why a `tokenizer.json` and not a `.tiktoken`
///
/// A `.tiktoken` line is `base64(token) rank`, and that one rank is both the
/// token's id **and** its merge priority. Every OpenAI vocabulary is built so
/// those coincide; a HuggingFace BPE need not be. Gemma 4's does not — 465
/// places where a later merge yields a lower id, and 514,906 merges collapsing
/// onto 236,339 distinct tokens, so the two orders are not even the same
/// length. Forcing them into one column mistokenizes 8.1% of real documents.
/// Such a vocabulary has to ship the file that states both, which is its
/// `tokenizer.json`.
///
/// # Format of the merge list
///
/// ```text
/// "SPLNTRM1"        magic, 8 bytes
/// count             u32, little-endian, number of merges
/// varint(id)*       count ids, in merge-priority order
/// ```
///
/// Ids, not token bytes: every merge result is itself a vocabulary entry, so
/// the bytes are already in the companion vocabulary and repeating them would
/// double the payload for nothing. That the two files agree is guaranteed by
/// their coming from one `tokenizer.json` in one pass.
///
/// Duplicates are dropped, first occurrence winning, because that is what the
/// consumer does with them: ranks are assigned with `insert_if_absent`, so a
/// token's rank is fixed by where it *first* appears and every later mention
/// changes nothing. Dropping them here shrinks Gemma 4's list from 514,906
/// entries to 236,339 and leaves the ranks identical — only the relative order
/// of distinct tokens is ever read.
pub fn pack_hf_json(json: &[u8]) -> Result<(Vec<u8>, Vec<u8>), String> {
    let root: serde_json::Value =
        serde_json::from_slice(json).map_err(|e| format!("tokenizer.json: {e}"))?;
    let model = root.get("model").ok_or("tokenizer.json: no `model`")?;

    let vocab = model
        .get("vocab")
        .and_then(serde_json::Value::as_object)
        .ok_or("tokenizer.json: `model.vocab` is not an object")?;

    let mut entries: Vec<(&str, u32)> = Vec::with_capacity(vocab.len());
    for (token, id) in vocab {
        let id = id
            .as_u64()
            .ok_or_else(|| format!("tokenizer.json: id for {token:?} is not a number"))?;
        entries.push((token.as_str(), id as u32));
    }
    entries.sort_unstable_by_key(|&(_, id)| id);

    let mut packed = Vec::with_capacity(MAGIC.len() + 4 + json.len() / 4);
    packed.extend_from_slice(MAGIC);
    packed.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (token, id) in &entries {
        write_varint(&mut packed, *id as u64);
        write_varint(&mut packed, token.len() as u64);
        packed.extend_from_slice(token.as_bytes());
    }

    // `merges` is either ["a b", ...] or [["a", "b"], ...] depending on the
    // `tokenizers` version that wrote the file; both spell the same pair.
    let merges = model
        .get("merges")
        .and_then(serde_json::Value::as_array)
        .ok_or("tokenizer.json: `model.merges` is not an array")?;

    let by_token: std::collections::HashMap<&str, u32> = entries.into_iter().collect();
    let mut seen = std::collections::HashSet::with_capacity(merges.len());
    // (result id, byte length of the left half). The split is what makes the
    // rule recoverable rather than just its outcome: `result[..split]` and
    // `result[split..]` are the two operands, and an operand is how a reader
    // tells an ATOM — a vocabulary entry that merges are built FROM — apart from
    // an entry no merge touches at all. See the format note on `MERGES_MAGIC`.
    let mut rules: Vec<(u32, usize)> = Vec::with_capacity(merges.len());
    let mut joined = String::new();
    for (i, entry) in merges.iter().enumerate() {
        joined.clear();
        let split = match entry {
            serde_json::Value::String(s) => {
                let (a, b) = s
                    .split_once(' ')
                    .ok_or_else(|| format!("merge {i}: no space between the pair"))?;
                joined.push_str(a);
                joined.push_str(b);
                a.len()
            }
            serde_json::Value::Array(pair) if pair.len() == 2 => {
                let a = pair[0]
                    .as_str()
                    .ok_or_else(|| format!("merge {i}: half is not a string"))?;
                let b = pair[1]
                    .as_str()
                    .ok_or_else(|| format!("merge {i}: half is not a string"))?;
                joined.push_str(a);
                joined.push_str(b);
                a.len()
            }
            _ => return Err(format!("merge {i}: neither \"a b\" nor [\"a\", \"b\"]")),
        };
        if split == 0 || split >= joined.len() {
            return Err(format!("merge {i}: an empty half"));
        }
        // A merge whose result is not itself a vocabulary entry can never be
        // performed, so it carries no rank and is dropped rather than given one.
        let Some(&id) = by_token.get(joined.as_str()) else {
            continue;
        };
        if seen.insert(id) {
            rules.push((id, split));
        }
    }

    if rules.is_empty() {
        return Err("no merges".to_string());
    }
    let mut packed_merges = Vec::with_capacity(MERGES_MAGIC.len() + 4 + rules.len() * 4);
    packed_merges.extend_from_slice(MERGES_MAGIC);
    packed_merges.extend_from_slice(&(rules.len() as u32).to_le_bytes());
    for (id, split) in rules {
        write_varint(&mut packed_merges, id as u64);
        write_varint(&mut packed_merges, split as u64);
    }

    Ok((packed, packed_merges))
}

/// Pack `<manifest>/vocabs/<stem>.json` into `$OUT_DIR/<stem>.splv` and
/// `$OUT_DIR/<stem>.splm`, and tell cargo to rerun when the json changes.
///
/// The whole of a merges-carrying vocabulary crate's build script.
pub fn pack_hf_json_into_out_dir(stem: &str) {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let out = std::env::var("OUT_DIR").expect("OUT_DIR");
    let src = format!("{manifest}/vocabs/{stem}.json");
    println!("cargo::rerun-if-changed={src}");

    let json = std::fs::read(&src).unwrap_or_else(|e| panic!("{src}: {e}"));
    let (vocab, merges) = pack_hf_json(&json).unwrap_or_else(|e| panic!("{src}: {e}"));
    let dst = format!("{out}/{stem}.splv");
    std::fs::write(&dst, vocab).unwrap_or_else(|e| panic!("{dst}: {e}"));
    let dst = format!("{out}/{stem}.splm");
    std::fs::write(&dst, merges).unwrap_or_else(|e| panic!("{dst}: {e}"));
}

fn write_varint(out: &mut Vec<u8>, mut n: u64) {
    loop {
        let byte = (n & 0x7F) as u8;
        n >>= 7;
        out.push(byte | if n != 0 { 0x80 } else { 0 });
        if n == 0 {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rank_above_127_takes_two_varint_bytes() {
        // "a" is rank 300, so the varint is 0xAC 0x02 rather than one byte.
        let packed = pack(b"YQ== 300").expect("packs");
        assert_eq!(&packed[..8], MAGIC);
        assert_eq!(&packed[8..12], &1u32.to_le_bytes());
        assert_eq!(&packed[12..], &[0xAC, 0x02, 0x01, b'a']);
    }

    #[test]
    fn the_empty_token_survives() {
        // Whisper's rank 50256 is an empty token: a line that starts with the
        // separator. Splitting on the first space would read "50256" as the
        // token and find no rank.
        let packed = pack(b" 50256").expect("packs");
        assert_eq!(&packed[12..], &[0xD0, 0x88, 0x03, 0x00]);
    }

    #[test]
    fn blank_lines_are_skipped_not_counted() {
        let packed = pack(b"YQ== 0\n\nYg== 1\n").expect("packs");
        assert_eq!(&packed[8..12], &2u32.to_le_bytes());
    }

    #[test]
    fn a_line_without_a_rank_is_an_error() {
        assert!(pack(b"YQ==").is_err());
    }

    const TINY: &str = r#"{"model":{"vocab":{"a":0,"b":1,"ab":2,"aba":3},
        "merges":["a b","ab a","a b"]}}"#;

    #[test]
    fn a_tokenizer_json_packs_vocabulary_and_merges() {
        let (vocab, merges) = pack_hf_json(TINY.as_bytes()).expect("packs");
        assert_eq!(&vocab[..8], MAGIC);
        assert_eq!(&vocab[8..12], &4u32.to_le_bytes());
        assert_eq!(&merges[..8], MERGES_MAGIC);
        // "a b" -> ab (id 2), "ab a" -> aba (id 3), and the repeated "a b" is
        // dropped: its rank is already fixed by the first occurrence.
        assert_eq!(&merges[8..12], &2u32.to_le_bytes());
        // Each rule is `id split`: "a"+"b" -> id 2 splitting after 1 byte,
        // "ab"+"a" -> id 3 splitting after 2.
        assert_eq!(&merges[12..], &[2, 1, 3, 2]);
    }

    /// The split is the left half's length, so a reader can recover the pair a
    /// rule joined rather than only the token it produced. Without it there is
    /// no way to tell a vocabulary entry that merges build FROM from one that no
    /// merge touches, and the second kind must not be encodable.
    #[test]
    fn a_merge_records_where_its_halves_met() {
        let json = r#"{"model":{"vocab":{"a":0,"bc":1,"abc":2},"merges":["a bc"]}}"#;
        let (_, merges) = pack_hf_json(json.as_bytes()).expect("packs");
        assert_eq!(&merges[12..], &[2, 1]);
    }

    /// A pair with an empty half names no two tokens, so it is not a rule.
    #[test]
    fn a_merge_with_an_empty_half_is_an_error() {
        let json = r#"{"model":{"vocab":{"a":0,"ab":1},"merges":[["a",""]]}}"#;
        assert!(pack_hf_json(json.as_bytes()).is_err());
    }

    /// Both spellings the `tokenizers` versions emit mean the same pair.
    #[test]
    fn the_pair_and_string_merge_spellings_agree() {
        let pairs = r#"{"model":{"vocab":{"a":0,"b":1,"ab":2},"merges":[["a","b"]]}}"#;
        let string = r#"{"model":{"vocab":{"a":0,"b":1,"ab":2},"merges":["a b"]}}"#;
        assert_eq!(
            pack_hf_json(pairs.as_bytes()).unwrap(),
            pack_hf_json(string.as_bytes()).unwrap()
        );
    }

    /// The vocabulary is packed in id order however the json happens to list
    /// it — a json object preserves insertion order, which need not be id order.
    #[test]
    fn the_vocabulary_is_packed_in_id_order() {
        let shuffled = r#"{"model":{"vocab":{"ab":2,"a":0,"b":1},"merges":["a b"]}}"#;
        let ordered = r#"{"model":{"vocab":{"a":0,"b":1,"ab":2},"merges":["a b"]}}"#;
        assert_eq!(
            pack_hf_json(shuffled.as_bytes()).unwrap(),
            pack_hf_json(ordered.as_bytes()).unwrap()
        );
    }

    /// A merge whose result is not a vocabulary entry can never be performed,
    /// so it carries no rank — dropping it is not the same as failing.
    #[test]
    fn a_merge_with_no_resulting_token_is_skipped() {
        let json = r#"{"model":{"vocab":{"a":0,"b":1,"ab":2},"merges":["a a","a b"]}}"#;
        let (_, merges) = pack_hf_json(json.as_bytes()).expect("packs");
        assert_eq!(&merges[8..12], &1u32.to_le_bytes());
        assert_eq!(&merges[12..], &[2, 1]);
    }
}
