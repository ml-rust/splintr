//! Turn a vocabulary text file into the binary form splintr's loader reads.
//!
//! Three inputs, one output shape: a `.tiktoken` ([`pack`]), an `.mbpe`
//! ([`pack_mbpe`], specified in `docs/mbpe.md`), or a HuggingFace
//! `tokenizer.json` ([`pack_hf_json`]). The last two also produce a packed
//! merge order, for a vocabulary whose merge priority is not its id order.
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
//! reviewable and diffable, and for a `.tiktoken` is the very format
//! `Tokenizer::from_file` accepts — and the packed form is derived at build
//! time, so it cannot disagree with the text it came from.
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
/// Such a vocabulary has to state both, and its `tokenizer.json` is the file
/// upstream publishes that does.
///
/// This is the *conversion* path, not the shipping one: what a vocabulary crate
/// ships is the `.mbpe` derived from it (a seventh the size, packing to the same
/// bytes — see [`pack_mbpe`]), and that conversion is checked against this
/// function.
///
/// # Format of the merge list
///
/// See [`MERGES_MAGIC`]. Rules are `(id, split)`: ids rather than token bytes,
/// since every merge result is itself a vocabulary entry and repeating its
/// bytes would double the payload for nothing. That the two files agree is
/// guaranteed by their coming from one `tokenizer.json` in one pass.
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

/// Pack an `.mbpe` file into `(vocabulary, merges)` — the same two binaries
/// [`pack_hf_json`] produces, from a file a fiftieth the size.
///
/// # The format
///
/// ```text
/// mbpe 1
/// vocab 262144
/// merges 236339
/// <blank line>
/// <262144 token lines, position = id>
/// <236339 merge lines, "id split", in priority order>
/// ```
///
/// Header lines are `key value` in any order, terminated by one blank line.
/// Both keys are required and an unknown key is an **error**, never something
/// to skip: a reader that ignored what it did not recognize would half-read a
/// later version of this format and tokenize plausibly and wrongly, which is the
/// failure the format exists to argue against. A v1 reader refuses a v2 file.
///
/// A token line is the token's bytes with four escapes: `\\` for a backslash,
/// `\n` for a line feed, `\r` for a carriage return, and `\xNN` (two lowercase
/// hex digits) for any byte that is not part of well-formed UTF-8. Everything
/// else is literal, which is what makes the file greppable — and cheap, since
/// base64 costs a third for nothing when 262,113 of Gemma 4's 262,144 pieces
/// need no escape at all.
///
/// A merge line is `id split`, both decimal: `id` is the vocabulary id of what
/// the merge produces and `split` the byte length of its left operand, so the
/// pair is `token[..split]` and `token[split..]`. The split is what separates an
/// atom from an orphan — see [`MERGES_MAGIC`] — and is the one thing the merge
/// order cannot be stored without.
///
/// # Ids by position
///
/// A token's id is its line number, so the file spends no bytes stating what is
/// already implied. That is only safe for a vocabulary whose ids are contiguous
/// from zero, and a file that is not is rejected here rather than silently
/// renumbered: a hole would shift every id after it, which is wrong ids with no
/// error. The header's `vocab` count is that check — it must equal the number of
/// token lines exactly.
pub fn pack_mbpe(text: &[u8]) -> Result<(Vec<u8>, Vec<u8>), String> {
    let mut lines = text.split(|&b| b == b'\n').map(strip_cr);

    if lines.next() != Some(b"mbpe 1") {
        return Err("not an mbpe file: line 1 is not `mbpe 1`".to_string());
    }
    let (vocab_count, merge_count) = read_header(&mut lines)?;
    if vocab_count == 0 {
        return Err("no entries".to_string());
    }
    // An `.mbpe` states a vocabulary AND a merge order; a vocabulary that needs
    // no merge order is a `.tiktoken` and packs through `pack`.
    if merge_count == 0 {
        return Err("no merges".to_string());
    }

    let mut packed = Vec::with_capacity(MAGIC.len() + 4 + text.len());
    packed.extend_from_slice(MAGIC);
    packed.extend_from_slice(&(vocab_count as u32).to_le_bytes());
    let mut token = Vec::new();
    for id in 0..vocab_count {
        let line = lines
            .next()
            .ok_or_else(|| format!("token {id}: file ends after {id} of {vocab_count} tokens"))?;
        token.clear();
        unescape_into(line, &mut token).map_err(|e| format!("token {id}: {e}"))?;
        write_varint(&mut packed, id as u64);
        write_varint(&mut packed, token.len() as u64);
        packed.extend_from_slice(&token);
    }

    let mut packed_merges = Vec::with_capacity(MERGES_MAGIC.len() + 4 + merge_count * 4);
    packed_merges.extend_from_slice(MERGES_MAGIC);
    packed_merges.extend_from_slice(&(merge_count as u32).to_le_bytes());
    for rule in 0..merge_count {
        let line = lines.next().ok_or_else(|| {
            format!("merge {rule}: file ends after {rule} of {merge_count} merges")
        })?;
        let (id, split) = merge_line(line).map_err(|e| format!("merge {rule}: {e}"))?;
        if id as usize >= vocab_count {
            return Err(format!("merge {rule}: id {id} is not a vocabulary entry"));
        }
        if split == 0 {
            return Err(format!("merge {rule}: an empty left half"));
        }
        write_varint(&mut packed_merges, id as u64);
        write_varint(&mut packed_merges, split as u64);
    }

    // Anything past the last merge is a file that disagrees with its own header,
    // and the disagreement could as easily be in the counts as in the body.
    if lines.any(|line| !line.is_empty()) {
        return Err("trailing content after the declared merges".to_string());
    }
    Ok((packed, packed_merges))
}

/// Pack `<manifest>/vocabs/<stem>.mbpe` into `$OUT_DIR/<stem>.splv` and
/// `$OUT_DIR/<stem>.splm`, and tell cargo to rerun when the text changes.
///
/// The whole of a merges-carrying vocabulary crate's build script.
pub fn pack_mbpe_into_out_dir(stem: &str) {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let out = std::env::var("OUT_DIR").expect("OUT_DIR");
    let src = format!("{manifest}/vocabs/{stem}.mbpe");
    println!("cargo::rerun-if-changed={src}");

    let text = std::fs::read(&src).unwrap_or_else(|e| panic!("{src}: {e}"));
    let (vocab, merges) = pack_mbpe(&text).unwrap_or_else(|e| panic!("{src}: {e}"));
    let dst = format!("{out}/{stem}.splv");
    std::fs::write(&dst, vocab).unwrap_or_else(|e| panic!("{dst}: {e}"));
    let dst = format!("{out}/{stem}.splm");
    std::fs::write(&dst, merges).unwrap_or_else(|e| panic!("{dst}: {e}"));
}

/// Render `token` as one `.mbpe` line, the inverse of the unescaping
/// [`pack_mbpe`] performs.
///
/// Public because the file has to be *written* by something, and a writer that
/// does not share the reader's escape table is how the two drift apart.
pub fn escape(token: &[u8]) -> String {
    let mut out = String::with_capacity(token.len() + 8);
    let mut at = 0;
    while at < token.len() {
        match token[at] {
            b'\\' => {
                out.push_str("\\\\");
                at += 1;
            }
            b'\n' => {
                out.push_str("\\n");
                at += 1;
            }
            b'\r' => {
                out.push_str("\\r");
                at += 1;
            }
            _ => match utf8_char(&token[at..]) {
                // Literal wherever the bytes are well-formed UTF-8, which is all
                // of every published vocabulary but the tiktoken ones.
                Some(text) => {
                    out.push_str(text);
                    at += text.len();
                }
                None => {
                    out.push_str(&format!("\\x{:02x}", token[at]));
                    at += 1;
                }
            },
        }
    }
    out
}

/// The longest well-formed UTF-8 character at the head of `bytes`, if it starts
/// with one.
fn utf8_char(bytes: &[u8]) -> Option<&str> {
    let len = match bytes[0] {
        b if b < 0x80 => 1,
        b if b >> 5 == 0b110 => 2,
        b if b >> 4 == 0b1110 => 3,
        b if b >> 3 == 0b11110 => 4,
        _ => return None,
    };
    std::str::from_utf8(bytes.get(..len)?).ok()
}

/// Decode one escaped token line into `out`.
fn unescape_into(line: &[u8], out: &mut Vec<u8>) -> Result<(), String> {
    let mut at = 0;
    while at < line.len() {
        if line[at] != b'\\' {
            out.push(line[at]);
            at += 1;
            continue;
        }
        match line.get(at + 1) {
            Some(b'\\') => out.push(b'\\'),
            Some(b'n') => out.push(b'\n'),
            Some(b'r') => out.push(b'\r'),
            Some(b'x') => {
                let hex = line
                    .get(at + 2..at + 4)
                    .ok_or_else(|| "a `\\x` escape needs two hex digits".to_string())?;
                let hex = std::str::from_utf8(hex).map_err(|e| e.to_string())?;
                out.push(
                    u8::from_str_radix(hex, 16)
                        .map_err(|_| format!("`\\x{hex}` is not two hex digits"))?,
                );
                at += 4;
                continue;
            }
            Some(other) => return Err(format!("unknown escape `\\{}`", *other as char)),
            None => return Err("a line ends in a lone backslash".to_string()),
        }
        at += 2;
    }
    Ok(())
}

/// Read the header block: `key value` lines in any order, terminated by one
/// blank line.
///
/// Every known key is required and an unknown key is an **error**, not
/// something to skip. A format whose readers ignore what they do not recognize
/// is a format that silently half-reads its own future versions, and silently
/// ignoring a declared field is the single failure this project keeps finding in
/// other people's tokenizers. So a v1 reader refuses a v2 file outright rather
/// than reading the part it happens to understand.
fn read_header<'a>(lines: &mut impl Iterator<Item = &'a [u8]>) -> Result<(usize, usize), String> {
    let (mut vocab, mut merges) = (None, None);
    loop {
        let line = lines.next().ok_or("the header is not terminated")?;
        if line.is_empty() {
            break;
        }
        let line = std::str::from_utf8(line).map_err(|e| format!("header: {e}"))?;
        let (key, value) = line
            .split_once(' ')
            .ok_or_else(|| format!("header line {line:?} is not `key value`"))?;
        let count = |slot: &mut Option<usize>| -> Result<(), String> {
            let n = value
                .parse()
                .map_err(|e| format!("`{key}` header: {e}"))
                .map(Some)?;
            match slot.replace(n.expect("just parsed")) {
                Some(_) => Err(format!("`{key}` declared twice")),
                None => Ok(()),
            }
        };
        match key {
            "vocab" => count(&mut vocab)?,
            "merges" => count(&mut merges)?,
            other => return Err(format!("unknown header key {other:?}")),
        }
    }
    Ok((
        vocab.ok_or("no `vocab` header")?,
        merges.ok_or("no `merges` header")?,
    ))
}

/// Read an `id split` merge line.
fn merge_line(line: &[u8]) -> Result<(u32, usize), String> {
    let line = std::str::from_utf8(line).map_err(|e| e.to_string())?;
    let (id, split) = line
        .split_once(' ')
        .ok_or_else(|| format!("expected `id split`, found {line:?}"))?;
    Ok((
        id.parse().map_err(|e| format!("id: {e}"))?,
        split.parse().map_err(|e| format!("split: {e}"))?,
    ))
}

/// A line as the file meant it, whichever line ending it was written with.
fn strip_cr(line: &[u8]) -> &[u8] {
    line.strip_suffix(b"\r").unwrap_or(line)
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

    const TINY_MBPE: &str = "mbpe 1\nvocab 4\nmerges 2\n\na\nb\nab\naba\n2 1\n3 2\n";

    /// The whole point: an `.mbpe` and the `tokenizer.json` it was derived from
    /// pack to the same two binaries, byte for byte. Anything less and the
    /// smaller file would be a different vocabulary.
    #[test]
    fn an_mbpe_packs_to_the_same_bytes_as_the_json_it_came_from() {
        assert_eq!(
            pack_mbpe(TINY_MBPE.as_bytes()).expect("packs"),
            pack_hf_json(TINY.as_bytes()).expect("packs")
        );
    }

    /// Ids come from position, so a vocabulary the header miscounts is rejected
    /// rather than read with every id after the miscount shifted.
    #[test]
    fn a_header_count_that_disagrees_with_the_body_is_an_error() {
        let short = "mbpe 1\nvocab 5\nmerges 2\n\na\nb\nab\naba\n2 1\n3 2\n";
        let long = "mbpe 1\nvocab 3\nmerges 2\n\na\nb\nab\naba\n2 1\n3 2\n";
        assert!(pack_mbpe(short.as_bytes()).is_err());
        assert!(pack_mbpe(long.as_bytes()).is_err());
    }

    #[test]
    fn a_file_without_the_magic_is_not_an_mbpe() {
        let no_magic = "vocab 1\nmerges 1\n\na\n0 1\n";
        assert!(pack_mbpe(no_magic.as_bytes()).is_err());
        let wrong_version = "mbpe 2\nvocab 1\nmerges 1\n\na\n0 1\n";
        assert!(pack_mbpe(wrong_version.as_bytes()).is_err());
    }

    /// A merge naming an id the vocabulary does not have is a file describing
    /// two different vocabularies.
    #[test]
    fn a_merge_outside_the_vocabulary_is_an_error() {
        let out_of_range = "mbpe 1\nvocab 4\nmerges 1\n\na\nb\nab\naba\n9 1\n";
        assert!(pack_mbpe(out_of_range.as_bytes()).is_err());
    }

    /// The four escapes, and the rule that everything else stays literal.
    #[test]
    fn escaping_round_trips_every_byte_shape() {
        for token in [
            b"plain".to_vec(),
            b"back\\slash".to_vec(),
            b"line\nfeed".to_vec(),
            b"carriage\rreturn".to_vec(),
            // Well-formed UTF-8 stays literal, however far outside ASCII.
            "▁caf\u{e9} \u{1f680} \u{4e2d}\u{6587}".as_bytes().to_vec(),
            // A lone continuation byte is not UTF-8 and must be escaped —
            // cl100k_base alone has 773 tokens that are not valid UTF-8.
            vec![0xE2, 0x96],
            vec![0x80, b'a', 0xFF],
            Vec::new(),
        ] {
            let line = escape(&token);
            let mut back = Vec::new();
            unescape_into(line.as_bytes(), &mut back).expect("unescapes");
            assert_eq!(back, token, "round trip failed for {line:?}");
            assert!(!line.contains('\n'), "escaped form spans lines: {line:?}");
        }
    }

    /// Escapes are read as escapes, not as their spelling.
    #[test]
    fn an_escaped_token_packs_to_the_bytes_it_names() {
        let text = "mbpe 1\nvocab 3\nmerges 1\n\n\\n\n\\x80\n\\n\\x80\n2 1\n";
        let (vocab, _) = pack_mbpe(text.as_bytes()).expect("packs");
        // ids 0..2 with lengths 1, 1, 2: `\n`, `\x80`, and their concatenation.
        assert_eq!(&vocab[12..], &[0, 1, b'\n', 1, 1, 0x80, 2, 2, b'\n', 0x80]);
    }

    /// The header is `key value` in any order — nothing about the two keys
    /// implies a sequence, and a writer that emits them in the other one is not
    /// writing a different file.
    #[test]
    fn header_keys_may_come_in_any_order() {
        let shuffled = "mbpe 1\nmerges 2\nvocab 4\n\na\nb\nab\naba\n2 1\n3 2\n";
        assert_eq!(
            pack_mbpe(shuffled.as_bytes()).expect("packs"),
            pack_mbpe(TINY_MBPE.as_bytes()).expect("packs")
        );
    }

    /// The forward-compatibility contract, and the reason it is this way round:
    /// a reader that skipped what it did not recognize would half-read a later
    /// version of this format and tokenize plausibly and wrongly, which is the
    /// exact failure this project keeps finding in other people's tokenizers.
    #[test]
    fn an_unknown_header_key_is_refused_not_skipped() {
        let v2 = "mbpe 1\nvocab 4\nmerges 2\nsuffix </w>\n\na\nb\nab\naba\n2 1\n3 2\n";
        assert!(pack_mbpe(v2.as_bytes()).is_err());
    }

    #[test]
    fn an_unknown_escape_is_an_error() {
        let text = "mbpe 1\nvocab 1\nmerges 1\n\n\\q\n0 1\n";
        assert!(pack_mbpe(text.as_bytes()).is_err());
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
