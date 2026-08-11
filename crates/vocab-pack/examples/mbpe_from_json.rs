//! Derive an `.mbpe` from a HuggingFace `tokenizer.json`.
//!
//! ```text
//! cargo run -p splintr-vocab-pack --example mbpe_from_json -- <tokenizer.json> <out.mbpe>
//! ```
//!
//! Rust rather than a script under `scripts/`, unlike every other vocabulary
//! this repository derives, for one reason: the writer and the reader then share
//! one escape table (`splintr_vocab_pack::escape` against the unescaping inside
//! `pack_mbpe`) instead of two implementations in two languages that agree until
//! they don't.
//!
//! What it checks before writing, because a derived vocabulary that is wrong is
//! worse than one that is large:
//!
//! - ids are contiguous from zero, since `.mbpe` implies them by position;
//! - the file it wrote packs to the same two binaries the json does, byte for
//!   byte. That is the whole correctness claim, and it is verified here rather
//!   than asserted.

use splintr_vocab_pack::{escape, pack_hf_json, pack_mbpe};
use std::collections::HashMap;

fn main() {
    let mut args = std::env::args().skip(1);
    let (src, dst) = match (args.next(), args.next()) {
        (Some(src), Some(dst)) => (src, dst),
        _ => {
            eprintln!("usage: mbpe_from_json <tokenizer.json> <out.mbpe>");
            std::process::exit(2);
        }
    };

    let json = std::fs::read(&src).unwrap_or_else(|e| panic!("{src}: {e}"));
    let text = convert(&json).unwrap_or_else(|e| panic!("{src}: {e}"));

    // The claim this file exists to make: same vocabulary, smaller file.
    let from_json = pack_hf_json(&json).unwrap_or_else(|e| panic!("{src}: {e}"));
    let from_mbpe = pack_mbpe(text.as_bytes()).unwrap_or_else(|e| panic!("{dst}: {e}"));
    assert!(
        from_json == from_mbpe,
        "the derived .mbpe does not pack to the same bytes as its json"
    );

    std::fs::write(&dst, &text).unwrap_or_else(|e| panic!("{dst}: {e}"));
    println!(
        "{dst}: {} bytes, from {} ({:.1}x smaller); packs byte-identically",
        text.len(),
        json.len(),
        json.len() as f64 / text.len() as f64
    );
}

fn convert(json: &[u8]) -> Result<String, String> {
    let root: serde_json::Value = serde_json::from_slice(json).map_err(|e| e.to_string())?;
    let model = root.get("model").ok_or("no `model`")?;
    let vocab = model
        .get("vocab")
        .and_then(serde_json::Value::as_object)
        .ok_or("`model.vocab` is not an object")?;

    let mut entries: Vec<(&str, u32)> = Vec::with_capacity(vocab.len());
    for (token, id) in vocab {
        let id = id
            .as_u64()
            .ok_or_else(|| format!("id for {token:?} is not a number"))?;
        entries.push((token.as_str(), id as u32));
    }
    entries.sort_unstable_by_key(|&(_, id)| id);

    // Ids are implied by position, so a hole would shift every id after it.
    // Refused here, at the one place that could introduce it, rather than left
    // for a reader to notice.
    for (at, &(token, id)) in entries.iter().enumerate() {
        if id as usize != at {
            return Err(format!(
                "ids are not contiguous from zero: {token:?} is id {id} at position {at}. \
                 `.mbpe` implies ids by position and cannot state this vocabulary."
            ));
        }
    }

    let merges = model
        .get("merges")
        .and_then(serde_json::Value::as_array)
        .ok_or("`model.merges` is not an array")?;
    let by_token: HashMap<&str, u32> = entries.iter().copied().collect();

    let mut rules: Vec<(u32, usize)> = Vec::with_capacity(merges.len());
    let mut seen = std::collections::HashSet::with_capacity(merges.len());
    let mut joined = String::new();
    for (i, entry) in merges.iter().enumerate() {
        let (a, b) = match entry {
            serde_json::Value::String(s) => s
                .split_once(' ')
                .ok_or_else(|| format!("merge {i}: no space between the pair"))?,
            serde_json::Value::Array(pair) if pair.len() == 2 => (
                pair[0]
                    .as_str()
                    .ok_or_else(|| format!("merge {i}: half is not a string"))?,
                pair[1]
                    .as_str()
                    .ok_or_else(|| format!("merge {i}: half is not a string"))?,
            ),
            _ => return Err(format!("merge {i}: neither \"a b\" nor [\"a\", \"b\"]")),
        };
        if a.is_empty() || b.is_empty() {
            return Err(format!("merge {i}: an empty half"));
        }
        joined.clear();
        joined.push_str(a);
        joined.push_str(b);
        // A merge whose result is not a vocabulary entry can never be performed
        // and carries no rank — dropped, exactly as `pack_hf_json` drops it, so
        // the two agree on the list as well as on its order.
        let Some(&id) = by_token.get(joined.as_str()) else {
            continue;
        };
        if seen.insert(id) {
            rules.push((id, a.len()));
        }
    }
    if rules.is_empty() {
        return Err("no merges".to_string());
    }

    let mut out = String::with_capacity(json.len() / 6);
    out.push_str("mbpe 1\n");
    out.push_str(&format!("vocab {}\n", entries.len()));
    out.push_str(&format!("merges {}\n", rules.len()));
    out.push('\n');
    for (token, _) in &entries {
        out.push_str(&escape(token.as_bytes()));
        out.push('\n');
    }
    for (id, split) in &rules {
        out.push_str(&format!("{id} {split}\n"));
    }
    Ok(out)
}
