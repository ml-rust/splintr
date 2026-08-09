//! Rust-to-Rust tokenizer comparison: no Python, no FFI, no object marshalling.
//!
//! The Python perf suite measures what a Python caller gets, where most of a
//! batch call is CPython building lists. This measures the kernels themselves,
//! which is the only way to know whether splintr's merge loop and scanners are
//! actually competitive or merely hidden behind a shared FFI tax.
//!
//! Engines are grouped by what they can load, and only compared inside a group:
//!
//!   ranks  a `.tiktoken` rank file plus that vocabulary's pre-tokenizer
//!          expression — splintr, tiktoken-rs, riptoken
//!   json   a HuggingFace `tokenizer.json` — splintr, HF tokenizers,
//!          basetenkenizer
//!   baked  the vocabulary compiled into the crate — bpe-openai, which offers
//!          no way to load one
//!
//! Every engine's ids are checked against splintr's before any timing is read.

use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::HashMap;
use std::time::Instant;

const ROUNDS: usize = 3;

type Encode = Box<dyn Fn(&str) -> Vec<u32> + Sync>;
/// Both shapes, because the crates disagree: splintr's `Tokenizer` wants
/// `&[String]` while everything else wants `&[&str]`. Handing each the form it
/// declares keeps a conversion out of every engine's measured time.
type EncodeAll = Box<dyn Fn(&[String], &[&str]) -> usize + Sync>;

struct Engine {
    name: &'static str,
    /// How the vocabulary reached it, which is what makes a group comparable.
    family: &'static str,
    /// Whether the parallel number comes from the library's own batch call or
    /// from rayon in this file. A library without a batch API is not slower for
    /// lacking one — in Rust you just parallelize it yourself — but the
    /// distinction belongs in the report.
    own_batch: bool,
    encode: Encode,
    encode_all: EncodeAll,
}

fn read_corpus(path: &str) -> Vec<String> {
    std::fs::read_to_string(path)
        .expect("corpus")
        .split("\n\u{0}\n")
        .map(str::to_owned)
        .collect()
}

fn tiktoken_ranks(path: &str) -> FxHashMap<Vec<u8>, u32> {
    splintr::core::load_tiktoken_bpe_file(path).expect("rank file")
}

/// Engines that could not load this vocabulary at all, with the reason. A crate
/// that cannot read a file is reported as absent, the way the Python suite's
/// `—` cells work — never as a slow result, and never as a crashed run.
type Skipped = Vec<(&'static str, String)>;

fn build(
    vocab: &str,
    ranks_path: &str,
    json_path: &str,
    pattern: &'static str,
) -> (Vec<Engine>, Skipped) {
    let mut engines: Vec<Engine> = Vec::new();
    let mut skipped: Skipped = Vec::new();

    // --- ranks family -------------------------------------------------------
    {
        let ranks = splintr::core::load_tiktoken_bpe_file(ranks_path).expect("rank file");
        let tok = splintr::Tokenizer::new(ranks, FxHashMap::default(), pattern).expect("splintr");
        engines.push(Engine {
            name: "splintr",
            family: "ranks",
            own_batch: true,
            encode: {
                let t = tok.clone();
                Box::new(move |s| t.encode(s))
            },
            encode_all: {
                let t = tok.clone();
                Box::new(move |owned, _| t.encode_batch(owned).iter().map(Vec::len).sum())
            },
        });
    }
    match tiktoken_rs::CoreBPE::new(tiktoken_ranks(ranks_path), FxHashMap::default(), pattern) {
        Err(e) => skipped.push(("tiktoken-rs", e.to_string())),
        Ok(bpe) => {
            let bpe = std::sync::Arc::new(bpe);
            engines.push(Engine {
                name: "tiktoken-rs",
                family: "ranks",
                own_batch: false, // no batch API; rayon below is the idiomatic use
                encode: {
                    let b = bpe.clone();
                    Box::new(move |s| b.encode_ordinary(s))
                },
                encode_all: {
                    let b = bpe.clone();
                    Box::new(move |_, ts| ts.par_iter().map(|t| b.encode_ordinary(t).len()).sum())
                },
            });
        }
    }
    match riptoken::CoreBPE::new(tiktoken_ranks(ranks_path), FxHashMap::default(), pattern) {
        Err(e) => skipped.push(("riptoken", e.to_string())),
        Ok(bpe) => {
            let bpe = std::sync::Arc::new(bpe);
            engines.push(Engine {
                name: "riptoken",
                family: "ranks",
                own_batch: true,
                encode: {
                    let b = bpe.clone();
                    Box::new(move |s| b.encode_ordinary(s))
                },
                encode_all: {
                    let b = bpe.clone();
                    Box::new(move |_, ts| b.encode_ordinary_batch(ts).iter().map(Vec::len).sum())
                },
            });
        }
    }

    // --- json family --------------------------------------------------------
    // `-` means the vocabulary publishes no `tokenizer.json` at all, which is
    // true of Kimi. Saying so once beats pointing every json engine at a path
    // that does not exist and letting each report an I/O error — three
    // identical "No such file or directory" lines read like splintr broke,
    // when the file was never supposed to be there.
    let publishes_json = json_path != "-";
    if !publishes_json {
        skipped.push((
            "json family",
            "no tokenizer.json is published for this vocabulary".to_string(),
        ));
    }

    if publishes_json {
        match splintr::from_json_path(json_path) {
            Err(e) => skipped.push(("splintr (json)", e.to_string())),
            Ok(tok) => {
                let tok = std::sync::Arc::new(tok);
                engines.push(Engine {
                    name: "splintr",
                    family: "json",
                    own_batch: true,
                    encode: {
                        let t = tok.clone();
                        Box::new(move |s| t.encode_raw(s))
                    },
                    encode_all: {
                        let t = tok.clone();
                        Box::new(move |_, ts| t.encode_batch(ts).iter().map(Vec::len).sum())
                    },
                });
            }
        }
        match tokenizers::Tokenizer::from_file(json_path) {
            Err(e) => skipped.push(("HF tokenizers", e.to_string())),
            Ok(tok) => {
                let tok = std::sync::Arc::new(tok);
                engines.push(Engine {
                    name: "HF tokenizers",
                    family: "json",
                    own_batch: true,
                    encode: {
                        let t = tok.clone();
                        Box::new(move |s| {
                            t.encode_fast(s, false)
                                .expect("hf encode")
                                .get_ids()
                                .to_vec()
                        })
                    },
                    encode_all: {
                        let t = tok.clone();
                        Box::new(move |_, ts| {
                            t.encode_batch_fast(ts.to_vec(), false)
                                .expect("hf batch")
                                .iter()
                                .map(|e| e.get_ids().len())
                                .sum()
                        })
                    },
                });
            }
        }
        match basetenkenizer::Tokenizer::from_file(std::path::Path::new(json_path)) {
            Err(e) => skipped.push(("basetenkenizer", format!("{e:?}"))),
            Ok(tok) => {
                let tok = std::sync::Arc::new(tok);
                engines.push(Engine {
                    name: "basetenkenizer",
                    family: "json",
                    own_batch: true,
                    encode: {
                        let t = tok.clone();
                        Box::new(move |s| t.encode(s).expect("bt encode"))
                    },
                    encode_all: {
                        let t = tok.clone();
                        Box::new(move |_, ts| {
                            t.encode_batch(ts, false)
                                .expect("bt batch")
                                .iter()
                                .map(Vec::len)
                                .sum()
                        })
                    },
                });
            }
        }
    }

    // --- baked family -------------------------------------------------------
    // bpe-openai compiles its vocabularies in and offers no loader, so it can
    // only appear for the two it ships.
    let baked = match vocab {
        "cl100k" => Some(bpe_openai::cl100k_base()),
        "o200k" => Some(bpe_openai::o200k_base()),
        _ => None,
    };
    if let Some(t) = baked {
        engines.push(Engine {
            // Joins the rank-file table rather than standing alone: it carries
            // its own copy of this vocabulary instead of reading ours, and the
            // parity check is what licenses the comparison — its ids match, so
            // it is tokenizing the same text with the same vocabulary.
            name: "bpe-openai",
            family: "ranks",
            own_batch: false,
            encode: Box::new(move |s| t.encode(s)),
            encode_all: Box::new(move |_, ts| ts.par_iter().map(|s| t.encode(s).len()).sum()),
        });
    }

    (engines, skipped)
}

/// Minimal JSON string escaping — the report reads these reasons verbatim and a
/// crate's error text is arbitrary.
fn serde_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' | '\r' | '\t' => out.push(' '),
            c if (c as u32) < 0x20 => out.push(' '),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn bench(f: impl Fn()) -> f64 {
    f(); // warm
    let mut best = f64::MAX;
    for _ in 0..ROUNDS {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

/// How long one pass over a family's slice should take. Long enough that
/// start-up and the last straggler thread stop mattering, short enough that the
/// slowest engine does not decide the runtime of the whole suite.
const TARGET_SECS: f64 = 2.0;
/// Never measure less than this, however slow the engine: below it, one long
/// document lands in one thread and the number becomes noise.
const MIN_SLICE_BYTES: usize = 8 << 20;

/// The prefix of `refs` whose bytes sum to at least `want`, and its true size.
fn slice_of<'a>(refs: &'a [&'a str], want: usize) -> (&'a [&'a str], usize) {
    let (mut end, mut acc) = (0, 0usize);
    while end < refs.len() && acc < want {
        acc += refs[end].len();
        end += 1;
    }
    (&refs[..end], acc)
}

/// Corpus size for one family, set by its **slowest** engine.
///
/// Every engine in a table has to see the same input or the columns are not a
/// comparison, so the slice cannot be chosen per engine. It also cannot be one
/// size for the whole suite: HF `tokenizers` runs at ~4 MB/s where splintr runs
/// at ~64, so a slice that keeps HF's table honest wastes minutes on families
/// it does not appear in. Probing settles it — a cheap timed pass over a small
/// prefix estimates each engine's rate, and the family takes what the weakest
/// one can cover in `TARGET_SECS`.
fn family_slice<'a>(
    engines: &[&Engine],
    docs: &'a [String],
    refs: &'a [&'a str],
    parallel: bool,
) -> (usize, &'static str) {
    let (probe, probe_bytes) = slice_of(refs, MIN_SLICE_BYTES / 4);
    let probe_docs = &docs[..probe.len()];
    let mut slowest_mbps = f64::MAX;
    let mut slowest = "";
    for e in engines {
        let secs = if parallel {
            let t = Instant::now();
            std::hint::black_box((e.encode_all)(probe_docs, probe));
            t.elapsed().as_secs_f64()
        } else {
            let t = Instant::now();
            for s in probe {
                std::hint::black_box((e.encode)(s));
            }
            t.elapsed().as_secs_f64()
        };
        let mbps = probe_bytes as f64 / 1e6 / secs.max(1e-9);
        if mbps < slowest_mbps {
            slowest_mbps = mbps;
            slowest = e.name;
        }
    }
    let want = (slowest_mbps * 1e6 * TARGET_SECS) as usize;
    (want.max(MIN_SLICE_BYTES), slowest)
}

/// 64 MB of stack for every thread that touches a tokenizer.
///
/// HF `tokenizers` overflows the default 2 MB stack on WikiText documents
/// (~18 KB) and takes the process down with it — a stack overflow aborts and
/// cannot be caught. Measuring it at all means giving it room. splintr and the
/// rest are unaffected by the larger stack, so this changes no other number.
const STACK: usize = 64 << 20;

fn main() {
    rayon::ThreadPoolBuilder::new()
        .stack_size(STACK)
        .build_global()
        .expect("rayon pool");
    // The serial pass runs on this thread, so it needs the same headroom.
    std::thread::Builder::new()
        .stack_size(STACK)
        .spawn(run)
        .expect("worker")
        .join()
        .expect("worker panicked");
}

fn run() {
    let mut args = std::env::args().skip(1);
    let vocab = args.next().expect("vocab name");
    let ranks_path = args.next().expect("ranks path");
    let json_path = args.next().expect("json path");
    let corpus_path = args.next().expect("corpus path");

    let pattern: &'static str = match vocab.as_str() {
        "cl100k" => splintr::CL100K_BASE_PATTERN,
        "o200k" => splintr::O200K_BASE_PATTERN,
        "qwen3" => splintr::QWEN2_PATTERN,
        "kimi" => splintr::KIMI_PATTERN,
        other => panic!("no pattern for {other}"),
    };

    let docs = read_corpus(corpus_path.as_str());
    let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
    let total_bytes: usize = refs.iter().map(|s| s.len()).sum();

    eprintln!(
        "corpus {:.1} MB / {} docs; {} threads",
        total_bytes as f64 / 1e6,
        refs.len(),
        rayon::current_num_threads(),
    );

    let (engines, skipped) = build(&vocab, &ranks_path, &json_path, pattern);
    for (name, why) in &skipped {
        eprintln!("skip {name} — {why}");
    }

    // Parity first: a timing table over engines that disagree is not a
    // comparison. splintr's ranks build is the reference.
    let sample: Vec<&str> = refs
        .iter()
        .step_by(refs.len() / 25)
        .take(25)
        .copied()
        .collect();
    let reference: Vec<Vec<u32>> = sample.iter().map(|s| (engines[0].encode)(s)).collect();
    let mut agree: HashMap<&str, bool> = HashMap::new();
    for e in &engines {
        let got: Vec<Vec<u32>> = sample.iter().map(|s| (e.encode)(s)).collect();
        agree.insert(e.name, got == reference);
    }

    let skipped_json: Vec<String> = skipped
        .iter()
        .map(|(n, w)| format!("{{\"engine\":\"{n}\",\"reason\":{}}}", serde_escape(w)))
        .collect();
    println!(
        "{{\"vocab\":\"{vocab}\",\"skipped\":[{}],\"results\":[",
        skipped_json.join(",")
    );
    let mut first = true;
    let mut families: Vec<&str> = Vec::new();
    for e in &engines {
        if !families.contains(&e.family) {
            families.push(e.family);
        }
    }
    for family in families {
        let members: Vec<&Engine> = engines.iter().filter(|e| e.family == family).collect();

        let (serial_want, serial_slowest) = family_slice(&members, &docs, &refs, false);
        let (par_want, par_slowest) = family_slice(&members, &docs, &refs, true);
        let (serial_refs, serial_bytes) = slice_of(&refs, serial_want);
        let (par_refs, par_bytes) = slice_of(&refs, par_want);
        let par_docs = &docs[..par_refs.len()];
        eprintln!(
            "{family}: serial {:.1} MB (paced by {serial_slowest}), \
             parallel {:.1} MB (paced by {par_slowest})",
            serial_bytes as f64 / 1e6,
            par_bytes as f64 / 1e6,
        );

        for e in &members {
            let s = bench(|| {
                for t in serial_refs {
                    std::hint::black_box((e.encode)(t));
                }
            });
            let p = bench(|| {
                std::hint::black_box((e.encode_all)(par_docs, par_refs));
            });
            if !first {
                println!(",");
            }
            first = false;
            print!(
                "{{\"engine\":\"{}\",\"family\":\"{}\",\"own_batch\":{},\"agrees\":{},\
             \"serial_mb\":{:.1},\"par_mb\":{:.1},\
             \"serial_mbps\":{:.1},\"par_mbps\":{:.1},\"par_gbps\":{:.2}}}",
                e.name,
                e.family,
                e.own_batch,
                agree[e.name],
                serial_bytes as f64 / 1e6,
                par_bytes as f64 / 1e6,
                serial_bytes as f64 / 1e6 / s,
                par_bytes as f64 / 1e6 / p,
                par_bytes as f64 / 1e9 / p,
            );
        }
    }
    println!("\n]}}");
}
