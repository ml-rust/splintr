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
//! Which vocabularies run is `.github/perf-vocabs.tsv`, passed in one row at a
//! time; every corpus named on the command line is measured separately, because
//! a tokenizer's cost differs by script far more than it differs by engine.
//!
//! Ids are checked against the group's own oracle before any timing is read —
//! HuggingFace `tokenizers` for `json`, since that is the implementation the
//! format is defined by, and splintr's build for `ranks`, where there is no
//! upstream to defer to.

use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::HashMap;
use std::time::Instant;

/// Timed passes per measurement, after a warm-up. The best of them is taken.
///
/// Three. Trimming this to two was tried and put back: what the run costs is
/// set by how big a slice [`TARGET_SECS`] picks, not by how many times it is
/// walked, and the third pass is what protects a jittery engine from being
/// reported at a rate it does not have — `tokie` alone varied 97..234 MB/s
/// across consecutive runs of the identical binary.
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
    /// Milliseconds to build this engine from the file on disk. Reported
    /// because it is what a short-lived process actually pays: a serving
    /// binary that loads a 30 MB `tokenizer.json` per worker cares more about
    /// this than about the last few MB/s.
    load_ms: f64,
    encode: Encode,
    encode_all: EncodeAll,
}

/// `f`'s value and how long it took, in milliseconds.
fn timed<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let at = Instant::now();
    let value = f();
    (value, at.elapsed().as_secs_f64() * 1e3)
}

/// An engine that loaded this vocabulary and then failed to encode with it.
///
/// Kept apart from [`Skipped`] because the two say different things, and the
/// report spells them differently. A skipped engine never claimed the shape —
/// there is no such file, or its loader refused — and reads `—`. This one
/// accepted the vocabulary and broke on the text, which reads `x`: a bug in
/// that engine, not a gap in its advertised support.
type Broken = Vec<(&'static str, &'static str, String)>;

/// Take out any engine that cannot survive encoding a short probe, and say why.
///
/// Constructing successfully is not the same as working. `tiktoken-rs` builds
/// happily from Tekken's rank file and then panics inside `encode_ordinary`
/// with "no entry found for key" — a rank file whose bytes it cannot all look
/// up. A panic in one engine used to take the whole run with it, so a single
/// crate's gap in coverage meant no numbers at all for anyone, on any
/// vocabulary after it.
fn take_broken(engines: &mut Vec<Engine>) -> Broken {
    const PROBE: &str = "The quick brown fox — 日本語 42\n\tindent\n";
    let previous = std::panic::take_hook();
    // The default hook prints the panic and its location, which for an engine
    // we are deliberately probing is noise in the middle of the run log.
    std::panic::set_hook(Box::new(|_| {}));
    let mut broken: Broken = Vec::new();
    engines.retain(|e| {
        let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (e.encode)(PROBE)));
        match attempt {
            Ok(_) => true,
            Err(payload) => {
                let why = payload
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "panicked".to_string());
                broken.push((e.name, e.family, format!("panicked while encoding: {why}")));
                false
            }
        }
    });
    std::panic::set_hook(previous);
    broken
}

/// kitoken's ids as `u32`.
///
/// `encode` returns a `Result` where the rest of this file returns ids — an
/// error here means no tokens for that text, which the parity check then
/// reports as a disagreement rather than letting it pass as an
/// empty-but-equal answer.
fn kitoken_ids(tok: &kitoken::Kitoken, text: &str) -> Vec<u32> {
    tok.encode(text, false).unwrap_or_default()
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

fn build(ranks_path: &str, json_path: &str, pattern: &'static str) -> (Vec<Engine>, Skipped) {
    let mut engines: Vec<Engine> = Vec::new();
    let mut skipped: Skipped = Vec::new();

    // --- ranks family -------------------------------------------------------
    // `-` means this vocabulary publishes no `.tiktoken` rank file, the mirror
    // of `-` for json below. Most of the families in the manifest ship a
    // `tokenizer.json` and nothing else, and a missing rank file is a family
    // that does not apply here — not an error, and not a reason to leave the
    // model out of the report entirely, which is what a mandatory rank file
    // did.
    let publishes_ranks = ranks_path != "-";
    if publishes_ranks {
        let (tok, load_ms) = timed(|| {
            let ranks = splintr::core::load_tiktoken_bpe_file(ranks_path).expect("rank file");
            splintr::Tokenizer::new(ranks, FxHashMap::default(), pattern).expect("splintr")
        });
        engines.push(Engine {
            name: "splintr",
            family: "ranks",
            own_batch: true,
            load_ms,
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
    // kitoken reads a rank file directly. Note it takes no pre-tokenizer
    // expression: it derives one, where every other member of this family is
    // handed the same string. If that derivation differs from the vocabulary's
    // real one the ids will differ, and the parity column is what says so —
    // the row is reported either way rather than quietly omitted.
    let (built, load_ms) =
        timed(|| publishes_ranks.then(|| kitoken::Kitoken::from_tiktoken_file(ranks_path)));
    match built {
        None => {}
        Some(Err(e)) => skipped.push(("kitoken", e.to_string())),
        Some(Ok(tok)) => {
            let tok = std::sync::Arc::new(tok);
            engines.push(Engine {
                name: "kitoken",
                family: "ranks",
                own_batch: false, // no batch API; rayon below is the idiomatic use
                load_ms,
                encode: {
                    let t = tok.clone();
                    Box::new(move |s| kitoken_ids(&t, s))
                },
                encode_all: {
                    let t = tok.clone();
                    Box::new(move |_, ts| ts.par_iter().map(|s| kitoken_ids(&t, s).len()).sum())
                },
            });
        }
    }
    let (built, load_ms) = timed(|| {
        publishes_ranks.then(|| {
            riptoken::CoreBPE::new(tiktoken_ranks(ranks_path), FxHashMap::default(), pattern)
        })
    });
    match built {
        None => {}
        Some(Err(e)) => skipped.push(("riptoken", e.to_string())),
        Some(Ok(bpe)) => {
            let bpe = std::sync::Arc::new(bpe);
            engines.push(Engine {
                name: "riptoken",
                family: "ranks",
                own_batch: true,
                load_ms,
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
        let (built, load_ms) = timed(|| splintr::from_json_path(json_path));
        match built {
            Err(e) => skipped.push(("splintr (json)", e.to_string())),
            Ok(tok) => {
                let tok = std::sync::Arc::new(tok);
                engines.push(Engine {
                    name: "splintr",
                    family: "json",
                    own_batch: true,
                    load_ms,
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
        let (built, load_ms) = timed(|| tokenizers::Tokenizer::from_file(json_path));
        match built {
            Err(e) => skipped.push(("HF tokenizers", e.to_string())),
            Ok(tok) => {
                let tok = std::sync::Arc::new(tok);
                engines.push(Engine {
                    name: "HF tokenizers",
                    family: "json",
                    own_batch: true,
                    load_ms,
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
        let (built, load_ms) = timed(|| kitoken::Kitoken::from_tokenizers_file(json_path));
        match built {
            Err(e) => skipped.push(("kitoken", e.to_string())),
            Ok(tok) => {
                let tok = std::sync::Arc::new(tok);
                engines.push(Engine {
                    name: "kitoken",
                    family: "json",
                    own_batch: false,
                    load_ms,
                    encode: {
                        let t = tok.clone();
                        Box::new(move |s| kitoken_ids(&t, s))
                    },
                    encode_all: {
                        let t = tok.clone();
                        Box::new(move |_, ts| ts.par_iter().map(|s| kitoken_ids(&t, s).len()).sum())
                    },
                });
            }
        }
        let (built, load_ms) = timed(|| tokie::Tokenizer::from_json(json_path));
        match built {
            Err(e) => skipped.push(("tokie", e.to_string())),
            Ok(tok) => {
                let tok = std::sync::Arc::new(tok);
                engines.push(Engine {
                    name: "tokie",
                    family: "json",
                    own_batch: false,
                    load_ms,
                    // `encode_ids` rather than `encode`: the ids-only path, so
                    // tokie is not charged for the offsets nobody else here is
                    // building either.
                    encode: {
                        let t = tok.clone();
                        Box::new(move |s| t.encode_ids(s, false))
                    },
                    encode_all: {
                        let t = tok.clone();
                        Box::new(move |_, ts| {
                            ts.par_iter().map(|s| t.encode_ids(s, false).len()).sum()
                        })
                    },
                });
            }
        }
        let (built, load_ms) =
            timed(|| basetenkenizer::Tokenizer::from_file(std::path::Path::new(json_path)));
        match built {
            Err(e) => skipped.push(("basetenkenizer", format!("{e:?}"))),
            Ok(tok) => {
                let tok = std::sync::Arc::new(tok);
                engines.push(Engine {
                    name: "basetenkenizer",
                    family: "json",
                    own_batch: true,
                    load_ms,
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

/// Shortest pass worth timing. Below it the clock and the scheduler are a
/// visible fraction of the answer.
const MIN_PASS_SECS: f64 = 0.15;

/// Seconds for one pass over the slice, as the best of [`ROUNDS`] attempts.
///
/// The minimum, not the mean. Interference from the scheduler and from
/// whatever else the machine is doing is one-sided — it can only make a pass
/// slower — so the best pass is the closest thing to what the code costs, and
/// an average would report how busy the runner was as though it were a
/// property of the tokenizer.
///
/// Engines are measured one after another rather than interleaved, so a
/// transient lands inside whichever engine's window it happens to hit and
/// looks like that engine being erratic. Taking the minimum is what keeps that
/// out of the number; the Python suite interleaves its rounds across engines
/// for the same reason.
///
/// A pass too quick to measure is repeated until it is not, and the total
/// divided back down. The corpora are a few megabytes each, and the fastest
/// engines cross one in under a millisecond on all cores — gemma's batch path
/// managed 0.8 ms — which is a figure made mostly of timer resolution and
/// whatever else the runner was doing. Repetition is safe here precisely
/// because every pass is post-warm-up: the chunk cache is already full, so the
/// second pass over a slice costs what the first one did.
fn bench(f: impl Fn()) -> f64 {
    let at = Instant::now();
    f(); // warm, and an estimate of what one pass costs
    let once = at.elapsed().as_secs_f64();
    let passes = match once > 0.0 {
        true => ((MIN_PASS_SECS / once).ceil() as usize).clamp(1, 4096),
        false => 4096,
    };

    let mut best = f64::MAX;
    for _ in 0..ROUNDS {
        let t = Instant::now();
        for _ in 0..passes {
            f();
        }
        best = best.min(t.elapsed().as_secs_f64() / passes as f64);
    }
    best
}

/// How long one pass over a family's slice should take, for the slowest engine
/// in it. Long enough that start-up and the last straggler thread stop
/// mattering, short enough that the slowest engine does not decide the runtime
/// of the whole suite.
///
/// This is the single knob that sets what the run costs, because the slice it
/// picks is the slice *every* engine in the family then walks. HuggingFace
/// `tokenizers` reads ~5 MB/s where splintr reads ~190, so at two seconds it
/// alone spent four seconds per family per corpus and the whole manifest took
/// ten minutes. A third of a second still swamps timer noise — the gaps here
/// are twofold and larger, not a few percent.
const TARGET_SECS: f64 = 0.35;
/// Never measure less than this, however slow the engine: below it, one long
/// document lands in one thread and the number becomes noise.
const MIN_SLICE_BYTES: usize = 256 << 10;

/// Prefix used to estimate each engine's rate before sizing the real slice.
/// Only the ratio between engines matters here, so it can be small — and it
/// must be, since every engine pays it twice per family per corpus.
const PROBE_BYTES: usize = 256 << 10;

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
    let (probe, probe_bytes) = slice_of(refs, PROBE_BYTES);
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

/// splintr's own constant for a pre-tokenizer expression, by the name the
/// manifest uses.
///
/// Resolved here rather than transcribed into the manifest for the reason the
/// manifest itself records: a hand-typed pattern silently benchmarks a
/// different pre-tokenizer for every engine at once, and parity cannot catch it
/// because they all agree on the wrong answer.
fn pattern_named(name: &str) -> &'static str {
    match name {
        "CL100K_BASE_PATTERN" => splintr::CL100K_BASE_PATTERN,
        "O200K_BASE_PATTERN" => splintr::O200K_BASE_PATTERN,
        "QWEN2_PATTERN" => splintr::QWEN2_PATTERN,
        "KIMI_PATTERN" => splintr::KIMI_PATTERN,
        "LLAMA3_PATTERN" => splintr::LLAMA3_PATTERN,
        "MISTRAL_V3_PATTERN" => splintr::MISTRAL_V3_PATTERN,
        "GPT2_PATTERN" => splintr::GPT2_PATTERN,
        // Only reached with a rank file, which is the only thing a pattern is
        // for. Without one the value is never read.
        "-" => splintr::GPT2_PATTERN,
        other => panic!("no splintr constant named {other}"),
    }
}

/// The engine a family's ids are checked against.
///
/// For `json` that is HuggingFace `tokenizers`, the implementation the format
/// is defined by — checking splintr against splintr proves nothing, and a
/// vocabulary loaded two different ways is not the same question. For `ranks`
/// it is splintr's own build, because the family's other members read the very
/// rank file it does and there is no upstream to defer to.
fn oracle_for(family: &str) -> &'static str {
    match family {
        "json" => "HF tokenizers",
        _ => "splintr",
    }
}

fn run() {
    let mut args = std::env::args().skip(1);
    let vocab = args.next().expect("vocab name");
    let ranks_path = args.next().expect("ranks path");
    let json_path = args.next().expect("json path");
    let pattern = pattern_named(&args.next().expect("pattern name"));
    let corpora: Vec<String> = args.collect();
    assert!(!corpora.is_empty(), "at least one corpus path");

    let (mut engines, skipped) = build(&ranks_path, &json_path, pattern);
    let broken = take_broken(&mut engines);
    for (name, why) in &skipped {
        eprintln!("skip {name} — {why}");
    }
    for (name, family, why) in &broken {
        eprintln!("broken {name} ({family}) — {why}");
    }

    let skipped_json: Vec<String> = skipped
        .iter()
        .map(|(n, w)| format!("{{\"engine\":\"{n}\",\"reason\":{}}}", serde_escape(w)))
        .collect();
    println!(
        "{{\"vocab\":\"{vocab}\",\"skipped\":[{}],\"results\":[",
        skipped_json.join(",")
    );

    let mut families: Vec<&str> = Vec::new();
    for family in engines
        .iter()
        .map(|e| e.family)
        .chain(broken.iter().map(|(_, f, _)| *f))
    {
        if !families.contains(&family) {
            families.push(family);
        }
    }

    // One record per engine per family per corpus. A tokenizer's cost is not
    // one number — the same change measured -17% on English and -2.5% on
    // Chinese — so the corpora are never averaged into one figure here. The
    // report aggregates for its summary and keeps this detail in the artifact.
    let mut first = true;
    for corpus_path in &corpora {
        let corpus = std::path::Path::new(corpus_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(corpus_path.as_str())
            .to_string();
        let docs = read_corpus(corpus_path.as_str());
        let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
        let total_bytes: usize = refs.iter().map(|s| s.len()).sum();
        eprintln!(
            "{corpus}: {:.1} MB / {} docs; {} threads",
            total_bytes as f64 / 1e6,
            refs.len(),
            rayon::current_num_threads(),
        );

        // Parity first: a timing table over engines that disagree is not a
        // comparison. Checked per corpus, because a disagreement is usually
        // script-specific — an engine that matches on English and breaks on
        // Devanagari would pass a check run only on the first corpus.
        let step = (refs.len() / 25).max(1);
        let sample: Vec<&str> = refs.iter().step_by(step).take(25).copied().collect();
        let mut agree: HashMap<&str, bool> = HashMap::new();
        for family in &families {
            let members: Vec<&Engine> = engines.iter().filter(|e| e.family == *family).collect();
            let oracle = members
                .iter()
                .find(|e| e.name == oracle_for(family))
                .or(members.first());
            let Some(oracle) = oracle else { continue };
            let reference: Vec<Vec<u32>> = sample.iter().map(|s| (oracle.encode)(s)).collect();
            for e in &members {
                let got: Vec<Vec<u32>> = sample.iter().map(|s| (e.encode)(s)).collect();
                agree.insert(e.name, got == reference);
            }
        }

        for family in &families {
            // An engine that loaded this vocabulary and then broke on it still
            // gets a cell — it belongs in the table saying `x`, not left out
            // as though it had never claimed the shape.
            for (name, fam, why) in broken.iter().filter(|(_, f, _)| f == family) {
                if !first {
                    println!(",");
                }
                first = false;
                print!(
                    "{{\"engine\":\"{name}\",\"family\":\"{fam}\",\"corpus\":\"{corpus}\",\
                     \"broken\":true,\"reason\":{}}}",
                    serde_escape(why)
                );
            }

            let members: Vec<&Engine> = engines.iter().filter(|e| e.family == *family).collect();
            if members.is_empty() {
                continue;
            }

            let (serial_want, serial_slowest) = family_slice(&members, &docs, &refs, false);
            let (par_want, par_slowest) = family_slice(&members, &docs, &refs, true);
            let (serial_refs, serial_bytes) = slice_of(&refs, serial_want);
            let (par_refs, par_bytes) = slice_of(&refs, par_want);
            let par_docs = &docs[..par_refs.len()];
            eprintln!(
                "  {family}: serial {:.1} MB (paced by {serial_slowest}), \
                 parallel {:.1} MB (paced by {par_slowest})",
                serial_bytes as f64 / 1e6,
                par_bytes as f64 / 1e6,
            );

            for e in &members {
                // Tokens are counted once, outside the timed passes, and
                // reported because equal throughput on unequal token counts is
                // not equal work: a vocabulary that emits 20% more tokens for
                // the same bytes did 20% less compression, and the reader
                // cannot see that from MB/s alone.
                let tokens: usize = serial_refs.iter().map(|t| (e.encode)(t).len()).sum();
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
                    "{{\"engine\":\"{}\",\"family\":\"{}\",\"corpus\":\"{}\",\
                 \"own_batch\":{},\"agrees\":{},\"load_ms\":{:.1},\
                 \"serial_mb\":{:.1},\"par_mb\":{:.1},\"tokens\":{},\
                 \"bytes_per_token\":{:.2},\
                 \"serial_mbps\":{:.1},\"par_mbps\":{:.1},\"par_gbps\":{:.2}}}",
                    e.name,
                    e.family,
                    corpus,
                    e.own_batch,
                    agree.get(e.name).copied().unwrap_or(false),
                    e.load_ms,
                    serial_bytes as f64 / 1e6,
                    par_bytes as f64 / 1e6,
                    tokens,
                    serial_bytes as f64 / tokens.max(1) as f64,
                    serial_bytes as f64 / 1e6 / s,
                    par_bytes as f64 / 1e6 / p,
                    par_bytes as f64 / 1e9 / p,
                );
            }
        }
    }
    println!("\n]}}");
}
