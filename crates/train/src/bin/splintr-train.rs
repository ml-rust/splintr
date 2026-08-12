//! `splintr-train` — train a tokenizer vocabulary from text files.
//!
//! Lives in this crate rather than in `splintr` because the tokenizer crate has
//! no binary target and should not grow an argument parser to gain one.

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand, ValueEnum};
use splintr_train::{
    write, BpeTrainer, Corpus, PreTok, Seeding, TrainError, UnigramTrainer, WordPieceTrainer,
};

#[derive(Parser)]
#[command(
    name = "splintr-train",
    about = "Train a tokenizer vocabulary from text files",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Byte-pair encoding: segmented by replaying merges. The tiktoken and
    /// GPT-family shape.
    Bpe {
        #[command(flatten)]
        common: Common,
        /// Where the vocabulary goes. `.tiktoken` writes ranks, `.json` writes a
        /// full tokenizer.json.
        #[arg(short, long)]
        output: PathBuf,
        /// Seed symbols from raw bytes (nothing is unspellable) or from
        /// characters (needed when the pre-tokenizer is byte-level).
        #[arg(long, value_enum, default_value_t = SeedArg::Bytes)]
        seeding: SeedArg,
    },
    /// WordPiece: segmented by greedy longest match. The BERT-family shape.
    #[command(name = "wordpiece")]
    WordPiece {
        #[command(flatten)]
        common: Common,
        /// `.txt` writes a plain vocab list, `.json` writes a tokenizer.json.
        #[arg(short, long)]
        output: PathBuf,
        /// Drop pieces the segmenter can never emit. Removes almost all of them
        /// and costs a few percent in tokens — see the crate docs.
        #[arg(long)]
        prune: bool,
    },
    /// Unigram: segmented by maximising a sum of log-probabilities. The
    /// SentencePiece shape, and the best of the three at small vocabularies.
    Unigram {
        #[command(flatten)]
        common: Common,
        /// `.spm` writes SentencePiece text, `.json` writes a tokenizer.json.
        #[arg(short, long)]
        output: PathBuf,
    },
}

#[derive(clap::Args)]
struct Common {
    /// Text files to train on. Each line is one document.
    #[arg(required = true)]
    input: Vec<PathBuf>,
    /// Tokens to produce, special tokens included.
    #[arg(short = 'n', long, default_value_t = 32_000)]
    vocab_size: usize,
    /// Special tokens, numbered ahead of (WordPiece, Unigram) or above (BPE) the
    /// pieces.
    #[arg(short, long)]
    special: Vec<String>,
    /// Words occurring fewer than this many times are ignored.
    #[arg(long, default_value_t = 2)]
    min_frequency: u64,
    /// How the text is cut into words before training.
    #[arg(long, value_enum, default_value_t = PreTokArg::Whitespace)]
    pre_tokenizer: PreTokArg,
    /// The expression for `--pre-tokenizer pattern`.
    #[arg(long, default_value = r"\s*\S+")]
    pattern: String,
    /// Mark every word start with U+2581. Required for anything a
    /// SentencePiece-style segmenter will load, and the default for `unigram`.
    #[arg(long)]
    metaspace: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum SeedArg {
    Bytes,
    Chars,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum PreTokArg {
    /// Split on whitespace, isolating punctuation.
    Whitespace,
    /// GPT-2 byte level. Use `--seeding chars` with it.
    ByteLevel,
    /// Split on `--pattern`, tiktoken-style.
    Pattern,
    /// No splitting: each line is one word. Rarely what you want.
    None,
}

impl From<PreTokArg> for PreTok {
    fn from(arg: PreTokArg) -> Self {
        match arg {
            PreTokArg::Whitespace => PreTok::Whitespace,
            PreTokArg::ByteLevel => PreTok::ByteLevel,
            PreTokArg::Pattern => PreTok::Pattern(String::new()),
            PreTokArg::None => PreTok::None,
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), TrainError> {
    let cli = Cli::parse();
    match cli.command {
        Command::Bpe {
            common,
            output,
            seeding,
        } => {
            let counts = read(&common, false)?;
            let vocab = BpeTrainer::builder()
                .vocab_size(common.vocab_size.saturating_sub(common.special.len()))
                .min_frequency(common.min_frequency)
                .special_tokens(common.special.clone())
                .seeding(match seeding {
                    SeedArg::Bytes => Seeding::Bytes,
                    SeedArg::Chars => Seeding::Chars,
                })
                .build()
                .train(&counts)?;

            match extension(&output) {
                "json" => write::bpe_json_file(&vocab, &Default::default(), &output)?,
                _ => write::tiktoken_file(&vocab, &output)?,
            }
            report(&output, vocab.pieces().len() + vocab.specials().len());
            if extension(&output) != "json" {
                // A rank file carries no pattern, and loading it with the wrong
                // one silently produces different ids — so the recipe is written
                // beside it rather than left for the user to remember.
                let sidecar = output.with_extension(format!("{}.recipe.json", extension(&output)));
                if write::recipe_json_file(&vocab, &sidecar)? {
                    println!("recipe (needed to load this file): {}", sidecar.display());
                }
                println!("pattern (needed to load this file): {}", pattern(&common));
            }
        }

        Command::WordPiece {
            common,
            output,
            prune,
        } => {
            let counts = read(&common, false)?;
            let vocab = WordPieceTrainer::builder()
                .vocab_size(common.vocab_size)
                .min_frequency(common.min_frequency)
                .special_tokens(common.special.clone())
                .prune(prune.then(Default::default))
                .build()
                .train(&counts)?;

            match extension(&output) {
                "json" => write::wordpiece_json_file(&vocab, &Default::default(), &output)?,
                _ => write::vocab_txt_file(&vocab, &output)?,
            }
            report(&output, vocab.len());
        }

        Command::Unigram { common, output } => {
            // Unigram output is loaded by SentencePiece-shaped segmenters, which
            // prepend the marker before matching — so it is marked by default
            // rather than on request.
            let counts = read(&common, true)?;
            let vocab = UnigramTrainer::builder()
                .vocab_size(common.vocab_size)
                .min_frequency(common.min_frequency)
                .special_tokens(common.special.clone())
                .build()
                .train(&counts)?;

            match extension(&output) {
                "json" => write::unigram_json_file(&vocab, &Default::default(), &output)?,
                _ => write::spm_file(&vocab, &output)?,
            }
            report(&output, vocab.len());
        }
    }
    Ok(())
}

/// The pre-tokenizer expression in force, for reporting.
fn pattern(common: &Common) -> String {
    match common.pre_tokenizer {
        PreTokArg::Pattern => common.pattern.clone(),
        PreTokArg::Whitespace => "whitespace + isolated punctuation".to_string(),
        PreTokArg::ByteLevel => "byte level".to_string(),
        PreTokArg::None => "none".to_string(),
    }
}

/// Read every input file into word counts.
fn read(common: &Common, force_metaspace: bool) -> Result<splintr_train::WordCounts, TrainError> {
    let pre = match common.pre_tokenizer {
        PreTokArg::Pattern => PreTok::Pattern(common.pattern.clone()),
        other => PreTok::from(other),
    };

    let mut corpus = Corpus::with_pre_tok(pre)?;
    if common.metaspace || force_metaspace {
        corpus = corpus.with_metaspace();
    }
    for path in &common.input {
        corpus.feed_file(path)?;
    }

    let counts = corpus.into_counts();
    println!(
        "read {} file(s): {} distinct words, {} occurrences",
        common.input.len(),
        counts.len(),
        counts.total()
    );
    Ok(counts)
}

fn extension(path: &Path) -> &str {
    path.extension().and_then(|e| e.to_str()).unwrap_or("")
}

fn report(output: &Path, tokens: usize) {
    println!("wrote {} tokens to {}", tokens, output.display());
}
