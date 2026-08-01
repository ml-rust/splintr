pub mod core;
#[cfg(feature = "python")]
mod python;

pub use core::{
    AddedToken, AddedTokenSet, ByteLevelStreamingDecoder, SentencePieceError,
    SentencePieceTokenizer, SpmError, SpmPrefixScheme, SpmTokenizer, StreamingDecoder, Tokenize,
    TokenizeError, Tokenizer, TokenizerError, WordPieceError, WordPieceTokenizer,
    CL100K_BASE_PATTERN, DEEPSEEK_V3_PATTERNS, GPT2_PATTERN, LLAMA3_PATTERN, MISTRAL_V3_PATTERN,
    O200K_BASE_PATTERN, QWEN2_PATTERN, SENTENCEPIECE_PATTERN,
};

// Re-export pretrained tokenizer API
pub use core::pretrained;
// Re-export Whisper tokenizer API
pub use core::whisper;
pub use core::{
    base_vocab_size, base_vocab_size_by_name, bos_token_id, bos_token_id_by_name,
    cl100k_base_special_tokens, deepseek_v3_special_tokens, eos_token_id, eos_token_id_by_name,
    from_pretrained, from_vocab, llama3_special_tokens, o200k_base_special_tokens, pad_token_id,
    patterns, special_tokens, uses_byte_level, PretrainedVocab,
};
pub use core::{whisper_special_tokens, WhisperVariant};
// Re-export the universal loaded-tokenizer type and its special-token policy,
// plus the two loaders that produce them: a HuggingFace `tokenizer.json`, and a
// GGUF file's embedded vocabulary as extracted by the caller.
pub use core::{
    from_gguf_vocab, from_json_bytes, from_json_path, AnyTokenizer, Backend, GgufVocab,
    GgufVocabError, HfJsonError, PolicyError, SpecialMode, SpecialPolicy,
};

/// Splintr - Fast Rust tokenizer (BPE + SentencePiece + WordPiece) with Python bindings
///
/// A high-performance tokenizer featuring:
/// - Regexr with JIT and SIMD (default, pure Rust)
/// - Optional PCRE2 with JIT (requires `pcre2` feature)
/// - Rayon parallelism for multi-core encoding
/// - Linked-list BPE algorithm (avoids O(N²) on pathological inputs)
/// - SentencePiece unigram with Viterbi maximum-score segmentation (true Unigram) and byte fallback
/// - WordPiece tokenizer for BERT-family models with `##` continuation prefix
/// - FxHashMap for fast lookups
/// - Aho-Corasick for fast special token matching
/// - LRU cache for frequently encoded chunks
/// - UTF-8 streaming decoder for LLM output
/// - Agent tokens for chat/reasoning/tool-use applications
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python::PyTokenizer>()?;
    m.add_class::<python::PySentencePieceTokenizer>()?;
    m.add_class::<python::PySpmTokenizer>()?;
    m.add_class::<python::PyWordPieceTokenizer>()?;
    m.add_class::<python::PyAnyTokenizer>()?;
    m.add_class::<python::PyStreamingDecoder>()?;
    m.add_class::<python::PyByteLevelStreamingDecoder>()?;
    m.add_function(wrap_pyfunction!(python::from_json, m)?)?;
    m.add_function(wrap_pyfunction!(python::from_json_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(python::base_vocab_size, m)?)?;
    // Register all agent token classes (auto-generated from scripts/generate_agent_tokens.py)
    python::register_agent_tokens(m)?;
    m.add("CL100K_BASE_PATTERN", CL100K_BASE_PATTERN)?;
    m.add("O200K_BASE_PATTERN", O200K_BASE_PATTERN)?;
    m.add("LLAMA3_PATTERN", LLAMA3_PATTERN)?;
    // Whether the optional `pcre2` regex backend was compiled in. Exposed so a
    // caller (or a test) can query the capability directly instead of inferring
    // it from an error message, which would silently start reporting "absent"
    // if that message were ever reworded.
    m.add("HAS_PCRE2", cfg!(feature = "pcre2"))?;
    Ok(())
}
