pub mod core;
#[cfg(feature = "python")]
mod python;

pub use core::{
    ByteLevelStreamingDecoder, StreamingDecoder, Tokenizer, TokenizerError, CL100K_BASE_PATTERN,
    LLAMA3_PATTERN, O200K_BASE_PATTERN,
};

// Re-export pretrained tokenizer API
pub use core::pretrained;
pub use core::{
    bos_token_id, cl100k_base_special_tokens, deepseek_v3_special_tokens, eos_token_id,
    eos_token_id_by_name, from_pretrained, from_vocab, llama3_special_tokens,
    o200k_base_special_tokens, pad_token_id, pattern, special_tokens, uses_byte_level,
    PretrainedVocab,
};

/// Splintr - Fast Rust BPE tokenizer with Python bindings
///
/// A high-performance tokenizer featuring:
/// - Regexr with JIT and SIMD (default, pure Rust)
/// - Optional PCRE2 with JIT (requires `pcre2` feature)
/// - Rayon parallelism for multi-core encoding
/// - Linked-list BPE algorithm (avoids O(N²) on pathological inputs)
/// - FxHashMap for fast lookups
/// - Aho-Corasick for fast special token matching
/// - LRU cache for frequently encoded chunks
/// - UTF-8 streaming decoder for LLM output
/// - Agent tokens for chat/reasoning/tool-use applications
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python::PyTokenizer>()?;
    m.add_class::<python::PyStreamingDecoder>()?;
    m.add_class::<python::PyByteLevelStreamingDecoder>()?;
    // Register all agent token classes (auto-generated from scripts/generate_agent_tokens.py)
    python::register_agent_tokens(m)?;
    m.add("CL100K_BASE_PATTERN", CL100K_BASE_PATTERN)?;
    m.add("O200K_BASE_PATTERN", O200K_BASE_PATTERN)?;
    m.add("LLAMA3_PATTERN", LLAMA3_PATTERN)?;
    Ok(())
}
