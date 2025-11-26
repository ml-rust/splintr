pub mod core;
mod python;

use pyo3::prelude::*;

pub use core::{
    ByteLevelStreamingDecoder, StreamingDecoder, Tokenizer, TokenizerError, CL100K_BASE_PATTERN,
    LLAMA3_PATTERN, O200K_BASE_PATTERN,
};

/// Splintr - Fast Rust BPE tokenizer with Python bindings
///
/// A high-performance tokenizer featuring:
/// - PCRE2 with JIT compilation (2-4x faster than fancy-regex)
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
    m.add_class::<python::PyCL100KAgentTokens>()?;
    m.add_class::<python::PyO200KAgentTokens>()?;
    m.add_class::<python::PyLlama3AgentTokens>()?;
    m.add_class::<python::PyDeepSeekV3AgentTokens>()?;
    m.add("CL100K_BASE_PATTERN", CL100K_BASE_PATTERN)?;
    m.add("O200K_BASE_PATTERN", O200K_BASE_PATTERN)?;
    m.add("LLAMA3_PATTERN", LLAMA3_PATTERN)?;
    Ok(())
}
