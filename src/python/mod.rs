mod bindings;

use crate::core::{CL100K_BASE_PATTERN, LLAMA3_PATTERN, O200K_BASE_PATTERN};
pub use bindings::{
    PyByteLevelStreamingDecoder, PyCL100KAgentTokens, PyDeepSeekV3AgentTokens, PyLlama3AgentTokens,
    PyO200KAgentTokens, PyStreamingDecoder, PyTokenizer,
};

use pyo3::prelude::*;

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
    m.add_class::<PyTokenizer>()?;
    m.add_class::<PyStreamingDecoder>()?;
    m.add_class::<PyByteLevelStreamingDecoder>()?;
    m.add_class::<PyCL100KAgentTokens>()?;
    m.add_class::<PyO200KAgentTokens>()?;
    m.add_class::<PyLlama3AgentTokens>()?;
    m.add_class::<PyDeepSeekV3AgentTokens>()?;
    m.add("CL100K_BASE_PATTERN", CL100K_BASE_PATTERN)?;
    m.add("O200K_BASE_PATTERN", O200K_BASE_PATTERN)?;
    m.add("LLAMA3_PATTERN", LLAMA3_PATTERN)?;
    Ok(())
}
