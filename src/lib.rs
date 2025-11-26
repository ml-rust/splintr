pub mod core;
mod python;

use pyo3::prelude::*;

pub use core::{
    StreamingDecoder, Tokenizer, TokenizerError, CL100K_BASE_PATTERN, O200K_BASE_PATTERN,
};

/// splintr - Fast Rust BPE tokenizer with Python bindings
///
/// A high-performance tokenizer featuring:
/// - PCRE2 with JIT compilation (2-4x faster than fancy-regex)
/// - Rayon parallelism for multi-core encoding
/// - Linked-list BPE algorithm (avoids O(N²) on pathological inputs)
/// - FxHashMap for fast lookups
/// - Aho-Corasick for fast special token matching
/// - LRU cache for frequently encoded chunks
/// - UTF-8 streaming decoder for LLM output
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python::PyTokenizer>()?;
    m.add_class::<python::PyStreamingDecoder>()?;
    m.add("CL100K_BASE_PATTERN", CL100K_BASE_PATTERN)?;
    m.add("O200K_BASE_PATTERN", O200K_BASE_PATTERN)?;
    Ok(())
}
