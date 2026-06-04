//! Core tokenization engine for splintr.
//!
//! This module contains multi-backend tokenizer implementations:
//! - Byte-pair encoding using a linked-list algorithm (O(N) merges vs O(N²) for vectors)
//! - SentencePiece unigram tokenizer with score-based tie-breaking
//! - WordPiece tokenizer for BERT-family models with `##` continuation prefix
//! - Vocabulary loading from tiktoken format
//! - UTF-8 safe streaming decoder for LLM output
//! - Main tokenizer interface with LRU caching and Rayon parallelism
//! - Unified [`Tokenize`] trait for backend-agnostic usage
//!
//! # Architecture
//!
//! The core is organized into four main components:
//!
//! - [`Tokenizer`]: Main tokenizer struct with encoding/decoding API, LRU cache,
//!   and Aho-Corasick special token matching. Uses regexr backend by default,
//!   with optional PCRE2 backend via `.pcre2(true)` (requires `pcre2` feature).
//! - `bpe`: Low-level byte-pair encoding algorithm using linked-list approach
//! - `vocab`: Vocabulary loading utilities for tiktoken format
//! - [`StreamingDecoder`]: UTF-8 safe streaming decoder for token-by-token LLM output
//! - [`ByteLevelStreamingDecoder`]: Streaming decoder for ByteLevel tokenizers (DeepSeek, GPT-2)
//!
//! # Performance Optimizations
//!
//! - **Regexr with JIT**: Pure Rust regex engine with SIMD acceleration (default)
//! - **Optional PCRE2 with JIT**: requires `pcre2` feature
//! - **Rayon parallelism**: Multi-core encoding for batch operations
//! - **FxHashMap**: Faster hashing than standard HashMap for string keys
//! - **Aho-Corasick**: O(N) multi-pattern matching for special tokens
//! - **LRU Cache**: Avoids redundant BPE computation for repeated chunks

mod bpe;
pub mod byte_level;
pub mod pretrained;
pub mod sentencepiece;
mod streaming;
pub mod tokenize;
mod tokenizer;
mod vocab;
pub mod whisper;
pub mod wordpiece;

pub use bpe::byte_pair_encode;
pub use byte_level::{byte_level_decode, byte_level_decode_bytes, byte_level_encode};
pub use pretrained::{
    bos_token_id, bos_token_id_by_name, cl100k_base_special_tokens, deepseek_v3_special_tokens,
    eos_token_id, eos_token_id_by_name, from_pretrained, from_vocab, llama3_special_tokens,
    o200k_base_special_tokens, pad_token_id, pattern, special_tokens, uses_byte_level,
    PretrainedVocab,
};
pub use sentencepiece::{SentencePieceError, SentencePieceTokenizer};
pub use streaming::{ByteLevelStreamingDecoder, StreamingDecoder};
pub use tokenize::{Tokenize, TokenizeError};
pub use tokenizer::{
    cl100k_agent_tokens, o200k_agent_tokens, Tokenizer, TokenizerError, CL100K_BASE_PATTERN,
    GPT2_PATTERN, LLAMA3_PATTERN, MISTRAL_V3_PATTERN, O200K_BASE_PATTERN, SENTENCEPIECE_PATTERN,
};
pub use vocab::{build_decoder, load_tiktoken_bpe, load_tiktoken_bpe_file, VocabError};
pub use whisper::{
    from_tokenizer_json_bytes as whisper_from_tokenizer_json_bytes,
    from_tokenizer_json_path as whisper_from_tokenizer_json, whisper_special_tokens,
    WhisperTokenizerError, WhisperVariant, WHISPER_LANGUAGES_V1V2, WHISPER_LANGUAGES_V3,
};
pub use wordpiece::WordPieceTokenizer;
