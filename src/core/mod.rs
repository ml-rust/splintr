//! Core tokenization engine for splintr.
//!
//! This module contains multi-backend tokenizer implementations:
//! - Byte-pair encoding using a linked-list algorithm (O(N) merges vs O(N²) for vectors)
//! - SentencePiece unigram tokenizer (Viterbi maximum-score segmentation) with byte fallback
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
//! - [`StreamingDecoder`]: UTF-8 safe streaming decoder for token-by-token LLM output,
//!   built by [`Tokenizer::streaming_decoder`] so it always matches the vocabulary
//!   it decodes (ByteLevel or raw) and always agrees with [`Tokenizer::decode`]
//!
//! # Performance Optimizations
//!
//! - **Regexr with JIT**: Pure Rust regex engine with SIMD acceleration (default)
//! - **Optional PCRE2 with JIT**: requires `pcre2` feature
//! - **Rayon parallelism**: Multi-core encoding for batch operations
//! - **FxHashMap**: Faster hashing than standard HashMap for string keys
//! - **Aho-Corasick**: O(N) multi-pattern matching for special tokens
//! - **LRU Cache**: Avoids redundant BPE computation for repeated chunks

mod added;
mod any_tokenizer;
pub(crate) mod batch;
mod bpe;
pub mod byte_level;
mod decoder;
pub mod gguf;
pub mod hf_json;
mod metaspace;
pub mod normalizer;
mod policy;
pub mod precompiled;
pub mod pretokenizer;
pub mod pretrained;
pub(crate) mod scratch;
pub mod sentencepiece;
pub mod spm;
pub(crate) mod streaming;
mod token_bytes;
pub mod tokenize;
mod tokenizer;
mod vocab;
pub mod whisper;
pub mod wordpiece;

pub use added::{AddedToken, AddedTokenSet};
pub use any_tokenizer::{AnyTokenizer, Backend};
pub use bpe::byte_pair_encode;
pub use byte_level::{byte_level_decode, byte_level_decode_bytes, byte_level_encode};
pub use gguf::{from_gguf_vocab, GgufVocab, GgufVocabError};
pub use hf_json::{from_json_bytes, from_json_path, HfJsonError};
pub use normalizer::{NormOp, Normalizer};
pub use policy::{PolicyError, SpecialDecode, SpecialMode, SpecialPolicy};
pub use precompiled::Precompiled;
pub use pretokenizer::{PreTokStage, PreTokenizer, SplitBehavior, SplitPattern};
pub use pretrained::{
    base_vocab_size, base_vocab_size_by_name, bos_token_id, bos_token_id_by_name,
    cl100k_base_special_tokens, deepseek_v3_special_tokens, eos_token_id, eos_token_id_by_name,
    from_pretrained, from_vocab, glm4_special_tokens, gpt_oss_special_tokens,
    llama3_special_tokens, o200k_base_special_tokens, pad_token_id, patterns, qwen3_special_tokens,
    special_tokens, uses_byte_level, PretrainedVocab,
};
pub use sentencepiece::{SentencePieceError, SentencePieceTokenizer};
pub use spm::{SpmError, SpmPrefixScheme, SpmTokenizer, NEVER_MERGE};
pub use streaming::StreamingDecoder;
pub use token_bytes::{encoder_from_owned, Decoder, Encoder, TokenBytes};
pub use tokenize::{Tokenize, TokenizeError};
pub use tokenizer::{
    cl100k_agent_tokens, deepseek_v3_agent_tokens, glm4_agent_tokens, gpt_oss_agent_tokens,
    kimi_k2_agent_tokens, kimi_k3_agent_tokens, llama3_agent_tokens, mistral_v1_agent_tokens,
    mistral_v2_agent_tokens, mistral_v3_agent_tokens, o200k_agent_tokens, qwen3_agent_tokens,
    ByteFallback, Tokenizer, TokenizerError, CL100K_BASE_PATTERN, DEEPSEEK_V3_PATTERNS,
    GPT2_PATTERN, KIMI_PATTERN, LLAMA3_PATTERN, MISTRAL_V3_PATTERN, NO_SPLIT_PATTERN,
    O200K_BASE_PATTERN, QWEN2_PATTERN, SENTENCEPIECE_PATTERN,
};
pub use vocab::{
    build_decoder, load_packed_bpe, load_packed_bpe_borrowed, load_spm_vocab, load_tiktoken_bpe,
    load_tiktoken_bpe_file, place_special_pieces, VocabError,
};
pub use whisper::{
    whisper_special_tokens, WhisperVariant, WHISPER_LANGUAGES_V1V2, WHISPER_LANGUAGES_V3,
};
pub use wordpiece::{WordPieceError, WordPieceTokenizer};
