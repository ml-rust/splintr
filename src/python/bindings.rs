//! Python bindings for the splintr tokenizer.
//!
//! This module provides PyO3 wrappers around the core Rust tokenizer,
//! exposing a Python-friendly API while maintaining Rust performance.
//!
//! # Bundled Vocabularies
//!
//! The module includes pre-loaded vocabularies for:
//! - `cl100k_base`: GPT-4, GPT-3.5-turbo (~100k tokens)
//! - `o200k_base`: GPT-4o (~200k tokens)
//!
//! # Thread Safety
//!
//! The tokenizer is thread-safe and can be shared across Python threads.
//! Batch operations use Rayon for true parallelism, bypassing the GIL
//! during Rust computation.
//!
//! # Example
//!
//! ```python
//! from splintr import Tokenizer
//!
//! # Load pretrained model
//! tokenizer = Tokenizer.from_pretrained("cl100k_base")
//!
//! # Encode/decode
//! tokens = tokenizer.encode("Hello, world!")
//! text = tokenizer.decode(tokens)
//!
//! # Streaming decode for LLM output
//! decoder = tokenizer.streaming_decoder()
//! for token_id in token_stream:
//!     if text := decoder.add_token(token_id):
//!         print(text, end="", flush=True)
//! ```

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustc_hash::FxHashMap;

use crate::core::{
    byte_level_decode_bytes, Tokenizer, CL100K_BASE_PATTERN, LLAMA3_PATTERN, O200K_BASE_PATTERN,
};

/// Bundled cl100k_base vocabulary (GPT-4, GPT-3.5-turbo)
const CL100K_BASE_VOCAB: &[u8] = include_bytes!("../../python/splintr/vocabs/cl100k_base.tiktoken");

/// Bundled o200k_base vocabulary (GPT-4o)
const O200K_BASE_VOCAB: &[u8] = include_bytes!("../../python/splintr/vocabs/o200k_base.tiktoken");

/// Bundled Llama 3 vocabulary (Llama 3/3.1/3.2/3.3)
const LLAMA3_VOCAB: &[u8] = include_bytes!("../../python/splintr/vocabs/llama3.tiktoken");

/// Bundled DeepSeek V3 vocabulary
const DEEPSEEK_V3_VOCAB: &[u8] = include_bytes!("../../python/splintr/vocabs/deepseek_v3.tiktoken");

// =============================================================================
// Special Token Definitions
// =============================================================================
//
// This section defines special tokens for cl100k_base and o200k_base tokenizers.
// These tokens are used for chat formatting, agent architectures, and multimodal
// applications.
//
// ## Token ID Allocation
//
// OpenAI Reserved:
//   - cl100k_base: 100257-100276
//   - o200k_base:  199999-200018
//
// Agent Extensions (added by splintr):
//   - cl100k_base: 100277-100324 (48 tokens)
//   - o200k_base:  200019-200066 (48 tokens)
//
// ## Python Usage
//
// ```python
// from splintr import Tokenizer
//
// tokenizer = Tokenizer.from_pretrained("cl100k_base")
//
// # Encode with special tokens recognized
// tokens = tokenizer.encode_with_special("<|system|>You are helpful.<|user|>Hi!")
//
// # Decode back to text
// text = tokenizer.decode(tokens)
// ```
//
// ## Agent Token Categories
//
// ### 1. Conversation Structure
// Standard ChatML-style tokens for multi-turn conversations:
// - `<|system|>`: System instructions defining assistant behavior
// - `<|user|>`: User input/queries
// - `<|assistant|>`: Assistant responses
// - `<|im_start|>`: Generic message start (ChatML format)
// - `<|im_end|>`: Generic message end (ChatML format)
//
// Example:
// ```
// <|im_start|>system
// You are a helpful assistant.<|im_end|>
// <|im_start|>user
// Hello!<|im_end|>
// <|im_start|>assistant
// Hi there!<|im_end|>
// ```
//
// ### 2. Reasoning/Thinking (Chain-of-Thought)
// For System 2 reasoning similar to DeepSeek-R1 or OpenAI o1:
// - `<|think|>` / `<|/think|>`: Internal reasoning block
//
// Example:
// ```
// <|think|>
// Let me analyze this step by step...
// First, I need to consider X.
// <|/think|>
// The answer is Y.
// ```
//
// ### 3. ReAct Agent Loop
// For ReAct (Reason + Act) agent architectures:
// - `<|plan|>` / `<|/plan|>`: High-level strategy
// - `<|step|>` / `<|/step|>`: Individual action step
// - `<|act|>` / `<|/act|>`: Action intent
// - `<|observe|>` / `<|/observe|>`: Environment feedback
//
// Example:
// ```
// <|plan|>Search for info, then summarize<|/plan|>
// <|act|>search("climate change")<|/act|>
// <|observe|>Found 3 articles...<|/observe|>
// ```
//
// ### 4. Tool/Function Calling
// Structured tool use with error handling:
// - `<|function|>` / `<|/function|>`: Function call (name + args)
// - `<|result|>` / `<|/result|>`: Successful return value
// - `<|error|>` / `<|/error|>`: Execution error (enables retry)
//
// Example:
// ```
// <|function|>{"name": "get_weather", "args": {"city": "London"}}<|/function|>
// <|result|>{"temp": 18, "condition": "cloudy"}<|/result|>
// ```
//
// ### 5. Code Execution
// Jupyter notebook-style code interpreter:
// - `<|code|>` / `<|/code|>`: Code block
// - `<|output|>` / `<|/output|>`: Execution output
// - `<|lang|>` / `<|/lang|>`: Language identifier
//
// Example:
// ```
// <|code|><|lang|>python<|/lang|>
// print(2 + 2)
// <|/code|>
// <|output|>4<|/output|>
// ```
//
// ### 6. RAG/Citations
// Retrieval-Augmented Generation with source attribution:
// - `<|context|>` / `<|/context|>`: Retrieved context
// - `<|quote|>` / `<|/quote|>`: Direct quotation
// - `<|cite|>` / `<|/cite|>`: Citation reference
// - `<|source|>` / `<|/source|>`: Source metadata
//
// Example:
// ```
// <|context|><|source|>doc_123<|/source|>
// The Earth orbits the Sun.<|/context|>
// According to <|cite|>doc_123<|/cite|>, <|quote|>The Earth orbits the Sun.<|/quote|>
// ```
//
// ### 7. Memory/State
// Long-term memory persistence:
// - `<|memory|>` / `<|/memory|>`: Store information
// - `<|recall|>` / `<|/recall|>`: Retrieve stored info
//
// ### 8. Control Tokens
// Sequence control:
// - `<|pad|>`: Padding for batch alignment
// - `<|stop|>`: Generation stop signal
// - `<|sep|>`: Segment separator
//
// ### 9. Multimodal
// Non-text content placeholders:
// - `<|image|>` / `<|/image|>`: Image data
// - `<|audio|>` / `<|/audio|>`: Audio data
// - `<|video|>` / `<|/video|>`: Video data
//
// ### 10. Document Structure
// Semantic layout for parsing structured documents:
// - `<|title|>` / `<|/title|>`: Document/section title
// - `<|section|>` / `<|/section|>`: Semantic section boundary
// - `<|summary|>` / `<|/summary|>`: Condensed content summary
//
// Example:
// ```
// <|title|>Introduction<|/title|>
// <|section|>Content here...<|/section|>
// <|summary|>Key points: X, Y, Z<|/summary|>
// ```
//
// =============================================================================

/// Get the standard special tokens for cl100k_base encoding.
///
/// Returns a map of special token strings to their token IDs for GPT-4
/// and GPT-3.5-turbo models.
///
/// ## OpenAI Standard Tokens (100257-100276)
/// - `<|endoftext|>`: End of text marker (100257)
/// - `<|fim_prefix|>`: Fill-in-the-middle prefix (100258)
/// - `<|fim_middle|>`: Fill-in-the-middle middle (100259)
/// - `<|fim_suffix|>`: Fill-in-the-middle suffix (100260)
/// - `<|endofprompt|>`: End of prompt marker (100276)
///
/// ## Agent Tokens (100277-100324)
/// Extended vocabulary for chat and agent applications. See module docs above.
fn cl100k_base_special_tokens() -> FxHashMap<String, u32> {
    let mut special = FxHashMap::default();
    // OpenAI standard special tokens (100257-100276)
    special.insert("<|endoftext|>".to_string(), 100257);
    special.insert("<|fim_prefix|>".to_string(), 100258);
    special.insert("<|fim_middle|>".to_string(), 100259);
    special.insert("<|fim_suffix|>".to_string(), 100260);
    special.insert("<|endofprompt|>".to_string(), 100276);

    // Agent tokens (100277+) - These extend the vocabulary without conflicting
    // with OpenAI's reserved range

    // Core conversation structure
    special.insert("<|system|>".to_string(), 100277);
    special.insert("<|user|>".to_string(), 100278);
    special.insert("<|assistant|>".to_string(), 100279);
    special.insert("<|im_start|>".to_string(), 100280);
    special.insert("<|im_end|>".to_string(), 100281);

    // Reasoning/thinking tokens (System 2 / Chain-of-Thought)
    special.insert("<|think|>".to_string(), 100282);
    special.insert("<|/think|>".to_string(), 100283);

    // ReAct agent loop tokens
    special.insert("<|plan|>".to_string(), 100284);
    special.insert("<|/plan|>".to_string(), 100285);
    special.insert("<|step|>".to_string(), 100286);
    special.insert("<|/step|>".to_string(), 100287);
    special.insert("<|act|>".to_string(), 100288);
    special.insert("<|/act|>".to_string(), 100289);
    special.insert("<|observe|>".to_string(), 100290);
    special.insert("<|/observe|>".to_string(), 100291);

    // Tool/function calling
    special.insert("<|function|>".to_string(), 100292);
    special.insert("<|/function|>".to_string(), 100293);
    special.insert("<|result|>".to_string(), 100294);
    special.insert("<|/result|>".to_string(), 100295);
    special.insert("<|error|>".to_string(), 100296);
    special.insert("<|/error|>".to_string(), 100297);

    // Code execution
    special.insert("<|code|>".to_string(), 100298);
    special.insert("<|/code|>".to_string(), 100299);
    special.insert("<|output|>".to_string(), 100300);
    special.insert("<|/output|>".to_string(), 100301);
    special.insert("<|lang|>".to_string(), 100302);
    special.insert("<|/lang|>".to_string(), 100303);

    // RAG/context injection
    special.insert("<|context|>".to_string(), 100304);
    special.insert("<|/context|>".to_string(), 100305);
    special.insert("<|quote|>".to_string(), 100306);
    special.insert("<|/quote|>".to_string(), 100307);
    special.insert("<|cite|>".to_string(), 100308);
    special.insert("<|/cite|>".to_string(), 100309);
    special.insert("<|source|>".to_string(), 100310);
    special.insert("<|/source|>".to_string(), 100311);

    // Memory/state management
    special.insert("<|memory|>".to_string(), 100312);
    special.insert("<|/memory|>".to_string(), 100313);
    special.insert("<|recall|>".to_string(), 100314);
    special.insert("<|/recall|>".to_string(), 100315);

    // Control tokens
    special.insert("<|pad|>".to_string(), 100316);
    special.insert("<|stop|>".to_string(), 100317);
    special.insert("<|sep|>".to_string(), 100318);

    // Multimodal placeholders
    special.insert("<|image|>".to_string(), 100319);
    special.insert("<|/image|>".to_string(), 100320);
    special.insert("<|audio|>".to_string(), 100321);
    special.insert("<|/audio|>".to_string(), 100322);
    special.insert("<|video|>".to_string(), 100323);
    special.insert("<|/video|>".to_string(), 100324);

    // Document structure (semantic layout for parsing structured data)
    special.insert("<|title|>".to_string(), 100325);
    special.insert("<|/title|>".to_string(), 100326);
    special.insert("<|section|>".to_string(), 100327);
    special.insert("<|/section|>".to_string(), 100328);
    special.insert("<|summary|>".to_string(), 100329);
    special.insert("<|/summary|>".to_string(), 100330);

    special
}

/// Get the standard special tokens for o200k_base encoding (GPT-4o).
///
/// Returns a simplified set of special tokens for GPT-4o models:
/// - `<|endoftext|>`: End of text marker
/// - `<|endofprompt|>`: End of prompt marker
fn o200k_base_special_tokens() -> FxHashMap<String, u32> {
    let mut special = FxHashMap::default();
    // OpenAI standard special tokens (199999-200018)
    special.insert("<|endoftext|>".to_string(), 199999);
    special.insert("<|endofprompt|>".to_string(), 200018);

    // Agent tokens (200019+) - These extend the vocabulary without conflicting
    // with OpenAI's reserved range

    // Core conversation structure
    special.insert("<|system|>".to_string(), 200019);
    special.insert("<|user|>".to_string(), 200020);
    special.insert("<|assistant|>".to_string(), 200021);
    special.insert("<|im_start|>".to_string(), 200022);
    special.insert("<|im_end|>".to_string(), 200023);

    // Reasoning/thinking tokens (System 2 / Chain-of-Thought)
    special.insert("<|think|>".to_string(), 200024);
    special.insert("<|/think|>".to_string(), 200025);

    // ReAct agent loop tokens
    special.insert("<|plan|>".to_string(), 200026);
    special.insert("<|/plan|>".to_string(), 200027);
    special.insert("<|step|>".to_string(), 200028);
    special.insert("<|/step|>".to_string(), 200029);
    special.insert("<|act|>".to_string(), 200030);
    special.insert("<|/act|>".to_string(), 200031);
    special.insert("<|observe|>".to_string(), 200032);
    special.insert("<|/observe|>".to_string(), 200033);

    // Tool/function calling
    special.insert("<|function|>".to_string(), 200034);
    special.insert("<|/function|>".to_string(), 200035);
    special.insert("<|result|>".to_string(), 200036);
    special.insert("<|/result|>".to_string(), 200037);
    special.insert("<|error|>".to_string(), 200038);
    special.insert("<|/error|>".to_string(), 200039);

    // Code execution
    special.insert("<|code|>".to_string(), 200040);
    special.insert("<|/code|>".to_string(), 200041);
    special.insert("<|output|>".to_string(), 200042);
    special.insert("<|/output|>".to_string(), 200043);
    special.insert("<|lang|>".to_string(), 200044);
    special.insert("<|/lang|>".to_string(), 200045);

    // RAG/context injection
    special.insert("<|context|>".to_string(), 200046);
    special.insert("<|/context|>".to_string(), 200047);
    special.insert("<|quote|>".to_string(), 200048);
    special.insert("<|/quote|>".to_string(), 200049);
    special.insert("<|cite|>".to_string(), 200050);
    special.insert("<|/cite|>".to_string(), 200051);
    special.insert("<|source|>".to_string(), 200052);
    special.insert("<|/source|>".to_string(), 200053);

    // Memory/state management
    special.insert("<|memory|>".to_string(), 200054);
    special.insert("<|/memory|>".to_string(), 200055);
    special.insert("<|recall|>".to_string(), 200056);
    special.insert("<|/recall|>".to_string(), 200057);

    // Control tokens
    special.insert("<|pad|>".to_string(), 200058);
    special.insert("<|stop|>".to_string(), 200059);
    special.insert("<|sep|>".to_string(), 200060);

    // Multimodal placeholders
    special.insert("<|image|>".to_string(), 200061);
    special.insert("<|/image|>".to_string(), 200062);
    special.insert("<|audio|>".to_string(), 200063);
    special.insert("<|/audio|>".to_string(), 200064);
    special.insert("<|video|>".to_string(), 200065);
    special.insert("<|/video|>".to_string(), 200066);

    // Document structure (semantic layout for parsing structured data)
    special.insert("<|title|>".to_string(), 200067);
    special.insert("<|/title|>".to_string(), 200068);
    special.insert("<|section|>".to_string(), 200069);
    special.insert("<|/section|>".to_string(), 200070);
    special.insert("<|summary|>".to_string(), 200071);
    special.insert("<|/summary|>".to_string(), 200072);

    special
}

/// Get the standard special tokens for Llama 3/3.1/3.2/3.3 encoding.
///
/// Returns a map of special token strings to their token IDs for Meta's
/// Llama 3 family of models.
///
/// ## Meta Standard Tokens (128000-128010, 128256)
/// - `<|begin_of_text|>`: Start of text marker (128000)
/// - `<|end_of_text|>`: End of text marker (128001)
/// - `<|finetune_right_pad_id|>`: Padding token for fine-tuning (128004)
/// - `<|step_id|>`: Step marker for reasoning (128005, added in 3.2-Vision)
/// - `<|start_header_id|>`: Header start marker (128006)
/// - `<|end_header_id|>`: Header end marker (128007)
/// - `<|eom_id|>`: End of message marker for tool use (128008, added in 3.1)
/// - `<|eot_id|>`: End of turn marker (128009)
/// - `<|python_tag|>`: Python code interpreter marker (128010, added in 3.1)
/// - `<|image|>`: Image placeholder for vision models (128256, added in 3.2-Vision)
///
/// ## Agent Tokens (128300-128353)
/// Extended vocabulary for chat and agent applications. See module docs above.
fn llama3_special_tokens() -> FxHashMap<String, u32> {
    let mut special = FxHashMap::default();

    // Meta standard special tokens (128000-128010)
    // Based on official Llama 3.3 and 3.2-Vision tokenizer_config.json
    special.insert("<|begin_of_text|>".to_string(), 128000);
    special.insert("<|end_of_text|>".to_string(), 128001);
    special.insert("<|reserved_special_token_0|>".to_string(), 128002);
    special.insert("<|reserved_special_token_1|>".to_string(), 128003);
    special.insert("<|finetune_right_pad_id|>".to_string(), 128004); // Added in Llama 3.1
    special.insert("<|step_id|>".to_string(), 128005); // Added in Llama 3.2-Vision (was reserved_special_token_2)
    special.insert("<|start_header_id|>".to_string(), 128006);
    special.insert("<|end_header_id|>".to_string(), 128007);
    special.insert("<|eom_id|>".to_string(), 128008); // Added in Llama 3.1 - End of Message
    special.insert("<|eot_id|>".to_string(), 128009);
    special.insert("<|python_tag|>".to_string(), 128010); // Added in Llama 3.1 - Code interpreter

    // Agent tokens (128300+) - These extend the vocabulary without conflicting
    // with Meta's reserved range

    // Core conversation structure
    special.insert("<|system|>".to_string(), 128300);
    special.insert("<|user|>".to_string(), 128301);
    special.insert("<|assistant|>".to_string(), 128302);
    special.insert("<|im_start|>".to_string(), 128303);
    special.insert("<|im_end|>".to_string(), 128304);

    // Reasoning/thinking tokens (System 2 / Chain-of-Thought)
    special.insert("<|think|>".to_string(), 128305);
    special.insert("<|/think|>".to_string(), 128306);

    // ReAct agent loop tokens
    special.insert("<|plan|>".to_string(), 128307);
    special.insert("<|/plan|>".to_string(), 128308);
    special.insert("<|step|>".to_string(), 128309);
    special.insert("<|/step|>".to_string(), 128310);
    special.insert("<|act|>".to_string(), 128311);
    special.insert("<|/act|>".to_string(), 128312);
    special.insert("<|observe|>".to_string(), 128313);
    special.insert("<|/observe|>".to_string(), 128314);

    // Tool/function calling
    special.insert("<|function|>".to_string(), 128315);
    special.insert("<|/function|>".to_string(), 128316);
    special.insert("<|result|>".to_string(), 128317);
    special.insert("<|/result|>".to_string(), 128318);
    special.insert("<|error|>".to_string(), 128319);
    special.insert("<|/error|>".to_string(), 128320);

    // Code execution
    special.insert("<|code|>".to_string(), 128321);
    special.insert("<|/code|>".to_string(), 128322);
    special.insert("<|output|>".to_string(), 128323);
    special.insert("<|/output|>".to_string(), 128324);
    special.insert("<|lang|>".to_string(), 128325);
    special.insert("<|/lang|>".to_string(), 128326);

    // RAG/context injection
    special.insert("<|context|>".to_string(), 128327);
    special.insert("<|/context|>".to_string(), 128328);
    special.insert("<|quote|>".to_string(), 128329);
    special.insert("<|/quote|>".to_string(), 128330);
    special.insert("<|cite|>".to_string(), 128331);
    special.insert("<|/cite|>".to_string(), 128332);
    special.insert("<|source|>".to_string(), 128333);
    special.insert("<|/source|>".to_string(), 128334);

    // Memory/state management
    special.insert("<|memory|>".to_string(), 128335);
    special.insert("<|/memory|>".to_string(), 128336);
    special.insert("<|recall|>".to_string(), 128337);
    special.insert("<|/recall|>".to_string(), 128338);

    // Control tokens
    special.insert("<|pad|>".to_string(), 128339);
    special.insert("<|stop|>".to_string(), 128340);
    special.insert("<|sep|>".to_string(), 128341);

    // Multimodal placeholders - aligned with official Meta <|image|> token (128256)
    special.insert("<|image|>".to_string(), 128256); // Official Meta token from Llama 3.2-Vision
    special.insert("<|/image|>".to_string(), 128257);
    special.insert("<|audio|>".to_string(), 128258);
    special.insert("<|/audio|>".to_string(), 128259);
    special.insert("<|video|>".to_string(), 128260);
    special.insert("<|/video|>".to_string(), 128261);

    // Document structure (semantic layout for parsing structured data)
    special.insert("<|title|>".to_string(), 128348);
    special.insert("<|/title|>".to_string(), 128349);
    special.insert("<|section|>".to_string(), 128350);
    special.insert("<|/section|>".to_string(), 128351);
    special.insert("<|summary|>".to_string(), 128352);
    special.insert("<|/summary|>".to_string(), 128353);

    special
}

/// Get the standard special tokens for DeepSeek V3 encoding.
///
/// Returns a map of special token strings to their token IDs for DeepSeek V3 models.
///
/// ## DeepSeek Native Tokens (0-2, 128798-128814)
/// - `<｜begin▁of▁sentence｜>`: BOS token (0)
/// - `<｜end▁of▁sentence｜>`: EOS token (1)
/// - `<｜▁pad▁｜>`: PAD token (2)
/// - `<think>` / `</think>`: Thinking tokens (128798-128799)
/// - `<｜fim▁hole｜>` / `<｜fim▁begin｜>` / `<｜fim▁end｜>`: FIM tokens (128800-128802)
/// - `<｜User｜>`: User message marker (128803)
/// - `<｜Assistant｜>`: Assistant message marker (128804)
/// - `<|EOT|>`: End of turn (128805)
/// - Tool calling tokens (128806-128814)
///
/// ## Agent Tokens (128900-128953)
/// Extended vocabulary for chat and agent applications. See module docs above.
fn deepseek_v3_special_tokens() -> FxHashMap<String, u32> {
    let mut special = FxHashMap::default();

    // DeepSeek native special tokens (0-2)
    special.insert("<｜begin▁of▁sentence｜>".to_string(), 0);
    special.insert("<｜end▁of▁sentence｜>".to_string(), 1);
    special.insert("<｜▁pad▁｜>".to_string(), 2);

    // Thinking tokens (128798-128799)
    special.insert("<think>".to_string(), 128798);
    special.insert("</think>".to_string(), 128799);

    // FIM (Fill-in-the-Middle) tokens (128800-128802)
    special.insert("<｜fim▁hole｜>".to_string(), 128800);
    special.insert("<｜fim▁begin｜>".to_string(), 128801);
    special.insert("<｜fim▁end｜>".to_string(), 128802);

    // Chat tokens (128803-128805)
    special.insert("<｜User｜>".to_string(), 128803);
    special.insert("<｜Assistant｜>".to_string(), 128804);
    special.insert("<|EOT|>".to_string(), 128805);

    // Tool calling tokens (128806-128814)
    special.insert("<｜tool▁calls▁begin｜>".to_string(), 128806);
    special.insert("<｜tool▁calls▁end｜>".to_string(), 128807);
    special.insert("<｜tool▁call▁begin｜>".to_string(), 128808);
    special.insert("<｜tool▁call▁end｜>".to_string(), 128809);
    special.insert("<｜tool▁outputs▁begin｜>".to_string(), 128810);
    special.insert("<｜tool▁outputs▁end｜>".to_string(), 128811);
    special.insert("<｜tool▁output▁begin｜>".to_string(), 128812);
    special.insert("<｜tool▁output▁end｜>".to_string(), 128813);
    special.insert("<｜tool▁sep｜>".to_string(), 128814);

    // Agent tokens (128900+) - These extend the vocabulary without conflicting
    // with DeepSeek's reserved range (128000-128814)

    // Core conversation structure
    special.insert("<|system|>".to_string(), 128900);
    special.insert("<|user|>".to_string(), 128901);
    special.insert("<|assistant|>".to_string(), 128902);
    special.insert("<|im_start|>".to_string(), 128903);
    special.insert("<|im_end|>".to_string(), 128904);

    // Reasoning/thinking tokens (System 2 / Chain-of-Thought)
    special.insert("<|think|>".to_string(), 128905);
    special.insert("<|/think|>".to_string(), 128906);

    // ReAct agent loop tokens
    special.insert("<|plan|>".to_string(), 128907);
    special.insert("<|/plan|>".to_string(), 128908);
    special.insert("<|step|>".to_string(), 128909);
    special.insert("<|/step|>".to_string(), 128910);
    special.insert("<|act|>".to_string(), 128911);
    special.insert("<|/act|>".to_string(), 128912);
    special.insert("<|observe|>".to_string(), 128913);
    special.insert("<|/observe|>".to_string(), 128914);

    // Tool/function calling
    special.insert("<|function|>".to_string(), 128915);
    special.insert("<|/function|>".to_string(), 128916);
    special.insert("<|result|>".to_string(), 128917);
    special.insert("<|/result|>".to_string(), 128918);
    special.insert("<|error|>".to_string(), 128919);
    special.insert("<|/error|>".to_string(), 128920);

    // Code execution
    special.insert("<|code|>".to_string(), 128921);
    special.insert("<|/code|>".to_string(), 128922);
    special.insert("<|output|>".to_string(), 128923);
    special.insert("<|/output|>".to_string(), 128924);
    special.insert("<|lang|>".to_string(), 128925);
    special.insert("<|/lang|>".to_string(), 128926);

    // RAG/context injection
    special.insert("<|context|>".to_string(), 128927);
    special.insert("<|/context|>".to_string(), 128928);
    special.insert("<|quote|>".to_string(), 128929);
    special.insert("<|/quote|>".to_string(), 128930);
    special.insert("<|cite|>".to_string(), 128931);
    special.insert("<|/cite|>".to_string(), 128932);
    special.insert("<|source|>".to_string(), 128933);
    special.insert("<|/source|>".to_string(), 128934);

    // Memory/state management
    special.insert("<|memory|>".to_string(), 128935);
    special.insert("<|/memory|>".to_string(), 128936);
    special.insert("<|recall|>".to_string(), 128937);
    special.insert("<|/recall|>".to_string(), 128938);

    // Control tokens
    special.insert("<|pad|>".to_string(), 128939);
    special.insert("<|stop|>".to_string(), 128940);
    special.insert("<|sep|>".to_string(), 128941);

    // Multimodal placeholders
    special.insert("<|image|>".to_string(), 128942);
    special.insert("<|/image|>".to_string(), 128943);
    special.insert("<|audio|>".to_string(), 128944);
    special.insert("<|/audio|>".to_string(), 128945);
    special.insert("<|video|>".to_string(), 128946);
    special.insert("<|/video|>".to_string(), 128947);

    // Document structure (semantic layout for parsing structured data)
    special.insert("<|title|>".to_string(), 128948);
    special.insert("<|/title|>".to_string(), 128949);
    special.insert("<|section|>".to_string(), 128950);
    special.insert("<|/section|>".to_string(), 128951);
    special.insert("<|summary|>".to_string(), 128952);
    special.insert("<|/summary|>".to_string(), 128953);

    special
}

/// Python wrapper for the Rust Tokenizer.
#[pyclass(name = "Tokenizer")]
pub struct PyTokenizer {
    inner: Tokenizer,
}

#[pymethods]
impl PyTokenizer {
    /// Create a new tokenizer from a vocabulary file.
    ///
    /// Args:
    ///     vocab_path: Path to a tiktoken-format vocabulary file
    ///     pattern: PCRE2 regex pattern for tokenization
    ///     special_tokens: Optional dict of special tokens to IDs
    #[new]
    #[pyo3(signature = (vocab_path, pattern, special_tokens=None))]
    fn new(
        vocab_path: &str,
        pattern: &str,
        special_tokens: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let special = parse_special_tokens(special_tokens)?;

        let inner = Tokenizer::from_file(vocab_path, pattern, special)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        Ok(Self { inner })
    }

    /// Create a tokenizer from a pretrained model name.
    ///
    /// Currently supported:
    /// - "cl100k_base" (GPT-4, GPT-3.5-turbo)
    /// - "o200k_base" (GPT-4o)
    /// - "llama3" / "llama3.1" / "llama3.2" / "llama3.3" (Meta Llama 3 family)
    /// - "deepseek_v3" / "deepseek-v3" (DeepSeek V3)
    ///
    /// Args:
    ///     name: Model name (e.g., "cl100k_base", "o200k_base", "llama3", "deepseek_v3")
    ///
    /// Returns:
    ///     Tokenizer instance
    #[staticmethod]
    fn from_pretrained(name: &str) -> PyResult<Self> {
        match name {
            "cl100k_base" => {
                let special = cl100k_base_special_tokens();
                let inner = Tokenizer::from_bytes(CL100K_BASE_VOCAB, CL100K_BASE_PATTERN, special)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(Self { inner })
            }
            "o200k_base" => {
                let special = o200k_base_special_tokens();
                let inner = Tokenizer::from_bytes(O200K_BASE_VOCAB, O200K_BASE_PATTERN, special)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(Self { inner })
            }
            "llama3" | "llama3.1" | "llama3.2" | "llama3.3" => {
                let special = llama3_special_tokens();
                let inner = Tokenizer::from_bytes(LLAMA3_VOCAB, LLAMA3_PATTERN, special)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(Self { inner })
            }
            "deepseek_v3" | "deepseek-v3" => {
                let special = deepseek_v3_special_tokens();
                // DeepSeek uses ByteLevel BPE encoding
                let inner =
                    Tokenizer::from_bytes_byte_level(DEEPSEEK_V3_VOCAB, LLAMA3_PATTERN, special)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(Self { inner })
            }
            _ => Err(PyValueError::new_err(format!(
                "Unknown pretrained model: {}. Supported: cl100k_base, o200k_base, llama3, llama3.1, llama3.2, llama3.3, deepseek_v3",
                name
            ))),
        }
    }

    /// Create a tokenizer from raw vocabulary bytes.
    ///
    /// Args:
    ///     vocab_data: Raw bytes of tiktoken-format vocabulary
    ///     pattern: PCRE2 regex pattern for tokenization
    ///     special_tokens: Optional dict of special tokens to IDs
    #[staticmethod]
    #[pyo3(signature = (vocab_data, pattern, special_tokens=None))]
    fn from_bytes(
        vocab_data: &[u8],
        pattern: &str,
        special_tokens: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let special = parse_special_tokens(special_tokens)?;

        let inner = Tokenizer::from_bytes(vocab_data, pattern, special)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(Self { inner })
    }

    /// Encode text to token IDs.
    ///
    /// Special tokens in the input are treated as regular text.
    /// This method uses sequential encoding which is optimal for most use cases.
    ///
    /// Args:
    ///     text: Input text to encode
    ///
    /// Returns:
    ///     List of token IDs
    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }

    /// Encode text to token IDs using Rayon parallel processing.
    ///
    /// This method parallelizes the BPE encoding of individual chunks using Rayon.
    /// It has higher overhead than `encode()` due to thread pool coordination,
    /// but can be faster for very large texts (typically >1MB) where the
    /// parallelization benefit outweighs the overhead.
    ///
    /// For most use cases, prefer `encode()` (sequential) or `encode_batch()`
    /// (parallel across multiple texts).
    ///
    /// Args:
    ///     text: Input text to encode
    ///
    /// Returns:
    ///     List of token IDs
    fn encode_rayon(&self, text: &str) -> Vec<u32> {
        self.inner.encode_rayon(text)
    }

    /// Encode text with special token handling.
    ///
    /// Special tokens in the input are encoded directly without BPE.
    ///
    /// Args:
    ///     text: Input text to encode
    ///
    /// Returns:
    ///     List of token IDs
    fn encode_with_special(&self, text: &str) -> Vec<u32> {
        self.inner.encode_with_special(text)
    }

    /// Decode token IDs to a string.
    ///
    /// Args:
    ///     tokens: List of token IDs
    ///
    /// Returns:
    ///     Decoded string
    ///
    /// Raises:
    ///     ValueError: If decoded bytes are not valid UTF-8
    fn decode(&self, tokens: Vec<u32>) -> PyResult<String> {
        self.inner
            .decode(&tokens)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Decode token IDs to bytes.
    ///
    /// Args:
    ///     tokens: List of token IDs
    ///
    /// Returns:
    ///     Decoded bytes
    fn decode_bytes(&self, tokens: Vec<u32>) -> Vec<u8> {
        self.inner.decode_bytes(&tokens)
    }

    /// Decode token IDs to string, replacing invalid UTF-8.
    ///
    /// Args:
    ///     tokens: List of token IDs
    ///
    /// Returns:
    ///     Decoded string with replacement characters for invalid UTF-8
    fn decode_lossy(&self, tokens: Vec<u32>) -> String {
        self.inner.decode_lossy(&tokens)
    }

    /// Batch encode multiple texts in parallel.
    ///
    /// Uses Rayon to parallelize encoding across texts.
    ///
    /// Args:
    ///     texts: List of texts to encode
    ///
    /// Returns:
    ///     List of token ID lists
    fn encode_batch(&self, texts: Vec<String>) -> Vec<Vec<u32>> {
        self.inner.encode_batch(&texts)
    }

    /// Batch encode multiple texts with special token handling.
    ///
    /// Args:
    ///     texts: List of texts to encode
    ///
    /// Returns:
    ///     List of token ID lists
    fn encode_batch_with_special(&self, texts: Vec<String>) -> Vec<Vec<u32>> {
        self.inner.encode_batch_with_special(&texts)
    }

    /// Batch decode multiple token lists in parallel.
    ///
    /// Uses Rayon to parallelize decoding across token lists.
    ///
    /// Args:
    ///     token_lists: List of token ID lists
    ///
    /// Returns:
    ///     List of decoded strings
    ///
    /// Raises:
    ///     ValueError: If any decoded bytes are not valid UTF-8
    fn decode_batch(&self, token_lists: Vec<Vec<u32>>) -> PyResult<Vec<String>> {
        self.inner
            .decode_batch(&token_lists)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Batch decode multiple token lists, replacing invalid UTF-8.
    ///
    /// Args:
    ///     token_lists: List of token ID lists
    ///
    /// Returns:
    ///     List of decoded strings with replacement characters for invalid UTF-8
    fn decode_batch_lossy(&self, token_lists: Vec<Vec<u32>>) -> Vec<String> {
        self.inner.decode_batch_lossy(&token_lists)
    }

    /// Get the vocabulary size (including special tokens).
    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// Create a streaming decoder for UTF-8 safe token-by-token decoding.
    ///
    /// Useful for streaming LLM output where token boundaries may not
    /// align with UTF-8 character boundaries.
    ///
    /// Returns:
    ///     StreamingDecoder instance
    ///
    /// Example:
    ///     decoder = tokenizer.streaming_decoder()
    ///     for token_id in token_stream:
    ///         if text := decoder.add_token(token_id):
    ///             print(text, end="", flush=True)
    ///     print(decoder.flush())
    fn streaming_decoder(&self) -> PyStreamingDecoder {
        PyStreamingDecoder::new(
            self.inner.decoder().clone(),
            self.inner.special_tokens_decoder().clone(),
        )
    }

    /// Create a ByteLevel streaming decoder for UTF-8 safe token-by-token decoding.
    ///
    /// This decoder is designed for tokenizers using ByteLevel BPE encoding
    /// (GPT-2, Llama, DeepSeek V3) where tokens represent ByteLevel-encoded
    /// characters that need to be decoded back to raw bytes before UTF-8 assembly.
    ///
    /// Returns:
    ///     ByteLevelStreamingDecoder instance
    ///
    /// Example:
    ///     tokenizer = Tokenizer.from_pretrained("deepseek_v3")
    ///     decoder = tokenizer.byte_level_streaming_decoder()
    ///     for token_id in token_stream:
    ///         if text := decoder.add_token(token_id):
    ///             print(text, end="", flush=True)
    ///     print(decoder.flush())
    fn byte_level_streaming_decoder(&self) -> PyByteLevelStreamingDecoder {
        PyByteLevelStreamingDecoder::new(
            self.inner.decoder().clone(),
            self.inner.special_tokens_decoder().clone(),
        )
    }

    /// Clear the encoding cache.
    fn clear_cache(&self) {
        self.inner.clear_cache();
    }

    /// Get the number of entries in the cache.
    #[getter]
    fn cache_len(&self) -> usize {
        self.inner.cache_len()
    }

    /// String representation.
    fn __repr__(&self) -> String {
        format!("Tokenizer(vocab_size={})", self.inner.vocab_size())
    }
}

/// Parse special tokens from Python dict to FxHashMap.
fn parse_special_tokens(
    special_tokens: Option<&Bound<'_, PyDict>>,
) -> PyResult<FxHashMap<String, u32>> {
    let mut result = FxHashMap::default();

    if let Some(dict) = special_tokens {
        for (key, value) in dict.iter() {
            let k: String = key.extract()?;
            let v: u32 = value.extract()?;
            result.insert(k, v);
        }
    }

    Ok(result)
}

/// Python wrapper for streaming decoder.
///
/// Handles UTF-8 safe streaming decode for token-by-token LLM output.
/// Buffers incomplete UTF-8 sequences and only emits complete characters.
#[pyclass(name = "StreamingDecoder")]
pub struct PyStreamingDecoder {
    decoder: FxHashMap<u32, Vec<u8>>,
    special_decoder: FxHashMap<u32, String>,
    buffer: Vec<u8>,
}

#[pymethods]
impl PyStreamingDecoder {
    /// Add a token and return any complete UTF-8 characters.
    ///
    /// Args:
    ///     token_id: The token ID to decode
    ///
    /// Returns:
    ///     String of complete characters, or None if still buffering
    fn add_token(&mut self, token_id: u32) -> Option<String> {
        // Get bytes for this token
        let bytes = if let Some(b) = self.decoder.get(&token_id) {
            b.as_slice()
        } else if let Some(s) = self.special_decoder.get(&token_id) {
            s.as_bytes()
        } else {
            return None;
        };

        // Add to buffer
        self.buffer.extend_from_slice(bytes);

        // Try to extract complete UTF-8 characters
        self.extract_complete_utf8()
    }

    /// Add multiple tokens at once and return complete UTF-8 characters.
    ///
    /// Args:
    ///     token_ids: List of token IDs to decode
    ///
    /// Returns:
    ///     String of complete characters, or None if still buffering
    fn add_tokens(&mut self, token_ids: Vec<u32>) -> Option<String> {
        for token_id in token_ids {
            let bytes = if let Some(b) = self.decoder.get(&token_id) {
                b.as_slice()
            } else if let Some(s) = self.special_decoder.get(&token_id) {
                s.as_bytes()
            } else {
                continue;
            };

            self.buffer.extend_from_slice(bytes);
        }

        self.extract_complete_utf8()
    }

    /// Flush any remaining buffered bytes.
    ///
    /// If there are incomplete UTF-8 sequences in the buffer, they will be
    /// replaced with the Unicode replacement character (U+FFFD).
    ///
    /// Returns:
    ///     Any remaining buffered content
    fn flush(&mut self) -> String {
        if self.buffer.is_empty() {
            return String::new();
        }

        let result = String::from_utf8_lossy(&self.buffer).into_owned();
        self.buffer.clear();
        result
    }

    /// Reset the decoder state, discarding any buffered bytes.
    fn reset(&mut self) {
        self.buffer.clear();
    }

    /// Check if there are buffered bytes waiting for completion.
    #[getter]
    fn has_pending(&self) -> bool {
        !self.buffer.is_empty()
    }

    /// Get the number of pending bytes in the buffer.
    #[getter]
    fn pending_bytes(&self) -> usize {
        self.buffer.len()
    }

    fn __repr__(&self) -> String {
        format!("StreamingDecoder(pending_bytes={})", self.buffer.len())
    }
}

impl PyStreamingDecoder {
    fn new(decoder: FxHashMap<u32, Vec<u8>>, special_decoder: FxHashMap<u32, String>) -> Self {
        Self {
            decoder,
            special_decoder,
            buffer: Vec::with_capacity(16),
        }
    }

    fn extract_complete_utf8(&mut self) -> Option<String> {
        if self.buffer.is_empty() {
            return None;
        }

        let valid_len = self.find_valid_utf8_len();

        if valid_len == 0 {
            return None;
        }

        let valid_bytes: Vec<u8> = self.buffer.drain(..valid_len).collect();
        // SAFETY: We've verified this is valid UTF-8
        let result = unsafe { String::from_utf8_unchecked(valid_bytes) };

        Some(result)
    }

    fn find_valid_utf8_len(&self) -> usize {
        let bytes = &self.buffer;
        let len = bytes.len();

        if len == 0 {
            return 0;
        }

        // First, try to validate the entire buffer
        if std::str::from_utf8(bytes).is_ok() {
            return len;
        }

        // Find how many bytes at the end might be an incomplete sequence
        for incomplete_len in 1..=3.min(len) {
            let check_len = len - incomplete_len;
            if check_len == 0 {
                continue;
            }

            if std::str::from_utf8(&bytes[..check_len]).is_ok()
                && Self::could_be_incomplete_sequence(&bytes[check_len..])
            {
                return check_len;
            }
        }

        // If nothing works, find the last position that's valid
        for i in (0..len).rev() {
            if std::str::from_utf8(&bytes[..=i]).is_ok() {
                return i + 1;
            }
        }

        0
    }

    fn could_be_incomplete_sequence(bytes: &[u8]) -> bool {
        if bytes.is_empty() {
            return false;
        }

        let first = bytes[0];

        match first {
            // 2-byte sequence: 110xxxxx
            0xC0..=0xDF => bytes.len() < 2,
            // 3-byte sequence: 1110xxxx
            0xE0..=0xEF => bytes.len() < 3,
            // 4-byte sequence: 11110xxx
            0xF0..=0xF7 => bytes.len() < 4,
            // Continuation byte or invalid
            _ => false,
        }
    }
}

/// Python wrapper for ByteLevel streaming decoder.
///
/// Handles UTF-8 safe streaming decode for token-by-token LLM output from
/// ByteLevel-encoded tokenizers (GPT-2, Llama, DeepSeek V3). First decodes
/// ByteLevel encoding to raw bytes, then assembles into valid UTF-8 strings.
#[pyclass(name = "ByteLevelStreamingDecoder")]
pub struct PyByteLevelStreamingDecoder {
    decoder: FxHashMap<u32, Vec<u8>>,
    special_decoder: FxHashMap<u32, String>,
    buffer: Vec<u8>,
}

#[pymethods]
impl PyByteLevelStreamingDecoder {
    /// Add a token and return any complete UTF-8 characters.
    ///
    /// The token's ByteLevel-encoded bytes are first decoded to raw bytes,
    /// then assembled into valid UTF-8 strings.
    ///
    /// Args:
    ///     token_id: The token ID to decode
    ///
    /// Returns:
    ///     String of complete characters, or None if still buffering
    fn add_token(&mut self, token_id: u32) -> Option<String> {
        if let Some(encoded_bytes) = self.decoder.get(&token_id) {
            // Decode ByteLevel encoding to raw bytes
            if let Some(raw_bytes) = byte_level_decode_bytes(encoded_bytes) {
                self.buffer.extend_from_slice(&raw_bytes);
            } else {
                // Fallback: treat as raw bytes if ByteLevel decode fails
                self.buffer.extend_from_slice(encoded_bytes);
            }
        } else if let Some(special) = self.special_decoder.get(&token_id) {
            // Special tokens are NOT ByteLevel-encoded, add directly
            self.buffer.extend_from_slice(special.as_bytes());
        } else {
            return None;
        }

        self.extract_complete_utf8()
    }

    /// Add multiple tokens at once and return complete UTF-8 characters.
    ///
    /// Args:
    ///     token_ids: List of token IDs to decode
    ///
    /// Returns:
    ///     String of complete characters, or None if still buffering
    fn add_tokens(&mut self, token_ids: Vec<u32>) -> Option<String> {
        for token_id in token_ids {
            if let Some(encoded_bytes) = self.decoder.get(&token_id) {
                if let Some(raw_bytes) = byte_level_decode_bytes(encoded_bytes) {
                    self.buffer.extend_from_slice(&raw_bytes);
                } else {
                    self.buffer.extend_from_slice(encoded_bytes);
                }
            } else if let Some(special) = self.special_decoder.get(&token_id) {
                self.buffer.extend_from_slice(special.as_bytes());
            }
        }

        self.extract_complete_utf8()
    }

    /// Flush any remaining buffered bytes.
    ///
    /// If there are incomplete UTF-8 sequences in the buffer, they will be
    /// replaced with the Unicode replacement character (U+FFFD).
    ///
    /// Returns:
    ///     Any remaining buffered content
    fn flush(&mut self) -> String {
        if self.buffer.is_empty() {
            return String::new();
        }

        let result = String::from_utf8_lossy(&self.buffer).into_owned();
        self.buffer.clear();
        result
    }

    /// Reset the decoder state, discarding any buffered bytes.
    fn reset(&mut self) {
        self.buffer.clear();
    }

    /// Check if there are buffered bytes waiting for completion.
    #[getter]
    fn has_pending(&self) -> bool {
        !self.buffer.is_empty()
    }

    /// Get the number of pending bytes in the buffer.
    #[getter]
    fn pending_bytes(&self) -> usize {
        self.buffer.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "ByteLevelStreamingDecoder(pending_bytes={})",
            self.buffer.len()
        )
    }
}

impl PyByteLevelStreamingDecoder {
    fn new(decoder: FxHashMap<u32, Vec<u8>>, special_decoder: FxHashMap<u32, String>) -> Self {
        Self {
            decoder,
            special_decoder,
            buffer: Vec::with_capacity(16),
        }
    }

    fn extract_complete_utf8(&mut self) -> Option<String> {
        if self.buffer.is_empty() {
            return None;
        }

        let valid_len = self.find_valid_utf8_len();

        if valid_len == 0 {
            return None;
        }

        let valid_bytes: Vec<u8> = self.buffer.drain(..valid_len).collect();
        // SAFETY: We've verified this is valid UTF-8
        let result = unsafe { String::from_utf8_unchecked(valid_bytes) };

        Some(result)
    }

    fn find_valid_utf8_len(&self) -> usize {
        let bytes = &self.buffer;
        let len = bytes.len();

        if len == 0 {
            return 0;
        }

        // First, try to validate the entire buffer
        if std::str::from_utf8(bytes).is_ok() {
            return len;
        }

        // Find how many bytes at the end might be an incomplete sequence
        for incomplete_len in 1..=3.min(len) {
            let check_len = len - incomplete_len;
            if check_len == 0 {
                continue;
            }

            if std::str::from_utf8(&bytes[..check_len]).is_ok()
                && Self::could_be_incomplete_sequence(&bytes[check_len..])
            {
                return check_len;
            }
        }

        // If nothing works, find the last position that's valid
        for i in (0..len).rev() {
            if std::str::from_utf8(&bytes[..=i]).is_ok() {
                return i + 1;
            }
        }

        0
    }

    fn could_be_incomplete_sequence(bytes: &[u8]) -> bool {
        if bytes.is_empty() {
            return false;
        }

        let first = bytes[0];

        match first {
            0xC0..=0xDF => bytes.len() < 2,
            0xE0..=0xEF => bytes.len() < 3,
            0xF0..=0xF7 => bytes.len() < 4,
            _ => false,
        }
    }
}

// =============================================================================
// Agent Token Constants for Python
// =============================================================================

/// Agent token IDs for cl100k_base (GPT-4, GPT-3.5-turbo).
///
/// Provides constant token IDs for building chat models, reasoning systems,
/// and autonomous agents. Token IDs start at 100277 to avoid conflicts with
/// OpenAI's reserved range (100257-100276).
///
/// # Python Example
///
/// ```python
/// from splintr import CL100K_AGENT_TOKENS
///
/// # Get token IDs
/// system_id = CL100K_AGENT_TOKENS.SYSTEM      # 100277
/// think_id = CL100K_AGENT_TOKENS.THINK        # 100282
///
/// # Use with tokenizer
/// tokenizer = Tokenizer.from_pretrained("cl100k_base")
/// tokens = tokenizer.encode_with_special("<|think|>reasoning<|/think|>")
/// assert CL100K_AGENT_TOKENS.THINK in tokens
/// ```
#[pyclass(name = "CL100K_AGENT_TOKENS", frozen)]
pub struct PyCL100KAgentTokens;

#[pymethods]
impl PyCL100KAgentTokens {
    // =========================================================================
    // Conversation Structure (100277-100281)
    // =========================================================================

    /// System message marker - defines assistant behavior (100277)
    #[classattr]
    const SYSTEM: u32 = 100277;
    /// User message marker - human input (100278)
    #[classattr]
    const USER: u32 = 100278;
    /// Assistant message marker - AI responses (100279)
    #[classattr]
    const ASSISTANT: u32 = 100279;
    /// ChatML message start delimiter (100280)
    #[classattr]
    const IM_START: u32 = 100280;
    /// ChatML message end delimiter (100281)
    #[classattr]
    const IM_END: u32 = 100281;

    // =========================================================================
    // Reasoning/Thinking (100282-100283)
    // =========================================================================

    /// Start of thinking block - Chain-of-Thought reasoning (100282)
    #[classattr]
    const THINK: u32 = 100282;
    /// End of thinking block (100283)
    #[classattr]
    const THINK_END: u32 = 100283;

    // =========================================================================
    // ReAct Agent Loop (100284-100291)
    // =========================================================================

    /// Start of planning phase (100284)
    #[classattr]
    const PLAN: u32 = 100284;
    /// End of planning phase (100285)
    #[classattr]
    const PLAN_END: u32 = 100285;
    /// Start of step (100286)
    #[classattr]
    const STEP: u32 = 100286;
    /// End of step (100287)
    #[classattr]
    const STEP_END: u32 = 100287;
    /// Start of action (100288)
    #[classattr]
    const ACT: u32 = 100288;
    /// End of action (100289)
    #[classattr]
    const ACT_END: u32 = 100289;
    /// Start of observation (100290)
    #[classattr]
    const OBSERVE: u32 = 100290;
    /// End of observation (100291)
    #[classattr]
    const OBSERVE_END: u32 = 100291;

    // =========================================================================
    // Tool/Function Calling (100292-100297)
    // =========================================================================

    /// Start of function call (100292)
    #[classattr]
    const FUNCTION: u32 = 100292;
    /// End of function call (100293)
    #[classattr]
    const FUNCTION_END: u32 = 100293;
    /// Start of function result (100294)
    #[classattr]
    const RESULT: u32 = 100294;
    /// End of function result (100295)
    #[classattr]
    const RESULT_END: u32 = 100295;
    /// Start of error block (100296)
    #[classattr]
    const ERROR: u32 = 100296;
    /// End of error block (100297)
    #[classattr]
    const ERROR_END: u32 = 100297;

    // =========================================================================
    // Code Execution (100298-100303)
    // =========================================================================

    /// Start of code block (100298)
    #[classattr]
    const CODE: u32 = 100298;
    /// End of code block (100299)
    #[classattr]
    const CODE_END: u32 = 100299;
    /// Start of output (100300)
    #[classattr]
    const OUTPUT: u32 = 100300;
    /// End of output (100301)
    #[classattr]
    const OUTPUT_END: u32 = 100301;
    /// Start of language tag (100302)
    #[classattr]
    const LANG: u32 = 100302;
    /// End of language tag (100303)
    #[classattr]
    const LANG_END: u32 = 100303;

    // =========================================================================
    // RAG/Citations (100304-100311)
    // =========================================================================

    /// Start of context block (100304)
    #[classattr]
    const CONTEXT: u32 = 100304;
    /// End of context block (100305)
    #[classattr]
    const CONTEXT_END: u32 = 100305;
    /// Start of quote (100306)
    #[classattr]
    const QUOTE: u32 = 100306;
    /// End of quote (100307)
    #[classattr]
    const QUOTE_END: u32 = 100307;
    /// Start of citation (100308)
    #[classattr]
    const CITE: u32 = 100308;
    /// End of citation (100309)
    #[classattr]
    const CITE_END: u32 = 100309;
    /// Start of source (100310)
    #[classattr]
    const SOURCE: u32 = 100310;
    /// End of source (100311)
    #[classattr]
    const SOURCE_END: u32 = 100311;

    // =========================================================================
    // Memory/State (100312-100315)
    // =========================================================================

    /// Start of memory block (100312)
    #[classattr]
    const MEMORY: u32 = 100312;
    /// End of memory block (100313)
    #[classattr]
    const MEMORY_END: u32 = 100313;
    /// Start of recall block (100314)
    #[classattr]
    const RECALL: u32 = 100314;
    /// End of recall block (100315)
    #[classattr]
    const RECALL_END: u32 = 100315;

    // =========================================================================
    // Control Tokens (100316-100318)
    // =========================================================================

    /// Padding token (100316)
    #[classattr]
    const PAD: u32 = 100316;
    /// Stop token (100317)
    #[classattr]
    const STOP: u32 = 100317;
    /// Separator token (100318)
    #[classattr]
    const SEP: u32 = 100318;

    // =========================================================================
    // Multimodal (100319-100324)
    // =========================================================================

    /// Start of image (100319)
    #[classattr]
    const IMAGE: u32 = 100319;
    /// End of image (100320)
    #[classattr]
    const IMAGE_END: u32 = 100320;
    /// Start of audio (100321)
    #[classattr]
    const AUDIO: u32 = 100321;
    /// End of audio (100322)
    #[classattr]
    const AUDIO_END: u32 = 100322;
    /// Start of video (100323)
    #[classattr]
    const VIDEO: u32 = 100323;
    /// End of video (100324)
    #[classattr]
    const VIDEO_END: u32 = 100324;

    // =========================================================================
    // Document Structure (100325-100330)
    // =========================================================================

    /// Start of title - document/section title (100325)
    #[classattr]
    const TITLE: u32 = 100325;
    /// End of title (100326)
    #[classattr]
    const TITLE_END: u32 = 100326;
    /// Start of section - semantic document section (100327)
    #[classattr]
    const SECTION: u32 = 100327;
    /// End of section (100328)
    #[classattr]
    const SECTION_END: u32 = 100328;
    /// Start of summary - condensed content summary (100329)
    #[classattr]
    const SUMMARY: u32 = 100329;
    /// End of summary (100330)
    #[classattr]
    const SUMMARY_END: u32 = 100330;
}

/// Agent token IDs for o200k_base (GPT-4o).
///
/// Provides constant token IDs for building chat models, reasoning systems,
/// and autonomous agents. Token IDs start at 200019 to avoid conflicts with
/// OpenAI's reserved range (199999-200018).
///
/// # Python Example
///
/// ```python
/// from splintr import O200K_AGENT_TOKENS
///
/// # Get token IDs
/// system_id = O200K_AGENT_TOKENS.SYSTEM      # 200019
/// think_id = O200K_AGENT_TOKENS.THINK        # 200024
/// ```
#[pyclass(name = "O200K_AGENT_TOKENS", frozen)]
pub struct PyO200KAgentTokens;

#[pymethods]
impl PyO200KAgentTokens {
    // =========================================================================
    // Conversation Structure (200019-200023)
    // =========================================================================

    /// System message marker - defines assistant behavior (200019)
    #[classattr]
    const SYSTEM: u32 = 200019;
    /// User message marker - human input (200020)
    #[classattr]
    const USER: u32 = 200020;
    /// Assistant message marker - AI responses (200021)
    #[classattr]
    const ASSISTANT: u32 = 200021;
    /// ChatML message start delimiter (200022)
    #[classattr]
    const IM_START: u32 = 200022;
    /// ChatML message end delimiter (200023)
    #[classattr]
    const IM_END: u32 = 200023;

    // =========================================================================
    // Reasoning/Thinking (200024-200025)
    // =========================================================================

    /// Start of thinking block - Chain-of-Thought reasoning (200024)
    #[classattr]
    const THINK: u32 = 200024;
    /// End of thinking block (200025)
    #[classattr]
    const THINK_END: u32 = 200025;

    // =========================================================================
    // ReAct Agent Loop (200026-200033)
    // =========================================================================

    /// Start of planning phase (200026)
    #[classattr]
    const PLAN: u32 = 200026;
    /// End of planning phase (200027)
    #[classattr]
    const PLAN_END: u32 = 200027;
    /// Start of step (200028)
    #[classattr]
    const STEP: u32 = 200028;
    /// End of step (200029)
    #[classattr]
    const STEP_END: u32 = 200029;
    /// Start of action (200030)
    #[classattr]
    const ACT: u32 = 200030;
    /// End of action (200031)
    #[classattr]
    const ACT_END: u32 = 200031;
    /// Start of observation (200032)
    #[classattr]
    const OBSERVE: u32 = 200032;
    /// End of observation (200033)
    #[classattr]
    const OBSERVE_END: u32 = 200033;

    // =========================================================================
    // Tool/Function Calling (200034-200039)
    // =========================================================================

    /// Start of function call (200034)
    #[classattr]
    const FUNCTION: u32 = 200034;
    /// End of function call (200035)
    #[classattr]
    const FUNCTION_END: u32 = 200035;
    /// Start of function result (200036)
    #[classattr]
    const RESULT: u32 = 200036;
    /// End of function result (200037)
    #[classattr]
    const RESULT_END: u32 = 200037;
    /// Start of error block (200038)
    #[classattr]
    const ERROR: u32 = 200038;
    /// End of error block (200039)
    #[classattr]
    const ERROR_END: u32 = 200039;

    // =========================================================================
    // Code Execution (200040-200045)
    // =========================================================================

    /// Start of code block (200040)
    #[classattr]
    const CODE: u32 = 200040;
    /// End of code block (200041)
    #[classattr]
    const CODE_END: u32 = 200041;
    /// Start of output (200042)
    #[classattr]
    const OUTPUT: u32 = 200042;
    /// End of output (200043)
    #[classattr]
    const OUTPUT_END: u32 = 200043;
    /// Start of language tag (200044)
    #[classattr]
    const LANG: u32 = 200044;
    /// End of language tag (200045)
    #[classattr]
    const LANG_END: u32 = 200045;

    // =========================================================================
    // RAG/Citations (200046-200053)
    // =========================================================================

    /// Start of context block (200046)
    #[classattr]
    const CONTEXT: u32 = 200046;
    /// End of context block (200047)
    #[classattr]
    const CONTEXT_END: u32 = 200047;
    /// Start of quote (200048)
    #[classattr]
    const QUOTE: u32 = 200048;
    /// End of quote (200049)
    #[classattr]
    const QUOTE_END: u32 = 200049;
    /// Start of citation (200050)
    #[classattr]
    const CITE: u32 = 200050;
    /// End of citation (200051)
    #[classattr]
    const CITE_END: u32 = 200051;
    /// Start of source (200052)
    #[classattr]
    const SOURCE: u32 = 200052;
    /// End of source (200053)
    #[classattr]
    const SOURCE_END: u32 = 200053;

    // =========================================================================
    // Memory/State (200054-200057)
    // =========================================================================

    /// Start of memory block (200054)
    #[classattr]
    const MEMORY: u32 = 200054;
    /// End of memory block (200055)
    #[classattr]
    const MEMORY_END: u32 = 200055;
    /// Start of recall block (200056)
    #[classattr]
    const RECALL: u32 = 200056;
    /// End of recall block (200057)
    #[classattr]
    const RECALL_END: u32 = 200057;

    // =========================================================================
    // Control Tokens (200058-200060)
    // =========================================================================

    /// Padding token (200058)
    #[classattr]
    const PAD: u32 = 200058;
    /// Stop token (200059)
    #[classattr]
    const STOP: u32 = 200059;
    /// Separator token (200060)
    #[classattr]
    const SEP: u32 = 200060;

    // =========================================================================
    // Multimodal (200061-200066)
    // =========================================================================

    /// Start of image (200061)
    #[classattr]
    const IMAGE: u32 = 200061;
    /// End of image (200062)
    #[classattr]
    const IMAGE_END: u32 = 200062;
    /// Start of audio (200063)
    #[classattr]
    const AUDIO: u32 = 200063;
    /// End of audio (200064)
    #[classattr]
    const AUDIO_END: u32 = 200064;
    /// Start of video (200065)
    #[classattr]
    const VIDEO: u32 = 200065;
    /// End of video (200066)
    #[classattr]
    const VIDEO_END: u32 = 200066;

    // =========================================================================
    // Document Structure (200067-200072)
    // =========================================================================

    /// Start of title - document/section title (200067)
    #[classattr]
    const TITLE: u32 = 200067;
    /// End of title (200068)
    #[classattr]
    const TITLE_END: u32 = 200068;
    /// Start of section - semantic document section (200069)
    #[classattr]
    const SECTION: u32 = 200069;
    /// End of section (200070)
    #[classattr]
    const SECTION_END: u32 = 200070;
    /// Start of summary - condensed content summary (200071)
    #[classattr]
    const SUMMARY: u32 = 200071;
    /// End of summary (200072)
    #[classattr]
    const SUMMARY_END: u32 = 200072;
}

/// Agent token IDs for Llama 3/3.1/3.2/3.3.
///
/// Provides constant token IDs for building chat models, reasoning systems,
/// and autonomous agents. Includes both official Meta tokens (128000-128010)
/// and splintr agent tokens (128300+).
///
/// # Python Example
///
/// ```python
/// from splintr import LLAMA3_AGENT_TOKENS
///
/// # Official Meta tokens
/// bos = LLAMA3_AGENT_TOKENS.BEGIN_OF_TEXT     # 128000
/// eot = LLAMA3_AGENT_TOKENS.EOT_ID            # 128009
/// python = LLAMA3_AGENT_TOKENS.PYTHON_TAG     # 128010
///
/// # Splintr agent tokens
/// system_id = LLAMA3_AGENT_TOKENS.SYSTEM      # 128300
/// think_id = LLAMA3_AGENT_TOKENS.THINK        # 128305
///
/// # Use with tokenizer
/// tokenizer = Tokenizer.from_pretrained("llama3")
/// tokens = tokenizer.encode_with_special("<|think|>reasoning<|/think|>")
/// assert LLAMA3_AGENT_TOKENS.THINK in tokens
/// ```
#[pyclass(name = "LLAMA3_AGENT_TOKENS", frozen)]
pub struct PyLlama3AgentTokens;

#[pymethods]
impl PyLlama3AgentTokens {
    // =========================================================================
    // Official Meta Special Tokens (128000-128010)
    // Based on Llama 3.3 tokenizer_config.json
    // =========================================================================

    /// Start of text marker (128000)
    #[classattr]
    const BEGIN_OF_TEXT: u32 = 128000;
    /// End of text marker (128001)
    #[classattr]
    const END_OF_TEXT: u32 = 128001;
    /// Padding token for fine-tuning - added in Llama 3.1 (128004)
    #[classattr]
    const FINETUNE_RIGHT_PAD_ID: u32 = 128004;
    /// Start of header marker (128006)
    #[classattr]
    const START_HEADER_ID: u32 = 128006;
    /// End of header marker (128007)
    #[classattr]
    const END_HEADER_ID: u32 = 128007;
    /// End of message marker for tool use - added in Llama 3.1 (128008)
    #[classattr]
    const EOM_ID: u32 = 128008;
    /// End of turn marker (128009)
    #[classattr]
    const EOT_ID: u32 = 128009;
    /// Python code interpreter marker - added in Llama 3.1 (128010)
    #[classattr]
    const PYTHON_TAG: u32 = 128010;

    // =========================================================================
    // Llama 3.2-Vision Tokens (128005)
    // =========================================================================

    /// Step marker for reasoning - added in Llama 3.2-Vision (128005)
    #[classattr]
    const STEP_ID: u32 = 128005;

    // =========================================================================
    // Conversation Structure (128300-128304)
    // =========================================================================

    /// System message marker - defines assistant behavior (128300)
    #[classattr]
    const SYSTEM: u32 = 128300;
    /// User message marker - human input (128301)
    #[classattr]
    const USER: u32 = 128301;
    /// Assistant message marker - AI responses (128302)
    #[classattr]
    const ASSISTANT: u32 = 128302;
    /// ChatML message start delimiter (128303)
    #[classattr]
    const IM_START: u32 = 128303;
    /// ChatML message end delimiter (128304)
    #[classattr]
    const IM_END: u32 = 128304;

    // =========================================================================
    // Reasoning/Thinking (128305-128306)
    // =========================================================================

    /// Start of thinking block - Chain-of-Thought reasoning (128305)
    #[classattr]
    const THINK: u32 = 128305;
    /// End of thinking block (128306)
    #[classattr]
    const THINK_END: u32 = 128306;

    // =========================================================================
    // ReAct Agent Loop (128307-128314)
    // =========================================================================

    /// Start of planning phase (128307)
    #[classattr]
    const PLAN: u32 = 128307;
    /// End of planning phase (128308)
    #[classattr]
    const PLAN_END: u32 = 128308;
    /// Start of step (128309)
    #[classattr]
    const STEP: u32 = 128309;
    /// End of step (128310)
    #[classattr]
    const STEP_END: u32 = 128310;
    /// Start of action (128311)
    #[classattr]
    const ACT: u32 = 128311;
    /// End of action (128312)
    #[classattr]
    const ACT_END: u32 = 128312;
    /// Start of observation (128313)
    #[classattr]
    const OBSERVE: u32 = 128313;
    /// End of observation (128314)
    #[classattr]
    const OBSERVE_END: u32 = 128314;

    // =========================================================================
    // Tool/Function Calling (128315-128320)
    // =========================================================================

    /// Start of function call (128315)
    #[classattr]
    const FUNCTION: u32 = 128315;
    /// End of function call (128316)
    #[classattr]
    const FUNCTION_END: u32 = 128316;
    /// Start of function result (128317)
    #[classattr]
    const RESULT: u32 = 128317;
    /// End of function result (128318)
    #[classattr]
    const RESULT_END: u32 = 128318;
    /// Start of error block (128319)
    #[classattr]
    const ERROR: u32 = 128319;
    /// End of error block (128320)
    #[classattr]
    const ERROR_END: u32 = 128320;

    // =========================================================================
    // Code Execution (128321-128326)
    // =========================================================================

    /// Start of code block (128321)
    #[classattr]
    const CODE: u32 = 128321;
    /// End of code block (128322)
    #[classattr]
    const CODE_END: u32 = 128322;
    /// Start of output (128323)
    #[classattr]
    const OUTPUT: u32 = 128323;
    /// End of output (128324)
    #[classattr]
    const OUTPUT_END: u32 = 128324;
    /// Start of language tag (128325)
    #[classattr]
    const LANG: u32 = 128325;
    /// End of language tag (128326)
    #[classattr]
    const LANG_END: u32 = 128326;

    // =========================================================================
    // RAG/Citations (128327-128334)
    // =========================================================================

    /// Start of context block (128327)
    #[classattr]
    const CONTEXT: u32 = 128327;
    /// End of context block (128328)
    #[classattr]
    const CONTEXT_END: u32 = 128328;
    /// Start of quote (128329)
    #[classattr]
    const QUOTE: u32 = 128329;
    /// End of quote (128330)
    #[classattr]
    const QUOTE_END: u32 = 128330;
    /// Start of citation (128331)
    #[classattr]
    const CITE: u32 = 128331;
    /// End of citation (128332)
    #[classattr]
    const CITE_END: u32 = 128332;
    /// Start of source (128333)
    #[classattr]
    const SOURCE: u32 = 128333;
    /// End of source (128334)
    #[classattr]
    const SOURCE_END: u32 = 128334;

    // =========================================================================
    // Memory/State (128335-128338)
    // =========================================================================

    /// Start of memory block (128335)
    #[classattr]
    const MEMORY: u32 = 128335;
    /// End of memory block (128336)
    #[classattr]
    const MEMORY_END: u32 = 128336;
    /// Start of recall block (128337)
    #[classattr]
    const RECALL: u32 = 128337;
    /// End of recall block (128338)
    #[classattr]
    const RECALL_END: u32 = 128338;

    // =========================================================================
    // Control Tokens (128339-128341)
    // =========================================================================

    /// Padding token (128339)
    #[classattr]
    const PAD: u32 = 128339;
    /// Stop token (128340)
    #[classattr]
    const STOP: u32 = 128340;
    /// Separator token (128341)
    #[classattr]
    const SEP: u32 = 128341;

    // =========================================================================
    // Multimodal (128256-128261)
    // Uses official Meta <|image|> token ID from Llama 3.2-Vision
    // =========================================================================

    /// Image placeholder - official Meta token from Llama 3.2-Vision (128256)
    #[classattr]
    const IMAGE: u32 = 128256;
    /// End of image (128257)
    #[classattr]
    const IMAGE_END: u32 = 128257;
    /// Start of audio (128258)
    #[classattr]
    const AUDIO: u32 = 128258;
    /// End of audio (128259)
    #[classattr]
    const AUDIO_END: u32 = 128259;
    /// Start of video (128260)
    #[classattr]
    const VIDEO: u32 = 128260;
    /// End of video (128261)
    #[classattr]
    const VIDEO_END: u32 = 128261;

    // =========================================================================
    // Document Structure (128348-128353)
    // =========================================================================

    /// Start of title - document/section title (128348)
    #[classattr]
    const TITLE: u32 = 128348;
    /// End of title (128349)
    #[classattr]
    const TITLE_END: u32 = 128349;
    /// Start of section - semantic document section (128350)
    #[classattr]
    const SECTION: u32 = 128350;
    /// End of section (128351)
    #[classattr]
    const SECTION_END: u32 = 128351;
    /// Start of summary - condensed content summary (128352)
    #[classattr]
    const SUMMARY: u32 = 128352;
    /// End of summary (128353)
    #[classattr]
    const SUMMARY_END: u32 = 128353;
}

/// Agent token IDs for DeepSeek V3.
///
/// Provides constant token IDs for building chat models, reasoning systems,
/// and autonomous agents. Includes both official DeepSeek tokens (0-2, 128798-128814)
/// and splintr agent tokens (128900+).
///
/// # Python Example
///
/// ```python
/// from splintr import DEEPSEEK_V3_AGENT_TOKENS
///
/// # Official DeepSeek tokens
/// bos = DEEPSEEK_V3_AGENT_TOKENS.BEGIN_OF_SENTENCE   # 0
/// eos = DEEPSEEK_V3_AGENT_TOKENS.END_OF_SENTENCE     # 1
/// user = DEEPSEEK_V3_AGENT_TOKENS.USER_NATIVE        # 128803
///
/// # Splintr agent tokens
/// system_id = DEEPSEEK_V3_AGENT_TOKENS.SYSTEM        # 128900
/// think_id = DEEPSEEK_V3_AGENT_TOKENS.THINK          # 128905
///
/// # Use with tokenizer
/// tokenizer = Tokenizer.from_pretrained("deepseek_v3")
/// tokens = tokenizer.encode_with_special("<think>reasoning</think>")
/// assert DEEPSEEK_V3_AGENT_TOKENS.THINK_NATIVE in tokens
/// ```
#[pyclass(name = "DEEPSEEK_V3_AGENT_TOKENS", frozen)]
pub struct PyDeepSeekV3AgentTokens;

#[pymethods]
impl PyDeepSeekV3AgentTokens {
    // =========================================================================
    // Official DeepSeek Native Tokens (0-2)
    // =========================================================================

    /// Begin of sentence token (0)
    #[classattr]
    const BEGIN_OF_SENTENCE: u32 = 0;
    /// End of sentence token (1)
    #[classattr]
    const END_OF_SENTENCE: u32 = 1;
    /// Padding token (2)
    #[classattr]
    const PAD_NATIVE: u32 = 2;

    // =========================================================================
    // DeepSeek Thinking Tokens (128798-128799)
    // =========================================================================

    /// Native thinking start token (128798)
    #[classattr]
    const THINK_NATIVE: u32 = 128798;
    /// Native thinking end token (128799)
    #[classattr]
    const THINK_END_NATIVE: u32 = 128799;

    // =========================================================================
    // DeepSeek FIM Tokens (128800-128802)
    // =========================================================================

    /// Fill-in-the-middle hole marker (128800)
    #[classattr]
    const FIM_HOLE: u32 = 128800;
    /// Fill-in-the-middle begin marker (128801)
    #[classattr]
    const FIM_BEGIN: u32 = 128801;
    /// Fill-in-the-middle end marker (128802)
    #[classattr]
    const FIM_END: u32 = 128802;

    // =========================================================================
    // DeepSeek Chat Tokens (128803-128805)
    // =========================================================================

    /// Native user message marker (128803)
    #[classattr]
    const USER_NATIVE: u32 = 128803;
    /// Native assistant message marker (128804)
    #[classattr]
    const ASSISTANT_NATIVE: u32 = 128804;
    /// End of turn marker (128805)
    #[classattr]
    const EOT: u32 = 128805;

    // =========================================================================
    // DeepSeek Tool Calling Tokens (128806-128814)
    // =========================================================================

    /// Tool calls begin marker (128806)
    #[classattr]
    const TOOL_CALLS_BEGIN: u32 = 128806;
    /// Tool calls end marker (128807)
    #[classattr]
    const TOOL_CALLS_END: u32 = 128807;
    /// Tool call begin marker (128808)
    #[classattr]
    const TOOL_CALL_BEGIN: u32 = 128808;
    /// Tool call end marker (128809)
    #[classattr]
    const TOOL_CALL_END: u32 = 128809;
    /// Tool outputs begin marker (128810)
    #[classattr]
    const TOOL_OUTPUTS_BEGIN: u32 = 128810;
    /// Tool outputs end marker (128811)
    #[classattr]
    const TOOL_OUTPUTS_END: u32 = 128811;
    /// Tool output begin marker (128812)
    #[classattr]
    const TOOL_OUTPUT_BEGIN: u32 = 128812;
    /// Tool output end marker (128813)
    #[classattr]
    const TOOL_OUTPUT_END: u32 = 128813;
    /// Tool separator (128814)
    #[classattr]
    const TOOL_SEP: u32 = 128814;

    // =========================================================================
    // Conversation Structure (128900-128904)
    // =========================================================================

    /// System message marker - defines assistant behavior (128900)
    #[classattr]
    const SYSTEM: u32 = 128900;
    /// User message marker - human input (128901)
    #[classattr]
    const USER: u32 = 128901;
    /// Assistant message marker - AI responses (128902)
    #[classattr]
    const ASSISTANT: u32 = 128902;
    /// ChatML message start delimiter (128903)
    #[classattr]
    const IM_START: u32 = 128903;
    /// ChatML message end delimiter (128904)
    #[classattr]
    const IM_END: u32 = 128904;

    // =========================================================================
    // Reasoning/Thinking (128905-128906)
    // =========================================================================

    /// Start of thinking block - Chain-of-Thought reasoning (128905)
    #[classattr]
    const THINK: u32 = 128905;
    /// End of thinking block (128906)
    #[classattr]
    const THINK_END: u32 = 128906;

    // =========================================================================
    // ReAct Agent Loop (128907-128914)
    // =========================================================================

    /// Start of planning phase (128907)
    #[classattr]
    const PLAN: u32 = 128907;
    /// End of planning phase (128908)
    #[classattr]
    const PLAN_END: u32 = 128908;
    /// Start of step (128909)
    #[classattr]
    const STEP: u32 = 128909;
    /// End of step (128910)
    #[classattr]
    const STEP_END: u32 = 128910;
    /// Start of action (128911)
    #[classattr]
    const ACT: u32 = 128911;
    /// End of action (128912)
    #[classattr]
    const ACT_END: u32 = 128912;
    /// Start of observation (128913)
    #[classattr]
    const OBSERVE: u32 = 128913;
    /// End of observation (128914)
    #[classattr]
    const OBSERVE_END: u32 = 128914;

    // =========================================================================
    // Tool/Function Calling (128915-128920)
    // =========================================================================

    /// Start of function call (128915)
    #[classattr]
    const FUNCTION: u32 = 128915;
    /// End of function call (128916)
    #[classattr]
    const FUNCTION_END: u32 = 128916;
    /// Start of function result (128917)
    #[classattr]
    const RESULT: u32 = 128917;
    /// End of function result (128918)
    #[classattr]
    const RESULT_END: u32 = 128918;
    /// Start of error block (128919)
    #[classattr]
    const ERROR: u32 = 128919;
    /// End of error block (128920)
    #[classattr]
    const ERROR_END: u32 = 128920;

    // =========================================================================
    // Code Execution (128921-128926)
    // =========================================================================

    /// Start of code block (128921)
    #[classattr]
    const CODE: u32 = 128921;
    /// End of code block (128922)
    #[classattr]
    const CODE_END: u32 = 128922;
    /// Start of output (128923)
    #[classattr]
    const OUTPUT: u32 = 128923;
    /// End of output (128924)
    #[classattr]
    const OUTPUT_END: u32 = 128924;
    /// Start of language tag (128925)
    #[classattr]
    const LANG: u32 = 128925;
    /// End of language tag (128926)
    #[classattr]
    const LANG_END: u32 = 128926;

    // =========================================================================
    // RAG/Citations (128927-128934)
    // =========================================================================

    /// Start of context block (128927)
    #[classattr]
    const CONTEXT: u32 = 128927;
    /// End of context block (128928)
    #[classattr]
    const CONTEXT_END: u32 = 128928;
    /// Start of quote (128929)
    #[classattr]
    const QUOTE: u32 = 128929;
    /// End of quote (128930)
    #[classattr]
    const QUOTE_END: u32 = 128930;
    /// Start of citation (128931)
    #[classattr]
    const CITE: u32 = 128931;
    /// End of citation (128932)
    #[classattr]
    const CITE_END: u32 = 128932;
    /// Start of source (128933)
    #[classattr]
    const SOURCE: u32 = 128933;
    /// End of source (128934)
    #[classattr]
    const SOURCE_END: u32 = 128934;

    // =========================================================================
    // Memory/State (128935-128938)
    // =========================================================================

    /// Start of memory block (128935)
    #[classattr]
    const MEMORY: u32 = 128935;
    /// End of memory block (128936)
    #[classattr]
    const MEMORY_END: u32 = 128936;
    /// Start of recall block (128937)
    #[classattr]
    const RECALL: u32 = 128937;
    /// End of recall block (128938)
    #[classattr]
    const RECALL_END: u32 = 128938;

    // =========================================================================
    // Control Tokens (128939-128941)
    // =========================================================================

    /// Padding token (128939)
    #[classattr]
    const PAD: u32 = 128939;
    /// Stop token (128940)
    #[classattr]
    const STOP: u32 = 128940;
    /// Separator token (128941)
    #[classattr]
    const SEP: u32 = 128941;

    // =========================================================================
    // Multimodal (128942-128947)
    // =========================================================================

    /// Start of image (128942)
    #[classattr]
    const IMAGE: u32 = 128942;
    /// End of image (128943)
    #[classattr]
    const IMAGE_END: u32 = 128943;
    /// Start of audio (128944)
    #[classattr]
    const AUDIO: u32 = 128944;
    /// End of audio (128945)
    #[classattr]
    const AUDIO_END: u32 = 128945;
    /// Start of video (128946)
    #[classattr]
    const VIDEO: u32 = 128946;
    /// End of video (128947)
    #[classattr]
    const VIDEO_END: u32 = 128947;

    // =========================================================================
    // Document Structure (128948-128953)
    // =========================================================================

    /// Start of title - document/section title (128948)
    #[classattr]
    const TITLE: u32 = 128948;
    /// End of title (128949)
    #[classattr]
    const TITLE_END: u32 = 128949;
    /// Start of section - semantic document section (128950)
    #[classattr]
    const SECTION: u32 = 128950;
    /// End of section (128951)
    #[classattr]
    const SECTION_END: u32 = 128951;
    /// Start of summary - condensed content summary (128952)
    #[classattr]
    const SUMMARY: u32 = 128952;
    /// End of summary (128953)
    #[classattr]
    const SUMMARY_END: u32 = 128953;
}
