# Splintr API Guide

This guide provides comprehensive documentation for using Splintr's Python and Rust APIs. For a quick start, see the [main README](../README.md).

## Table of Contents

- [Python API Reference](#python-api-reference)
  - [Tokenizer Class](#tokenizer-class) (BPE)
  - [Encoding Methods](#encoding-methods)
  - [Decoding Methods](#decoding-methods)
  - [Cache Management](#cache-management)
  - [SentencePiece Tokenizer Class](#sentencepiece-tokenizer-class) (Unigram)
  - [Loading any model from `tokenizer.json`](#loading-any-model-from-tokenizerjson)
- [Streaming Decoder](#streaming-decoder)
  - [Regular Streaming Decoder](#regular-streaming-decoder)
  - [ByteLevel Streaming Decoder](#bytelevel-streaming-decoder)
- [Rust API Reference](#rust-api-reference)
  - [Tokenize Trait](#tokenize-trait)
  - [BPE Tokenizer](#bpe-tokenizer)
  - [SentencePiece Tokenizer](#sentencepiece-tokenizer)
  - [WordPiece Tokenizer](#wordpiece-tokenizer)
- [Detailed Usage Examples](#detailed-usage-examples)
  - [Basic Encoding and Decoding](#basic-encoding-and-decoding)
  - [Batch Processing](#batch-processing)
  - [Special Tokens Usage](#special-tokens-usage)
  - [Agent Tokens Usage](#agent-tokens-usage)
  - [Streaming Examples](#streaming-examples)

## Python API Reference

### Tokenizer Class

The `Tokenizer` class is the main entry point for tokenization in Python.

#### Loading

**Load a pretrained model:**

```python
from splintr import Tokenizer

# Load pretrained model (includes vocabulary and special tokens)
tokenizer = Tokenizer.from_pretrained("cl100k_base")  # OpenAI GPT-4/3.5
tokenizer = Tokenizer.from_pretrained("o200k_base")   # OpenAI GPT-4o
tokenizer = Tokenizer.from_pretrained("llama3")       # Meta Llama 3 family
tokenizer = Tokenizer.from_pretrained("deepseek_v3")  # DeepSeek V3/R1
tokenizer = Tokenizer.from_pretrained("mistral_v1")   # Mistral 7B v0.1/v0.2, Mixtral 8x7B
tokenizer = Tokenizer.from_pretrained("mistral_v2")   # Mistral 7B v0.3, Codestral, Mixtral 8x22B
tokenizer = Tokenizer.from_pretrained("mistral_v3")   # Mistral NeMo, Large 2, Pixtral
tokenizer = Tokenizer.from_pretrained("whisper_v3")   # OpenAI Whisper multilingual (v1/v2/v3; bare "whisper" → v2)
```

> Whisper English-only checkpoints (`*.en`) use a different base BPE and are not bundled — load those with [`from_json`](#loading-any-model-from-tokenizerjson).

**Load from custom vocabulary file:**

```python
from splintr import Tokenizer, CL100K_BASE_PATTERN

tokenizer = Tokenizer(
    vocab_path="path/to/vocab.tiktoken",
    pattern=CL100K_BASE_PATTERN,
    special_tokens={"<|endoftext|>": 100257}
)
```

### Encoding Methods

#### `encode(text: str) -> list[int]`

Encode text to token IDs using sequential processing. This is optimal for most use cases with texts under 1MB.

```python
tokens = tokenizer.encode("Hello, world!")
print(tokens)  # [9906, 11, 1917, 0]
```

#### `encode_with_special(text: str) -> list[int]`

Encode text while recognizing special tokens in the input. Special tokens are matched and encoded as single tokens rather than being split.

```python
text = "<|endoftext|>This is a test"
tokens = tokenizer.encode_with_special(text)
# Special token <|endoftext|> becomes a single token ID
```

#### `encode_batch(texts: list[str]) -> list[list[int]]`

Encode multiple texts in parallel using Rayon. This is where Splintr really shines, achieving 10-12x speedup over sequential processing.

```python
texts = ["Hello, world!", "How are you?", "Machine learning is fun!"]
batch_tokens = tokenizer.encode_batch(texts)
# Returns: [[9906, 11, 1917, 0], [4438, 527, 499, 30], ...]
```

#### `encode_rayon(text: str) -> list[int]`

Encode a single text using Rayon's internal parallelization. This is only beneficial for very large texts (>1MB). For typical use cases, `encode()` is faster.

```python
# Only useful for very large texts
large_text = "..." * 1000000  # >1MB of text
tokens = tokenizer.encode_rayon(large_text)
```

### Decoding Methods

#### `decode(tokens: list[int]) -> str`

Decode token IDs back to text. Raises an error if the decoded bytes are not valid UTF-8.

```python
tokens = [9906, 11, 1917, 0]
text = tokenizer.decode(tokens)
print(text)  # "Hello, world!"
```

#### `decode_bytes(tokens: list[int]) -> bytes`

Decode token IDs to raw bytes without UTF-8 validation.

```python
tokens = [9906, 11, 1917, 0]
raw_bytes = tokenizer.decode_bytes(tokens)
print(raw_bytes)  # b'Hello, world!'
```

#### `decode_lossy(tokens: list[int]) -> str`

Decode token IDs to text, replacing any invalid UTF-8 sequences with the replacement character (�).

```python
tokens = [9906, 11, 1917, 0]
text = tokenizer.decode_lossy(tokens)
# Invalid UTF-8 sequences become �
```

### Properties

#### `vocab_size: int`

The total vocabulary size including special tokens.

```python
print(tokenizer.vocab_size)  # e.g., 100311 for cl100k_base with agent tokens
```

#### `cache_len: int`

The number of entries currently in the LRU cache.

```python
print(tokenizer.cache_len)  # Number of cached text chunks
```

### Cache Management

#### `clear_cache()`

Clear the LRU encoding cache. Useful if memory pressure is a concern.

```python
tokenizer.clear_cache()
```

### SentencePiece Tokenizer Class

The `SentencePieceTokenizer` class provides unigram tokenization for models using SentencePiece (e.g., loaded from GGUF files).

#### Creating

```python
from splintr import SentencePieceTokenizer

# Create from raw vocabulary data
tokenizer = SentencePieceTokenizer(
    tokens=["<unk>", "<s>", "</s>", "▁Hello", "▁world"],
    scores=[0.0, 0.0, 0.0, -1.2, -1.5],
    eos_token_id=2,
    bos_token_id=1,  # optional
)
```

#### `encode(text: str) -> list[int]`

Encode text using Viterbi maximum-score segmentation (true SentencePiece Unigram, not greedy), with byte fallback for unknown characters. Prepends BOS if configured.

```python
ids = tokenizer.encode("Hello world")
# [1, 3, 4]  (BOS + ▁Hello + ▁world)
```

#### `decode(ids: list[int]) -> str`

Decode token IDs to text. Skips BOS/EOS tokens, converts ▁ back to spaces.

```python
text = tokenizer.decode([1, 3, 4])
# "Hello world"
```

#### `decode_lossy(ids: list[int]) -> str`

Decode token IDs, silently skipping any invalid (out-of-range) IDs.

```python
text = tokenizer.decode_lossy([1, 3, 999, 4])
# "Hello world"  (999 is skipped)
```

#### Properties

- `vocab_size: int` — Total vocabulary size
- `eos_token_id: int` — End-of-sequence token ID
- `bos_token_id: int | None` — Beginning-of-sequence token ID (if configured)

#### Methods

- `is_eos(token_id: int) -> bool` — Check if a token is the EOS token

### Loading any model from `tokenizer.json`

For models not bundled with `from_pretrained`, load a HuggingFace `tokenizer.json`
with `from_json`. It reads everything from the file — split regex, byte-level
flag, BPE **merge order** (independent of token ids, so RoBERTa-style vocabs
work), the full ordered normalizer (including SentencePiece's `Precompiled`
charsmap), and special tokens — and dispatches on `model.type` to the matching
backend object:

```python
from splintr import from_json, from_json_bytes

tok = from_json("tokenizer.json")          # from a path
# tok = from_json_bytes(open("tokenizer.json","rb").read())  # from bytes

ids = tok.encode("Hello world")                  # content tokens (HF add_special_tokens=False)
ids = tok.encode_with_special_tokens("Hello")    # also applies the post_processor template
text = tok.decode(ids)
```

| `model.type` | Returned object        | Example models                     |
|--------------|------------------------|------------------------------------|
| `BPE`        | `Tokenizer`            | GPT-2, RoBERTa, Qwen, Whisper.en   |
| `Unigram`    | `SentencePieceTokenizer` | T5, Gemma, Albert, XLNet         |
| `WordPiece`  | `WordPieceTokenizer`   | BERT, DistilBERT, Electra          |

**Strict by design.** Rather than silently approximate an unsupported config (which
would produce wrong tokens with no signal), `from_json` raises:

- `UnsupportedModelType` — `model.type` is not BPE/Unigram/WordPiece
- `UnsupportedNormalizer` — an unrecognized normalizer step (dropping it would mis-normalize)
- `InvalidNormalizerRegex` — a `Replace` regex that fails to compile
- `UnsupportedPreTokenizer` — a declared pre-tokenizer with no recognized split (refusing to guess the pattern)

Output is verified id-for-id against HuggingFace `tokenizers` across all three
families. (Rust: `splintr::from_json_path` / `from_json_bytes`.)

## Streaming Decoder

Streaming decoders are essential for real-time LLM applications where tokens arrive one at a time. They handle the critical problem of BPE tokens not aligning with UTF-8 character boundaries.

### Regular Streaming Decoder

Use `streaming_decoder()` for standard tokenizers (cl100k_base, o200k_base, llama3).

#### Why You Need This

BPE tokens don't align with UTF-8 character boundaries. A multi-byte Unicode character like "世" (3 bytes: `0xE4 0xB8 0x96`) might split across tokens. The streaming decoder:

1. Buffers incomplete byte sequences across token boundaries
2. Only outputs text when complete UTF-8 characters are available
3. Prevents display corruption in streaming LLM output
4. Handles edge cases automatically

#### Basic Usage

```python
# Create a streaming decoder
decoder = tokenizer.streaming_decoder()

# Process tokens one at a time (typical LLM streaming scenario)
for token_id in token_stream:
    # Returns text only when complete UTF-8 characters are available
    if text := decoder.add_token(token_id):
        print(text, end="", flush=True)

# Flush any remaining buffered bytes at the end
print(decoder.flush())
```

#### Real-World Example

```python
import openai
from splintr import Tokenizer

tokenizer = Tokenizer.from_pretrained("cl100k_base")
decoder = tokenizer.streaming_decoder()

# Stream tokens from OpenAI API
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        # Process each token as it arrives
        token_ids = get_token_ids(chunk)  # pseudo-code
        for token_id in token_ids:
            if text := decoder.add_token(token_id):
                print(text, end="", flush=True)

# Don't forget to flush at the end
print(decoder.flush())
```

#### API Methods

**Core operations:**

- `add_token(token_id: int) -> str | None`: Add a token, return complete characters or None if buffering
- `add_tokens(token_ids: list[int]) -> str | None`: Add multiple tokens at once
- `flush() -> str`: Flush buffered bytes (incomplete sequences become �)
- `reset()`: Clear the buffer and start fresh

**Properties:**

- `has_pending: bool`: Whether there are buffered bytes waiting
- `pending_bytes: int`: Number of bytes currently buffered

### ByteLevel Streaming Decoder

For tokenizers using **ByteLevel BPE encoding** (DeepSeek V3, GPT-2), use `byte_level_streaming_decoder()` instead.

#### Why ByteLevel?

ByteLevel BPE encodes raw bytes (0-255) as printable Unicode characters (e.g., space `0x20` becomes `Ġ`). The ByteLevel streaming decoder handles this extra decoding step automatically:

1. Decodes ByteLevel-encoded token bytes back to raw bytes
2. Buffers incomplete UTF-8 sequences across token boundaries
3. Only outputs text when complete UTF-8 characters are available

See [bytelevel_bpe.md](bytelevel_bpe.md) for details on ByteLevel encoding.

#### Basic Usage

```python
from splintr import Tokenizer

# DeepSeek V3 uses ByteLevel BPE encoding
tokenizer = Tokenizer.from_pretrained("deepseek_v3")
decoder = tokenizer.byte_level_streaming_decoder()

# Process tokens one at a time
for token_id in token_stream:
    if text := decoder.add_token(token_id):
        print(text, end="", flush=True)

print(decoder.flush())
```

#### API Methods

The ByteLevel streaming decoder has the same API as the regular streaming decoder:

- `add_token(token_id: int) -> str | None`
- `add_tokens(token_ids: list[int]) -> str | None`
- `flush() -> str`
- `reset()`
- `has_pending: bool`
- `pending_bytes: int`

## Rust API Reference

The Rust API provides similar functionality with strongly-typed interfaces. For complete documentation, see [docs.rs/splintr](https://docs.rs/splintr).

### Setup

Add Splintr to your `Cargo.toml`:

```toml
[dependencies]
splintr = "*"  # or pin to a specific version
```

### Tokenize Trait

All tokenizer backends implement the `Tokenize` trait, enabling generic code:

```rust
use splintr::Tokenize;

fn count_tokens(tokenizer: &dyn Tokenize, text: &str) -> usize {
    tokenizer.encode(text).len()
}
```

**Methods:**

- `encode(&self, text: &str) -> Vec<u32>`: Encode text to token IDs
- `decode(&self, ids: &[u32]) -> Result<String, TokenizeError>`: Decode token IDs to text
- `vocab_size(&self) -> usize`: Vocabulary size

Implemented by `Tokenizer` (BPE), `SentencePieceTokenizer` (unigram), and `WordPieceTokenizer` (WordPiece).

### BPE Tokenizer

```rust
use splintr::{Tokenizer, CL100K_BASE_PATTERN};
use rustc_hash::FxHashMap;

// Load vocabulary and create tokenizer
let encoder = load_tiktoken_bpe_file("cl100k_base.tiktoken")?;
let special_tokens = FxHashMap::default();
let tokenizer = Tokenizer::new(encoder, special_tokens, CL100K_BASE_PATTERN)?;

// Encode text
let tokens = tokenizer.encode("Hello, world!");
println!("{:?}", tokens);

// Batch encode
let texts = vec!["Hello".to_string(), "World".to_string()];
let batch_tokens = tokenizer.encode_batch(&texts);
```

#### Encoding Methods

- `encode(&self, text: &str) -> Vec<u32>`: Sequential encoding (optimal for texts <1MB)
- `encode_with_special(&self, text: &str) -> Vec<u32>`: Encode with special token recognition
- `encode_batch(&self, texts: &[String]) -> Vec<Vec<u32>>`: Parallel encoding across texts
- `encode_rayon(&self, text: &str) -> Vec<u32>`: Parallel encoding within text (for texts >1MB)

#### Decoding Methods

- `decode(&self, tokens: &[u32]) -> Result<String, TokenizerError>`: Decode to UTF-8 string
- `decode_bytes(&self, tokens: &[u32]) -> Vec<u8>`: Decode to raw bytes
- `decode_lossy(&self, tokens: &[u32]) -> String`: Decode with replacement for invalid UTF-8

### SentencePiece Tokenizer

For models using SentencePiece unigram tokenization (e.g., Mistral V1/V2):

```rust
use splintr::SentencePieceTokenizer;

// Create from raw vocabulary data
let tokenizer = SentencePieceTokenizer::new(
    tokens,       // Vec<String> — token strings indexed by ID
    scores,       // Vec<f32> — per-token Unigram scores maximized by Viterbi (empty for uniform)
    Some(1),      // Optional BOS token ID
    2,            // EOS token ID
)?;

// Encode (prepends BOS if configured, uses ▁ word boundaries)
let ids = tokenizer.encode("Hello world");

// Decode (skips BOS/EOS, converts ▁ back to spaces)
let text = tokenizer.decode(&ids)?;

// Lossy decode (skips invalid token IDs instead of erroring)
let text = tokenizer.decode_lossy(&ids);
```

#### Methods

- `encode(&self, text: &str) -> Vec<u32>`: Viterbi maximum-score (true SentencePiece Unigram) segmentation with byte fallback
- `decode(&self, ids: &[u32]) -> Result<String, SentencePieceError>`: Decode to UTF-8 string
- `decode_lossy(&self, ids: &[u32]) -> String`: Decode, skipping invalid token IDs
- `vocab_size(&self) -> usize`: Vocabulary size
- `is_eos(&self, token_id: u32) -> bool`: Check if token is EOS
- `eos_token_id(&self) -> u32`: Get EOS token ID
- `bos_token_id(&self) -> Option<u32>`: Get BOS token ID

### WordPiece Tokenizer

For BERT-family models using WordPiece subword tokenization:

```rust
use splintr::{WordPieceTokenizer, Tokenize};

// Create from a flat vocabulary (index = token ID)
let vocab = vec![
    "[PAD]", "[UNK]", "[CLS]", "[SEP]",
    "hello", "world", "##ing", "##s",
].into_iter().map(String::from).collect();

let tokenizer = WordPieceTokenizer::new(
    vocab,    // Vec<String> — token strings indexed by ID
    1,        // UNK token ID
    200,      // Max word length before mapping to UNK
    true,     // Lowercase and strip accents (for uncased models)
);

// Encode (BasicTokenizer + WordPiece greedy longest-match)
let ids = tokenizer.encode("Hello world");

// Decode (reconstructs text, skips [CLS]/[SEP]/[PAD] special tokens)
let text = tokenizer.decode(&ids)?;
```

#### Methods

- `encode(&self, text: &str) -> Vec<u32>`: BasicTokenizer + WordPiece subword tokenization
- `decode(&self, ids: &[u32]) -> Result<String, TokenizeError>`: Decode, joining subwords and removing `##` prefixes
- `vocab_size(&self) -> usize`: Vocabulary size
- `cls_token_id(&self) -> Option<u32>`: `[CLS]` token ID
- `sep_token_id(&self) -> Option<u32>`: `[SEP]` token ID
- `pad_token_id(&self) -> Option<u32>`: `[PAD]` token ID
- `unk_token_id(&self) -> u32`: `[UNK]` token ID

### Error Handling

The Rust API uses `Result` types for operations that can fail:

```rust
match tokenizer.decode(&tokens) {
    Ok(text) => println!("Decoded: {}", text),
    Err(e) => eprintln!("Decoding error: {}", e),
}
```

## Detailed Usage Examples

### Basic Encoding and Decoding

```python
from splintr import Tokenizer

# Load tokenizer
tokenizer = Tokenizer.from_pretrained("cl100k_base")

# Simple encoding
text = "The quick brown fox jumps over the lazy dog."
tokens = tokenizer.encode(text)
print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Token count: {len(tokens)}")

# Simple decoding
decoded = tokenizer.decode(tokens)
print(f"Decoded: {decoded}")
assert decoded == text

# Handle different languages
chinese = "你好世界"
tokens_cn = tokenizer.encode(chinese)
print(f"Chinese tokens: {tokens_cn}")
decoded_cn = tokenizer.decode(tokens_cn)
print(f"Decoded Chinese: {decoded_cn}")
```

### Batch Processing

```python
from splintr import Tokenizer
import time

tokenizer = Tokenizer.from_pretrained("cl100k_base")

# Prepare a batch of texts
texts = [
    "First text to encode",
    "Second text to encode",
    "Third text with different content",
    "Fourth text for batch processing",
] * 100  # 400 texts

# Measure batch encoding performance
start = time.time()
batch_tokens = tokenizer.encode_batch(texts)
elapsed = time.time() - start

print(f"Encoded {len(texts)} texts in {elapsed:.3f}s")
print(f"Throughput: {len(texts)/elapsed:.1f} texts/second")

# Process results
for i, tokens in enumerate(batch_tokens[:5]):
    print(f"Text {i}: {len(tokens)} tokens")
```

### Special Tokens Usage

```python
from splintr import Tokenizer

tokenizer = Tokenizer.from_pretrained("cl100k_base")

# Encode without special token recognition
# The special token gets split into multiple tokens
text = "Start <|endoftext|> End"
tokens_no_special = tokenizer.encode(text)
print(f"Without special tokens: {len(tokens_no_special)} tokens")

# Encode with special token recognition
# The special token becomes a single token
tokens_with_special = tokenizer.encode_with_special(text)
print(f"With special tokens: {len(tokens_with_special)} tokens")

# Verify the difference
decoded = tokenizer.decode(tokens_with_special)
print(f"Decoded: {decoded}")
```

### Agent Tokens Usage

```python
from splintr import Tokenizer, CL100K_AGENT_TOKENS, LLAMA3_AGENT_TOKENS, DEEPSEEK_V3_AGENT_TOKENS

# OpenAI models with agent tokens
tokenizer_openai = Tokenizer.from_pretrained("cl100k_base")

# Chain-of-Thought reasoning
cot_text = "<|think|>Let me break this down step by step...<|/think|>The answer is 42."
tokens = tokenizer_openai.encode_with_special(cot_text)
print(f"Thinking token ID: {CL100K_AGENT_TOKENS.THINK}")
print(f"Thinking end token ID: {CL100K_AGENT_TOKENS.THINK_END}")

# ReAct agent pattern
react_text = """<|plan|>I need to search for information
<|act|>search("climate change")
<|observe|>Found 10 results...
<|think|>Based on these results..."""

tokens = tokenizer_openai.encode_with_special(react_text)
print(f"Encoded {len(tokens)} tokens")

# Function calling
function_text = """<|function|>calculate_sum
<|result|>42
<|/result|>"""

tokens = tokenizer_openai.encode_with_special(function_text)
print(f"Function token ID: {CL100K_AGENT_TOKENS.FUNCTION}")
print(f"Result token ID: {CL100K_AGENT_TOKENS.RESULT}")

# RAG with citations
rag_text = """<|context|>This is source material...
<|cite|>According to the documentation...
<|source|>docs.example.com"""

tokens = tokenizer_openai.encode_with_special(rag_text)
print(f"Context token ID: {CL100K_AGENT_TOKENS.CONTEXT}")
print(f"Cite token ID: {CL100K_AGENT_TOKENS.CITE}")

# Llama 3 models
tokenizer_llama = Tokenizer.from_pretrained("llama3")

# Use Llama 3 native tokens
llama_text = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nHello<|eot_id|>"
tokens = tokenizer_llama.encode_with_special(llama_text)
print(f"Llama begin_of_text: {LLAMA3_AGENT_TOKENS.BEGIN_OF_TEXT}")
print(f"Llama start_header_id: {LLAMA3_AGENT_TOKENS.START_HEADER_ID}")

# DeepSeek V3 models with native thinking tokens
tokenizer_deepseek = Tokenizer.from_pretrained("deepseek_v3")

# Use DeepSeek's native thinking tokens for R1-style reasoning
deepseek_text = "<think>Let me reason through this problem step by step...</think>The solution is X."
tokens = tokenizer_deepseek.encode_with_special(deepseek_text)
print(f"DeepSeek think token (native): {DEEPSEEK_V3_AGENT_TOKENS.THINK_NATIVE}")
print(f"DeepSeek think_end token (native): {DEEPSEEK_V3_AGENT_TOKENS.THINK_END_NATIVE}")

# DeepSeek V3 also has tool calling tokens
tool_text = """<｜tool▁calls▁begin｜>
<｜tool▁call▁begin｜>
function_name
<｜tool▁call▁end｜>
<｜tool▁calls▁end｜>"""

tokens = tokenizer_deepseek.encode_with_special(tool_text)
print(f"Encoded tool calling pattern with {len(tokens)} tokens")
```

### Streaming Examples

#### Streaming Decoder for Regular Tokenizers

```python
from splintr import Tokenizer
import time

tokenizer = Tokenizer.from_pretrained("cl100k_base")

# Simulate streaming token generation
text = "Hello, 世界! This is a test of streaming decoding with Unicode characters: 你好"
tokens = tokenizer.encode(text)

# Create streaming decoder
decoder = tokenizer.streaming_decoder()

print("Streaming output:")
for token in tokens:
    # Simulate network delay
    time.sleep(0.05)

    # Add token and print if we get complete characters
    if chunk := decoder.add_token(token):
        print(chunk, end="", flush=True)

# Flush any remaining bytes
if remaining := decoder.flush():
    print(remaining, end="", flush=True)

print("\n\nStreaming complete!")
```

#### ByteLevel Streaming Decoder for DeepSeek V3

```python
from splintr import Tokenizer
import time

tokenizer = Tokenizer.from_pretrained("deepseek_v3")

# Test text with Unicode
text = "DeepSeek V3 supports ByteLevel BPE! 中文测试"
tokens = tokenizer.encode(text)

# Create ByteLevel streaming decoder
decoder = tokenizer.byte_level_streaming_decoder()

print("ByteLevel streaming output:")
for token in tokens:
    time.sleep(0.05)

    if chunk := decoder.add_token(token):
        print(chunk, end="", flush=True)

# Flush remaining
if remaining := decoder.flush():
    print(remaining, end="", flush=True)

print("\n\nByteLevel streaming complete!")

# Check pending state
print(f"Has pending bytes: {decoder.has_pending}")
print(f"Pending byte count: {decoder.pending_bytes}")
```

#### Advanced Streaming with Error Handling

```python
from splintr import Tokenizer

tokenizer = Tokenizer.from_pretrained("cl100k_base")
decoder = tokenizer.streaming_decoder()

def stream_tokens(token_generator):
    """Stream tokens with proper error handling."""
    try:
        for token_id in token_generator:
            try:
                if text := decoder.add_token(token_id):
                    yield text
            except Exception as e:
                print(f"\nError processing token {token_id}: {e}")
                # Reset decoder and continue
                decoder.reset()
                continue

        # Always flush at the end
        if remaining := decoder.flush():
            yield remaining

    except Exception as e:
        print(f"\nFatal streaming error: {e}")
        # Final flush attempt
        try:
            if remaining := decoder.flush():
                yield remaining
        except:
            pass

# Use the streaming function
text = "Test streaming with proper error handling"
tokens = tokenizer.encode(text)

for chunk in stream_tokens(iter(tokens)):
    print(chunk, end="", flush=True)

print("\nDone!")
```

## Performance Tips

1. **Use `encode_batch()` for multiple texts**: This is where Splintr achieves 10-12x speedup. Always prefer batch encoding when you have multiple texts.

2. **Use `encode()` for single texts**: Don't use `encode_rayon()` unless your text is >1MB. The sequential implementation is faster for typical use cases.

3. **Cache frequently encoded text**: Splintr includes an LRU cache. If you're encoding the same text repeatedly, the cache will speed things up automatically.

4. **Clear cache if memory is tight**: Use `clear_cache()` if you're processing millions of unique texts and memory becomes a concern.

5. **Use streaming decoders for real-time output**: Don't decode each token individually. Use `streaming_decoder()` or `byte_level_streaming_decoder()` to handle UTF-8 boundaries correctly.

6. **Choose the right special token encoding**: Use `encode_with_special()` only when your text actually contains special tokens. For regular text, `encode()` is faster.

## Additional Resources

- [Main README](../README.md) - Quick start and overview
- [Special Tokens Documentation](special_tokens.md) - Complete agent tokens reference
- [ByteLevel BPE Documentation](bytelevel_bpe.md) - ByteLevel encoding details
- [API Documentation (Rust)](https://docs.rs/splintr) - Complete Rust API reference
- [GitHub Repository](https://github.com/ml-rust/splintr) - Source code and examples
