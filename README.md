# splintr

[![Crates.io](https://img.shields.io/crates/v/splintr.svg)](https://crates.io/crates/splintr)
[![PyPI](https://img.shields.io/pypi/v/splintr-rs.svg)](https://pypi.org/project/splintr-rs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A high-performance BPE tokenizer implemented in Rust with Python bindings, designed for efficient tokenization of text in machine learning applications, particularly for large language models.

## Features

splintr implements several optimizations that make tokenization faster and more efficient:

- **PCRE2 with JIT compilation**: Uses PCRE2's just-in-time compilation for regex matching, providing 2-4x speedup over fancy-regex on pattern matching operations
- **Rayon parallelism**: Leverages multiple CPU cores for encoding batches of text and individual regex chunks within each text
- **Linked-list BPE algorithm**: Implements BPE using a linked-list structure that avoids O(N²) complexity on pathological inputs with many repetitive patterns
- **FxHashMap**: Uses rustc's FxHasher for faster lookups compared to the default SipHash, trading cryptographic security for speed in non-adversarial contexts
- **Aho-Corasick for special tokens**: Employs the Aho-Corasick algorithm for fast multi-pattern matching of special tokens, avoiding regex alternation overhead
- **LRU cache**: Caches frequently encoded text chunks to avoid redundant BPE encoding operations
- **UTF-8 streaming decoder**: Safely handles token-by-token decoding for LLM output, buffering incomplete UTF-8 sequences across token boundaries

## Installation

### Python

```bash
pip install splintr-rs
```

### Rust

```toml
[dependencies]
splintr = "0.1.0-beta.1"
```

## Quick Start

### Python

```python
from splintr import Tokenizer

# Load a pretrained tokenizer
tokenizer = Tokenizer.from_pretrained("cl100k_base")

# Encode text to token IDs
tokens = tokenizer.encode("Hello, world!")
print(tokens)  # [9906, 11, 1917, 0]

# Decode token IDs back to text
text = tokenizer.decode(tokens)
print(text)  # "Hello, world!"

# Batch encode multiple texts in parallel
texts = ["Hello, world!", "How are you?", "Machine learning is fun!"]
batch_tokens = tokenizer.encode_batch(texts)
print(batch_tokens)  # [[9906, 11, 1917, 0], [4438, 527, 499, 30], ...]
```

### Rust

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

// Decode tokens
let text = tokenizer.decode(&tokens)?;
println!("{}", text);

// Batch encode
let texts = vec!["Hello".to_string(), "World".to_string()];
let batch_tokens = tokenizer.encode_batch(&texts);
```

## API Reference

### Python API

#### Tokenizer

**Loading a tokenizer:**

```python
# Load a pretrained model (includes vocabulary and special tokens)
tokenizer = Tokenizer.from_pretrained("cl100k_base")  # or "o200k_base"

# Load from a custom vocabulary file
tokenizer = Tokenizer(
    vocab_path="path/to/vocab.tiktoken",
    pattern=CL100K_BASE_PATTERN,
    special_tokens={"<|endoftext|>": 100257}
)
```

**Encoding:**

- `encode(text: str) -> list[int]`: Encode text to token IDs, treating special tokens as regular text
- `encode_with_special(text: str) -> list[int]`: Encode text, recognizing special tokens in the input
- `encode_batch(texts: list[str]) -> list[list[int]]`: Encode multiple texts in parallel

**Decoding:**

- `decode(tokens: list[int]) -> str`: Decode token IDs to text (raises error on invalid UTF-8)
- `decode_bytes(tokens: list[int]) -> bytes`: Decode token IDs to raw bytes
- `decode_lossy(tokens: list[int]) -> str`: Decode token IDs, replacing invalid UTF-8 with �

**Properties:**

- `vocab_size: int`: Total vocabulary size including special tokens
- `cache_len: int`: Number of entries in the LRU cache

**Cache management:**

- `clear_cache()`: Clear the encoding cache

#### StreamingDecoder

The streaming decoder is essential for real-time LLM applications where you receive tokens one at a time and need to display text incrementally:

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

**Why use streaming decoder?**

BPE tokens don't always align with UTF-8 character boundaries. For example, a multi-byte Unicode character like "世" (3 bytes: `0xE4 0xB8 0x96`) might be split across multiple tokens. The streaming decoder buffers incomplete byte sequences and only outputs text when complete characters are available, preventing display corruption.

**Methods:**

- `add_token(token_id: int) -> str | None`: Add a token and return complete characters, or None if still buffering
- `add_tokens(token_ids: list[int]) -> str | None`: Add multiple tokens at once
- `flush() -> str`: Flush remaining buffered bytes (incomplete sequences become �)
- `reset()`: Clear the buffer and start fresh

**Properties:**

- `has_pending: bool`: Whether there are buffered bytes waiting for completion
- `pending_bytes: int`: Number of bytes currently buffered

### Rust API

The Rust API provides similar functionality with strongly-typed interfaces. See the [API documentation](https://docs.rs/splintr) for detailed information.

## Streaming Decoder

The streaming decoder is particularly important when working with LLM APIs that stream tokens:

```python
import openai
from splintr import Tokenizer

tokenizer = Tokenizer.from_pretrained("cl100k_base")
decoder = tokenizer.streaming_decoder()

# Example with OpenAI streaming API
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        # Get token IDs from the API (pseudo-code, actual API may vary)
        token_ids = get_token_ids(chunk)

        for token_id in token_ids:
            if text := decoder.add_token(token_id):
                print(text, end="", flush=True)

# Don't forget to flush at the end
print(decoder.flush())
```

This approach ensures that:
1. Users see text as soon as complete characters are available
2. Multi-byte Unicode characters display correctly
3. No corruption occurs at token boundaries

## Performance

Benchmarks performed on Linux (6.16.8-arch3-1) with 24 CPU cores, comparing splintr to tiktoken (the reference Python implementation).

### Single Text Encoding

Performance on various text types:

| Content Type | Size | splintr (ms) | tiktoken (ms) | Speedup |
|--------------|------|--------------|---------------|---------|
| Long English | 450,000 chars | 7.94 | 19.91 | **2.5x** |
| Python Code | 59,200 chars | 1.67 | 5.90 | **3.5x** |
| JSON | 29,000 chars | 1.20 | 2.76 | **2.3x** |
| Numbers | 55,000 chars | 2.27 | 6.09 | **2.7x** |
| Whitespace-heavy | 50,000 chars | 1.36 | 4.91 | **3.6x** |
| Chinese | 11,500 chars | 1.09 | 1.45 | **1.3x** |

### Batch Encoding

Batch operations show significant speedup through parallelism:

| Configuration | splintr parallel (ms) | tiktoken (ms) | Speedup vs tiktoken |
|---------------|----------------------|---------------|---------------------|
| 10 × 1,000 chars | 0.25 | 0.48 | **1.9x** |
| 100 × 1,000 chars | 1.11 | 4.66 | **4.2x** |
| 1,000 × 100 chars | 1.42 | 6.95 | **4.9x** |
| 100 × 10,000 chars | 8.24 | 45.72 | **5.5x** |

**Parallel speedup within splintr:**
- 100 × 1,000 chars: 8.6x faster (parallel vs sequential)
- 1,000 × 100 chars: 16.8x faster (parallel vs sequential)

### Running Benchmarks

To reproduce these benchmarks or test on your own hardware:

```bash
# Clone the repository
git clone https://github.com/farhan/splintr.git
cd splintr

# Install dependencies (requires Python 3.8+)
pip install -e .
pip install tiktoken

# Run the benchmark suite
cd benchmarks
python benchmark.py --model cl100k_base --output results/my_benchmark.json

# View results
cat results/my_benchmark.md
```

The benchmark suite tests:
- Single text encoding across various content types (English, code, multilingual, etc.)
- Batch encoding with different batch sizes and text lengths
- Streaming decoder performance
- Special token handling

You can customize the benchmark by modifying `benchmark.py` or adding your own test data in the `data/` directory.

## Supported Models

| Model | Use Case | Vocabulary Size | Special Tokens | Import Constant |
|-------|----------|----------------|----------------|-----------------|
| `cl100k_base` | GPT-4, GPT-3.5-turbo | ~100,000 | 5 | `CL100K_BASE_PATTERN` |
| `o200k_base` | GPT-4o | ~200,000 | 2 | `O200K_BASE_PATTERN` |

**Special tokens:**

- **cl100k_base**: `<|endoftext|>`, `<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`, `<|endofprompt|>`
- **o200k_base**: `<|endoftext|>`, `<|endofprompt|>`

## Use Cases

splintr is designed for:

- **LLM applications**: Tokenizing prompts and streaming decoder for real-time output display
- **Training pipelines**: Fast batch encoding of large datasets for model training
- **Token counting**: Estimating API costs or enforcing token limits
- **Text preprocessing**: Converting text to tokens for embedding models or other NLP tasks

## Contributing

Contributions are welcome! Here's how you can help:

1. **Report bugs**: Open an issue with a minimal reproduction case
2. **Suggest features**: Describe your use case and why the feature would be helpful
3. **Submit pull requests**:
   - Add tests for new functionality
   - Run `cargo test` and `cargo clippy` before submitting
   - Update documentation as needed

### Development Setup

```bash
# Clone the repository
git clone https://github.com/farhan/splintr.git
cd splintr

# Install pre-commit hook (recommended)
cp hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Build the Rust library
cargo build --release

# Build Python bindings
pip install maturin
maturin develop --release

# Run tests
cargo test                    # Rust tests
cargo clippy --all-targets    # Linting
cargo fmt --all --check       # Format check
```

The pre-commit hook automatically runs formatting, clippy, and tests before each commit.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

splintr builds upon concepts from:
- [tiktoken](https://github.com/openai/tiktoken) - OpenAI's reference BPE tokenizer
- [tokenizers](https://github.com/huggingface/tokenizers) - Hugging Face's tokenization library

The performance optimizations are informed by profiling real-world usage patterns in LLM applications.
