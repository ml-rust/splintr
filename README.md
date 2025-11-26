# Splintr

[![Crates.io](https://img.shields.io/crates/v/splintr.svg)](https://crates.io/crates/splintr)
[![PyPI](https://img.shields.io/pypi/v/splintr-rs.svg)](https://pypi.org/project/splintr-rs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A high-performance BPE tokenizer implemented in Rust with Python bindings, designed for efficient tokenization of text in machine learning applications, particularly for large language models.

## Features

Splintr implements several optimizations that make tokenization faster and more efficient:

- **PCRE2 with JIT compilation**: Uses PCRE2's just-in-time compilation for regex matching, providing 2-4x speedup over fancy-regex on pattern matching operations
- **Rayon parallelism**: Leverages multiple CPU cores for encoding batches of text and individual regex chunks within each text
- **Linked-list BPE algorithm**: Implements BPE using a linked-list structure that avoids O(N²) complexity on pathological inputs with many repetitive patterns
- **FxHashMap**: Uses rustc's FxHasher for faster lookups compared to the default SipHash, trading cryptographic security for speed in non-adversarial contexts
- **Aho-Corasick for special tokens**: Employs the Aho-Corasick algorithm for fast multi-pattern matching of special tokens, avoiding regex alternation overhead
- **LRU cache**: Caches frequently encoded text chunks to avoid redundant BPE encoding operations
- **UTF-8 streaming decoder**: Safely handles token-by-token decoding for LLM output, buffering incomplete UTF-8 sequences across token boundaries
- **Extended agent tokens**: 54 special tokens for chat models, Chain-of-Thought reasoning, ReAct agents, tool calling, RAG citations, and multimodal applications (see [Special Tokens](docs/special_tokens.md))

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

- `encode(text: str) -> list[int]`: Encode text to token IDs (sequential, optimal for most use cases)
- `encode_with_special(text: str) -> list[int]`: Encode text, recognizing special tokens in the input
- `encode_batch(texts: list[str]) -> list[list[int]]`: Encode multiple texts in parallel (uses Rayon)
- `encode_rayon(text: str) -> list[int]`: Encode using Rayon parallelization (only beneficial for texts >1MB)

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

The Rust API provides similar functionality with strongly-typed interfaces:

**Encoding:**

- `encode(&self, text: &str) -> Vec<u32>`: Sequential encoding (optimal for texts <1MB)
- `encode_with_special(&self, text: &str) -> Vec<u32>`: Encode with special token recognition
- `encode_batch(&self, texts: &[String]) -> Vec<Vec<u32>>`: Parallel encoding across texts
- `encode_rayon(&self, text: &str) -> Vec<u32>`: Parallel encoding within text (for texts >1MB)

**Decoding:**

- `decode(&self, tokens: &[u32]) -> Result<String, TokenizerError>`: Decode to UTF-8 string
- `decode_bytes(&self, tokens: &[u32]) -> Vec<u8>`: Decode to raw bytes
- `decode_lossy(&self, tokens: &[u32]) -> String`: Decode with replacement for invalid UTF-8

See the [API documentation](https://docs.rs/splintr) for detailed information.

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

Benchmarks performed on Linux (6.16.8-arch3-1) with 24 CPU cores, comparing splintr against tiktoken (reference Python implementation), Hugging Face tokenizers, and TokenDagger.

### Single Text Encoding

Splintr achieves **3-4x faster** single-text encoding compared to tiktoken across various text sizes:

![Single Text Encoding Comparison](images/benchmark_single.png)

**Latency by text type:**

![Latency Comparison](images/benchmark_single_latency.png)

Splintr consistently maintains lower latency across different content types (Python code, JSON, English prose, Chinese text), making it ideal for interactive applications and real-time processing.

### Batch Encoding

For batch operations, splintr achieves **10-12x speedup** over tiktoken by parallelizing across texts:

![Batch Encoding Throughput](images/benchmark_batch.png)

| Configuration    | Splintr      | Tiktoken   | Speedup  |
| ---------------- | ------------ | ---------- | -------- |
| 1,000 × 100 chars | 111 MB/s    | 9 MB/s     | **12.3x** |
| 100 × 1,000 chars | 89 MB/s     | 8 MB/s     | **11.1x** |
| 10 × 10,000 chars | 72 MB/s     | 7 MB/s     | **10.3x** |

![Batch Speedup vs Tiktoken](images/benchmark_batch_speedup.png)

The batch encoding speedup scales effectively across different batch configurations, with higher speedups on larger batches where parallelization overhead is amortized.

### Design Decision: Sequential by Default

Splintr uses **sequential encoding for single texts** and **parallel encoding across batches**. This design choice is based on empirical benchmarking:

![Sequential vs Rayon Internal Parallelization](images/benchmark_splintr.png)

**Key findings:**

- **Sequential is faster** for texts up to ~1MB (typical LLM use case)
- Rayon's parallelization overhead only pays off at ~1MB+ text sizes
- Most real-world inputs (prompts, documents, code) are well under 1MB
- `encode()` uses sequential processing for optimal single-text performance
- `encode_batch()` parallelizes across multiple texts for maximum throughput

This architecture ensures splintr is optimized for the most common tokenization patterns in LLM applications while still providing excellent batch performance for data processing pipelines.

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

## Supported Vocabularies

| Vocabulary    | Used By              | Vocabulary Size | Special Tokens | Import Constant       |
| ------------- | -------------------- | --------------- | -------------- | --------------------- |
| `cl100k_base` | GPT-4, GPT-3.5-turbo | ~100,000        | 5 + 54 agent   | `CL100K_BASE_PATTERN` |
| `o200k_base`  | GPT-4o               | ~200,000        | 2 + 54 agent   | `O200K_BASE_PATTERN`  |

More vocabularies will be added in future releases.

**OpenAI standard tokens:**

- **cl100k_base**: `<|endoftext|>`, `<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`, `<|endofprompt|>`
- **o200k_base**: `<|endoftext|>`, `<|endofprompt|>`

**Agent tokens (54 per model):**

Splintr extends both vocabularies with tokens for building agent systems. See [docs/special_tokens.md](docs/special_tokens.md) for complete documentation.

```python
from splintr import Tokenizer, CL100K_AGENT_TOKENS

tokenizer = Tokenizer.from_pretrained("cl100k_base")

# Encode with special tokens
text = "<|think|>Let me reason...<|/think|>The answer is 42."
tokens = tokenizer.encode_with_special(text)

# Access token IDs programmatically
print(CL100K_AGENT_TOKENS.THINK)      # 100282
print(CL100K_AGENT_TOKENS.FUNCTION)   # 100292
```

| Category     | Tokens                                              | Purpose                    |
| ------------ | --------------------------------------------------- | -------------------------- |
| Conversation | `system`, `user`, `assistant`, `im_start`, `im_end` | ChatML format              |
| Thinking     | `think`                                             | Chain-of-Thought reasoning |
| ReAct        | `plan`, `step`, `act`, `observe`                    | Agent action loops         |
| Tools        | `function`, `result`, `error`                       | Function calling           |
| Code         | `code`, `output`, `lang`                            | Code execution             |
| RAG          | `context`, `quote`, `cite`, `source`                | Citations                  |
| Memory       | `memory`, `recall`                                  | State persistence          |
| Control      | `pad`, `stop`, `sep`                                | Sequence control           |
| Multimodal   | `image`, `audio`, `video`                           | Non-text content           |
| Document     | `title`, `section`, `summary`                       | Structured docs            |

## Use Cases

Splintr is designed for:

- **LLM applications**: Tokenizing prompts and streaming decoder for real-time output display
- **Agent systems**: Building ReAct agents, tool-calling systems, and Chain-of-Thought reasoning
- **Training pipelines**: Fast batch encoding of large datasets for model training
- **RAG applications**: Structured context injection with citation support
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

## Citation

If you use Splintr in your research, please cite:

```bibtex
@software{splintr,
  author = {Farhan Syah},
  title = {Splintr: High-Performance BPE Tokenizer},
  year = {2025},
  url = {https://github.com/farhan-syah/splintr}
}
```

## Acknowledgments

Splintr builds upon concepts from:

- [tiktoken](https://github.com/openai/tiktoken) - OpenAI's reference BPE tokenizer
- [tokenizers](https://github.com/huggingface/tokenizers) - Hugging Face's tokenization library

The performance optimizations are informed by profiling real-world usage patterns in LLM applications.
