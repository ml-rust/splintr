![Splintr](images/splntr.png)

[![Crates.io](https://img.shields.io/crates/v/splintr.svg)](https://crates.io/crates/splintr) [![PyPI](https://img.shields.io/pypi/v/splintr-rs.svg)](https://pypi.org/project/splintr-rs/) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**A high-performance BPE tokenizer built with Rust with Python bindings, focused on speed, safety, and resource optimization.**

## The Problem

Tokenization is everywhere in modern AI. Whether you're building LLM applications, training models, or processing data pipelines, you're tokenizing text constantly. But existing tokenizers have a problem: they're slow.

When you need to tokenize batches of prompts, documents, or training data, you're stuck waiting. Python-based tokenizers can't fully leverage modern multi-core CPUs. You need something faster.

## The Solution

Splintr brings Rust performance to Python. Built from the ground up for speed and efficiency:

![Batch Encoding Throughput](images/benchmark_batch.png)

| Configuration | Splintr      | Tiktoken | HuggingFace | TokenDagger |
| ------------- | ------------ | -------- | ----------- | ----------- |
| 1,000 texts   | **111 MB/s** | 9 MB/s   | 28 MB/s     | 9 MB/s      |
| 500 texts     | **107 MB/s** | 10 MB/s  | 27 MB/s     | 8 MB/s      |
| 100 texts     | **69 MB/s**  | 7 MB/s   | 20 MB/s     | 6 MB/s      |

**10-12x faster than tiktoken. 4x faster than HuggingFace. Built in Rust, accessible from Python.**

## Quick Start

### Python

```bash
pip install splintr-rs
```

```python
from splintr import Tokenizer

# Load a pretrained vocabulary (OpenAI)
tokenizer = Tokenizer.from_pretrained("cl100k_base")

# Or load Llama 3 tokenizer (Meta) - supports all versions up to Llama 3.3
# tokenizer = Tokenizer.from_pretrained("llama3")

# Encode text to token IDs
tokens = tokenizer.encode("Hello, world!")
print(tokens)  # [9906, 11, 1917, 0]

# Decode token IDs back to text
text = tokenizer.decode(tokens)
print(text)  # "Hello, world!"

# Batch encode multiple texts in parallel (this is where it shines)
texts = ["Hello, world!", "How are you?", "Machine learning is fun!"]
batch_tokens = tokenizer.encode_batch(texts)
print(batch_tokens)  # [[9906, 11, 1917, 0], [4438, 527, 499, 30], ...]
```

### Rust

```toml
[dependencies]
splintr = "0.3.0"
```

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

## Key Features

**Performance where it matters:**

- **12x faster batch encoding** - Parallel processing across multiple texts using Rayon
- **3-4x faster single text encoding** - Optimized sequential algorithm for typical use cases
- **Smart parallelization** - Sequential for small texts (<1MB), parallel for large datasets
- **LRU caching** - Avoid redundant encoding of frequently seen text chunks

**Built for production:**

- **Compatible vocabularies** - Supports cl100k_base, o200k_base (OpenAI), and Llama 3 family (Meta), with a familiar API
- **Streaming decoder** - Real-time LLM output display with proper UTF-8 handling
- **54 agent tokens** - Built-in support for chat, CoT reasoning, ReAct agents, tool calling, RAG citations
- **Battle-tested algorithms** - PCRE2 with JIT, Aho-Corasick for special tokens, linked-list BPE

**Cross-platform:**

- Python bindings via PyO3 (Linux, macOS, Windows)
- Native Rust library for maximum performance

## Performance Deep Dive

All benchmarks performed on Linux (6.16.8-arch3-1) with 24 CPU cores, comparing against tiktoken (reference Python implementation), Hugging Face tokenizers, and TokenDagger.

### Single Text Encoding

For single texts, splintr achieves **3-4x faster** encoding across various text sizes:

![Single Text Encoding Comparison](images/benchmark_single.png)

**Latency by content type:**

![Latency Comparison](images/benchmark_single_latency.png)

Consistent low latency across Python code, JSON, English prose, and Chinese text makes splintr ideal for interactive applications and real-time processing.

### Batch Encoding

The real magic happens with batches. Splintr parallelizes across texts to achieve **10-12x speedup**:

![Batch Speedup vs Tiktoken](images/benchmark_batch_speedup.png)

Higher speedups on larger batches where parallelization overhead is amortized. Perfect for:

- Training data preprocessing
- Bulk document tokenization
- API batch processing
- Data pipeline throughput

### Design Decision: Sequential by Default

Splintr uses **sequential encoding for single texts** and **parallel encoding across batches** based on empirical benchmarking:

![Sequential vs Rayon Internal Parallelization](images/benchmark_splintr.png)

**Key findings:**

- Sequential is faster for texts up to ~1MB (typical LLM prompts and documents)
- Rayon's parallelization overhead only pays off at ~1MB+ text sizes
- Most real-world inputs are well under 1MB
- `encode()` uses sequential processing for optimal single-text performance
- `encode_batch()` parallelizes across multiple texts for maximum throughput
- `encode_rayon()` available for the rare cases where you have >1MB single texts

This architecture ensures splintr is optimized for the most common tokenization patterns in LLM applications.

### Running Benchmarks Yourself

```bash
# Clone and install
git clone https://github.com/farhan-syah/splintr.git
cd splintr
pip install -e .
pip install tiktoken

# Run the benchmark suite
cd benchmarks
python benchmark.py --model cl100k_base --output results/my_benchmark.json

# View results
cat results/my_benchmark.md
```

The benchmark suite tests single text encoding, batch encoding, streaming decoder performance, and special token handling across various content types.

## Streaming Decoder

The streaming decoder is essential for real-time LLM applications where tokens arrive one at a time:

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

### Why You Need This

BPE tokens don't align with UTF-8 character boundaries. A multi-byte Unicode character like "世" (3 bytes: `0xE4 0xB8 0x96`) might split across tokens. The streaming decoder:

1. Buffers incomplete byte sequences across token boundaries
2. Only outputs text when complete UTF-8 characters are available
3. Prevents display corruption in streaming LLM output
4. Handles edge cases automatically

### Real-World Example

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

### API Methods

**Core operations:**

- `add_token(token_id: int) -> str | None`: Add a token, return complete characters or None if buffering
- `add_tokens(token_ids: list[int]) -> str | None`: Add multiple tokens at once
- `flush() -> str`: Flush buffered bytes (incomplete sequences become �)
- `reset()`: Clear the buffer and start fresh

**Properties:**

- `has_pending: bool`: Whether there are buffered bytes waiting
- `pending_bytes: int`: Number of bytes currently buffered

## API Reference

### Python API

#### Tokenizer

**Loading:**

```python
# Load pretrained model (includes vocabulary and special tokens)
tokenizer = Tokenizer.from_pretrained("cl100k_base")  # or "o200k_base", "llama3"

# Load from custom vocabulary file
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

See the [API documentation](https://docs.rs/splintr) for complete details.

## Supported Vocabularies

| Vocabulary    | Used By                       | Vocabulary Size | Special Tokens | Import Constant       |
| ------------- | ----------------------------- | --------------- | -------------- | --------------------- |
| `cl100k_base` | GPT-4, GPT-3.5-turbo          | ~100,000        | 5 + 54 agent   | `CL100K_BASE_PATTERN` |
| `o200k_base`  | GPT-4o                        | ~200,000        | 2 + 54 agent   | `O200K_BASE_PATTERN`  |
| `llama3`      | Llama 3, 3.1, 3.2, 3.3 (Meta) | ~128,000        | 11 + 54 agent  | `LLAMA3_PATTERN`      |

**OpenAI standard tokens:**

- **cl100k_base**: `<|endoftext|>`, `<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`, `<|endofprompt|>`
- **o200k_base**: `<|endoftext|>`, `<|endofprompt|>`

**Meta Llama 3 standard tokens:**

- **llama3**: `<|begin_of_text|>`, `<|end_of_text|>`, `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`, `<|eom_id|>` (3.1+), `<|python_tag|>` (3.1+), `<|step_id|>` (3.2-Vision), `<|image|>` (3.2-Vision)

### Agent Tokens (54 per model)

Splintr extends all vocabularies with tokens for building agent systems. See [docs/special_tokens.md](docs/special_tokens.md) for complete documentation.

```python
from splintr import Tokenizer, CL100K_AGENT_TOKENS, LLAMA3_AGENT_TOKENS

# OpenAI models
tokenizer = Tokenizer.from_pretrained("cl100k_base")
text = "<|think|>Let me reason...<|/think|>The answer is 42."
tokens = tokenizer.encode_with_special(text)
print(CL100K_AGENT_TOKENS.THINK)      # 100282
print(CL100K_AGENT_TOKENS.FUNCTION)   # 100292

# Llama 3 models (vocabulary includes all special tokens up to Llama 3.3)
tokenizer = Tokenizer.from_pretrained("llama3")
tokens = tokenizer.encode_with_special(text)
print(LLAMA3_AGENT_TOKENS.THINK)           # 128305
print(LLAMA3_AGENT_TOKENS.FUNCTION)        # 128315
print(LLAMA3_AGENT_TOKENS.BEGIN_OF_TEXT)   # 128000 (official Meta token)
print(LLAMA3_AGENT_TOKENS.IMAGE)           # 128256 (official Meta 3.2-Vision token)
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

## How It Works

Splintr implements several optimizations that make tokenization faster:

- **PCRE2 with JIT compilation**: 2-4x speedup on regex pattern matching
- **Rayon parallelism**: Leverages multiple CPU cores for batch encoding
- **Linked-list BPE algorithm**: Avoids O(N²) complexity on pathological inputs
- **FxHashMap**: Faster lookups than default SipHash for non-adversarial contexts
- **Aho-Corasick for special tokens**: Fast multi-pattern matching without regex alternation
- **LRU cache**: Avoids redundant BPE encoding of frequently seen chunks

## Use Cases

**LLM Applications:**

- Tokenizing prompts with 3-4x lower latency
- Streaming decoder for real-time output display
- Token counting for API cost estimation

**Agent Systems:**

- Building ReAct agents with structured reasoning tokens
- Tool-calling systems with function tokens
- Chain-of-Thought reasoning with thinking tokens

**Training Pipelines:**

- Fast batch encoding of large datasets (10-12x speedup)
- Preprocessing millions of documents efficiently
- Parallel tokenization across distributed systems

**RAG Applications:**

- Structured context injection with citation tokens
- Document chunking with section markers
- Source tracking through tokenization

**Data Processing:**

- Bulk document tokenization
- Multi-language text processing
- Real-time text preprocessing

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
git clone https://github.com/farhan-syah/splintr.git
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

## Acknowledgments

Splintr builds upon concepts from:

- [tiktoken](https://github.com/openai/tiktoken) - OpenAI's reference BPE tokenizer
- [tokenizers](https://github.com/huggingface/tokenizers) - Hugging Face's tokenization library

The performance optimizations are informed by profiling real-world usage patterns in LLM applications.

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
