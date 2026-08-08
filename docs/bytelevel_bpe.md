# ByteLevel BPE Encoding

This document describes the ByteLevel BPE encoding used by DeepSeek V3 and other models that employ this tokenization strategy.

## Overview

ByteLevel BPE (Byte-Pair Encoding) differs from standard BPE used by OpenAI and Llama tokenizers. Instead of operating directly on Unicode characters, ByteLevel BPE first converts text to raw bytes and then applies BPE on a byte-level vocabulary.

## How It Works

1. **Text to Bytes**: Input text is first converted to UTF-8 bytes
2. **Byte Mapping**: Raw bytes (0-255) are mapped to printable Unicode characters
3. **BPE Encoding**: Standard BPE is applied to the mapped characters
4. **Decoding**: Reverse the process to get back the original text

## Byte-to-Character Mapping

ByteLevel encoding uses a GPT-2 style mapping to ensure all bytes can be represented as printable characters:

The "printable" bytes map to themselves; the remaining 68 bytes are reassigned, in ascending byte order, to the contiguous block starting at U+0100:

| Byte Range                                 | Decimal            | Mapping                                         |
| ------------------------------------------ | ------------------ | ----------------------------------------------- |
| 0x21-0x7E                                  | 33-126             | Direct — printable ASCII (`!` to `~`)           |
| 0xA1-0xAC                                  | 161-172            | Direct — Latin-1 (`¡` to `¬`)                   |
| 0xAE-0xFF                                  | 174-255            | Direct — Latin-1 (`®` to `ÿ`)                   |
| 0x00-0x20, 0x7F-0xA0, 0xAD (the 68 others) | 0-32, 127-160, 173 | Remapped, in byte order, to U+0100…U+0143 (Ā…Ń) |

### Mapping Examples

| Byte (Hex) | Byte (Dec) | Character  | Description                         |
| ---------- | ---------- | ---------- | ----------------------------------- |
| 0x00       | 0          | Ā (U+0100) | Null byte (first remapped)          |
| 0x0A       | 10         | Ċ (U+010A) | Newline                             |
| 0x20       | 32         | Ġ (U+0120) | Space (last of the 0x00-0x20 block) |
| 0x21       | 33         | !          | Direct (unchanged)                  |
| 0x41       | 65         | A          | Direct (unchanged)                  |
| 0x7E       | 126        | ~          | Direct (unchanged)                  |
| 0x7F       | 127        | ġ (U+0121) | DEL (first remapped after 0x20)     |
| 0xAD       | 173        | Ń (U+0143) | Soft hyphen (last remapped)         |
| 0xFF       | 255        | ÿ (U+00FF) | Direct — Latin-1 maps to itself     |

## Why ByteLevel Encoding?

### Advantages

1. **Complete Coverage**: Can tokenize any byte sequence, including binary data
2. **No Unknown Tokens**: Every possible input has a valid tokenization
3. **Language Agnostic**: Works with any language or script without special handling
4. **Compact Vocabulary**: 256 base tokens cover all possible bytes

### Comparison with Standard BPE

| Aspect           | Standard BPE          | ByteLevel BPE       |
| ---------------- | --------------------- | ------------------- |
| Base vocabulary  | Unicode characters    | 256 bytes           |
| Unknown handling | Special `<unk>` token | Never needed        |
| Non-UTF8 input   | May fail              | Always works        |
| Vocabulary size  | Usually larger        | Can be more compact |

## Models Using ByteLevel BPE

- **DeepSeek V3**: 128,000 BPE tokens with ByteLevel encoding
- **GPT-2**: Original implementation of ByteLevel BPE
- **RoBERTa**: Uses GPT-2 style ByteLevel encoding
- **BART**: ByteLevel BPE for both encoder and decoder

## Implementation in Splintr

Splintr provides transparent ByteLevel encoding support for DeepSeek V3:

### Python

```python
from splintr import Tokenizer

# Load DeepSeek V3 tokenizer (ByteLevel encoding handled automatically)
tokenizer = Tokenizer.from_pretrained("deepseek_v3")

# Encoding works the same as other tokenizers
text = "Hello, 世界! 🌍"
tokens = tokenizer.encode(text)
decoded = tokenizer.decode(tokens)
assert decoded == text

# ByteLevel encoding handles any UTF-8 text
chinese = "你好世界"
tokens = tokenizer.encode(chinese)  # [30594, 3427]

# Even handles edge cases like mixed scripts
mixed = "café naïve 日本語"
tokens = tokenizer.encode(mixed)
assert tokenizer.decode(tokens) == mixed
```

### Rust

```rust
use splintr::{Tokenizer, DEEPSEEK_V3_PATTERN};

// ByteLevel encoding is handled by the tokenizer
let tokenizer = Tokenizer::from_pretrained("deepseek_v3").unwrap();

let text = "Hello, 世界!";
let tokens = tokenizer.encode(text);
let decoded = tokenizer.decode(&tokens).unwrap();
assert_eq!(decoded, text);
```

## Technical Details

### The Byte Mapping Function

The mapping from bytes to characters follows this logic:

```python
def bytes_to_unicode():
    """Create byte-to-unicode mapping (GPT-2 style)."""
    # Printable ASCII characters stay as-is
    bs = list(range(ord("!"), ord("~") + 1))  # 33-126
    bs += list(range(ord("¡"), ord("¬") + 1))  # 161-172
    bs += list(range(ord("®"), ord("ÿ") + 1))  # 174-255

    cs = bs[:]
    n = 0
    # Map remaining bytes (0-32, 127-160, 173) to U+0100+
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1

    return dict(zip(bs, [chr(c) for c in cs]))
```

### Space Handling

In ByteLevel BPE, the space character (0x20) is mapped to `Ġ` (U+0120). This is why you may see vocabulary entries like:

- `ĠHello` - "Hello" with leading space
- `Ġthe` - "the" with leading space
- `Ġ` - standalone space

This convention allows the tokenizer to distinguish between word-initial and word-internal tokens.

## Streaming Decoder

When streaming LLM output token-by-token, a ByteLevel vocabulary needs one decoding step more than a raw one. `streaming_decoder()` handles it automatically.

### The Problem

BPE tokens don't align with UTF-8 character boundaries. For ByteLevel tokenizers, there's an additional layer of complexity:

1. Tokens are in ByteLevel representation (e.g., `Ġ` for space)
2. Multi-byte UTF-8 characters may split across tokens
3. Both layers must be handled correctly for streaming output

### The Solution

There is one decoder class and one call, on every tokenizer class. It is built from the tokenizer's own configuration, so a ByteLevel vocabulary gets the ByteLevel unmapping without the caller asking for it:

```python
from splintr import Tokenizer

# DeepSeek V3 uses ByteLevel BPE
tokenizer = Tokenizer.from_pretrained("deepseek_v3")

# Same call as for cl100k_base — the tokenizer supplies the ByteLevel rule
decoder = tokenizer.streaming_decoder()

# Process tokens as they arrive from LLM
for token_id in token_stream:
    if text := decoder.add_token(token_id):
        print(text, end="", flush=True)

# Flush remaining buffered bytes
print(decoder.flush())
```

This is why there is no decoder to choose. A decoder paired with the wrong kind of vocabulary produced mojibake silently — running UTF-8 assembly over `Ġ`-spelled bytes, or ByteLevel unmapping over raw ones — and that pairing is no longer expressible, in Rust or in Python.

### How It Works

On a ByteLevel vocabulary the decoder performs two-stage decoding:

1. **ByteLevel Decode**: Converts ByteLevel-encoded token bytes back to raw bytes
   - `Ġ` (U+0120) → `0x20` (space)
   - `Ċ` (U+010A) → `0x0A` (newline)
   - Regular ASCII stays unchanged

2. **UTF-8 Assembly**: Buffers raw bytes until complete UTF-8 characters are available
   - Handles multi-byte characters split across token boundaries
   - Only outputs when valid UTF-8 characters can be formed

The first stage is skipped for a raw vocabulary (cl100k_base, o200k_base), and the same machinery additionally resolves `<0xNN>` byte fallback and the `▁` metaspace substitution where the tokenizer declares them. Whichever rules apply, `"".join(chunks) + flush()` equals `decode(ids)`.

### API

```python
decoder = tokenizer.streaming_decoder()

# Add single token
text = decoder.add_token(token_id)  # Returns str or None

# Add multiple tokens
text = decoder.add_tokens([token_id1, token_id2])

# Flush remaining bytes (incomplete sequences become U+FFFD)
remaining = decoder.flush()

# Reset decoder state
decoder.reset()

# Check buffer status
decoder.has_pending    # bool: True if bytes are buffered
decoder.pending_bytes  # int: Number of buffered bytes
```

## See Also

- [Special Tokens Reference](special_tokens.md) - DeepSeek V3 special tokens
- [README.md](../README.md) - Project overview and quick start
