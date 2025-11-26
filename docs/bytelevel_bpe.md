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

| Byte Range | Decimal Range | Mapping                                          |
| ---------- | ------------- | ------------------------------------------------ |
| 0x21-0x7E  | 33-126        | Direct ASCII (printable characters `!` to `~`)   |
| 0x00-0x20  | 0-32          | Mapped to Unicode range U+0100-U+0120 (Ā to Š)   |
| 0x7F-0xFF  | 127-255       | Mapped to Unicode range U+0121-U+01A0 (š to Ơ)   |

### Mapping Examples

| Byte (Hex) | Byte (Dec) | Character | Description        |
| ---------- | ---------- | --------- | ------------------ |
| 0x00       | 0          | Ā (U+0100)| Null byte          |
| 0x0A       | 10         | Ċ (U+010A)| Newline            |
| 0x20       | 32         | Š (U+0120)| Space              |
| 0x21       | 33         | !         | Direct (unchanged) |
| 0x41       | 65         | A         | Direct (unchanged) |
| 0x7E       | 126        | ~         | Direct (unchanged) |
| 0x7F       | 127        | š (U+0121)| DEL character      |
| 0xFF       | 255        | Ơ (U+01A0)| Max byte value     |

## Why ByteLevel Encoding?

### Advantages

1. **Complete Coverage**: Can tokenize any byte sequence, including binary data
2. **No Unknown Tokens**: Every possible input has a valid tokenization
3. **Language Agnostic**: Works with any language or script without special handling
4. **Compact Vocabulary**: 256 base tokens cover all possible bytes

### Comparison with Standard BPE

| Aspect              | Standard BPE          | ByteLevel BPE         |
| ------------------- | --------------------- | --------------------- |
| Base vocabulary     | Unicode characters    | 256 bytes             |
| Unknown handling    | Special `<unk>` token | Never needed          |
| Non-UTF8 input      | May fail              | Always works          |
| Vocabulary size     | Usually larger        | Can be more compact   |

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

In ByteLevel BPE, the space character (0x20) is mapped to `Š` (U+0120). This is why you may see vocabulary entries like:

- `ŠHello` - "Hello" with leading space
- `Šthe` - "the" with leading space
- `Š` - standalone space

This convention allows the tokenizer to distinguish between word-initial and word-internal tokens.

## See Also

- [Special Tokens Reference](special_tokens.md) - DeepSeek V3 special tokens
- [README.md](../README.md) - Project overview and quick start
