"""
Splintr - Fast Rust BPE tokenizer with Python bindings

A high-performance tokenizer featuring:
- Regexr with JIT and SIMD (default, pure Rust)
- Optional PCRE2 with JIT (requires pcre2 feature)
- Rayon parallelism for multi-core encoding
- Linked-list BPE algorithm (avoids O(N^2) on pathological inputs)
- FxHashMap for fast lookups
- Aho-Corasick for fast special token matching
- LRU cache for frequently encoded chunks
- UTF-8 streaming decoder for LLM output
- Agent tokens for chat/reasoning/tool-use applications

Supported tokenizers:
- cl100k_base: GPT-4, GPT-3.5-turbo
- o200k_base: GPT-4o
- llama3/llama3.1/llama3.2/llama3.3: Meta Llama 3 family
- deepseek_v3/deepseek-v3: DeepSeek V3
- mistral_v1: Mistral 7B v0.1/v0.2, Mixtral 8x7B
- mistral_v2: Mistral 7B v0.3, Mixtral 8x22B, Codestral
- mistral_v3: Mistral NeMo, Large 2, Pixtral (Tekken)

Usage:
    from splintr import Tokenizer

    # Load pretrained model (uses regexr by default)
    tokenizer = Tokenizer.from_pretrained("cl100k_base")  # GPT-4
    tokenizer = Tokenizer.from_pretrained("llama3")       # Llama 3
    tokenizer = Tokenizer.from_pretrained("deepseek_v3")  # DeepSeek V3
    tokenizer = Tokenizer.from_pretrained("mistral_v1")   # Mistral 7B v0.1/v0.2
    tokenizer = Tokenizer.from_pretrained("mistral_v2")   # Mistral 7B v0.3
    tokenizer = Tokenizer.from_pretrained("mistral_v3")   # Mistral NeMo (Tekken)

    # Use PCRE2 backend (requires pcre2 feature)
    # tokenizer = Tokenizer.from_pretrained("cl100k_base").pcre2(True)

    # Encode text
    tokens = tokenizer.encode("Hello, world!")
    print(tokens)

    # Decode tokens
    text = tokenizer.decode(tokens)
    print(text)

    # Batch encode (parallel)
    batch_tokens = tokenizer.encode_batch(["Hello", "World"])

    # Streaming decode (for LLM output)
    decoder = tokenizer.streaming_decoder()
    for token_id in token_stream:
        if text := decoder.add_token(token_id):
            print(text, end="", flush=True)
    print(decoder.flush())

    # ByteLevel streaming decode (for DeepSeek V3, GPT-2)
    tokenizer = Tokenizer.from_pretrained("deepseek_v3")
    decoder = tokenizer.byte_level_streaming_decoder()
    for token_id in token_stream:
        if text := decoder.add_token(token_id):
            print(text, end="", flush=True)
    print(decoder.flush())

Agent Tokens:
    from splintr import (
        Tokenizer,
        CL100K_AGENT_TOKENS, O200K_AGENT_TOKENS,
        LLAMA3_AGENT_TOKENS, DEEPSEEK_V3_AGENT_TOKENS,
        MISTRAL_V1_AGENT_TOKENS, MISTRAL_V2_AGENT_TOKENS, MISTRAL_V3_AGENT_TOKENS,
    )

    tokenizer = Tokenizer.from_pretrained("cl100k_base")
    print(CL100K_AGENT_TOKENS.THINK)      # 100282
    print(CL100K_AGENT_TOKENS.FUNCTION)   # 100292

    # For Llama 3
    tokenizer = Tokenizer.from_pretrained("llama3")
    print(LLAMA3_AGENT_TOKENS.THINK)      # 128305

    # For DeepSeek V3 (includes native tokens)
    tokenizer = Tokenizer.from_pretrained("deepseek_v3")
    print(DEEPSEEK_V3_AGENT_TOKENS.THINK_NATIVE)  # 128798 (native <think>)
    print(DEEPSEEK_V3_AGENT_TOKENS.USER_NATIVE)   # 128803 (native <｜User｜>)

    # For Mistral models (V1/V2/V3 have different base IDs)
    tokenizer = Tokenizer.from_pretrained("mistral_v1")
    print(MISTRAL_V1_AGENT_TOKENS.THINK)      # 32005
    tokenizer = Tokenizer.from_pretrained("mistral_v2")
    print(MISTRAL_V2_AGENT_TOKENS.THINK)      # 32773 (note: after control tokens at 3-9)
    tokenizer = Tokenizer.from_pretrained("mistral_v3")
    print(MISTRAL_V3_AGENT_TOKENS.THINK)      # 131077

    # Encode with special tokens
    tokens = tokenizer.encode_with_special("<|think|>reasoning<|/think|>")
    assert LLAMA3_AGENT_TOKENS.THINK in tokens

    # Token categories:
    # - Conversation: SYSTEM, USER, ASSISTANT, IM_START, IM_END
    # - Thinking: THINK, THINK_END (Chain-of-Thought)
    # - ReAct: PLAN, STEP, ACT, OBSERVE (+ _END variants)
    # - Tools: FUNCTION, RESULT, ERROR (+ _END variants)
    # - Code: CODE, OUTPUT, LANG (+ _END variants)
    # - RAG: CONTEXT, QUOTE, CITE, SOURCE (+ _END variants)
    # - Memory: MEMORY, RECALL (+ _END variants)
    # - Control: PAD, STOP, SEP
    # - Multimodal: IMAGE, AUDIO, VIDEO (+ _END variants)
    # - Document: TITLE, SECTION, SUMMARY (+ _END variants)
"""

from ._core import (
    Tokenizer,
    StreamingDecoder,
    ByteLevelStreamingDecoder,
    CL100K_BASE_PATTERN,
    O200K_BASE_PATTERN,
    LLAMA3_PATTERN,
    CL100K_AGENT_TOKENS,
    O200K_AGENT_TOKENS,
    LLAMA3_AGENT_TOKENS,
    DEEPSEEK_V3_AGENT_TOKENS,
    MISTRAL_V1_AGENT_TOKENS,
    MISTRAL_V2_AGENT_TOKENS,
    MISTRAL_V3_AGENT_TOKENS,
)

__all__ = [
    "Tokenizer",
    "StreamingDecoder",
    "ByteLevelStreamingDecoder",
    "CL100K_BASE_PATTERN",
    "O200K_BASE_PATTERN",
    "LLAMA3_PATTERN",
    "CL100K_AGENT_TOKENS",
    "O200K_AGENT_TOKENS",
    "LLAMA3_AGENT_TOKENS",
    "DEEPSEEK_V3_AGENT_TOKENS",
    "MISTRAL_V1_AGENT_TOKENS",
    "MISTRAL_V2_AGENT_TOKENS",
    "MISTRAL_V3_AGENT_TOKENS",
]
__version__ = "0.7.0"
