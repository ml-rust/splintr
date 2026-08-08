"""
Splintr - Fast Rust tokenizer (BPE + SentencePiece + WordPiece) with Python bindings

A high-performance tokenizer featuring:
- Regexr with JIT and SIMD (default, pure Rust)
- Optional PCRE2 with JIT (requires pcre2 feature)
- Rayon parallelism for multi-core encoding
- Linked-list BPE algorithm (avoids O(N^2) on pathological inputs)
- SentencePiece unigram with Viterbi maximum-score segmentation (true Unigram) and byte fallback
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
- qwen3/qwen/qwen2/qwen2.5/baichuan_m2: Qwen 2/3 (Baichuan-M2 ships it unchanged)
- glm4/glm/glm-4/glm4.5/glm-4.5: GLM-4/4.5
- gpt-oss/gpt_oss/o200k_harmony: OpenAI gpt-oss (o200k_base ranks + harmony tokens)
- kimi/kimi_k2/kimi_k3: Kimi (Moonshot AI); K2 and K3 share ranks, differ in markers
- mistral_v1: Mistral 7B v0.1/v0.2, Mixtral 8x7B
- mistral_v2: Mistral 7B v0.3, Mixtral 8x22B, Codestral
- mistral_v3: Mistral NeMo, Large 2, Pixtral (Tekken)
- whisper/whisper_v1/whisper_v2/whisper_v3: OpenAI Whisper multilingual (bundled;
  bare "whisper" -> v2).

from_pretrained returns an AnyTokenizer for every bundled vocabulary. It delegates
to the same core loader the Rust API uses, so a name produces the same ids on both
sides of the binding; .family names the backend it dispatched to ("BPE" for the
byte-level vocabularies, "Spm" for the SentencePiece ones, which are merged as
pieces since byte-level merging cannot build the word-boundary marker). Because
every loader turns added-token matching on, encode matches a special token spelled
out in the text -- use encode_ordinary to decline, encode_allowed_special for a
named subset.

Any other model: load its HuggingFace tokenizer.json with splintr.from_json(path).
It returns an AnyTokenizer, the universal loaded-tokenizer handle: it dispatches
internally to a BPE / Unigram / WordPiece / SPM backend (see .family) while
keeping the file's special-token policy and its declared decoder pipeline, so
decode matches HuggingFace. Use this for Whisper English-only checkpoints, BERT,
T5, Gemma, Qwen, etc.

Usage:
    from splintr import Tokenizer

    # Load pretrained model (uses regexr by default)
    tokenizer = Tokenizer.from_pretrained("cl100k_base")  # GPT-4
    tokenizer = Tokenizer.from_pretrained("llama3")       # Llama 3
    tokenizer = Tokenizer.from_pretrained("deepseek_v3")  # DeepSeek V3
    tokenizer = Tokenizer.from_pretrained("mistral_v1")   # Mistral 7B v0.1/v0.2
    tokenizer = Tokenizer.from_pretrained("mistral_v2")   # Mistral 7B v0.3
    tokenizer = Tokenizer.from_pretrained("mistral_v3")   # Mistral NeMo (Tekken)
    tokenizer = Tokenizer.from_pretrained("whisper_v3")   # Whisper large-v3 (multilingual)

    # Any other model: load its HuggingFace tokenizer.json directly
    from splintr import from_json
    tokenizer = from_json("path/to/tokenizer.json")  # BERT, T5, Gemma, Whisper.en, ...

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

    # Streaming decode (for LLM output). One decoder serves every vocabulary:
    # it is built from the tokenizer's own decode rules, so a ByteLevel
    # vocabulary (DeepSeek V3, GPT-2) needs no different call.
    decoder = tokenizer.streaming_decoder()
    for token_id in token_stream:
        if text := decoder.add_token(token_id):
            print(text, end="", flush=True)
    print(decoder.flush())

SentencePiece Unigram (for GGUF models):
    from splintr import SentencePieceTokenizer

    tokenizer = SentencePieceTokenizer(
        tokens=["<unk>", "<s>", "</s>", "▁Hello", "▁world"],
        scores=[0.0, 0.0, 0.0, -1.2, -1.5],
        eos_token_id=2,
        bos_token_id=1,
    )
    ids = tokenizer.encode("Hello world")
    text = tokenizer.decode(ids)

Agent Tokens:
    from splintr import (
        Tokenizer,
        CL100K_AGENT_TOKENS, O200K_AGENT_TOKENS,
        LLAMA3_AGENT_TOKENS, DEEPSEEK_V3_AGENT_TOKENS,
        QWEN3_AGENT_TOKENS, GLM4_AGENT_TOKENS, GPT_OSS_AGENT_TOKENS,
        KIMI_K2_AGENT_TOKENS, KIMI_K3_AGENT_TOKENS,
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

# Qwen and GLM already name some of the agent tokens; those constants carry the
# model's own id rather than a splintr-appended one.
print(QWEN3_AGENT_TOKENS.IM_START)    # 151644 (Qwen's own)
print(QWEN3_AGENT_TOKENS.THINK)       # 151674 (splintr's <|think|>)
print(GLM4_AGENT_TOKENS.SYSTEM)       # 151335 (GLM's own)

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
    SentencePieceTokenizer,
    SpmTokenizer,
    WordPieceTokenizer,
    AnyTokenizer,
    from_json,
    from_json_bytes,
    base_vocab_size,
    StreamingDecoder,
    CL100K_BASE_PATTERN,
    KIMI_PATTERN,
    QWEN2_PATTERN,
    O200K_BASE_PATTERN,
    LLAMA3_PATTERN,
    CL100K_AGENT_TOKENS,
    O200K_AGENT_TOKENS,
    LLAMA3_AGENT_TOKENS,
    DEEPSEEK_V3_AGENT_TOKENS,
    QWEN3_AGENT_TOKENS,
    GLM4_AGENT_TOKENS,
    GPT_OSS_AGENT_TOKENS,
    KIMI_K2_AGENT_TOKENS,
    KIMI_K3_AGENT_TOKENS,
    MISTRAL_V1_AGENT_TOKENS,
    MISTRAL_V2_AGENT_TOKENS,
    MISTRAL_V3_AGENT_TOKENS,
)

__all__ = [
    "Tokenizer",
    "SentencePieceTokenizer",
    "SpmTokenizer",
    "WordPieceTokenizer",
    "AnyTokenizer",
    "from_json",
    "from_json_bytes",
    "base_vocab_size",
    "StreamingDecoder",
    "CL100K_BASE_PATTERN",
    "KIMI_PATTERN",
    "QWEN2_PATTERN",
    "O200K_BASE_PATTERN",
    "LLAMA3_PATTERN",
    "CL100K_AGENT_TOKENS",
    "O200K_AGENT_TOKENS",
    "LLAMA3_AGENT_TOKENS",
    "DEEPSEEK_V3_AGENT_TOKENS",
    "QWEN3_AGENT_TOKENS",
    "GLM4_AGENT_TOKENS",
    "GPT_OSS_AGENT_TOKENS",
    "KIMI_K2_AGENT_TOKENS",
    "KIMI_K3_AGENT_TOKENS",
    "MISTRAL_V1_AGENT_TOKENS",
    "MISTRAL_V2_AGENT_TOKENS",
    "MISTRAL_V3_AGENT_TOKENS",
]
__version__ = "0.15.0"
