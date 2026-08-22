"""Type stubs for the compiled extension module `splintr._core`.

`splintr/__init__.py` re-exports everything here, so this file is where the
Python API is described and `__init__.pyi` is a re-export list. Keeping the two
apart is what makes `from splintr import _core` -- which `python/tests/conftest.py`
does, to read `HAS_PCRE2` -- check rather than fail on a missing submodule.

The agent-token classes below the sentinel are generated. See the note there.
"""

from collections.abc import Sequence
from typing import ClassVar, Final, NoReturn, Self, final

# Emitted alongside the agent-token classes below: the vocabularies are half of
# it, so a vocabulary added without regenerating would leave this list short.
# BEGIN GENERATED: stub-all

__all__ = [
    "AnyTokenizer",
    "CL100K_AGENT_TOKENS",
    "CL100K_BASE_PATTERN",
    "CODELLAMA_AGENT_TOKENS",
    "DEEPSEEK_V3_AGENT_TOKENS",
    "GEMMA2_AGENT_TOKENS",
    "GEMMA3_AGENT_TOKENS",
    "GEMMA4_AGENT_TOKENS",
    "GLM4_AGENT_TOKENS",
    "GPT2_PATTERN",
    "GPT_OSS_AGENT_TOKENS",
    "HAS_PCRE2",
    "KIMI_K2_AGENT_TOKENS",
    "KIMI_K3_AGENT_TOKENS",
    "KIMI_PATTERN",
    "LLAMA2_AGENT_TOKENS",
    "LLAMA3_AGENT_TOKENS",
    "LLAMA3_PATTERN",
    "MISTRAL_V1_AGENT_TOKENS",
    "MISTRAL_V2_AGENT_TOKENS",
    "MISTRAL_V3_AGENT_TOKENS",
    "MISTRAL_V3_PATTERN",
    "MODERNBERT_AGENT_TOKENS",
    "O200K_AGENT_TOKENS",
    "O200K_BASE_PATTERN",
    "OLMO2_AGENT_TOKENS",
    "PHI4_AGENT_TOKENS",
    "QWEN2_PATTERN",
    "QWEN3_AGENT_TOKENS",
    "SentencePieceTokenizer",
    "SpmTokenizer",
    "StreamingDecoder",
    "Tokenizer",
    "WordPieceTokenizer",
    "base_vocab_size",
    "from_json",
    "from_json_bytes",
]

# END GENERATED: stub-all

CL100K_BASE_PATTERN: Final[str]
KIMI_PATTERN: Final[str]
QWEN2_PATTERN: Final[str]
O200K_BASE_PATTERN: Final[str]
LLAMA3_PATTERN: Final[str]
MISTRAL_V3_PATTERN: Final[str]
GPT2_PATTERN: Final[str]

# Whether the optional PCRE2 backend was compiled into this build.
HAS_PCRE2: Final[bool]


@final
class StreamingDecoder:
    # Built by `Tokenizer.streaming_decoder()`; PyO3 registers no constructor.
    def __new__(cls) -> NoReturn: ...
    def add_token(self, token_id: int) -> str | None: ...
    def add_tokens(self, token_ids: Sequence[int]) -> str | None: ...
    def flush(self) -> str: ...
    def reset(self) -> None: ...
    @property
    def has_pending(self) -> bool: ...
    @property
    def pending_bytes(self) -> int: ...


@final
class Tokenizer:
    def __new__(cls, vocab_path: str, pattern: str, special_tokens: dict[str, int] | None = None) -> Self: ...
    @staticmethod
    def from_pretrained(name: str) -> AnyTokenizer: ...
    @staticmethod
    def from_bytes(vocab_data: bytes, pattern: str, special_tokens: dict[str, int] | None = None) -> Tokenizer: ...
    def pcre2(self, use_pcre2: bool = True) -> Tokenizer: ...
    def jit(self, use_jit: bool = True) -> Tokenizer: ...
    def encode(self, text: str) -> list[int]: ...
    def encode_raw(self, text: str) -> list[int]: ...
    def encode_rayon(self, text: str) -> list[int]: ...
    def encode_with_special(self, text: str) -> list[int]: ...
    def encode_ordinary(self, text: str) -> list[int]: ...
    def encode_allowed_special(self, text: str, allowed_special: Sequence[str]) -> list[int]: ...
    def decode(self, tokens: Sequence[int]) -> str: ...
    def decode_with_special(self, tokens: Sequence[int]) -> str: ...
    def decode_bytes(self, tokens: Sequence[int]) -> bytes: ...
    def decode_lossy(self, tokens: Sequence[int]) -> str: ...
    def decode_token_bytes(self, id: int) -> bytes: ...
    def decode_token(self, id: int) -> str: ...
    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]: ...
    def encode_flat(self, text: str) -> bytes: ...
    def encode_batch_flat(self, texts: Sequence[str]) -> tuple[bytes, bytes]: ...
    def encode_batch_with_special(self, texts: Sequence[str]) -> list[list[int]]: ...
    def decode_batch(self, token_lists: Sequence[Sequence[int]]) -> list[str]: ...
    def decode_batch_lossy(self, token_lists: Sequence[Sequence[int]]) -> list[str]: ...
    @property
    def vocab_size(self) -> int: ...
    def streaming_decoder(self) -> StreamingDecoder: ...
    def streaming_decoder_with_special(self) -> StreamingDecoder: ...
    def clear_cache(self) -> None: ...
    @property
    def cache_len(self) -> int: ...


@final
class SentencePieceTokenizer:
    def __new__(cls, tokens: Sequence[str], scores: Sequence[float], eos_token_id: int, bos_token_id: int | None = None) -> Self: ...
    def encode(self, text: str) -> list[int]: ...
    def encode_raw(self, text: str) -> list[int]: ...
    def encode_with_special(self, text: str) -> list[int]: ...
    def encode_ordinary(self, text: str) -> list[int]: ...
    def encode_allowed_special(self, text: str, allowed_special: Sequence[str]) -> list[int]: ...
    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]: ...
    def decode(self, ids: Sequence[int]) -> str: ...
    def decode_with_special(self, ids: Sequence[int]) -> str: ...
    def decode_lossy(self, ids: Sequence[int]) -> str: ...
    def decode_token_bytes(self, id: int) -> bytes: ...
    def decode_token(self, id: int) -> str: ...
    @property
    def vocab_size(self) -> int: ...
    def is_eos(self, token_id: int) -> bool: ...
    @property
    def eos_token_id(self) -> int: ...
    @property
    def bos_token_id(self) -> int | None: ...
    def streaming_decoder(self) -> StreamingDecoder: ...
    def streaming_decoder_with_special(self) -> StreamingDecoder: ...


@final
class SpmTokenizer:
    def __new__(cls, tokens: Sequence[str], scores: Sequence[float], bos_token_id: int | None = None, eos_token_id: int | None = None) -> Self: ...
    def encode(self, text: str) -> list[int]: ...
    def encode_raw(self, text: str) -> list[int]: ...
    def encode_with_special(self, text: str) -> list[int]: ...
    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]: ...
    def encode_ordinary(self, text: str) -> list[int]: ...
    def encode_allowed_special(self, text: str, allowed_special: Sequence[str]) -> list[int]: ...
    def decode(self, ids: Sequence[int]) -> str: ...
    def decode_with_special(self, ids: Sequence[int]) -> str: ...
    def decode_token_bytes(self, id: int) -> bytes: ...
    def decode_token(self, id: int) -> str: ...
    @property
    def vocab_size(self) -> int: ...
    @property
    def eos_token_id(self) -> int | None: ...
    @property
    def bos_token_id(self) -> int | None: ...
    def streaming_decoder(self) -> StreamingDecoder: ...
    def streaming_decoder_with_special(self) -> StreamingDecoder: ...


@final
class WordPieceTokenizer:
    def __new__(cls, vocab: Sequence[str], unk_token_id: int, max_word_len: int = 100, do_lower_case: bool = False, strip_accents: bool | None = None) -> Self: ...
    def encode(self, text: str) -> list[int]: ...
    def encode_raw(self, text: str) -> list[int]: ...
    def encode_with_special(self, text: str) -> list[int]: ...
    def encode_ordinary(self, text: str) -> list[int]: ...
    def encode_allowed_special(self, text: str, allowed_special: Sequence[str]) -> list[int]: ...
    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]: ...
    def decode(self, ids: Sequence[int]) -> str: ...
    def decode_with_special(self, ids: Sequence[int]) -> str: ...
    def decode_token_bytes(self, id: int) -> bytes: ...
    def decode_token(self, id: int) -> str: ...
    # A method, not a property -- unlike every other tokenizer here.
    def vocab_size(self) -> int: ...
    @property
    def unk_token_id(self) -> int: ...
    @property
    def cls_token_id(self) -> int | None: ...
    @property
    def sep_token_id(self) -> int | None: ...
    @property
    def pad_token_id(self) -> int | None: ...
    def streaming_decoder(self) -> StreamingDecoder: ...
    def streaming_decoder_with_special(self) -> StreamingDecoder: ...


@final
class AnyTokenizer:
    # Returned by `from_pretrained` / `from_json`; PyO3 registers no constructor.
    def __new__(cls) -> NoReturn: ...
    def encode(self, text: str) -> list[int]: ...
    def encode_raw(self, text: str) -> list[int]: ...
    def encode_with_special(self, text: str) -> list[int]: ...
    def encode_ordinary(self, text: str) -> list[int]: ...
    def encode_allowed_special(self, text: str, allowed_special: Sequence[str]) -> list[int]: ...
    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]: ...
    def encode_flat(self, text: str) -> bytes: ...
    def encode_batch_flat(self, texts: Sequence[str]) -> tuple[bytes, bytes]: ...
    def encode_batch_with_special(self, texts: Sequence[str]) -> list[list[int]]: ...
    def encode_rayon(self, text: str) -> list[int]: ...
    def decode(self, ids: Sequence[int]) -> str: ...
    def decode_with_special(self, ids: Sequence[int]) -> str: ...
    def decode_batch(self, token_lists: Sequence[Sequence[int]]) -> list[str]: ...
    def decode_bytes(self, tokens: Sequence[int]) -> bytes: ...
    def decode_lossy(self, tokens: Sequence[int]) -> str: ...
    def decode_batch_lossy(self, token_lists: Sequence[Sequence[int]]) -> list[str]: ...
    def decode_token_bytes(self, id: int) -> bytes: ...
    def decode_token(self, id: int) -> str: ...
    def pcre2(self, use_pcre2: bool = True) -> AnyTokenizer: ...
    def jit(self, use_jit: bool = True) -> AnyTokenizer: ...
    def streaming_decoder(self) -> StreamingDecoder: ...
    def streaming_decoder_with_special(self) -> StreamingDecoder: ...
    def clear_cache(self) -> None: ...
    @property
    def cache_len(self) -> int: ...
    @property
    def vocab_size(self) -> int: ...
    def is_eos(self, token_id: int) -> bool: ...
    @property
    def eos_token_id(self) -> int | None: ...
    def special_token_id(self, name: str) -> int | None: ...
    def special_tokens(self) -> dict[str, int]: ...
    @property
    def family(self) -> str: ...


def from_json(path: str) -> AnyTokenizer: ...
def from_json_bytes(data: bytes) -> AnyTokenizer: ...
def base_vocab_size(name: str) -> int: ...


# The agent-token classes are emitted from the same table as the Rust modules
# and the PyO3 classes, by `scripts/generate_agent_tokens.py --update-stub`.
# Edit that script, not this region -- CI diffs it and fails on drift.
# BEGIN GENERATED: agent-tokens

@final
class CL100K_AGENT_TOKENS:
    """cl100k_base (GPT-4, GPT-3.5-turbo) agent token ids (100277-100330)."""

    # Conversation & Roles (100277-100281)
    SYSTEM: ClassVar[int]
    USER: ClassVar[int]
    ASSISTANT: ClassVar[int]
    IM_START: ClassVar[int]
    IM_END: ClassVar[int]

    # Reasoning/Thinking (100282-100283)
    THINK: ClassVar[int]
    THINK_END: ClassVar[int]

    # ReAct Agent Loop (100284-100291)
    PLAN: ClassVar[int]
    PLAN_END: ClassVar[int]
    STEP: ClassVar[int]
    STEP_END: ClassVar[int]
    ACT: ClassVar[int]
    ACT_END: ClassVar[int]
    OBSERVE: ClassVar[int]
    OBSERVE_END: ClassVar[int]

    # Tool/Function Calling (100292-100297)
    FUNCTION: ClassVar[int]
    FUNCTION_END: ClassVar[int]
    RESULT: ClassVar[int]
    RESULT_END: ClassVar[int]
    ERROR: ClassVar[int]
    ERROR_END: ClassVar[int]

    # Code Execution (100298-100303)
    CODE: ClassVar[int]
    CODE_END: ClassVar[int]
    OUTPUT: ClassVar[int]
    OUTPUT_END: ClassVar[int]
    LANG: ClassVar[int]
    LANG_END: ClassVar[int]

    # RAG & Citations (100304-100311)
    CONTEXT: ClassVar[int]
    CONTEXT_END: ClassVar[int]
    QUOTE: ClassVar[int]
    QUOTE_END: ClassVar[int]
    CITE: ClassVar[int]
    CITE_END: ClassVar[int]
    SOURCE: ClassVar[int]
    SOURCE_END: ClassVar[int]

    # Memory/State Management (100312-100315)
    MEMORY: ClassVar[int]
    MEMORY_END: ClassVar[int]
    RECALL: ClassVar[int]
    RECALL_END: ClassVar[int]

    # Control Tokens (100316-100318)
    PAD: ClassVar[int]
    STOP: ClassVar[int]
    SEP: ClassVar[int]

    # Multimodal Placeholders (100319-100324)
    IMAGE: ClassVar[int]
    IMAGE_END: ClassVar[int]
    AUDIO: ClassVar[int]
    AUDIO_END: ClassVar[int]
    VIDEO: ClassVar[int]
    VIDEO_END: ClassVar[int]

    # Document Structure (100325-100330)
    TITLE: ClassVar[int]
    TITLE_END: ClassVar[int]
    SECTION: ClassVar[int]
    SECTION_END: ClassVar[int]
    SUMMARY: ClassVar[int]
    SUMMARY_END: ClassVar[int]


@final
class O200K_AGENT_TOKENS:
    """o200k_base (GPT-4o) agent token ids (200019-200072)."""

    # Conversation & Roles (200019-200023)
    SYSTEM: ClassVar[int]
    USER: ClassVar[int]
    ASSISTANT: ClassVar[int]
    IM_START: ClassVar[int]
    IM_END: ClassVar[int]

    # Reasoning/Thinking (200024-200025)
    THINK: ClassVar[int]
    THINK_END: ClassVar[int]

    # ReAct Agent Loop (200026-200033)
    PLAN: ClassVar[int]
    PLAN_END: ClassVar[int]
    STEP: ClassVar[int]
    STEP_END: ClassVar[int]
    ACT: ClassVar[int]
    ACT_END: ClassVar[int]
    OBSERVE: ClassVar[int]
    OBSERVE_END: ClassVar[int]

    # Tool/Function Calling (200034-200039)
    FUNCTION: ClassVar[int]
    FUNCTION_END: ClassVar[int]
    RESULT: ClassVar[int]
    RESULT_END: ClassVar[int]
    ERROR: ClassVar[int]
    ERROR_END: ClassVar[int]

    # Code Execution (200040-200045)
    CODE: ClassVar[int]
    CODE_END: ClassVar[int]
    OUTPUT: ClassVar[int]
    OUTPUT_END: ClassVar[int]
    LANG: ClassVar[int]
    LANG_END: ClassVar[int]

    # RAG & Citations (200046-200053)
    CONTEXT: ClassVar[int]
    CONTEXT_END: ClassVar[int]
    QUOTE: ClassVar[int]
    QUOTE_END: ClassVar[int]
    CITE: ClassVar[int]
    CITE_END: ClassVar[int]
    SOURCE: ClassVar[int]
    SOURCE_END: ClassVar[int]

    # Memory/State Management (200054-200057)
    MEMORY: ClassVar[int]
    MEMORY_END: ClassVar[int]
    RECALL: ClassVar[int]
    RECALL_END: ClassVar[int]

    # Control Tokens (200058-200060)
    PAD: ClassVar[int]
    STOP: ClassVar[int]
    SEP: ClassVar[int]

    # Multimodal Placeholders (200061-200066)
    IMAGE: ClassVar[int]
    IMAGE_END: ClassVar[int]
    AUDIO: ClassVar[int]
    AUDIO_END: ClassVar[int]
    VIDEO: ClassVar[int]
    VIDEO_END: ClassVar[int]

    # Document Structure (200067-200072)
    TITLE: ClassVar[int]
    TITLE_END: ClassVar[int]
    SECTION: ClassVar[int]
    SECTION_END: ClassVar[int]
    SUMMARY: ClassVar[int]
    SUMMARY_END: ClassVar[int]


@final
class LLAMA3_AGENT_TOKENS:
    """Llama 3 family agent token ids (128300-128353)."""

    # Model-specific native tokens
    BEGIN_OF_TEXT: ClassVar[int]
    END_OF_TEXT: ClassVar[int]
    FINETUNE_RIGHT_PAD_ID: ClassVar[int]
    STEP_ID: ClassVar[int]
    START_HEADER_ID: ClassVar[int]
    END_HEADER_ID: ClassVar[int]
    EOM_ID: ClassVar[int]
    EOT_ID: ClassVar[int]
    PYTHON_TAG: ClassVar[int]
    IMAGE: ClassVar[int]
    IMAGE_END: ClassVar[int]
    AUDIO: ClassVar[int]
    AUDIO_END: ClassVar[int]
    VIDEO: ClassVar[int]
    VIDEO_END: ClassVar[int]

    # Conversation & Roles (128300-128304)
    SYSTEM: ClassVar[int]
    USER: ClassVar[int]
    ASSISTANT: ClassVar[int]
    IM_START: ClassVar[int]
    IM_END: ClassVar[int]

    # Reasoning/Thinking (128305-128306)
    THINK: ClassVar[int]
    THINK_END: ClassVar[int]

    # ReAct Agent Loop (128307-128314)
    PLAN: ClassVar[int]
    PLAN_END: ClassVar[int]
    STEP: ClassVar[int]
    STEP_END: ClassVar[int]
    ACT: ClassVar[int]
    ACT_END: ClassVar[int]
    OBSERVE: ClassVar[int]
    OBSERVE_END: ClassVar[int]

    # Tool/Function Calling (128315-128320)
    FUNCTION: ClassVar[int]
    FUNCTION_END: ClassVar[int]
    RESULT: ClassVar[int]
    RESULT_END: ClassVar[int]
    ERROR: ClassVar[int]
    ERROR_END: ClassVar[int]

    # Code Execution (128321-128326)
    CODE: ClassVar[int]
    CODE_END: ClassVar[int]
    OUTPUT: ClassVar[int]
    OUTPUT_END: ClassVar[int]
    LANG: ClassVar[int]
    LANG_END: ClassVar[int]

    # RAG & Citations (128327-128334)
    CONTEXT: ClassVar[int]
    CONTEXT_END: ClassVar[int]
    QUOTE: ClassVar[int]
    QUOTE_END: ClassVar[int]
    CITE: ClassVar[int]
    CITE_END: ClassVar[int]
    SOURCE: ClassVar[int]
    SOURCE_END: ClassVar[int]

    # Memory/State Management (128335-128338)
    MEMORY: ClassVar[int]
    MEMORY_END: ClassVar[int]
    RECALL: ClassVar[int]
    RECALL_END: ClassVar[int]

    # Control Tokens (128339-128341)
    PAD: ClassVar[int]
    STOP: ClassVar[int]
    SEP: ClassVar[int]

    # Document Structure (128348-128353)
    TITLE: ClassVar[int]
    TITLE_END: ClassVar[int]
    SECTION: ClassVar[int]
    SECTION_END: ClassVar[int]
    SUMMARY: ClassVar[int]
    SUMMARY_END: ClassVar[int]


@final
class DEEPSEEK_V3_AGENT_TOKENS:
    """DeepSeek V3/R1 agent token ids (128900-128953)."""

    # Model-specific native tokens
    BEGIN_OF_SENTENCE: ClassVar[int]
    END_OF_SENTENCE: ClassVar[int]
    PAD_NATIVE: ClassVar[int]
    THINK_NATIVE: ClassVar[int]
    THINK_END_NATIVE: ClassVar[int]
    FIM_HOLE: ClassVar[int]
    FIM_BEGIN: ClassVar[int]
    FIM_END: ClassVar[int]
    USER_NATIVE: ClassVar[int]
    ASSISTANT_NATIVE: ClassVar[int]
    EOT: ClassVar[int]
    TOOL_CALLS_BEGIN: ClassVar[int]
    TOOL_CALLS_END: ClassVar[int]
    TOOL_CALL_BEGIN: ClassVar[int]
    TOOL_CALL_END: ClassVar[int]
    TOOL_OUTPUTS_BEGIN: ClassVar[int]
    TOOL_OUTPUTS_END: ClassVar[int]
    TOOL_OUTPUT_BEGIN: ClassVar[int]
    TOOL_OUTPUT_END: ClassVar[int]
    TOOL_SEP: ClassVar[int]

    # Conversation & Roles (128900-128904)
    SYSTEM: ClassVar[int]
    USER: ClassVar[int]
    ASSISTANT: ClassVar[int]
    IM_START: ClassVar[int]
    IM_END: ClassVar[int]

    # Reasoning/Thinking (128905-128906)
    THINK: ClassVar[int]
    THINK_END: ClassVar[int]

    # ReAct Agent Loop (128907-128914)
    PLAN: ClassVar[int]
    PLAN_END: ClassVar[int]
    STEP: ClassVar[int]
    STEP_END: ClassVar[int]
    ACT: ClassVar[int]
    ACT_END: ClassVar[int]
    OBSERVE: ClassVar[int]
    OBSERVE_END: ClassVar[int]

    # Tool/Function Calling (128915-128920)
    FUNCTION: ClassVar[int]
    FUNCTION_END: ClassVar[int]
    RESULT: ClassVar[int]
    RESULT_END: ClassVar[int]
    ERROR: ClassVar[int]
    ERROR_END: ClassVar[int]

    # Code Execution (128921-128926)
    CODE: ClassVar[int]
    CODE_END: ClassVar[int]
    OUTPUT: ClassVar[int]
    OUTPUT_END: ClassVar[int]
    LANG: ClassVar[int]
    LANG_END: ClassVar[int]

    # RAG & Citations (128927-128934)
    CONTEXT: ClassVar[int]
    CONTEXT_END: ClassVar[int]
    QUOTE: ClassVar[int]
    QUOTE_END: ClassVar[int]
    CITE: ClassVar[int]
    CITE_END: ClassVar[int]
    SOURCE: ClassVar[int]
    SOURCE_END: ClassVar[int]

    # Memory/State Management (128935-128938)
    MEMORY: ClassVar[int]
    MEMORY_END: ClassVar[int]
    RECALL: ClassVar[int]
    RECALL_END: ClassVar[int]

    # Control Tokens (128939-128941)
    PAD: ClassVar[int]
    STOP: ClassVar[int]
    SEP: ClassVar[int]

    # Multimodal Placeholders (128942-128947)
    IMAGE: ClassVar[int]
    IMAGE_END: ClassVar[int]
    AUDIO: ClassVar[int]
    AUDIO_END: ClassVar[int]
    VIDEO: ClassVar[int]
    VIDEO_END: ClassVar[int]

    # Document Structure (128948-128953)
    TITLE: ClassVar[int]
    TITLE_END: ClassVar[int]
    SECTION: ClassVar[int]
    SECTION_END: ClassVar[int]
    SUMMARY: ClassVar[int]
    SUMMARY_END: ClassVar[int]


@final
class MISTRAL_V1_AGENT_TOKENS:
    """Mistral V1 (7B v0.1/v0.2, Mixtral 8x7B) agent token ids (32000-32053)."""

    # Conversation & Roles (32000-32004)
    SYSTEM: ClassVar[int]
    USER: ClassVar[int]
    ASSISTANT: ClassVar[int]
    IM_START: ClassVar[int]
    IM_END: ClassVar[int]

    # Reasoning/Thinking (32005-32006)
    THINK: ClassVar[int]
    THINK_END: ClassVar[int]

    # ReAct Agent Loop (32007-32014)
    PLAN: ClassVar[int]
    PLAN_END: ClassVar[int]
    STEP: ClassVar[int]
    STEP_END: ClassVar[int]
    ACT: ClassVar[int]
    ACT_END: ClassVar[int]
    OBSERVE: ClassVar[int]
    OBSERVE_END: ClassVar[int]

    # Tool/Function Calling (32015-32020)
    FUNCTION: ClassVar[int]
    FUNCTION_END: ClassVar[int]
    RESULT: ClassVar[int]
    RESULT_END: ClassVar[int]
    ERROR: ClassVar[int]
    ERROR_END: ClassVar[int]

    # Code Execution (32021-32026)
    CODE: ClassVar[int]
    CODE_END: ClassVar[int]
    OUTPUT: ClassVar[int]
    OUTPUT_END: ClassVar[int]
    LANG: ClassVar[int]
    LANG_END: ClassVar[int]

    # RAG & Citations (32027-32034)
    CONTEXT: ClassVar[int]
    CONTEXT_END: ClassVar[int]
    QUOTE: ClassVar[int]
    QUOTE_END: ClassVar[int]
    CITE: ClassVar[int]
    CITE_END: ClassVar[int]
    SOURCE: ClassVar[int]
    SOURCE_END: ClassVar[int]

    # Memory/State Management (32035-32038)
    MEMORY: ClassVar[int]
    MEMORY_END: ClassVar[int]
    RECALL: ClassVar[int]
    RECALL_END: ClassVar[int]

    # Control Tokens (32039-32041)
    PAD: ClassVar[int]
    STOP: ClassVar[int]
    SEP: ClassVar[int]

    # Multimodal Placeholders (32042-32047)
    IMAGE: ClassVar[int]
    IMAGE_END: ClassVar[int]
    AUDIO: ClassVar[int]
    AUDIO_END: ClassVar[int]
    VIDEO: ClassVar[int]
    VIDEO_END: ClassVar[int]

    # Document Structure (32048-32053)
    TITLE: ClassVar[int]
    TITLE_END: ClassVar[int]
    SECTION: ClassVar[int]
    SECTION_END: ClassVar[int]
    SUMMARY: ClassVar[int]
    SUMMARY_END: ClassVar[int]


@final
class MISTRAL_V2_AGENT_TOKENS:
    """Mistral V2 (7B v0.3, Mixtral 8x22B, Codestral) agent token ids (32768-32821)."""

    # Conversation & Roles (32768-32772)
    SYSTEM: ClassVar[int]
    USER: ClassVar[int]
    ASSISTANT: ClassVar[int]
    IM_START: ClassVar[int]
    IM_END: ClassVar[int]

    # Reasoning/Thinking (32773-32774)
    THINK: ClassVar[int]
    THINK_END: ClassVar[int]

    # ReAct Agent Loop (32775-32782)
    PLAN: ClassVar[int]
    PLAN_END: ClassVar[int]
    STEP: ClassVar[int]
    STEP_END: ClassVar[int]
    ACT: ClassVar[int]
    ACT_END: ClassVar[int]
    OBSERVE: ClassVar[int]
    OBSERVE_END: ClassVar[int]

    # Tool/Function Calling (32783-32788)
    FUNCTION: ClassVar[int]
    FUNCTION_END: ClassVar[int]
    RESULT: ClassVar[int]
    RESULT_END: ClassVar[int]
    ERROR: ClassVar[int]
    ERROR_END: ClassVar[int]

    # Code Execution (32789-32794)
    CODE: ClassVar[int]
    CODE_END: ClassVar[int]
    OUTPUT: ClassVar[int]
    OUTPUT_END: ClassVar[int]
    LANG: ClassVar[int]
    LANG_END: ClassVar[int]

    # RAG & Citations (32795-32802)
    CONTEXT: ClassVar[int]
    CONTEXT_END: ClassVar[int]
    QUOTE: ClassVar[int]
    QUOTE_END: ClassVar[int]
    CITE: ClassVar[int]
    CITE_END: ClassVar[int]
    SOURCE: ClassVar[int]
    SOURCE_END: ClassVar[int]

    # Memory/State Management (32803-32806)
    MEMORY: ClassVar[int]
    MEMORY_END: ClassVar[int]
    RECALL: ClassVar[int]
    RECALL_END: ClassVar[int]

    # Control Tokens (32807-32809)
    PAD: ClassVar[int]
    STOP: ClassVar[int]
    SEP: ClassVar[int]

    # Multimodal Placeholders (32810-32815)
    IMAGE: ClassVar[int]
    IMAGE_END: ClassVar[int]
    AUDIO: ClassVar[int]
    AUDIO_END: ClassVar[int]
    VIDEO: ClassVar[int]
    VIDEO_END: ClassVar[int]

    # Document Structure (32816-32821)
    TITLE: ClassVar[int]
    TITLE_END: ClassVar[int]
    SECTION: ClassVar[int]
    SECTION_END: ClassVar[int]
    SUMMARY: ClassVar[int]
    SUMMARY_END: ClassVar[int]


@final
class MISTRAL_V3_AGENT_TOKENS:
    """Mistral V3/Tekken (NeMo, Large 2, Pixtral) agent token ids (131072-131125)."""

    # Model-specific native tokens
    INST: ClassVar[int]
    INST_END: ClassVar[int]
    AVAILABLE_TOOLS: ClassVar[int]
    AVAILABLE_TOOLS_END: ClassVar[int]
    TOOL_RESULTS: ClassVar[int]
    TOOL_RESULTS_END: ClassVar[int]
    TOOL_CALLS: ClassVar[int]

    # Conversation & Roles (131072-131076)
    SYSTEM: ClassVar[int]
    USER: ClassVar[int]
    ASSISTANT: ClassVar[int]
    IM_START: ClassVar[int]
    IM_END: ClassVar[int]

    # Reasoning/Thinking (131077-131078)
    THINK: ClassVar[int]
    THINK_END: ClassVar[int]

    # ReAct Agent Loop (131079-131086)
    PLAN: ClassVar[int]
    PLAN_END: ClassVar[int]
    STEP: ClassVar[int]
    STEP_END: ClassVar[int]
    ACT: ClassVar[int]
    ACT_END: ClassVar[int]
    OBSERVE: ClassVar[int]
    OBSERVE_END: ClassVar[int]

    # Tool/Function Calling (131087-131092)
    FUNCTION: ClassVar[int]
    FUNCTION_END: ClassVar[int]
    RESULT: ClassVar[int]
    RESULT_END: ClassVar[int]
    ERROR: ClassVar[int]
    ERROR_END: ClassVar[int]

    # Code Execution (131093-131098)
    CODE: ClassVar[int]
    CODE_END: ClassVar[int]
    OUTPUT: ClassVar[int]
    OUTPUT_END: ClassVar[int]
    LANG: ClassVar[int]
    LANG_END: ClassVar[int]

    # RAG & Citations (131099-131106)
    CONTEXT: ClassVar[int]
    CONTEXT_END: ClassVar[int]
    QUOTE: ClassVar[int]
    QUOTE_END: ClassVar[int]
    CITE: ClassVar[int]
    CITE_END: ClassVar[int]
    SOURCE: ClassVar[int]
    SOURCE_END: ClassVar[int]

    # Memory/State Management (131107-131110)
    MEMORY: ClassVar[int]
    MEMORY_END: ClassVar[int]
    RECALL: ClassVar[int]
    RECALL_END: ClassVar[int]

    # Control Tokens (131111-131113)
    PAD: ClassVar[int]
    STOP: ClassVar[int]
    SEP: ClassVar[int]

    # Multimodal Placeholders (131114-131119)
    IMAGE: ClassVar[int]
    IMAGE_END: ClassVar[int]
    AUDIO: ClassVar[int]
    AUDIO_END: ClassVar[int]
    VIDEO: ClassVar[int]
    VIDEO_END: ClassVar[int]

    # Document Structure (131120-131125)
    TITLE: ClassVar[int]
    TITLE_END: ClassVar[int]
    SECTION: ClassVar[int]
    SECTION_END: ClassVar[int]
    SUMMARY: ClassVar[int]
    SUMMARY_END: ClassVar[int]


@final
class QWEN3_AGENT_TOKENS:
    """Qwen 2/3 (also Baichuan-M2) agent token ids (151669-151722)."""

    # Model-specific native tokens
    ENDOFTEXT: ClassVar[int]
    IM_START: ClassVar[int]
    IM_END: ClassVar[int]
    OBJECT_REF_START: ClassVar[int]
    OBJECT_REF_END: ClassVar[int]
    BOX_START: ClassVar[int]
    BOX_END: ClassVar[int]
    QUAD_START: ClassVar[int]
    QUAD_END: ClassVar[int]
    VISION_START: ClassVar[int]
    VISION_END: ClassVar[int]
    VISION_PAD: ClassVar[int]
    IMAGE_PAD: ClassVar[int]
    VIDEO_PAD: ClassVar[int]
    TOOL_CALL: ClassVar[int]
    TOOL_CALL_END: ClassVar[int]
    FIM_PREFIX: ClassVar[int]
    FIM_MIDDLE: ClassVar[int]
    FIM_SUFFIX: ClassVar[int]
    FIM_PAD: ClassVar[int]
    REPO_NAME: ClassVar[int]
    FILE_SEP: ClassVar[int]
    TOOL_RESPONSE: ClassVar[int]
    TOOL_RESPONSE_END: ClassVar[int]
    THINK_NATIVE: ClassVar[int]
    THINK_END_NATIVE: ClassVar[int]

    # Conversation & Roles (151669-151673)
    SYSTEM: ClassVar[int]
    USER: ClassVar[int]
    ASSISTANT: ClassVar[int]

    # Reasoning/Thinking (151674-151675)
    THINK: ClassVar[int]
    THINK_END: ClassVar[int]

    # ReAct Agent Loop (151676-151683)
    PLAN: ClassVar[int]
    PLAN_END: ClassVar[int]
    STEP: ClassVar[int]
    STEP_END: ClassVar[int]
    ACT: ClassVar[int]
    ACT_END: ClassVar[int]
    OBSERVE: ClassVar[int]
    OBSERVE_END: ClassVar[int]

    # Tool/Function Calling (151684-151689)
    FUNCTION: ClassVar[int]
    FUNCTION_END: ClassVar[int]
    RESULT: ClassVar[int]
    RESULT_END: ClassVar[int]
    ERROR: ClassVar[int]
    ERROR_END: ClassVar[int]

    # Code Execution (151690-151695)
    CODE: ClassVar[int]
    CODE_END: ClassVar[int]
    OUTPUT: ClassVar[int]
    OUTPUT_END: ClassVar[int]
    LANG: ClassVar[int]
    LANG_END: ClassVar[int]

    # RAG & Citations (151696-151703)
    CONTEXT: ClassVar[int]
    CONTEXT_END: ClassVar[int]
    QUOTE: ClassVar[int]
    QUOTE_END: ClassVar[int]
    CITE: ClassVar[int]
    CITE_END: ClassVar[int]
    SOURCE: ClassVar[int]
    SOURCE_END: ClassVar[int]

    # Memory/State Management (151704-151707)
    MEMORY: ClassVar[int]
    MEMORY_END: ClassVar[int]
    RECALL: ClassVar[int]
    RECALL_END: ClassVar[int]

    # Control Tokens (151708-151710)
    PAD: ClassVar[int]
    STOP: ClassVar[int]
    SEP: ClassVar[int]

    # Multimodal Placeholders (151711-151716)
    IMAGE: ClassVar[int]
    IMAGE_END: ClassVar[int]
    AUDIO: ClassVar[int]
    AUDIO_END: ClassVar[int]
    VIDEO: ClassVar[int]
    VIDEO_END: ClassVar[int]

    # Document Structure (151717-151722)
    TITLE: ClassVar[int]
    TITLE_END: ClassVar[int]
    SECTION: ClassVar[int]
    SECTION_END: ClassVar[int]
    SUMMARY: ClassVar[int]
    SUMMARY_END: ClassVar[int]


@final
class GLM4_AGENT_TOKENS:
    """GLM-4/4.5 agent token ids (151365-151418)."""

    # Model-specific native tokens
    ENDOFTEXT: ClassVar[int]
    MASK: ClassVar[int]
    GMASK: ClassVar[int]
    SMASK: ClassVar[int]
    SOP: ClassVar[int]
    EOP: ClassVar[int]
    SYSTEM: ClassVar[int]
    USER: ClassVar[int]
    ASSISTANT: ClassVar[int]
    OBSERVATION: ClassVar[int]
    BEGIN_OF_IMAGE: ClassVar[int]
    END_OF_IMAGE: ClassVar[int]
    BEGIN_OF_VIDEO: ClassVar[int]
    END_OF_VIDEO: ClassVar[int]
    BEGIN_OF_AUDIO: ClassVar[int]
    END_OF_AUDIO: ClassVar[int]
    BEGIN_OF_TRANSCRIPTION: ClassVar[int]
    END_OF_TRANSCRIPTION: ClassVar[int]
    CODE_PREFIX: ClassVar[int]
    CODE_MIDDLE: ClassVar[int]
    CODE_SUFFIX: ClassVar[int]
    THINK_NATIVE: ClassVar[int]
    THINK_END_NATIVE: ClassVar[int]
    TOOL_CALL: ClassVar[int]
    TOOL_CALL_END: ClassVar[int]
    TOOL_RESPONSE: ClassVar[int]
    TOOL_RESPONSE_END: ClassVar[int]
    ARG_KEY: ClassVar[int]
    ARG_KEY_END: ClassVar[int]
    ARG_VALUE: ClassVar[int]
    ARG_VALUE_END: ClassVar[int]
    NOTHINK: ClassVar[int]
    BEGIN_OF_BOX: ClassVar[int]
    END_OF_BOX: ClassVar[int]
    IMAGE: ClassVar[int]
    VIDEO: ClassVar[int]

    # Conversation & Roles (151365-151369)
    IM_START: ClassVar[int]
    IM_END: ClassVar[int]

    # Reasoning/Thinking (151370-151371)
    THINK: ClassVar[int]
    THINK_END: ClassVar[int]

    # ReAct Agent Loop (151372-151379)
    PLAN: ClassVar[int]
    PLAN_END: ClassVar[int]
    STEP: ClassVar[int]
    STEP_END: ClassVar[int]
    ACT: ClassVar[int]
    ACT_END: ClassVar[int]
    OBSERVE: ClassVar[int]
    OBSERVE_END: ClassVar[int]

    # Tool/Function Calling (151380-151385)
    FUNCTION: ClassVar[int]
    FUNCTION_END: ClassVar[int]
    RESULT: ClassVar[int]
    RESULT_END: ClassVar[int]
    ERROR: ClassVar[int]
    ERROR_END: ClassVar[int]

    # Code Execution (151386-151391)
    CODE: ClassVar[int]
    CODE_END: ClassVar[int]
    OUTPUT: ClassVar[int]
    OUTPUT_END: ClassVar[int]
    LANG: ClassVar[int]
    LANG_END: ClassVar[int]

    # RAG & Citations (151392-151399)
    CONTEXT: ClassVar[int]
    CONTEXT_END: ClassVar[int]
    QUOTE: ClassVar[int]
    QUOTE_END: ClassVar[int]
    CITE: ClassVar[int]
    CITE_END: ClassVar[int]
    SOURCE: ClassVar[int]
    SOURCE_END: ClassVar[int]

    # Memory/State Management (151400-151403)
    MEMORY: ClassVar[int]
    MEMORY_END: ClassVar[int]
    RECALL: ClassVar[int]
    RECALL_END: ClassVar[int]

    # Control Tokens (151404-151406)
    PAD: ClassVar[int]
    STOP: ClassVar[int]
    SEP: ClassVar[int]

    # Multimodal Placeholders (151407-151412)
    IMAGE_END: ClassVar[int]
    AUDIO: ClassVar[int]
    AUDIO_END: ClassVar[int]
    VIDEO_END: ClassVar[int]

    # Document Structure (151413-151418)
    TITLE: ClassVar[int]
    TITLE_END: ClassVar[int]
    SECTION: ClassVar[int]
    SECTION_END: ClassVar[int]
    SUMMARY: ClassVar[int]
    SUMMARY_END: ClassVar[int]


@final
class GPT_OSS_AGENT_TOKENS:
    """OpenAI gpt-oss agent token ids (200019-200072)."""

    # Model-specific native tokens
    STARTOFTEXT: ClassVar[int]
    ENDOFTEXT: ClassVar[int]
    RETURN: ClassVar[int]
    CONSTRAIN: ClassVar[int]
    CHANNEL: ClassVar[int]
    START: ClassVar[int]
    END: ClassVar[int]
    MESSAGE: ClassVar[int]
    CALL: ClassVar[int]
    ENDOFPROMPT: ClassVar[int]

    # Conversation & Roles (200019-200023)
    SYSTEM: ClassVar[int]
    USER: ClassVar[int]
    ASSISTANT: ClassVar[int]
    IM_START: ClassVar[int]
    IM_END: ClassVar[int]

    # Reasoning/Thinking (200024-200025)
    THINK: ClassVar[int]
    THINK_END: ClassVar[int]

    # ReAct Agent Loop (200026-200033)
    PLAN: ClassVar[int]
    PLAN_END: ClassVar[int]
    STEP: ClassVar[int]
    STEP_END: ClassVar[int]
    ACT: ClassVar[int]
    ACT_END: ClassVar[int]
    OBSERVE: ClassVar[int]
    OBSERVE_END: ClassVar[int]

    # Tool/Function Calling (200034-200039)
    FUNCTION: ClassVar[int]
    FUNCTION_END: ClassVar[int]
    RESULT: ClassVar[int]
    RESULT_END: ClassVar[int]
    ERROR: ClassVar[int]
    ERROR_END: ClassVar[int]

    # Code Execution (200040-200045)
    CODE: ClassVar[int]
    CODE_END: ClassVar[int]
    OUTPUT: ClassVar[int]
    OUTPUT_END: ClassVar[int]
    LANG: ClassVar[int]
    LANG_END: ClassVar[int]

    # RAG & Citations (200046-200053)
    CONTEXT: ClassVar[int]
    CONTEXT_END: ClassVar[int]
    QUOTE: ClassVar[int]
    QUOTE_END: ClassVar[int]
    CITE: ClassVar[int]
    CITE_END: ClassVar[int]
    SOURCE: ClassVar[int]
    SOURCE_END: ClassVar[int]

    # Memory/State Management (200054-200057)
    MEMORY: ClassVar[int]
    MEMORY_END: ClassVar[int]
    RECALL: ClassVar[int]
    RECALL_END: ClassVar[int]

    # Control Tokens (200058-200060)
    PAD: ClassVar[int]
    STOP: ClassVar[int]
    SEP: ClassVar[int]

    # Multimodal Placeholders (200061-200066)
    IMAGE: ClassVar[int]
    IMAGE_END: ClassVar[int]
    AUDIO: ClassVar[int]
    AUDIO_END: ClassVar[int]
    VIDEO: ClassVar[int]
    VIDEO_END: ClassVar[int]

    # Document Structure (200067-200072)
    TITLE: ClassVar[int]
    TITLE_END: ClassVar[int]
    SECTION: ClassVar[int]
    SECTION_END: ClassVar[int]
    SUMMARY: ClassVar[int]
    SUMMARY_END: ClassVar[int]


@final
class KIMI_K2_AGENT_TOKENS:
    """Kimi K2 (K2, K2.5, K2.6, K2.7, Kimi-Linear) agent token ids (163840-163893)."""

    # Model-specific native tokens
    BOS: ClassVar[int]
    EOS: ClassVar[int]
    IM_END: ClassVar[int]
    IM_USER: ClassVar[int]
    IM_ASSISTANT: ClassVar[int]
    START_HEADER_ID: ClassVar[int]
    END_HEADER_ID: ClassVar[int]
    EOT: ClassVar[int]
    IM_SYSTEM: ClassVar[int]
    TOOL_CALLS_SECTION_BEGIN: ClassVar[int]
    TOOL_CALLS_SECTION_END: ClassVar[int]
    TOOL_CALL_BEGIN: ClassVar[int]
    TOOL_CALL_ARGUMENT_BEGIN: ClassVar[int]
    TOOL_CALL_END: ClassVar[int]
    IM_MIDDLE: ClassVar[int]
    MEDIA_BEGIN: ClassVar[int]
    MEDIA_CONTENT: ClassVar[int]
    MEDIA_END: ClassVar[int]
    MEDIA_PAD: ClassVar[int]
    THINK_NATIVE: ClassVar[int]
    THINK_END_NATIVE: ClassVar[int]
    UNK: ClassVar[int]
    PAD_NATIVE: ClassVar[int]

    # Conversation & Roles (163840-163844)
    SYSTEM: ClassVar[int]
    USER: ClassVar[int]
    ASSISTANT: ClassVar[int]
    IM_START: ClassVar[int]

    # Reasoning/Thinking (163845-163846)
    THINK: ClassVar[int]
    THINK_END: ClassVar[int]

    # ReAct Agent Loop (163847-163854)
    PLAN: ClassVar[int]
    PLAN_END: ClassVar[int]
    STEP: ClassVar[int]
    STEP_END: ClassVar[int]
    ACT: ClassVar[int]
    ACT_END: ClassVar[int]
    OBSERVE: ClassVar[int]
    OBSERVE_END: ClassVar[int]

    # Tool/Function Calling (163855-163860)
    FUNCTION: ClassVar[int]
    FUNCTION_END: ClassVar[int]
    RESULT: ClassVar[int]
    RESULT_END: ClassVar[int]
    ERROR: ClassVar[int]
    ERROR_END: ClassVar[int]

    # Code Execution (163861-163866)
    CODE: ClassVar[int]
    CODE_END: ClassVar[int]
    OUTPUT: ClassVar[int]
    OUTPUT_END: ClassVar[int]
    LANG: ClassVar[int]
    LANG_END: ClassVar[int]

    # RAG & Citations (163867-163874)
    CONTEXT: ClassVar[int]
    CONTEXT_END: ClassVar[int]
    QUOTE: ClassVar[int]
    QUOTE_END: ClassVar[int]
    CITE: ClassVar[int]
    CITE_END: ClassVar[int]
    SOURCE: ClassVar[int]
    SOURCE_END: ClassVar[int]

    # Memory/State Management (163875-163878)
    MEMORY: ClassVar[int]
    MEMORY_END: ClassVar[int]
    RECALL: ClassVar[int]
    RECALL_END: ClassVar[int]

    # Control Tokens (163879-163881)
    PAD: ClassVar[int]
    STOP: ClassVar[int]
    SEP: ClassVar[int]

    # Multimodal Placeholders (163882-163887)
    IMAGE: ClassVar[int]
    IMAGE_END: ClassVar[int]
    AUDIO: ClassVar[int]
    AUDIO_END: ClassVar[int]
    VIDEO: ClassVar[int]
    VIDEO_END: ClassVar[int]

    # Document Structure (163888-163893)
    TITLE: ClassVar[int]
    TITLE_END: ClassVar[int]
    SECTION: ClassVar[int]
    SECTION_END: ClassVar[int]
    SUMMARY: ClassVar[int]
    SUMMARY_END: ClassVar[int]


@final
class KIMI_K3_AGENT_TOKENS:
    """Kimi K3 agent token ids (163840-163893)."""

    # Model-specific native tokens
    BOS: ClassVar[int]
    EOS: ClassVar[int]
    END_OF_MSG: ClassVar[int]
    OPEN: ClassVar[int]
    CLOSE: ClassVar[int]
    SEP: ClassVar[int]
    START_HEADER_ID: ClassVar[int]
    END_HEADER_ID: ClassVar[int]
    EOT: ClassVar[int]
    MEDIA_BEGIN: ClassVar[int]
    MEDIA_CONTENT: ClassVar[int]
    MEDIA_END: ClassVar[int]
    MEDIA_PAD: ClassVar[int]
    OSAGENT_MODE: ClassVar[int]
    UNK: ClassVar[int]
    PAD_NATIVE: ClassVar[int]

    # Conversation & Roles (163840-163844)
    SYSTEM: ClassVar[int]
    USER: ClassVar[int]
    ASSISTANT: ClassVar[int]
    IM_START: ClassVar[int]
    IM_END: ClassVar[int]

    # Reasoning/Thinking (163845-163846)
    THINK: ClassVar[int]
    THINK_END: ClassVar[int]

    # ReAct Agent Loop (163847-163854)
    PLAN: ClassVar[int]
    PLAN_END: ClassVar[int]
    STEP: ClassVar[int]
    STEP_END: ClassVar[int]
    ACT: ClassVar[int]
    ACT_END: ClassVar[int]
    OBSERVE: ClassVar[int]
    OBSERVE_END: ClassVar[int]

    # Tool/Function Calling (163855-163860)
    FUNCTION: ClassVar[int]
    FUNCTION_END: ClassVar[int]
    RESULT: ClassVar[int]
    RESULT_END: ClassVar[int]
    ERROR: ClassVar[int]
    ERROR_END: ClassVar[int]

    # Code Execution (163861-163866)
    CODE: ClassVar[int]
    CODE_END: ClassVar[int]
    OUTPUT: ClassVar[int]
    OUTPUT_END: ClassVar[int]
    LANG: ClassVar[int]
    LANG_END: ClassVar[int]

    # RAG & Citations (163867-163874)
    CONTEXT: ClassVar[int]
    CONTEXT_END: ClassVar[int]
    QUOTE: ClassVar[int]
    QUOTE_END: ClassVar[int]
    CITE: ClassVar[int]
    CITE_END: ClassVar[int]
    SOURCE: ClassVar[int]
    SOURCE_END: ClassVar[int]

    # Memory/State Management (163875-163878)
    MEMORY: ClassVar[int]
    MEMORY_END: ClassVar[int]
    RECALL: ClassVar[int]
    RECALL_END: ClassVar[int]

    # Control Tokens (163879-163881)
    PAD: ClassVar[int]
    STOP: ClassVar[int]

    # Multimodal Placeholders (163882-163887)
    IMAGE: ClassVar[int]
    IMAGE_END: ClassVar[int]
    AUDIO: ClassVar[int]
    AUDIO_END: ClassVar[int]
    VIDEO: ClassVar[int]
    VIDEO_END: ClassVar[int]

    # Document Structure (163888-163893)
    TITLE: ClassVar[int]
    TITLE_END: ClassVar[int]
    SECTION: ClassVar[int]
    SECTION_END: ClassVar[int]
    SUMMARY: ClassVar[int]
    SUMMARY_END: ClassVar[int]


@final
class PHI4_AGENT_TOKENS:
    """Microsoft Phi-4 agent token ids (100352-100405)."""

    # Model-specific native tokens
    ENDOFTEXT: ClassVar[int]
    FIM_PREFIX: ClassVar[int]
    FIM_MIDDLE: ClassVar[int]
    FIM_SUFFIX: ClassVar[int]
    IM_START: ClassVar[int]
    IM_END: ClassVar[int]
    IM_SEP: ClassVar[int]
    ENDOFPROMPT: ClassVar[int]

    # Conversation & Roles (100352-100356)
    SYSTEM: ClassVar[int]
    USER: ClassVar[int]
    ASSISTANT: ClassVar[int]

    # Reasoning/Thinking (100357-100358)
    THINK: ClassVar[int]
    THINK_END: ClassVar[int]

    # ReAct Agent Loop (100359-100366)
    PLAN: ClassVar[int]
    PLAN_END: ClassVar[int]
    STEP: ClassVar[int]
    STEP_END: ClassVar[int]
    ACT: ClassVar[int]
    ACT_END: ClassVar[int]
    OBSERVE: ClassVar[int]
    OBSERVE_END: ClassVar[int]

    # Tool/Function Calling (100367-100372)
    FUNCTION: ClassVar[int]
    FUNCTION_END: ClassVar[int]
    RESULT: ClassVar[int]
    RESULT_END: ClassVar[int]
    ERROR: ClassVar[int]
    ERROR_END: ClassVar[int]

    # Code Execution (100373-100378)
    CODE: ClassVar[int]
    CODE_END: ClassVar[int]
    OUTPUT: ClassVar[int]
    OUTPUT_END: ClassVar[int]
    LANG: ClassVar[int]
    LANG_END: ClassVar[int]

    # RAG & Citations (100379-100386)
    CONTEXT: ClassVar[int]
    CONTEXT_END: ClassVar[int]
    QUOTE: ClassVar[int]
    QUOTE_END: ClassVar[int]
    CITE: ClassVar[int]
    CITE_END: ClassVar[int]
    SOURCE: ClassVar[int]
    SOURCE_END: ClassVar[int]

    # Memory/State Management (100387-100390)
    MEMORY: ClassVar[int]
    MEMORY_END: ClassVar[int]
    RECALL: ClassVar[int]
    RECALL_END: ClassVar[int]

    # Control Tokens (100391-100393)
    PAD: ClassVar[int]
    STOP: ClassVar[int]
    SEP: ClassVar[int]

    # Multimodal Placeholders (100394-100399)
    IMAGE: ClassVar[int]
    IMAGE_END: ClassVar[int]
    AUDIO: ClassVar[int]
    AUDIO_END: ClassVar[int]
    VIDEO: ClassVar[int]
    VIDEO_END: ClassVar[int]

    # Document Structure (100400-100405)
    TITLE: ClassVar[int]
    TITLE_END: ClassVar[int]
    SECTION: ClassVar[int]
    SECTION_END: ClassVar[int]
    SUMMARY: ClassVar[int]
    SUMMARY_END: ClassVar[int]


@final
class OLMO2_AGENT_TOKENS:
    """AI2 OLMo-2 agent token ids (100278-100331)."""

    # Model-specific native tokens
    EXTRA_ID_0: ClassVar[int]
    ENDOFTEXT: ClassVar[int]
    FIM_PREFIX: ClassVar[int]
    FIM_MIDDLE: ClassVar[int]
    FIM_SUFFIX: ClassVar[int]
    PHONE_NUMBER: ClassVar[int]
    EMAIL_ADDRESS: ClassVar[int]
    IP_ADDRESS: ClassVar[int]
    IM_START: ClassVar[int]
    IM_END: ClassVar[int]
    EXTRA_ID_1: ClassVar[int]
    EXTRA_ID_2: ClassVar[int]
    EXTRA_ID_3: ClassVar[int]
    EXTRA_ID_4: ClassVar[int]
    EXTRA_ID_5: ClassVar[int]
    EXTRA_ID_6: ClassVar[int]
    EXTRA_ID_7: ClassVar[int]
    EXTRA_ID_8: ClassVar[int]
    EXTRA_ID_9: ClassVar[int]
    EXTRA_ID_10: ClassVar[int]
    ENDOFPROMPT: ClassVar[int]
    PAD: ClassVar[int]

    # Conversation & Roles (100278-100282)
    SYSTEM: ClassVar[int]
    USER: ClassVar[int]
    ASSISTANT: ClassVar[int]

    # Reasoning/Thinking (100283-100284)
    THINK: ClassVar[int]
    THINK_END: ClassVar[int]

    # ReAct Agent Loop (100285-100292)
    PLAN: ClassVar[int]
    PLAN_END: ClassVar[int]
    STEP: ClassVar[int]
    STEP_END: ClassVar[int]
    ACT: ClassVar[int]
    ACT_END: ClassVar[int]
    OBSERVE: ClassVar[int]
    OBSERVE_END: ClassVar[int]

    # Tool/Function Calling (100293-100298)
    FUNCTION: ClassVar[int]
    FUNCTION_END: ClassVar[int]
    RESULT: ClassVar[int]
    RESULT_END: ClassVar[int]
    ERROR: ClassVar[int]
    ERROR_END: ClassVar[int]

    # Code Execution (100299-100304)
    CODE: ClassVar[int]
    CODE_END: ClassVar[int]
    OUTPUT: ClassVar[int]
    OUTPUT_END: ClassVar[int]
    LANG: ClassVar[int]
    LANG_END: ClassVar[int]

    # RAG & Citations (100305-100312)
    CONTEXT: ClassVar[int]
    CONTEXT_END: ClassVar[int]
    QUOTE: ClassVar[int]
    QUOTE_END: ClassVar[int]
    CITE: ClassVar[int]
    CITE_END: ClassVar[int]
    SOURCE: ClassVar[int]
    SOURCE_END: ClassVar[int]

    # Memory/State Management (100313-100316)
    MEMORY: ClassVar[int]
    MEMORY_END: ClassVar[int]
    RECALL: ClassVar[int]
    RECALL_END: ClassVar[int]

    # Control Tokens (100317-100319)
    STOP: ClassVar[int]
    SEP: ClassVar[int]

    # Multimodal Placeholders (100320-100325)
    IMAGE: ClassVar[int]
    IMAGE_END: ClassVar[int]
    AUDIO: ClassVar[int]
    AUDIO_END: ClassVar[int]
    VIDEO: ClassVar[int]
    VIDEO_END: ClassVar[int]

    # Document Structure (100326-100331)
    TITLE: ClassVar[int]
    TITLE_END: ClassVar[int]
    SECTION: ClassVar[int]
    SECTION_END: ClassVar[int]
    SUMMARY: ClassVar[int]
    SUMMARY_END: ClassVar[int]


@final
class LLAMA2_AGENT_TOKENS:
    """Llama 2 (also TinyLlama, Vicuna) agent token ids (32000-32053)."""

    # Conversation & Roles (32000-32004)
    SYSTEM: ClassVar[int]
    USER: ClassVar[int]
    ASSISTANT: ClassVar[int]
    IM_START: ClassVar[int]
    IM_END: ClassVar[int]

    # Reasoning/Thinking (32005-32006)
    THINK: ClassVar[int]
    THINK_END: ClassVar[int]

    # ReAct Agent Loop (32007-32014)
    PLAN: ClassVar[int]
    PLAN_END: ClassVar[int]
    STEP: ClassVar[int]
    STEP_END: ClassVar[int]
    ACT: ClassVar[int]
    ACT_END: ClassVar[int]
    OBSERVE: ClassVar[int]
    OBSERVE_END: ClassVar[int]

    # Tool/Function Calling (32015-32020)
    FUNCTION: ClassVar[int]
    FUNCTION_END: ClassVar[int]
    RESULT: ClassVar[int]
    RESULT_END: ClassVar[int]
    ERROR: ClassVar[int]
    ERROR_END: ClassVar[int]

    # Code Execution (32021-32026)
    CODE: ClassVar[int]
    CODE_END: ClassVar[int]
    OUTPUT: ClassVar[int]
    OUTPUT_END: ClassVar[int]
    LANG: ClassVar[int]
    LANG_END: ClassVar[int]

    # RAG & Citations (32027-32034)
    CONTEXT: ClassVar[int]
    CONTEXT_END: ClassVar[int]
    QUOTE: ClassVar[int]
    QUOTE_END: ClassVar[int]
    CITE: ClassVar[int]
    CITE_END: ClassVar[int]
    SOURCE: ClassVar[int]
    SOURCE_END: ClassVar[int]

    # Memory/State Management (32035-32038)
    MEMORY: ClassVar[int]
    MEMORY_END: ClassVar[int]
    RECALL: ClassVar[int]
    RECALL_END: ClassVar[int]

    # Control Tokens (32039-32041)
    PAD: ClassVar[int]
    STOP: ClassVar[int]
    SEP: ClassVar[int]

    # Multimodal Placeholders (32042-32047)
    IMAGE: ClassVar[int]
    IMAGE_END: ClassVar[int]
    AUDIO: ClassVar[int]
    AUDIO_END: ClassVar[int]
    VIDEO: ClassVar[int]
    VIDEO_END: ClassVar[int]

    # Document Structure (32048-32053)
    TITLE: ClassVar[int]
    TITLE_END: ClassVar[int]
    SECTION: ClassVar[int]
    SECTION_END: ClassVar[int]
    SUMMARY: ClassVar[int]
    SUMMARY_END: ClassVar[int]


@final
class CODELLAMA_AGENT_TOKENS:
    """Code Llama agent token ids (32016-32069)."""

    # Conversation & Roles (32016-32020)
    SYSTEM: ClassVar[int]
    USER: ClassVar[int]
    ASSISTANT: ClassVar[int]
    IM_START: ClassVar[int]
    IM_END: ClassVar[int]

    # Reasoning/Thinking (32021-32022)
    THINK: ClassVar[int]
    THINK_END: ClassVar[int]

    # ReAct Agent Loop (32023-32030)
    PLAN: ClassVar[int]
    PLAN_END: ClassVar[int]
    STEP: ClassVar[int]
    STEP_END: ClassVar[int]
    ACT: ClassVar[int]
    ACT_END: ClassVar[int]
    OBSERVE: ClassVar[int]
    OBSERVE_END: ClassVar[int]

    # Tool/Function Calling (32031-32036)
    FUNCTION: ClassVar[int]
    FUNCTION_END: ClassVar[int]
    RESULT: ClassVar[int]
    RESULT_END: ClassVar[int]
    ERROR: ClassVar[int]
    ERROR_END: ClassVar[int]

    # Code Execution (32037-32042)
    CODE: ClassVar[int]
    CODE_END: ClassVar[int]
    OUTPUT: ClassVar[int]
    OUTPUT_END: ClassVar[int]
    LANG: ClassVar[int]
    LANG_END: ClassVar[int]

    # RAG & Citations (32043-32050)
    CONTEXT: ClassVar[int]
    CONTEXT_END: ClassVar[int]
    QUOTE: ClassVar[int]
    QUOTE_END: ClassVar[int]
    CITE: ClassVar[int]
    CITE_END: ClassVar[int]
    SOURCE: ClassVar[int]
    SOURCE_END: ClassVar[int]

    # Memory/State Management (32051-32054)
    MEMORY: ClassVar[int]
    MEMORY_END: ClassVar[int]
    RECALL: ClassVar[int]
    RECALL_END: ClassVar[int]

    # Control Tokens (32055-32057)
    PAD: ClassVar[int]
    STOP: ClassVar[int]
    SEP: ClassVar[int]

    # Multimodal Placeholders (32058-32063)
    IMAGE: ClassVar[int]
    IMAGE_END: ClassVar[int]
    AUDIO: ClassVar[int]
    AUDIO_END: ClassVar[int]
    VIDEO: ClassVar[int]
    VIDEO_END: ClassVar[int]

    # Document Structure (32064-32069)
    TITLE: ClassVar[int]
    TITLE_END: ClassVar[int]
    SECTION: ClassVar[int]
    SECTION_END: ClassVar[int]
    SUMMARY: ClassVar[int]
    SUMMARY_END: ClassVar[int]


@final
class MODERNBERT_AGENT_TOKENS:
    """Answer.AI ModernBERT agent token ids (50368-50421)."""

    # Conversation & Roles (50368-50372)
    SYSTEM: ClassVar[int]
    USER: ClassVar[int]
    ASSISTANT: ClassVar[int]
    IM_START: ClassVar[int]
    IM_END: ClassVar[int]

    # Reasoning/Thinking (50373-50374)
    THINK: ClassVar[int]
    THINK_END: ClassVar[int]

    # ReAct Agent Loop (50375-50382)
    PLAN: ClassVar[int]
    PLAN_END: ClassVar[int]
    STEP: ClassVar[int]
    STEP_END: ClassVar[int]
    ACT: ClassVar[int]
    ACT_END: ClassVar[int]
    OBSERVE: ClassVar[int]
    OBSERVE_END: ClassVar[int]

    # Tool/Function Calling (50383-50388)
    FUNCTION: ClassVar[int]
    FUNCTION_END: ClassVar[int]
    RESULT: ClassVar[int]
    RESULT_END: ClassVar[int]
    ERROR: ClassVar[int]
    ERROR_END: ClassVar[int]

    # Code Execution (50389-50394)
    CODE: ClassVar[int]
    CODE_END: ClassVar[int]
    OUTPUT: ClassVar[int]
    OUTPUT_END: ClassVar[int]
    LANG: ClassVar[int]
    LANG_END: ClassVar[int]

    # RAG & Citations (50395-50402)
    CONTEXT: ClassVar[int]
    CONTEXT_END: ClassVar[int]
    QUOTE: ClassVar[int]
    QUOTE_END: ClassVar[int]
    CITE: ClassVar[int]
    CITE_END: ClassVar[int]
    SOURCE: ClassVar[int]
    SOURCE_END: ClassVar[int]

    # Memory/State Management (50403-50406)
    MEMORY: ClassVar[int]
    MEMORY_END: ClassVar[int]
    RECALL: ClassVar[int]
    RECALL_END: ClassVar[int]

    # Control Tokens (50407-50409)
    PAD: ClassVar[int]
    STOP: ClassVar[int]
    SEP: ClassVar[int]

    # Multimodal Placeholders (50410-50415)
    IMAGE: ClassVar[int]
    IMAGE_END: ClassVar[int]
    AUDIO: ClassVar[int]
    AUDIO_END: ClassVar[int]
    VIDEO: ClassVar[int]
    VIDEO_END: ClassVar[int]

    # Document Structure (50416-50421)
    TITLE: ClassVar[int]
    TITLE_END: ClassVar[int]
    SECTION: ClassVar[int]
    SECTION_END: ClassVar[int]
    SUMMARY: ClassVar[int]
    SUMMARY_END: ClassVar[int]


@final
class GEMMA2_AGENT_TOKENS:
    """Google Gemma 2 agent token ids (256000-256053)."""

    # Conversation & Roles (256000-256004)
    SYSTEM: ClassVar[int]
    USER: ClassVar[int]
    ASSISTANT: ClassVar[int]
    IM_START: ClassVar[int]
    IM_END: ClassVar[int]

    # Reasoning/Thinking (256005-256006)
    THINK: ClassVar[int]
    THINK_END: ClassVar[int]

    # ReAct Agent Loop (256007-256014)
    PLAN: ClassVar[int]
    PLAN_END: ClassVar[int]
    STEP: ClassVar[int]
    STEP_END: ClassVar[int]
    ACT: ClassVar[int]
    ACT_END: ClassVar[int]
    OBSERVE: ClassVar[int]
    OBSERVE_END: ClassVar[int]

    # Tool/Function Calling (256015-256020)
    FUNCTION: ClassVar[int]
    FUNCTION_END: ClassVar[int]
    RESULT: ClassVar[int]
    RESULT_END: ClassVar[int]
    ERROR: ClassVar[int]
    ERROR_END: ClassVar[int]

    # Code Execution (256021-256026)
    CODE: ClassVar[int]
    CODE_END: ClassVar[int]
    OUTPUT: ClassVar[int]
    OUTPUT_END: ClassVar[int]
    LANG: ClassVar[int]
    LANG_END: ClassVar[int]

    # RAG & Citations (256027-256034)
    CONTEXT: ClassVar[int]
    CONTEXT_END: ClassVar[int]
    QUOTE: ClassVar[int]
    QUOTE_END: ClassVar[int]
    CITE: ClassVar[int]
    CITE_END: ClassVar[int]
    SOURCE: ClassVar[int]
    SOURCE_END: ClassVar[int]

    # Memory/State Management (256035-256038)
    MEMORY: ClassVar[int]
    MEMORY_END: ClassVar[int]
    RECALL: ClassVar[int]
    RECALL_END: ClassVar[int]

    # Control Tokens (256039-256041)
    PAD: ClassVar[int]
    STOP: ClassVar[int]
    SEP: ClassVar[int]

    # Multimodal Placeholders (256042-256047)
    IMAGE: ClassVar[int]
    IMAGE_END: ClassVar[int]
    AUDIO: ClassVar[int]
    AUDIO_END: ClassVar[int]
    VIDEO: ClassVar[int]
    VIDEO_END: ClassVar[int]

    # Document Structure (256048-256053)
    TITLE: ClassVar[int]
    TITLE_END: ClassVar[int]
    SECTION: ClassVar[int]
    SECTION_END: ClassVar[int]
    SUMMARY: ClassVar[int]
    SUMMARY_END: ClassVar[int]


@final
class GEMMA3_AGENT_TOKENS:
    """Google Gemma 3 (also EmbeddingGemma) agent token ids (262144-262197)."""

    # Conversation & Roles (262144-262148)
    SYSTEM: ClassVar[int]
    USER: ClassVar[int]
    ASSISTANT: ClassVar[int]
    IM_START: ClassVar[int]
    IM_END: ClassVar[int]

    # Reasoning/Thinking (262149-262150)
    THINK: ClassVar[int]
    THINK_END: ClassVar[int]

    # ReAct Agent Loop (262151-262158)
    PLAN: ClassVar[int]
    PLAN_END: ClassVar[int]
    STEP: ClassVar[int]
    STEP_END: ClassVar[int]
    ACT: ClassVar[int]
    ACT_END: ClassVar[int]
    OBSERVE: ClassVar[int]
    OBSERVE_END: ClassVar[int]

    # Tool/Function Calling (262159-262164)
    FUNCTION: ClassVar[int]
    FUNCTION_END: ClassVar[int]
    RESULT: ClassVar[int]
    RESULT_END: ClassVar[int]
    ERROR: ClassVar[int]
    ERROR_END: ClassVar[int]

    # Code Execution (262165-262170)
    CODE: ClassVar[int]
    CODE_END: ClassVar[int]
    OUTPUT: ClassVar[int]
    OUTPUT_END: ClassVar[int]
    LANG: ClassVar[int]
    LANG_END: ClassVar[int]

    # RAG & Citations (262171-262178)
    CONTEXT: ClassVar[int]
    CONTEXT_END: ClassVar[int]
    QUOTE: ClassVar[int]
    QUOTE_END: ClassVar[int]
    CITE: ClassVar[int]
    CITE_END: ClassVar[int]
    SOURCE: ClassVar[int]
    SOURCE_END: ClassVar[int]

    # Memory/State Management (262179-262182)
    MEMORY: ClassVar[int]
    MEMORY_END: ClassVar[int]
    RECALL: ClassVar[int]
    RECALL_END: ClassVar[int]

    # Control Tokens (262183-262185)
    PAD: ClassVar[int]
    STOP: ClassVar[int]
    SEP: ClassVar[int]

    # Multimodal Placeholders (262186-262191)
    IMAGE: ClassVar[int]
    IMAGE_END: ClassVar[int]
    AUDIO: ClassVar[int]
    AUDIO_END: ClassVar[int]
    VIDEO: ClassVar[int]
    VIDEO_END: ClassVar[int]

    # Document Structure (262192-262197)
    TITLE: ClassVar[int]
    TITLE_END: ClassVar[int]
    SECTION: ClassVar[int]
    SECTION_END: ClassVar[int]
    SUMMARY: ClassVar[int]
    SUMMARY_END: ClassVar[int]


@final
class GEMMA4_AGENT_TOKENS:
    """Google Gemma 4 agent token ids (262144-262197)."""

    # Conversation & Roles (262144-262148)
    SYSTEM: ClassVar[int]
    USER: ClassVar[int]
    ASSISTANT: ClassVar[int]
    IM_START: ClassVar[int]
    IM_END: ClassVar[int]

    # Reasoning/Thinking (262149-262150)
    THINK: ClassVar[int]
    THINK_END: ClassVar[int]

    # ReAct Agent Loop (262151-262158)
    PLAN: ClassVar[int]
    PLAN_END: ClassVar[int]
    STEP: ClassVar[int]
    STEP_END: ClassVar[int]
    ACT: ClassVar[int]
    ACT_END: ClassVar[int]
    OBSERVE: ClassVar[int]
    OBSERVE_END: ClassVar[int]

    # Tool/Function Calling (262159-262164)
    FUNCTION: ClassVar[int]
    FUNCTION_END: ClassVar[int]
    RESULT: ClassVar[int]
    RESULT_END: ClassVar[int]
    ERROR: ClassVar[int]
    ERROR_END: ClassVar[int]

    # Code Execution (262165-262170)
    CODE: ClassVar[int]
    CODE_END: ClassVar[int]
    OUTPUT: ClassVar[int]
    OUTPUT_END: ClassVar[int]
    LANG: ClassVar[int]
    LANG_END: ClassVar[int]

    # RAG & Citations (262171-262178)
    CONTEXT: ClassVar[int]
    CONTEXT_END: ClassVar[int]
    QUOTE: ClassVar[int]
    QUOTE_END: ClassVar[int]
    CITE: ClassVar[int]
    CITE_END: ClassVar[int]
    SOURCE: ClassVar[int]
    SOURCE_END: ClassVar[int]

    # Memory/State Management (262179-262182)
    MEMORY: ClassVar[int]
    MEMORY_END: ClassVar[int]
    RECALL: ClassVar[int]
    RECALL_END: ClassVar[int]

    # Control Tokens (262183-262185)
    PAD: ClassVar[int]
    STOP: ClassVar[int]
    SEP: ClassVar[int]

    # Multimodal Placeholders (262186-262191)
    IMAGE: ClassVar[int]
    IMAGE_END: ClassVar[int]
    AUDIO: ClassVar[int]
    AUDIO_END: ClassVar[int]
    VIDEO: ClassVar[int]
    VIDEO_END: ClassVar[int]

    # Document Structure (262192-262197)
    TITLE: ClassVar[int]
    TITLE_END: ClassVar[int]
    SECTION: ClassVar[int]
    SECTION_END: ClassVar[int]
    SUMMARY: ClassVar[int]
    SUMMARY_END: ClassVar[int]

# END GENERATED: agent-tokens
