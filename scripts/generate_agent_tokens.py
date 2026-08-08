#!/usr/bin/env python3
"""
Generate agent token constants for Python bindings from a single source of truth.

This script generates Rust code for PyO3 Python bindings that exactly matches
the `insert_agent_tokens()` function in src/core/pretrained.rs.

Usage:
    python scripts/generate_agent_tokens.py > src/python/agent_tokens_generated.rs

The generated file should be included in bindings.rs via:
    include!("agent_tokens_generated.rs");
"""

# Single source of truth: Agent token definitions
# Format: (constant_name, token_string, offset, description)
AGENT_TOKENS = [
    # Core conversation structure (0-4)
    ("SYSTEM", "<|system|>", 0, "System role - system instructions"),
    ("USER", "<|user|>", 1, "User role - user input"),
    ("ASSISTANT", "<|assistant|>", 2, "Assistant role - model output"),
    ("IM_START", "<|im_start|>", 3, "Start of message - ChatML wrapper"),
    ("IM_END", "<|im_end|>", 4, "End of message - ChatML wrapper"),

    # Reasoning/thinking tokens (5-6)
    ("THINK", "<|think|>", 5, "Start of thinking - Chain-of-Thought"),
    ("THINK_END", "<|/think|>", 6, "End of thinking"),

    # ReAct agent loop tokens (7-14)
    ("PLAN", "<|plan|>", 7, "Start of plan - action planning"),
    ("PLAN_END", "<|/plan|>", 8, "End of plan"),
    ("STEP", "<|step|>", 9, "Start of step - individual action step"),
    ("STEP_END", "<|/step|>", 10, "End of step"),
    ("ACT", "<|act|>", 11, "Start of action - agent action"),
    ("ACT_END", "<|/act|>", 12, "End of action"),
    ("OBSERVE", "<|observe|>", 13, "Start of observation - environment feedback"),
    ("OBSERVE_END", "<|/observe|>", 14, "End of observation"),

    # Tool/function calling (15-20)
    ("FUNCTION", "<|function|>", 15, "Start of function call - function invocation"),
    ("FUNCTION_END", "<|/function|>", 16, "End of function call"),
    ("RESULT", "<|result|>", 17, "Start of function result - return value"),
    ("RESULT_END", "<|/result|>", 18, "End of function result"),
    ("ERROR", "<|error|>", 19, "Start of error - error message"),
    ("ERROR_END", "<|/error|>", 20, "End of error"),

    # Code execution (21-26)
    ("CODE", "<|code|>", 21, "Start of code - inline code execution"),
    ("CODE_END", "<|/code|>", 22, "End of code"),
    ("OUTPUT", "<|output|>", 23, "Start of output - execution output"),
    ("OUTPUT_END", "<|/output|>", 24, "End of output"),
    ("LANG", "<|lang|>", 25, "Start of language tag - code language"),
    ("LANG_END", "<|/lang|>", 26, "End of language tag"),

    # RAG/context injection (27-34)
    ("CONTEXT", "<|context|>", 27, "Start of context - retrieved context"),
    ("CONTEXT_END", "<|/context|>", 28, "End of context"),
    ("QUOTE", "<|quote|>", 29, "Start of quote - exact citation"),
    ("QUOTE_END", "<|/quote|>", 30, "End of quote"),
    ("CITE", "<|cite|>", 31, "Start of cite - citation reference"),
    ("CITE_END", "<|/cite|>", 32, "End of cite"),
    ("SOURCE", "<|source|>", 33, "Start of source - document source"),
    ("SOURCE_END", "<|/source|>", 34, "End of source"),

    # Memory/state management (35-38)
    ("MEMORY", "<|memory|>", 35, "Start of memory - persistent memory"),
    ("MEMORY_END", "<|/memory|>", 36, "End of memory"),
    ("RECALL", "<|recall|>", 37, "Start of recall - memory retrieval"),
    ("RECALL_END", "<|/recall|>", 38, "End of recall"),

    # Control tokens (39-41)
    ("PAD", "<|pad|>", 39, "Padding token"),
    ("STOP", "<|stop|>", 40, "Stop generation token"),
    ("SEP", "<|sep|>", 41, "Separator token"),

    # Multimodal placeholders (42-47)
    ("IMAGE", "<|image|>", 42, "Start of image - image placeholder"),
    ("IMAGE_END", "<|/image|>", 43, "End of image"),
    ("AUDIO", "<|audio|>", 44, "Start of audio - audio placeholder"),
    ("AUDIO_END", "<|/audio|>", 45, "End of audio"),
    ("VIDEO", "<|video|>", 46, "Start of video - video placeholder"),
    ("VIDEO_END", "<|/video|>", 47, "End of video"),

    # Document structure (48-53)
    ("TITLE", "<|title|>", 48, "Start of title - document/section title"),
    ("TITLE_END", "<|/title|>", 49, "End of title"),
    ("SECTION", "<|section|>", 50, "Start of section - semantic document section"),
    ("SECTION_END", "<|/section|>", 51, "End of section"),
    ("SUMMARY", "<|summary|>", 52, "Start of summary - condensed content summary"),
    ("SUMMARY_END", "<|/summary|>", 53, "End of summary"),
]

# Model configurations: (model_name, class_name, py_name, base_id, description, extra_tokens)
# NOTE: Class names must match existing exports in python/mod.rs
# py_name is the name exposed to Python (via #[pyclass(name = "...")])
# extra_tokens: list of (constant_name, token_id, description) for model-specific native tokens

# Llama3 official Meta tokens (from tokenizer_config.json)
LLAMA3_META_TOKENS = [
    ("BEGIN_OF_TEXT", 128000, "Start of text marker"),
    ("END_OF_TEXT", 128001, "End of text marker"),
    ("FINETUNE_RIGHT_PAD_ID", 128004, "Padding token for fine-tuning - added in Llama 3.1"),
    ("STEP_ID", 128005, "Step marker for reasoning - added in Llama 3.2-Vision"),
    ("START_HEADER_ID", 128006, "Start of header marker"),
    ("END_HEADER_ID", 128007, "End of header marker"),
    ("EOM_ID", 128008, "End of message marker for tool use - added in Llama 3.1"),
    ("EOT_ID", 128009, "End of turn marker"),
    ("PYTHON_TAG", 128010, "Python code interpreter marker - added in Llama 3.1"),
    # Llama 3.2-Vision multimodal tokens (aligned with official Meta tokens)
    ("IMAGE", 128256, "Image placeholder - Llama 3.2-Vision official"),
    ("IMAGE_END", 128257, "End of image placeholder"),
    ("AUDIO", 128258, "Audio placeholder"),
    ("AUDIO_END", 128259, "End of audio placeholder"),
    ("VIDEO", 128260, "Video placeholder"),
    ("VIDEO_END", 128261, "End of video placeholder"),
]

# DeepSeek V3 native tokens (from tokenizer_config.json)
DEEPSEEK_V3_NATIVE_TOKENS = [
    # Core special tokens (0-2)
    ("BEGIN_OF_SENTENCE", 0, "Begin of sentence marker"),
    ("END_OF_SENTENCE", 1, "End of sentence marker"),
    ("PAD_NATIVE", 2, "Native padding token"),
    # Thinking tokens (128798-128799)
    ("THINK_NATIVE", 128798, "Native <think> token"),
    ("THINK_END_NATIVE", 128799, "Native </think> token"),
    # FIM tokens (128800-128802)
    ("FIM_HOLE", 128800, "Fill-in-the-Middle hole"),
    ("FIM_BEGIN", 128801, "Fill-in-the-Middle begin"),
    ("FIM_END", 128802, "Fill-in-the-Middle end"),
    # Chat tokens (128803-128805)
    ("USER_NATIVE", 128803, "Native <｜User｜> token"),
    ("ASSISTANT_NATIVE", 128804, "Native <｜Assistant｜> token"),
    ("EOT", 128805, "End of turn <|EOT|>"),
    # Tool calling tokens (128806-128814)
    ("TOOL_CALLS_BEGIN", 128806, "Tool calls begin marker"),
    ("TOOL_CALLS_END", 128807, "Tool calls end marker"),
    ("TOOL_CALL_BEGIN", 128808, "Single tool call begin"),
    ("TOOL_CALL_END", 128809, "Single tool call end"),
    ("TOOL_OUTPUTS_BEGIN", 128810, "Tool outputs begin marker"),
    ("TOOL_OUTPUTS_END", 128811, "Tool outputs end marker"),
    ("TOOL_OUTPUT_BEGIN", 128812, "Single tool output begin"),
    ("TOOL_OUTPUT_END", 128813, "Single tool output end"),
    ("TOOL_SEP", 128814, "Tool separator"),
]

# Tokens to skip for Llama3 (they use native tokens at different positions)
LLAMA3_SKIP_TOKENS = {"IMAGE", "IMAGE_END", "AUDIO", "AUDIO_END", "VIDEO", "VIDEO_END"}

# Qwen 2/3 native tokens (from Qwen3's tokenizer.json added_tokens, 151643-151668).
#
# IM_START/IM_END carry no `_NATIVE` suffix and are listed here rather than
# generated from the agent block: Qwen names those two itself, so the agent
# token of that name *is* the model's id. THINK_NATIVE/THINK_END_NATIVE do carry
# the suffix, because Qwen's `<think>` is a different string from the agent
# `<|think|>` and both exist.
QWEN3_NATIVE_TOKENS = [
    ("ENDOFTEXT", 151643, "End of text marker"),
    ("IM_START", 151644, "Native <|im_start|> - ChatML message start"),
    ("IM_END", 151645, "Native <|im_end|> - ChatML message end"),
    ("OBJECT_REF_START", 151646, "Object reference begin"),
    ("OBJECT_REF_END", 151647, "Object reference end"),
    ("BOX_START", 151648, "Bounding box begin"),
    ("BOX_END", 151649, "Bounding box end"),
    ("QUAD_START", 151650, "Quadrilateral begin"),
    ("QUAD_END", 151651, "Quadrilateral end"),
    ("VISION_START", 151652, "Vision block begin"),
    ("VISION_END", 151653, "Vision block end"),
    ("VISION_PAD", 151654, "Vision padding"),
    ("IMAGE_PAD", 151655, "Image padding"),
    ("VIDEO_PAD", 151656, "Video padding"),
    ("TOOL_CALL", 151657, "Native <tool_call>"),
    ("TOOL_CALL_END", 151658, "Native </tool_call>"),
    ("FIM_PREFIX", 151659, "Fill-in-the-Middle prefix"),
    ("FIM_MIDDLE", 151660, "Fill-in-the-Middle middle"),
    ("FIM_SUFFIX", 151661, "Fill-in-the-Middle suffix"),
    ("FIM_PAD", 151662, "Fill-in-the-Middle padding"),
    ("REPO_NAME", 151663, "Repository name marker"),
    ("FILE_SEP", 151664, "File separator"),
    ("TOOL_RESPONSE", 151665, "Native <tool_response>"),
    ("TOOL_RESPONSE_END", 151666, "Native </tool_response>"),
    ("THINK_NATIVE", 151667, "Native <think> token"),
    ("THINK_END_NATIVE", 151668, "Native </think> token"),
]

# The two agent names Qwen defines itself; taken from QWEN3_NATIVE_TOKENS above
# instead of the agent block, so they resolve to the ids Qwen trained on.
QWEN3_SKIP_TOKENS = {"IM_START", "IM_END"}

# GLM-4/4.5 native tokens (from GLM-4.5's tokenizer.json, 151329-151364).
GLM4_NATIVE_TOKENS = [
    ("ENDOFTEXT", 151329, "End of text marker"),
    ("MASK", 151330, "[MASK]"),
    ("GMASK", 151331, "[gMASK]"),
    ("SMASK", 151332, "[sMASK]"),
    ("SOP", 151333, "Start of prefix <sop>"),
    ("EOP", 151334, "End of prefix <eop>"),
    ("SYSTEM", 151335, "Native <|system|> role marker"),
    ("USER", 151336, "Native <|user|> role marker"),
    ("ASSISTANT", 151337, "Native <|assistant|> role marker"),
    ("OBSERVATION", 151338, "Observation role marker"),
    ("BEGIN_OF_IMAGE", 151339, "Image block begin"),
    ("END_OF_IMAGE", 151340, "Image block end"),
    ("BEGIN_OF_VIDEO", 151341, "Video block begin"),
    ("END_OF_VIDEO", 151342, "Video block end"),
    ("BEGIN_OF_AUDIO", 151343, "Audio block begin"),
    ("END_OF_AUDIO", 151344, "Audio block end"),
    ("BEGIN_OF_TRANSCRIPTION", 151345, "Transcription begin"),
    ("END_OF_TRANSCRIPTION", 151346, "Transcription end"),
    ("CODE_PREFIX", 151347, "Code Fill-in-the-Middle prefix"),
    ("CODE_MIDDLE", 151348, "Code Fill-in-the-Middle middle"),
    ("CODE_SUFFIX", 151349, "Code Fill-in-the-Middle suffix"),
    ("THINK_NATIVE", 151350, "Native <think> token"),
    ("THINK_END_NATIVE", 151351, "Native </think> token"),
    ("TOOL_CALL", 151352, "Native <tool_call>"),
    ("TOOL_CALL_END", 151353, "Native </tool_call>"),
    ("TOOL_RESPONSE", 151354, "Native <tool_response>"),
    ("TOOL_RESPONSE_END", 151355, "Native </tool_response>"),
    ("ARG_KEY", 151356, "Tool argument key begin"),
    ("ARG_KEY_END", 151357, "Tool argument key end"),
    ("ARG_VALUE", 151358, "Tool argument value begin"),
    ("ARG_VALUE_END", 151359, "Tool argument value end"),
    ("NOTHINK", 151360, "/nothink directive"),
    ("BEGIN_OF_BOX", 151361, "Box begin"),
    ("END_OF_BOX", 151362, "Box end"),
    ("IMAGE", 151363, "Native <|image|> placeholder"),
    ("VIDEO", 151364, "Native <|video|> placeholder"),
]

# The five agent names GLM defines itself. `<|/image|>` and `<|/video|>` are
# NOT skipped: GLM names only the opening markers, so the closing ones stay in
# the agent block at their usual offsets.
GLM4_SKIP_TOKENS = {"SYSTEM", "USER", "ASSISTANT", "IMAGE", "VIDEO"}

# Kimi's named reserved tokens. Both generations reserve the whole 256-wide
# block at 163584-163839; these are the slots the model names, and everything
# else in it is a `<|reserved_token_N|>` placeholder. K2's table is K2.5's, which
# is a strict superset of plain K2's.
#
# `IM_END` here is Kimi's own 163586, not the agent token of the same name. The
# native entry is listed explicitly, so the generated constant carries the
# model's id — which is the same answer `insert_agent_tokens`'s collision rule
# reaches in `pretrained.rs`, from the other direction.
KIMI_K2_NATIVE_TOKENS = [
    ("BOS", 163584, "Beginning of sequence [BOS]"),
    ("EOS", 163585, "End of sequence [EOS]"),
    ("IM_END", 163586, "Native <|im_end|> - end of message"),
    ("IM_USER", 163587, "User turn marker"),
    ("IM_ASSISTANT", 163588, "Assistant turn marker"),
    ("START_HEADER_ID", 163590, "Header begin"),
    ("END_HEADER_ID", 163591, "Header end"),
    ("EOT", 163593, "End of turn [EOT]"),
    ("IM_SYSTEM", 163594, "System turn marker"),
    ("TOOL_CALLS_SECTION_BEGIN", 163595, "Tool calls section begin"),
    ("TOOL_CALLS_SECTION_END", 163596, "Tool calls section end"),
    ("TOOL_CALL_BEGIN", 163597, "Single tool call begin"),
    ("TOOL_CALL_ARGUMENT_BEGIN", 163598, "Tool call argument begin"),
    ("TOOL_CALL_END", 163599, "Single tool call end"),
    ("IM_MIDDLE", 163601, "Message middle marker"),
    ("MEDIA_BEGIN", 163602, "Media block begin"),
    ("MEDIA_CONTENT", 163603, "Media content"),
    ("MEDIA_END", 163604, "Media block end"),
    ("MEDIA_PAD", 163605, "Media padding"),
    ("THINK_NATIVE", 163606, "Native <think> token"),
    ("THINK_END_NATIVE", 163607, "Native </think> token"),
    ("UNK", 163838, "Unknown token [UNK]"),
    ("PAD_NATIVE", 163839, "Native padding [PAD]"),
]

# K3 renames the middle of the block and drops the tool-call markers.
KIMI_K3_NATIVE_TOKENS = [
    ("BOS", 163584, "Beginning of sequence [BOS]"),
    ("EOS", 163585, "End of sequence [EOS]"),
    ("END_OF_MSG", 163586, "End of message"),
    ("OPEN", 163587, "Open marker"),
    ("CLOSE", 163588, "Close marker"),
    ("SEP", 163589, "Separator"),
    ("START_HEADER_ID", 163590, "Header begin"),
    ("END_HEADER_ID", 163591, "Header end"),
    ("EOT", 163593, "End of turn [EOT]"),
    ("MEDIA_BEGIN", 163602, "Media block begin"),
    ("MEDIA_CONTENT", 163603, "Media content"),
    ("MEDIA_END", 163604, "Media block end"),
    ("MEDIA_PAD", 163605, "Media padding"),
    ("OSAGENT_MODE", 163649, "OS-agent mode marker"),
    ("UNK", 163838, "Unknown token [UNK]"),
    ("PAD_NATIVE", 163839, "Native padding [PAD]"),
]

# The agent-token names each Kimi generation defines itself, taken from the
# native tables above rather than the agent block: K2's `<|im_end|>` is 163586
# and K3's `<|sep|>` is 163589, both ids the checkpoint was trained on. Same rule
# as Qwen's `<|im_start|>` and GLM's `<|system|>`.
KIMI_K2_SKIP_TOKENS = {"IM_END"}
KIMI_K3_SKIP_TOKENS = {"SEP"}

# gpt-oss "harmony" native tokens (199998-200018). No name collides with an
# agent token, so nothing is skipped.
GPT_OSS_NATIVE_TOKENS = [
    ("STARTOFTEXT", 199998, "Start of text marker"),
    ("ENDOFTEXT", 199999, "End of text marker"),
    ("RETURN", 200002, "End of a final assistant turn"),
    ("CONSTRAIN", 200003, "Constrained-output marker"),
    ("CHANNEL", 200005, "Channel marker (analysis/commentary/final)"),
    ("START", 200006, "Message start"),
    ("END", 200007, "Message end"),
    ("MESSAGE", 200008, "Message body begin"),
    ("CALL", 200012, "Tool call marker"),
    ("ENDOFPROMPT", 200018, "End of prompt"),
]

# Mistral V3 control tokens (Tekken tokenizer)
MISTRAL_V3_CONTROL_TOKENS = [
    ("INST", 3, "[INST] instruction begin"),
    ("INST_END", 4, "[/INST] instruction end"),
    ("AVAILABLE_TOOLS", 5, "[AVAILABLE_TOOLS] tool list begin"),
    ("AVAILABLE_TOOLS_END", 6, "[/AVAILABLE_TOOLS] tool list end"),
    ("TOOL_RESULTS", 7, "[TOOL_RESULTS] tool results begin"),
    ("TOOL_RESULTS_END", 8, "[/TOOL_RESULTS] tool results end"),
    ("TOOL_CALLS", 9, "[TOOL_CALLS] tool calls marker"),
]

# One row per bundled vocabulary that carries agent tokens; Whisper is absent
# because it carries none. Both the Rust module and the Python class are emitted
# from this table, so the two languages cannot drift apart the way they did when
# Rust had constants for the two OpenAI vocabularies and Python had seven.
#
# Fields: (name, rust module, PyO3 struct, Python class, base id, description,
#          native tokens, agent names the vocabulary defines itself)
MODELS = [
    ("cl100k_base", "cl100k_agent_tokens", "PyCL100KAgentTokens", "CL100K_AGENT_TOKENS", 100277, "cl100k_base (GPT-4, GPT-3.5-turbo)", [], set()),
    ("o200k_base", "o200k_agent_tokens", "PyO200KAgentTokens", "O200K_AGENT_TOKENS", 200019, "o200k_base (GPT-4o)", [], set()),
    ("llama3", "llama3_agent_tokens", "PyLlama3AgentTokens", "LLAMA3_AGENT_TOKENS", 128300, "Llama 3 family", LLAMA3_META_TOKENS, LLAMA3_SKIP_TOKENS),
    ("deepseek_v3", "deepseek_v3_agent_tokens", "PyDeepSeekV3AgentTokens", "DEEPSEEK_V3_AGENT_TOKENS", 128900, "DeepSeek V3/R1", DEEPSEEK_V3_NATIVE_TOKENS, set()),
    ("mistral_v1", "mistral_v1_agent_tokens", "PyMistralV1AgentTokens", "MISTRAL_V1_AGENT_TOKENS", 32000, "Mistral V1 (7B v0.1/v0.2, Mixtral 8x7B)", [], set()),
    ("mistral_v2", "mistral_v2_agent_tokens", "PyMistralV2AgentTokens", "MISTRAL_V2_AGENT_TOKENS", 32768, "Mistral V2 (7B v0.3, Mixtral 8x22B, Codestral)", [], set()),
    ("mistral_v3", "mistral_v3_agent_tokens", "PyMistralV3AgentTokens", "MISTRAL_V3_AGENT_TOKENS", 131072, "Mistral V3/Tekken (NeMo, Large 2, Pixtral)", MISTRAL_V3_CONTROL_TOKENS, set()),
    ("qwen3", "qwen3_agent_tokens", "PyQwen3AgentTokens", "QWEN3_AGENT_TOKENS", 151669, "Qwen 2/3 (also Baichuan-M2)", QWEN3_NATIVE_TOKENS, QWEN3_SKIP_TOKENS),
    ("glm4", "glm4_agent_tokens", "PyGlm4AgentTokens", "GLM4_AGENT_TOKENS", 151365, "GLM-4/4.5", GLM4_NATIVE_TOKENS, GLM4_SKIP_TOKENS),
    ("gpt-oss", "gpt_oss_agent_tokens", "PyGptOssAgentTokens", "GPT_OSS_AGENT_TOKENS", 200019, "OpenAI gpt-oss", GPT_OSS_NATIVE_TOKENS, set()),
    ("kimi_k2", "kimi_k2_agent_tokens", "PyKimiK2AgentTokens", "KIMI_K2_AGENT_TOKENS", 163840, "Kimi K2 (K2, K2.5, K2.6, K2.7, Kimi-Linear)", KIMI_K2_NATIVE_TOKENS, KIMI_K2_SKIP_TOKENS),
    ("kimi_k3", "kimi_k3_agent_tokens", "PyKimiK3AgentTokens", "KIMI_K3_AGENT_TOKENS", 163840, "Kimi K3", KIMI_K3_NATIVE_TOKENS, KIMI_K3_SKIP_TOKENS),
]

# The ten categories the 54 agent tokens fall into, as (name, first offset,
# one-past-last offset). Shared by both emitters so the grouping is identical.
CATEGORIES = [
    ("Conversation & Roles", 0, 5),
    ("Reasoning/Thinking", 5, 7),
    ("ReAct Agent Loop", 7, 15),
    ("Tool/Function Calling", 15, 21),
    ("Code Execution", 21, 27),
    ("RAG & Citations", 27, 35),
    ("Memory/State Management", 35, 39),
    ("Control Tokens", 39, 42),
    ("Multimodal Placeholders", 42, 48),
    ("Document Structure", 48, 54),
]


def generate_class(model_name: str, class_name: str, py_name: str, base_id: int, description: str, extra_tokens: list, skip_tokens: set) -> str:
    """Generate a PyO3 class for agent tokens."""
    lines = []

    # Class docstring
    lines.append(f"/// {description} Agent Token IDs ({base_id}-{base_id + 53})")
    lines.append("///")
    lines.append(f"/// Access agent token IDs for {description}.")
    lines.append("///")
    lines.append("/// # Examples")
    lines.append("///")
    lines.append("/// ```python")
    lines.append(f"/// from splintr import Tokenizer, {py_name}")
    lines.append("///")
    lines.append(f'/// tokenizer = Tokenizer.from_pretrained("{model_name}")')
    lines.append(f"/// system_id = {py_name}.SYSTEM  # {base_id}")
    lines.append(f"/// think_id = {py_name}.THINK   # {base_id + 5}")
    lines.append("///")
    lines.append('/// text = "<|system|>You are a helpful assistant"')
    lines.append("/// tokens = tokenizer.encode_with_special(text)")
    lines.append(f"/// assert {py_name}.SYSTEM in tokens")
    lines.append("/// ```")

    # Class definition
    lines.append(f'#[pyclass(name = "{py_name}", frozen)]')
    lines.append(f"pub struct {class_name};")
    lines.append("")
    lines.append("#[pymethods]")
    lines.append(f"impl {class_name} {{")

    # Add model-specific native tokens first (if any)
    if extra_tokens:
        lines.append(f"    // {'=' * 73}")
        lines.append(f"    // Model-Specific Native Tokens")
        lines.append(f"    // {'=' * 73}")
        lines.append("")

        for const_name, token_id, desc in extra_tokens:
            lines.append(f"    /// {_rustdoc_escape(desc)} ({token_id})")
            lines.append("    #[classattr]")
            lines.append(f"    const {const_name}: u32 = {token_id};")
        lines.append("")

    for cat_name, start, end in CATEGORIES:
        lines.append(f"    // {'=' * 73}")
        lines.append(f"    // {cat_name} ({base_id + start}-{base_id + end - 1})")
        lines.append(f"    // {'=' * 73}")
        lines.append("")

        for const_name, token_str, offset, desc in AGENT_TOKENS:
            if start <= offset < end and const_name not in skip_tokens:
                token_id = base_id + offset
                lines.append(f"    /// {_rustdoc_escape(desc)} ({token_id})")
                lines.append("    #[classattr]")
                lines.append(f"    const {const_name}: u32 = {token_id};")
        lines.append("")

    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def generate_all() -> str:
    """Generate all agent token classes."""
    output = []

    output.append("// =============================================================================")
    output.append("// AUTO-GENERATED FILE - DO NOT EDIT MANUALLY")
    output.append("// Generated by: scripts/generate_agent_tokens.py")
    output.append("// Source of truth: AGENT_TOKENS in generate_agent_tokens.py")
    output.append("// =============================================================================")
    output.append("")
    output.append("// Note: pyo3::prelude::* is already imported in bindings.rs")
    output.append("")

    for _, _, class_name, py_name, base_id, description, extra_tokens, skip_tokens in MODELS:
        output.append(generate_class(_model_name_of(py_name), class_name, py_name, base_id, description, extra_tokens, skip_tokens))

    # Generate module registration helper
    output.append("/// Register all agent token classes with the Python module.")
    output.append("pub fn register_agent_tokens(m: &Bound<'_, PyModule>) -> PyResult<()> {")
    for _, _, class_name, _, _, _, _, _ in MODELS:
        output.append(f'    m.add_class::<{class_name}>()?;')
    output.append("    Ok(())")
    output.append("}")
    output.append("")

    return "\n".join(output)


def _model_name_of(py_name: str) -> str:
    """The `from_pretrained` name for a Python class name."""
    for model_name, _, _, name, _, _, _, _ in MODELS:
        if name == py_name:
            return model_name
    raise KeyError(py_name)


def generate_rust_module(
    model_name: str,
    mod_name: str,
    base_id: int,
    description: str,
    extra_tokens: list,
    skip_tokens: set,
) -> str:
    """Generate one `pub mod <vocab>_agent_tokens` of plain `u32` constants."""
    lines = []

    lines.append(f"/// Agent tokens for {description}.")
    lines.append("///")
    lines.append(
        f"/// The 54-token block starts at {base_id}. Every id here is also reachable"
    )
    lines.append(
        f'/// at runtime as `from_pretrained("{model_name}")?.special_token_id(name)`;'
    )
    lines.append("/// these constants are the compile-time form, so a typo is a compile")
    lines.append("/// error rather than a `None` to unwrap.")
    if skip_tokens:
        shared = ", ".join(sorted(skip_tokens))
        lines.append("///")
        lines.append(
            f"/// {description} defines {shared} itself, so those constants carry the"
        )
        lines.append(
            "/// vocabulary's own id — below `base_vocab_size`, the id the checkpoint"
        )
        lines.append("/// was trained on — rather than a splintr-appended one.")
    if extra_tokens:
        lines.append("///")
        lines.append(
            "/// The vocabulary's own markers are included too, so this module is its"
        )
        lines.append("/// whole special-token surface. A native name that would collide with")
        lines.append("/// an agent token of the same meaning takes a `_NATIVE` suffix.")
    lines.append(f"pub mod {mod_name} {{")

    if extra_tokens:
        lines.append("    // Vocabulary-native tokens")
        for const_name, token_id, desc in extra_tokens:
            lines.append(f"    /// {_rustdoc_escape(desc)}")
            lines.append(f"    pub const {const_name}: u32 = {token_id};")
        lines.append("")

    for cat_name, start, end in CATEGORIES:
        emitted = [
            (c, d, base_id + o)
            for c, _, o, d in AGENT_TOKENS
            if start <= o < end and c not in skip_tokens
        ]
        if not emitted:
            continue
        lines.append(f"    // {cat_name}")
        for const_name, desc, token_id in emitted:
            lines.append(f"    /// {_rustdoc_escape(desc)}")
            lines.append(f"    pub const {const_name}: u32 = {token_id};")
        lines.append("")

    if lines[-1] == "":
        lines.pop()
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def generate_all_rust() -> str:
    """Generate the Rust constants module for every vocabulary."""
    output = []
    output.append("// =============================================================================")
    output.append("// AUTO-GENERATED FILE - DO NOT EDIT MANUALLY")
    output.append("// Generated by: scripts/generate_agent_tokens.py --lang rust")
    output.append("// Source of truth: AGENT_TOKENS and MODELS in that script")
    output.append("// =============================================================================")
    output.append("//")
    output.append("// Compile-time agent-token ids, one module per bundled vocabulary.")
    output.append("// Whisper has no module: it carries no agent tokens.")
    output.append("//")
    output.append("// Included by agent_tokens.rs, which holds the tests for these ids.")
    output.append("")
    for model_name, mod_name, _, _, base_id, description, extra_tokens, skip_tokens in MODELS:
        output.append(
            generate_rust_module(model_name, mod_name, base_id, description, extra_tokens, skip_tokens)
        )
    return "\n".join(output)


import re

# ---------------------------------------------------------------------------
# Documentation tables
# ---------------------------------------------------------------------------
#
# The per-category tables in docs/special_tokens.md list the same 54 tokens this
# file already defines, so they were a fourth hand-maintained copy of one table
# — and behaved like one: they carried an id column per vocabulary, went stale
# the moment a vocabulary was added, and grew unreadable as the list grew. They
# are emitted from `AGENT_TOKENS` instead, keyed on the **offset**, which is the
# invariant: an id is its vocabulary's block start plus this offset, and the
# block starts are one small table rather than a column in ten.

DOC_PATH = "docs/special_tokens.md"
BEGIN = "<!-- BEGIN GENERATED: {} -->"
END = "<!-- END GENERATED: {} -->"


def _rustdoc_escape(text: str) -> str:
    """Make a description safe inside a Rust `///` comment.

    Token names are full of `<think>`, `[gMASK]` and `<sop>`, which rustdoc reads
    as HTML tags and intra-doc links — under `-D warnings` (what CI builds docs
    with) each is a hard error, so the crate simply fails to document. Wrapping
    each such span in backticks makes it a code span, which rustdoc leaves alone
    and which is how it should render anyway.
    """
    return re.sub(r"(<[^<>\s]+>|\[[^\[\]\s]+\])", r"`\1`", text)


def _md_escape(token: str) -> str:
    """A pipe inside a markdown table cell has to be escaped."""
    return token.replace("|", "\\|")


def category_table(start: int, end: int) -> str:
    """The `Token | Offset | Description` table for one category."""
    rows = [(t, o, d) for _, t, o, d in AGENT_TOKENS if start <= o < end]
    width = max(len(_md_escape(t)) + 2 for t, _, _ in rows)
    desc_width = max(len(d) for _, _, d in rows)
    lines = [
        f"| {'Token'.ljust(width)} | Offset | {'Description'.ljust(desc_width)} |",
        f"| {'-' * width} | ------ | {'-' * desc_width} |",
    ]
    for token, offset, desc in rows:
        cell = f"`{_md_escape(token)}`".ljust(width)
        lines.append(f"| {cell} | {offset:>6} | {desc.ljust(desc_width)} |")
    return "\n".join(lines)


def block_start_table() -> str:
    """Where each vocabulary's 54-id block begins — the other half of an id."""
    lines = [
        "| Vocabulary | Block starts at | Rust module | Python class |",
        "| --- | --- | --- | --- |",
    ]
    for model_name, mod_name, _, py_name, base_id, _, _, _ in MODELS:
        lines.append(
            f"| `{model_name}` | {base_id:,} | `{mod_name}` | `{py_name}` |"
        )
    lines.append("| `whisper` | — | — | — |")
    return "\n".join(lines)


def doc_regions() -> dict:
    """Region name -> generated markdown, for every sentinel-marked block."""
    regions = {"agent-token-block-starts": block_start_table()}
    for index, (_, start, end) in enumerate(CATEGORIES):
        regions[f"agent-tokens-category-{index}"] = category_table(start, end)
    return regions


def rewrite_docs(check_only: bool = False) -> int:
    """Replace every sentinel-marked region in the doc. Returns an exit code."""
    import sys

    with open(DOC_PATH, encoding="utf-8") as handle:
        original = handle.read()

    updated = original
    for name, body in doc_regions().items():
        begin, end = BEGIN.format(name), END.format(name)
        if begin not in updated or end not in updated:
            print(f"error: {DOC_PATH} has no region {name!r}", file=sys.stderr)
            return 1
        head, _, rest = updated.partition(begin)
        _, _, tail = rest.partition(end)
        updated = f"{head}{begin}\n\n{body}\n\n{end}{tail}"

    if updated == original:
        print(f"ok    {DOC_PATH} is up to date")
        return 0
    if check_only:
        print(
            f"error: {DOC_PATH} is stale — run "
            f"`python scripts/generate_agent_tokens.py --update-docs`",
            file=sys.stderr,
        )
        return 1
    with open(DOC_PATH, "w", encoding="utf-8") as handle:
        handle.write(updated)
    print(f"ok    rewrote {len(doc_regions())} regions in {DOC_PATH}")
    return 0


if __name__ == "__main__":
    import sys

    if "--update-docs" in sys.argv:
        raise SystemExit(rewrite_docs(check_only="--check" in sys.argv))

    lang = "python"
    if "--lang" in sys.argv:
        lang = sys.argv[sys.argv.index("--lang") + 1]
    if lang == "rust":
        print(generate_all_rust())
    elif lang == "python":
        print(generate_all())
    else:
        sys.exit(f"unknown --lang {lang!r}; expected 'rust' or 'python'")
