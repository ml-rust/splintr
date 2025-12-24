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

MODELS = [
    ("cl100k_base", "PyCL100KAgentTokens", "CL100K_AGENT_TOKENS", 100277, "cl100k_base (GPT-4, GPT-3.5-turbo)", [], set()),
    ("o200k_base", "PyO200KAgentTokens", "O200K_AGENT_TOKENS", 200019, "o200k_base (GPT-4o)", [], set()),
    ("llama3", "PyLlama3AgentTokens", "LLAMA3_AGENT_TOKENS", 128300, "Llama 3 family", LLAMA3_META_TOKENS, LLAMA3_SKIP_TOKENS),
    ("deepseek_v3", "PyDeepSeekV3AgentTokens", "DEEPSEEK_V3_AGENT_TOKENS", 128900, "DeepSeek V3/R1", DEEPSEEK_V3_NATIVE_TOKENS, set()),
    ("mistral_v1", "PyMistralV1AgentTokens", "MISTRAL_V1_AGENT_TOKENS", 32000, "Mistral V1 (7B v0.1/v0.2, Mixtral 8x7B)", [], set()),
    ("mistral_v2", "PyMistralV2AgentTokens", "MISTRAL_V2_AGENT_TOKENS", 32768, "Mistral V2 (7B v0.3, Mixtral 8x22B, Codestral)", [], set()),
    ("mistral_v3", "PyMistralV3AgentTokens", "MISTRAL_V3_AGENT_TOKENS", 131072, "Mistral V3/Tekken (NeMo, Large 2, Pixtral)", MISTRAL_V3_CONTROL_TOKENS, set()),
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
            lines.append(f"    /// {desc} ({token_id})")
            lines.append("    #[classattr]")
            lines.append(f"    const {const_name}: u32 = {token_id};")
        lines.append("")

    # Group standard agent tokens by category
    categories = [
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

    for cat_name, start, end in categories:
        lines.append(f"    // {'=' * 73}")
        lines.append(f"    // {cat_name} ({base_id + start}-{base_id + end - 1})")
        lines.append(f"    // {'=' * 73}")
        lines.append("")

        for const_name, token_str, offset, desc in AGENT_TOKENS:
            if start <= offset < end and const_name not in skip_tokens:
                token_id = base_id + offset
                lines.append(f"    /// {desc} ({token_id})")
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

    for model_name, class_name, py_name, base_id, description, extra_tokens, skip_tokens in MODELS:
        output.append(generate_class(model_name, class_name, py_name, base_id, description, extra_tokens, skip_tokens))

    # Generate module registration helper
    output.append("/// Register all agent token classes with the Python module.")
    output.append("pub fn register_agent_tokens(m: &Bound<'_, PyModule>) -> PyResult<()> {")
    for _, class_name, _, _, _, _, _ in MODELS:
        output.append(f'    m.add_class::<{class_name}>()?;')
    output.append("    Ok(())")
    output.append("}")
    output.append("")

    return "\n".join(output)


if __name__ == "__main__":
    print(generate_all())
