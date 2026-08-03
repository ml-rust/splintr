// =============================================================================
// Agent Token Constants (cl100k_base: 100277+, o200k_base: 200019+)
// =============================================================================
// These tokens extend the vocabulary for agent/chat applications without
// conflicting with OpenAI's reserved special token ranges.

/// Agent tokens for cl100k_base (GPT-4, GPT-3.5-turbo).
///
/// These special tokens extend the cl100k_base vocabulary for building chat models,
/// reasoning systems, and autonomous agents. Token IDs start at 100277 to avoid
/// conflicts with OpenAI's reserved range (100257-100276).
///
/// # Token Categories
///
/// ## Conversation Structure (100277-100281)
/// Standard ChatML-style tokens for multi-turn conversations:
/// - `<|system|>`: Marks system instructions that define assistant behavior
/// - `<|user|>`: Marks user input/queries
/// - `<|assistant|>`: Marks assistant responses
/// - `<|im_start|>`: Generic message start delimiter (ChatML format)
/// - `<|im_end|>`: Generic message end delimiter (ChatML format)
///
/// ## Reasoning/Thinking (100282-100283)
/// Chain-of-Thought (CoT) tokens for System 2 reasoning
///
/// ## ReAct Agent Loop (100284-100291)
/// Tokens for ReAct (Reason + Act) agent architectures
///
/// ## Tool/Function Calling (100292-100297)
/// Structured tool use with explicit success/error handling
///
/// ## Code Execution (100298-100303)
/// Jupyter notebook-style code interpreter flow
///
/// ## RAG/Citations (100304-100311)
/// Retrieval-Augmented Generation with source attribution
///
/// ## Memory/State (100312-100315)
/// Long-term memory and state persistence
///
/// ## Control Tokens (100316-100318)
/// Sequence control and formatting
///
/// ## Multimodal (100319-100324)
/// Placeholders for non-text content
///
/// ## Document Structure (100325-100330)
/// Semantic layout tokens for parsing structured documents
pub mod cl100k_agent_tokens {
    pub const SYSTEM: u32 = 100277;
    pub const USER: u32 = 100278;
    pub const ASSISTANT: u32 = 100279;
    pub const IM_START: u32 = 100280;
    pub const IM_END: u32 = 100281;
    pub const THINK: u32 = 100282;
    pub const THINK_END: u32 = 100283;
    pub const PLAN: u32 = 100284;
    pub const PLAN_END: u32 = 100285;
    pub const STEP: u32 = 100286;
    pub const STEP_END: u32 = 100287;
    pub const ACT: u32 = 100288;
    pub const ACT_END: u32 = 100289;
    pub const OBSERVE: u32 = 100290;
    pub const OBSERVE_END: u32 = 100291;
    pub const FUNCTION: u32 = 100292;
    pub const FUNCTION_END: u32 = 100293;
    pub const RESULT: u32 = 100294;
    pub const RESULT_END: u32 = 100295;
    pub const ERROR: u32 = 100296;
    pub const ERROR_END: u32 = 100297;
    pub const CODE: u32 = 100298;
    pub const CODE_END: u32 = 100299;
    pub const OUTPUT: u32 = 100300;
    pub const OUTPUT_END: u32 = 100301;
    pub const LANG: u32 = 100302;
    pub const LANG_END: u32 = 100303;
    pub const CONTEXT: u32 = 100304;
    pub const CONTEXT_END: u32 = 100305;
    pub const QUOTE: u32 = 100306;
    pub const QUOTE_END: u32 = 100307;
    pub const CITE: u32 = 100308;
    pub const CITE_END: u32 = 100309;
    pub const SOURCE: u32 = 100310;
    pub const SOURCE_END: u32 = 100311;
    pub const MEMORY: u32 = 100312;
    pub const MEMORY_END: u32 = 100313;
    pub const RECALL: u32 = 100314;
    pub const RECALL_END: u32 = 100315;
    pub const PAD: u32 = 100316;
    pub const STOP: u32 = 100317;
    pub const SEP: u32 = 100318;
    pub const IMAGE: u32 = 100319;
    pub const IMAGE_END: u32 = 100320;
    pub const AUDIO: u32 = 100321;
    pub const AUDIO_END: u32 = 100322;
    pub const VIDEO: u32 = 100323;
    pub const VIDEO_END: u32 = 100324;
    pub const TITLE: u32 = 100325;
    pub const TITLE_END: u32 = 100326;
    pub const SECTION: u32 = 100327;
    pub const SECTION_END: u32 = 100328;
    pub const SUMMARY: u32 = 100329;
    pub const SUMMARY_END: u32 = 100330;
}

/// Agent tokens for o200k_base (GPT-4o).
///
/// See [`cl100k_agent_tokens`] for detailed documentation on each token category.
/// The token semantics are identical; only the IDs differ.
pub mod o200k_agent_tokens {
    pub const SYSTEM: u32 = 200019;
    pub const USER: u32 = 200020;
    pub const ASSISTANT: u32 = 200021;
    pub const IM_START: u32 = 200022;
    pub const IM_END: u32 = 200023;
    pub const THINK: u32 = 200024;
    pub const THINK_END: u32 = 200025;
    pub const PLAN: u32 = 200026;
    pub const PLAN_END: u32 = 200027;
    pub const STEP: u32 = 200028;
    pub const STEP_END: u32 = 200029;
    pub const ACT: u32 = 200030;
    pub const ACT_END: u32 = 200031;
    pub const OBSERVE: u32 = 200032;
    pub const OBSERVE_END: u32 = 200033;
    pub const FUNCTION: u32 = 200034;
    pub const FUNCTION_END: u32 = 200035;
    pub const RESULT: u32 = 200036;
    pub const RESULT_END: u32 = 200037;
    pub const ERROR: u32 = 200038;
    pub const ERROR_END: u32 = 200039;
    pub const CODE: u32 = 200040;
    pub const CODE_END: u32 = 200041;
    pub const OUTPUT: u32 = 200042;
    pub const OUTPUT_END: u32 = 200043;
    pub const LANG: u32 = 200044;
    pub const LANG_END: u32 = 200045;
    pub const CONTEXT: u32 = 200046;
    pub const CONTEXT_END: u32 = 200047;
    pub const QUOTE: u32 = 200048;
    pub const QUOTE_END: u32 = 200049;
    pub const CITE: u32 = 200050;
    pub const CITE_END: u32 = 200051;
    pub const SOURCE: u32 = 200052;
    pub const SOURCE_END: u32 = 200053;
    pub const MEMORY: u32 = 200054;
    pub const MEMORY_END: u32 = 200055;
    pub const RECALL: u32 = 200056;
    pub const RECALL_END: u32 = 200057;
    pub const PAD: u32 = 200058;
    pub const STOP: u32 = 200059;
    pub const SEP: u32 = 200060;
    pub const IMAGE: u32 = 200061;
    pub const IMAGE_END: u32 = 200062;
    pub const AUDIO: u32 = 200063;
    pub const AUDIO_END: u32 = 200064;
    pub const VIDEO: u32 = 200065;
    pub const VIDEO_END: u32 = 200066;
    pub const TITLE: u32 = 200067;
    pub const TITLE_END: u32 = 200068;
    pub const SECTION: u32 = 200069;
    pub const SECTION_END: u32 = 200070;
    pub const SUMMARY: u32 = 200071;
    pub const SUMMARY_END: u32 = 200072;
}
