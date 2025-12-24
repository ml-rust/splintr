# Special Tokens Reference

This document describes the special tokens available in Splintr's `cl100k_base`, `o200k_base`, `llama3`, `deepseek_v3`, and `mistral_v1`/`mistral_v2`/`mistral_v3` tokenizers, including the extended agent token vocabulary.

## Table of Contents

- [Overview](#overview)
- [Design Rationale](#design-rationale)
- [Token ID Allocation](#token-id-allocation)
- [OpenAI Standard Tokens](#openai-standard-tokens)
- [Meta Llama 3 Standard Tokens](#meta-llama-3-standard-tokens)
- [DeepSeek V3 Standard Tokens](#deepseek-v3-standard-tokens)
- [Mistral Standard Tokens](#mistral-standard-tokens)
- [Agent Token Categories](#agent-token-categories)
  - [1. Conversation Structure](#1-conversation-structure)
  - [2. Reasoning / Chain-of-Thought](#2-reasoning--chain-of-thought)
  - [3. ReAct Agent Loop](#3-react-agent-loop)
  - [4. Tool / Function Calling](#4-tool--function-calling)
  - [5. Code Execution](#5-code-execution)
  - [6. RAG / Citations](#6-rag--citations)
  - [7. Memory / State](#7-memory--state)
  - [8. Control Tokens](#8-control-tokens)
  - [9. Multimodal](#9-multimodal)
  - [10. Document Structure](#10-document-structure)
- [Usage Examples](#usage-examples)
- [Python API Reference](#python-api-reference)
- [Rust API Reference](#rust-api-reference)

---

## Overview

Splintr extends the standard OpenAI/Llama/Deepseek tokenizer vocabularies with **54 additional special tokens** designed for building modern AI agent systems. These tokens provide semantic structure for:

- Multi-turn chat conversations (ChatML format)
- Chain-of-Thought reasoning (System 2 thinking)
- ReAct-style agent loops (Reason + Act)
- Tool/function calling with error handling
- Code execution environments
- Retrieval-Augmented Generation (RAG) with citations
- Long-term memory and state persistence
- Multimodal content placeholders
- Structured document parsing

---

## Design Rationale

### Why Special Tokens?

Special tokens serve as **semantic markers** that help models understand the structure and intent of different parts of the input. Unlike regular text that gets split into subword tokens, special tokens are:

1. **Atomic**: Always encoded as a single token ID, never split
2. **Unambiguous**: Cannot be confused with regular text
3. **Efficient**: Single token vs multiple tokens for delimiters
4. **Trainable**: Models can learn specific behaviors associated with each token

### Why Extend the Vocabulary?

OpenAI's standard tokenizers include only basic special tokens (`<|endoftext|>`, `<|fim_*|>`, etc.). Modern agent architectures require richer semantic markers to:

- **Separate concerns**: Distinguish thinking from output, actions from observations
- **Enable parsing**: Reliably extract structured data from model outputs
- **Support training**: Provide clear signals for fine-tuning agent behaviors
- **Maintain compatibility**: Work alongside existing tokenizer infrastructure

### Token Naming Convention

All tokens follow the `<|name|>` / `<|/name|>` pattern:

- Opening tags: `<|name|>` - marks the start of a semantic block
- Closing tags: `<|/name|>` - marks the end of a semantic block
- Standalone tokens: `<|name|>` - single markers (e.g., `<|pad|>`, `<|stop|>`)

This convention mirrors XML/HTML for familiarity while using `<|...|>` to avoid conflicts with actual markup in training data.

---

## Token ID Allocation

### Avoiding Conflicts

Token IDs are carefully allocated to avoid conflicts with reserved ranges:

| Model         | Regular Tokens | Reserved Range  | Agent Tokens      | Total     |
| ------------- | -------------- | --------------- | ----------------- | --------- |
| `cl100k_base` | 0-100,255      | 100,257-100,276 | 100,277-100,330   | 100,331   |
| `o200k_base`  | 0-199,997      | 199,999-200,018 | 200,019-200,072   | 200,073   |
| `llama3`      | 0-127,999      | 128,000-128,261 | 128,300-128,353   | 128,354   |
| `deepseek_v3` | 0-127,999      | 128,798-128,814 | 128,900-128,953   | 128,954   |
| `mistral_v1`  | 0-31,999       | 0-2             | 32,000-32,053     | 32,054    |
| `mistral_v2`  | 0-32,767       | 0-9             | 32,768-32,821     | 32,822    |
| `mistral_v3`  | 0-131,071      | 0-9             | 131,072-131,125   | 131,126   |

### Why These Ranges?

- **OpenAI compatibility**: Agent tokens start after OpenAI's last known special token
- **Meta compatibility**: Llama 3 agent tokens start at 128,300 to avoid Meta's reserved range (128,000-128,261)
- **Future-proofing**: Gap between standard tokens and agent tokens allows for future additions
- **Consistency**: Same token semantics map to different IDs per vocabulary, but maintain relative ordering

---

## OpenAI Standard Tokens

These tokens are part of the original OpenAI tokenizer specification:

### cl100k_base (GPT-4, GPT-3.5-turbo)

| Token               | ID     | Purpose                    |
| ------------------- | ------ | -------------------------- |
| `<\|endoftext\|>`   | 100257 | End of document marker     |
| `<\|fim_prefix\|>`  | 100258 | Fill-in-the-middle: prefix |
| `<\|fim_middle\|>`  | 100259 | Fill-in-the-middle: middle |
| `<\|fim_suffix\|>`  | 100260 | Fill-in-the-middle: suffix |
| `<\|endofprompt\|>` | 100276 | End of prompt marker       |

### o200k_base (GPT-4o)

| Token               | ID     | Purpose                |
| ------------------- | ------ | ---------------------- |
| `<\|endoftext\|>`   | 199999 | End of document marker |
| `<\|endofprompt\|>` | 200018 | End of prompt marker   |

---

## Meta Llama 3 Standard Tokens

These tokens are part of the official Meta Llama 3 tokenizer specification, with version-specific additions noted.

### llama3 (Supports Llama 3 through 3.3)

Splintr's `llama3` vocabulary includes the base 128,000 BPE tokens plus all special tokens from Llama 3.0 through 3.3, providing full compatibility with all Llama 3 model versions.

#### Core Tokens (Llama 3.0+)

| Token                   | ID     | Purpose               |
| ----------------------- | ------ | --------------------- |
| `<\|begin_of_text\|>`   | 128000 | Beginning of sequence |
| `<\|end_of_text\|>`     | 128001 | End of sequence       |
| `<\|start_header_id\|>` | 128006 | Start of role header  |
| `<\|end_header_id\|>`   | 128007 | End of role header    |
| `<\|eot_id\|>`          | 128009 | End of turn           |

#### Added in Llama 3.1

| Token                         | ID     | Purpose                       |
| ----------------------------- | ------ | ----------------------------- |
| `<\|finetune_right_pad_id\|>` | 128004 | Padding token for fine-tuning |
| `<\|eom_id\|>`                | 128008 | End of message (tool use)     |
| `<\|python_tag\|>`            | 128010 | Code interpreter marker       |

#### Added in Llama 3.2-Vision

| Token           | ID     | Purpose                    |
| --------------- | ------ | -------------------------- |
| `<\|step_id\|>` | 128005 | Step identifier for vision |
| `<\|image\|>`   | 128256 | Image content placeholder  |

### Llama 3 Chat Format

Llama 3 uses a header-based chat format different from ChatML:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The capital of France is Paris.<|eot_id|>
```

**Key differences from ChatML:**

- Uses `<|start_header_id|>role<|end_header_id|>` instead of `<|im_start|>role`
- Uses `<|eot_id|>` instead of `<|im_end|>`
- Double newline after header before content
- `<|begin_of_text|>` at sequence start

---

## DeepSeek V3 Standard Tokens

These tokens are part of the official DeepSeek V3 tokenizer specification.

> **Note**: DeepSeek V3 uses ByteLevel BPE encoding, which differs from standard BPE. See [ByteLevel BPE Encoding](bytelevel_bpe.md) for details on how bytes are mapped to tokens.

### deepseek_v3 (DeepSeek V3)

Splintr's `deepseek_v3` vocabulary includes the base 128,000 BPE tokens plus all native DeepSeek special tokens and Splintr agent tokens.

#### Core Tokens

| Token              | ID     | Purpose                   |
| ------------------ | ------ | ------------------------- |
| `<\|begin▁of▁sentence\|>` | 0      | Beginning of sequence (BOS) |
| `<\|end▁of▁sentence\|>`   | 1      | End of sequence (EOS)       |

**Note**: DeepSeek uses the special underscore character `▁` (U+2581, "LOWER ONE EIGHTH BLOCK") in token names, which differs from the regular underscore `_`.

#### Reasoning Tokens (Native DeepSeek)

| Token              | ID     | Purpose                   |
| ------------------ | ------ | ------------------------- |
| `<\|begin▁of▁thought\|>`  | 128798 | Start of thinking block   |
| `<\|end▁of▁thought\|>`    | 128799 | End of thinking block     |

#### Fill-in-the-Middle (FIM) Tokens

| Token               | ID     | Purpose                        |
| ------------------- | ------ | ------------------------------ |
| `<\|fim▁begin\|>`   | 128800 | Start of FIM context           |
| `<\|fim▁hole\|>`    | 128801 | Placeholder for code to insert |
| `<\|fim▁end\|>`     | 128802 | End of FIM context             |

#### Chat Tokens

| Token             | ID     | Purpose             |
| ----------------- | ------ | ------------------- |
| `<\|User\|>`      | 128803 | User turn marker    |
| `<\|Assistant\|>` | 128804 | Assistant turn marker |
| `<\|end▁of▁turn\|>` | 128805 | End of conversation turn |

#### Tool/Function Calling Tokens

| Token                      | ID     | Purpose                    |
| -------------------------- | ------ | -------------------------- |
| `<\|tool▁calls▁begin\|>`   | 128806 | Start of tool call block   |
| `<\|tool▁calls▁end\|>`     | 128807 | End of tool call block     |
| `<\|tool▁call▁begin\|>`    | 128808 | Start of individual call   |
| `<\|tool▁call▁end\|>`      | 128809 | End of individual call     |
| `<\|tool▁outputs▁begin\|>` | 128810 | Start of tool outputs      |
| `<\|tool▁outputs▁end\|>`   | 128811 | End of tool outputs        |
| `<\|tool▁output▁begin\|>`  | 128812 | Start of individual output |
| `<\|tool▁output▁end\|>`    | 128813 | End of individual output   |
| `<\|tool▁sep\|>`           | 128814 | Tool separator             |

### DeepSeek V3 Chat Format

DeepSeek V3 uses a simpler chat format compared to Llama 3:

```
<|begin▁of▁sentence|><|User|>Hello, how are you?<|Assistant|>I'm doing well, thank you! How can I help you today?<|end▁of▁turn|>
```

**Key differences from other formats:**

- Uses `<|User|>` and `<|Assistant|>` role markers (with capital letters)
- Uses `<|end▁of▁turn|>` to mark conversation turn boundaries
- Uses special underscore `▁` character in token names
- No separate header start/end tokens like Llama 3

### DeepSeek V3 Thinking Format

DeepSeek V3 has native support for chain-of-thought reasoning with dedicated tokens:

```
<|begin▁of▁thought|>
Let me think about this step by step...
1. First, I need to understand the question
2. Then, I'll analyze the options
3. Finally, I'll formulate my answer
<|end▁of▁thought|>

Based on my analysis, the answer is 42.
```

---

## Mistral Standard Tokens

Mistral AI has released multiple tokenizer versions with different vocabulary sizes and capabilities:

- **V1** (~32k tokens): Mistral 7B v0.1/v0.2, Mixtral 8x7B - Basic SentencePiece
- **V2** (~32,768 tokens): Mistral 7B v0.3, Codestral, Mixtral 8x22B - SentencePiece + Control Tokens
- **V3/Tekken** (~131k tokens): Mistral NeMo, Large 2, Pixtral - Tiktoken-based (NOT SentencePiece)

> **Note**: V1 and V2 use SentencePiece-style tokenization. V3/Tekken uses Tiktoken (similar to GPT-4o).

### mistral_v1 (Mistral 7B v0.1/v0.2, Mixtral 8x7B)

Splintr's `mistral_v1` vocabulary includes ~32,000 BPE tokens plus 54 agent tokens.

#### Core SentencePiece Tokens

| Token   | ID | Purpose                      |
| ------- | -- | ---------------------------- |
| `<unk>` | 0  | Unknown token                |
| `<s>`   | 1  | Beginning of sequence (BOS)  |
| `</s>`  | 2  | End of sequence (EOS)        |

### mistral_v2 (Mistral 7B v0.3, Codestral, Mixtral 8x22B)

V2 extends V1 with 768 control tokens for tool calling and instruction formatting.

#### Core SentencePiece Tokens (same as V1)

| Token   | ID | Purpose                      |
| ------- | -- | ---------------------------- |
| `<unk>` | 0  | Unknown token                |
| `<s>`   | 1  | Beginning of sequence (BOS)  |
| `</s>`  | 2  | End of sequence (EOS)        |

#### V2 Control Tokens

| Token                | ID | Purpose                    |
| -------------------- | -- | -------------------------- |
| `[INST]`             | 3  | Start of user instruction  |
| `[/INST]`            | 4  | End of user instruction    |
| `[TOOL_CALLS]`       | 5  | Tool calling block         |
| `[AVAILABLE_TOOLS]`  | 6  | Available tools definition |
| `[/AVAILABLE_TOOLS]` | 7  | End of tools definition    |
| `[TOOL_RESULTS]`     | 8  | Tool results block         |
| `[/TOOL_RESULTS]`    | 9  | End of tool results        |

### mistral_v3 (Mistral NeMo, Large 2, Pixtral)

V3 uses a completely different tokenizer architecture: **Tekken** (Tiktoken-based), with a much larger vocabulary (~131k tokens).

#### Core Tokens

| Token   | ID | Purpose                      |
| ------- | -- | ---------------------------- |
| `<unk>` | 0  | Unknown token                |
| `<s>`   | 1  | Beginning of sequence (BOS)  |
| `</s>`  | 2  | End of sequence (EOS)        |

V3 includes the same control tokens as V2 (`[INST]`, `[/INST]`, etc.) but uses Tiktoken encoding instead of SentencePiece.

### Mistral Chat Format

All Mistral versions use a similar instruction-based chat format:

```
<s>[INST] You are a helpful assistant. [/INST]</s>
<s>[INST] What is the capital of France? [/INST] The capital of France is Paris.</s>
```

**Key differences from other formats:**

- Uses `<s>` BOS and `</s>` EOS markers around each turn
- Uses `[INST]` and `[/INST]` markers for instructions
- V2+ treats `[INST]`, `[/INST]` as special tokens (single token IDs)
- V1 encodes them as regular text (multiple token IDs)

### SentencePiece Encoding (V1 and V2 only)

V1 and V2 use SentencePiece encoding:

```python
# Input: "Hello world"
# Encoding process:
#   1. Split by whitespace pattern
#   2. Prepend ▁ (U+2581) to word start: ["▁Hello", "▁world"]
#   3. Apply BPE merges to each chunk
# Decoding process:
#   1. Decode tokens to bytes
#   2. Replace ▁ with space: " Hello world"
```

V3 does NOT use SentencePiece - it uses Tiktoken (similar to o200k_base).

---

## Agent Token Categories

> **Mistral Agent Token IDs**: The tables below show `mistral_v1 ID`. For V2 and V3:
> - **V1**: Agent tokens start at 32,000
> - **V2**: Agent tokens start at 32,768 (add 768 to V1 IDs)
> - **V3**: Agent tokens start at 131,072

### 1. Conversation Structure

**Purpose**: Standard ChatML-style tokens for multi-turn conversations.

| Token             | cl100k ID | o200k ID | llama3 ID | deepseek_v3 ID | mistral_v1 ID | Description                                     |
| ----------------- | --------- | -------- | --------- | -------------- | ------------- | ----------------------------------------------- |
| `<\|system\|>`    | 100277    | 200019   | 128300    | 128900         | 32000         | System instructions defining assistant behavior |
| `<\|user\|>`      | 100278    | 200020   | 128301    | 128901         | 32001         | User input/queries                              |
| `<\|assistant\|>` | 100279    | 200021   | 128302    | 128902         | 32002         | Assistant responses                             |
| `<\|im_start\|>`  | 100280    | 200022   | 128303    | 128903         | 32003         | Generic message start (ChatML)                  |
| `<\|im_end\|>`    | 100281    | 200023   | 128304    | 128904         | 32004         | Generic message end (ChatML)                    |

**Rationale**: These tokens implement the [ChatML format](https://github.com/openai/openai-python/blob/main/chatml.md) used by OpenAI and adopted widely for chat model training. The `im_start`/`im_end` tokens provide a generic wrapper, while role-specific tokens (`system`, `user`, `assistant`) enable direct role marking.

**Example**:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
```

---

### 2. Reasoning / Chain-of-Thought

**Purpose**: Enable System 2 (slow, deliberate) reasoning similar to DeepSeek-R1 or OpenAI o1.

| Token          | cl100k ID | o200k ID | llama3 ID | deepseek_v3 ID | mistral_v1 ID | Description              |
| -------------- | --------- | -------- | --------- | -------------- | ---------- | ------------------------ |
| `<\|think\|>`  | 100282    | 200024   | 128305    | 128905         | 32005      | Start of reasoning block |
| `<\|/think\|>` | 100283    | 200025   | 128306    | 128906         | 32006      | End of reasoning block   |

**Rationale**: Chain-of-Thought (CoT) prompting significantly improves model performance on complex tasks. Dedicated thinking tokens allow:

- **Training**: Models learn to "think before answering"
- **Inference**: Thinking can be hidden from users in production
- **Analysis**: Reasoning traces can be extracted for debugging/evaluation

**Example**:

```
<|think|>
The user is asking about the capital of France.
I know that Paris is the capital and largest city of France.
It has been the capital since the 10th century.
<|/think|>
The capital of France is Paris.
```

---

### 3. ReAct Agent Loop

**Purpose**: Implement the ReAct (Reason + Act) paradigm for autonomous agents.

| Token            | cl100k ID | o200k ID | llama3 ID | deepseek_v3 ID | mistral_v1 ID | Description                     |
| ---------------- | --------- | -------- | --------- | -------------- | ------------------------------- |
| `<\|plan\|>`     | 100284    | 200026   | 128307    | 128907         | 32007      | High-level strategy formulation |
| `<\|/plan\|>`    | 100285    | 200027   | 128308    | 128908         | 32008      | End of plan                     |
| `<\|step\|>`     | 100286    | 200028   | 128309    | 128909         | 32009      | Individual step within plan     |
| `<\|/step\|>`    | 100287    | 200029   | 128310    | 128910         | 32010      | End of step                     |
| `<\|act\|>`      | 100288    | 200030   | 128311    | 128911         | 32011      | Action intent declaration       |
| `<\|/act\|>`     | 100289    | 200031   | 128312    | 128912         | 32012      | End of action                   |
| `<\|observe\|>`  | 100290    | 200032   | 128313    | 128913         | 32013      | Environment feedback            |
| `<\|/observe\|>` | 100291    | 200033   | 128314    | 128914         | 32014      | End of observation              |

**Rationale**: The [ReAct paper](https://arxiv.org/abs/2210.03629) demonstrated that interleaving reasoning and acting improves agent performance. These tokens create a structured loop:

1. **Plan**: Agent decides overall strategy
2. **Step**: Break plan into discrete actions
3. **Act**: Declare intent to perform action
4. **Observe**: Receive and process environment feedback
5. Repeat until task complete

**Example**:

```
<|plan|>
To answer this question, I need to:
1. Search for current weather data
2. Extract the temperature
3. Format the response
<|/plan|>
<|step|>Searching for weather data<|/step|>
<|act|>search("London weather today")<|/act|>
<|observe|>Temperature: 18°C, Condition: Partly cloudy<|/observe|>
<|step|>Formatting response<|/step|>
The current temperature in London is 18°C with partly cloudy skies.
```

---

### 4. Tool / Function Calling

**Purpose**: Structured tool use with explicit success/error handling.

| Token             | cl100k ID | o200k ID | llama3 ID | deepseek_v3 ID | mistral_v1 ID | Description                 |
| ----------------- | --------- | -------- | --------- | -------------- | --------------------------- |
| `<\|function\|>`  | 100292    | 200034   | 128315    | 128915         | 32015      | Function call specification |
| `<\|/function\|>` | 100293    | 200035   | 128316    | 128916         | 32016      | End of function call        |
| `<\|result\|>`    | 100294    | 200036   | 128317    | 128917         | 32017      | Successful return value     |
| `<\|/result\|>`   | 100295    | 200037   | 128318    | 128918         | 32018      | End of result               |
| `<\|error\|>`     | 100296    | 200038   | 128319    | 128919         | 32019      | Execution error             |
| `<\|/error\|>`    | 100297    | 200039   | 128320    | 128920         | 32020      | End of error                |

**Rationale**: Function calling is fundamental to agent capabilities. Separating `<|act|>` (intent) from `<|function|>` (technical payload) allows:

- **Intent**: "I want to check the weather" (`<|act|>`)
- **Implementation**: `{"name": "get_weather", "args": {...}}` (`<|function|>`)

The `<|error|>` token is critical for robust agents—it signals that the previous action failed, enabling retry logic without confusing errors with valid outputs.

**Example**:

```
<|function|>{"name": "get_weather", "args": {"city": "London", "units": "celsius"}}<|/function|>
<|result|>{"temperature": 18, "condition": "partly_cloudy", "humidity": 65}<|/result|>
```

**Error handling**:

```
<|function|>{"name": "get_stock_price", "args": {"symbol": "INVALID"}}<|/function|>
<|error|>{"code": "SYMBOL_NOT_FOUND", "message": "Stock symbol 'INVALID' not found"}<|/error|>
```

---

### 5. Code Execution

**Purpose**: Jupyter notebook-style code interpreter flow.

| Token           | cl100k ID | o200k ID | llama3 ID | deepseek_v3 ID | mistral_v1 ID | Description           |
| --------------- | --------- | -------- | --------- | -------------- | --------------------- |
| `<\|code\|>`    | 100298    | 200040   | 128321    | 128921         | 32021      | Code block to execute |
| `<\|/code\|>`   | 100299    | 200041   | 128322    | 128922         | 32022      | End of code block     |
| `<\|output\|>`  | 100300    | 200042   | 128323    | 128923         | 32023      | Execution output      |
| `<\|/output\|>` | 100301    | 200043   | 128324    | 128924         | 32024      | End of output         |
| `<\|lang\|>`    | 100302    | 200044   | 128325    | 128925         | 32025      | Language identifier   |
| `<\|/lang\|>`   | 100303    | 200045   | 128326    | 128926         | 32026      | End of language tag   |

**Rationale**: Code execution is a powerful agent capability. These tokens model the notebook paradigm:

- Code cells with explicit language tags
- Captured stdout/return values
- Clear separation between code and output

**Example**:

```
<|code|><|lang|>python<|/lang|>
import math

def calculate_circle_area(radius):
    return math.pi * radius ** 2

area = calculate_circle_area(5)
print(f"Area: {area:.2f}")
<|/code|>
<|output|>Area: 78.54<|/output|>
```

---

### 6. RAG / Citations

**Purpose**: Retrieval-Augmented Generation with source attribution.

| Token            | cl100k ID | o200k ID | llama3 ID | deepseek_v3 ID | mistral_v1 ID | Description             |
| ---------------- | --------- | -------- | --------- | -------------- | ----------------------- |
| `<\|context\|>`  | 100304    | 200046   | 128327    | 128927         | 32027      | Retrieved context block |
| `<\|/context\|>` | 100305    | 200047   | 128328    | 128928         | 32028      | End of context          |
| `<\|quote\|>`    | 100306    | 200048   | 128329    | 128929         | 32029      | Direct quotation        |
| `<\|/quote\|>`   | 100307    | 200049   | 128330    | 128930         | 32030      | End of quote            |
| `<\|cite\|>`     | 100308    | 200050   | 128331    | 128931         | 32031      | Citation reference      |
| `<\|/cite\|>`    | 100309    | 200051   | 128332    | 128932         | 32032      | End of citation         |
| `<\|source\|>`   | 100310    | 200052   | 128333    | 128933         | 32033      | Source metadata         |
| `<\|/source\|>`  | 100311    | 200053   | 128334    | 128934         | 32034      | End of source           |

**Rationale**: RAG systems retrieve relevant documents to ground model responses. These tokens enable:

- **Grounded generation**: Model sees retrieved context explicitly
- **Citation training**: Model learns to cite sources
- **Verification**: Outputs can be traced back to sources
- **Hallucination reduction**: Clear separation of retrieved vs generated content

**Example**:

```
<|context|>
<|source|>wikipedia:Paris<|/source|>
Paris is the capital and most populous city of France. With an official
estimated population of 2,102,650 residents in January 2023 in an area of
more than 105 km², Paris is the fourth-most populated city in the European Union.
<|/context|>

Based on the retrieved information, Paris is the capital of France with a
population of approximately <|quote|>2,102,650 residents<|/quote|>
<|cite|>wikipedia:Paris<|/cite|>.
```

---

### 7. Memory / State

**Purpose**: Long-term memory and state persistence across sessions.

| Token           | cl100k ID | o200k ID | llama3 ID | deepseek_v3 ID | mistral_v1 ID | Description         |
| --------------- | --------- | -------- | --------- | -------------- | ------------------- |
| `<\|memory\|>`  | 100312    | 200054   | 128335    | 128935         | 32035      | Store information   |
| `<\|/memory\|>` | 100313    | 200055   | 128336    | 128936         | 32036      | End of memory block |
| `<\|recall\|>`  | 100314    | 200056   | 128337    | 128937         | 32037      | Retrieved memory    |
| `<\|/recall\|>` | 100315    | 200057   | 128338    | 128938         | 32038      | End of recall       |

**Rationale**: Persistent memory enables agents to:

- Remember user preferences across conversations
- Build up knowledge over time
- Maintain continuity in long-running tasks

The separation of `memory` (write) and `recall` (read) mirrors database semantics.

**Example**:

```
<|memory|>User prefers concise responses. User's name is Alice.<|/memory|>

... later in conversation ...

<|recall|>User prefers concise responses. User's name is Alice.<|/recall|>
Hello Alice! Here's a brief answer: The capital of France is Paris.
```

---

### 8. Control Tokens

**Purpose**: Sequence control and formatting.

| Token        | cl100k ID | o200k ID | llama3 ID | deepseek_v3 ID | mistral_v1 ID | Description                 |
| ------------ | --------- | -------- | --------- | -------------- | --------------------------- |
| `<\|pad\|>`  | 100316    | 200058   | 128339    | 128939         | 32039      | Padding for batch alignment |
| `<\|stop\|>` | 100317    | 200059   | 128340    | 128940         | 32040      | Generation stop signal      |
| `<\|sep\|>`  | 100318    | 200060   | 128341    | 128941         | 32041      | Segment separator           |

**Rationale**: These are utility tokens for training and inference:

- **pad**: Aligns sequences in batches (has no semantic meaning)
- **stop**: Alternative to `<|endoftext|>` for stopping generation
- **sep**: Separates segments without implying document boundaries

---

### 9. Multimodal

**Purpose**: Placeholders for non-text content.

| Token          | cl100k ID | o200k ID | llama3 ID | deepseek_v3 ID | mistral_v1 ID | Description   |
| -------------- | --------- | -------- | --------- | -------------- | ------------- |
| `<\|image\|>`  | 100319    | 200061   | 128256\*  | 128942         | 32042      | Image content |
| `<\|/image\|>` | 100320    | 200062   | 128257    | 128943         | 32043      | End of image  |
| `<\|audio\|>`  | 100321    | 200063   | 128258    | 128944         | 32044      | Audio content |
| `<\|/audio\|>` | 100322    | 200064   | 128259    | 128945         | 32045      | End of audio  |
| `<\|video\|>`  | 100323    | 200065   | 128260    | 128946         | 32046      | Video content |
| `<\|/video\|>` | 100324    | 200066   | 128261    | 128947         | 32047      | End of video  |

\*Note: Llama 3's `<|image|>` token (128256) is aligned with the official Meta Llama 3.2-Vision token ID for compatibility.

**Rationale**: Multimodal models need to mark where non-text embeddings are inserted. These tokens serve as:

- **Placeholders**: Mark positions for embedding injection
- **Delimiters**: Wrap base64-encoded or referenced content
- **Training signals**: Help models learn cross-modal attention

**Example**:

```
Describe what you see in this image:
<|image|>base64_encoded_image_data_here<|/image|>

The image shows a sunset over the ocean with vibrant orange and purple colors.
```

---

### 10. Document Structure

**Purpose**: Semantic layout for parsing structured documents.

| Token            | cl100k ID | o200k ID | llama3 ID | deepseek_v3 ID | mistral_v1 ID | Description            |
| ---------------- | --------- | -------- | --------- | -------------- | ---------------------- |
| `<\|title\|>`    | 100325    | 200067   | 128348    | 128948         | 32048      | Document/section title |
| `<\|/title\|>`   | 100326    | 200068   | 128349    | 128949         | 32049      | End of title           |
| `<\|section\|>`  | 100327    | 200069   | 128350    | 128950         | 32050      | Semantic section       |
| `<\|/section\|>` | 100328    | 200070   | 128351    | 128951         | 32051      | End of section         |
| `<\|summary\|>`  | 100329    | 200071   | 128352    | 128952         | 32052      | Content summary        |
| `<\|/summary\|>` | 100330    | 200072   | 128353    | 128953         | 32053      | End of summary         |

**Rationale**: When processing structured documents (papers, reports, documentation), these tokens help:

- **Preserve structure**: Maintain document hierarchy in tokenized form
- **Enable extraction**: Reliably parse titles, sections, summaries
- **Support generation**: Train models to produce well-structured output

**Example**:

```
<|title|>Climate Change Impact Assessment<|/title|>

<|summary|>
This report examines the effects of climate change on coastal ecosystems,
finding significant impacts on biodiversity and recommending adaptive strategies.
<|/summary|>

<|section|>
<|title|>Introduction<|/title|>
Climate change represents one of the most significant challenges...
<|/section|>

<|section|>
<|title|>Methodology<|/title|>
We analyzed data from 50 coastal monitoring stations...
<|/section|>
```

---

## Usage Examples

### Python (OpenAI models)

```python
from splintr import Tokenizer, CL100K_AGENT_TOKENS

tokenizer = Tokenizer.from_pretrained("cl100k_base")

# Encode text with special tokens
text = "<|think|>Let me reason about this...<|/think|>The answer is 42."
tokens = tokenizer.encode_with_special(text)

# Check for specific tokens
if CL100K_AGENT_TOKENS.THINK in tokens:
    print("Contains thinking block")

# Decode back to text
decoded = tokenizer.decode(tokens)
assert decoded == text

# Access token IDs programmatically
print(f"THINK token ID: {CL100K_AGENT_TOKENS.THINK}")        # 100282
print(f"FUNCTION token ID: {CL100K_AGENT_TOKENS.FUNCTION}")  # 100292
```

### Python (DeepSeek V3 models)

```python
from splintr import Tokenizer, DEEPSEEK_V3_AGENT_TOKENS

# Load DeepSeek V3 tokenizer (includes ByteLevel BPE encoding)
tokenizer = Tokenizer.from_pretrained("deepseek_v3")

# Encode text with special tokens
text = "<|think|>Let me reason about this...<|/think|>The answer is 42."
tokens = tokenizer.encode_with_special(text)

# Check for specific tokens
if DEEPSEEK_V3_AGENT_TOKENS.THINK in tokens:
    print("Contains thinking block")

# Use native DeepSeek tokens
chat = "<|User|>Hello!<|Assistant|>Hi there!<|end▁of▁turn|>"
tokens = tokenizer.encode_with_special(chat)

# Access native DeepSeek token IDs
print(f"Native User token: {DEEPSEEK_V3_AGENT_TOKENS.DEEPSEEK_USER}")        # 128803
print(f"Native thinking: {DEEPSEEK_V3_AGENT_TOKENS.BEGIN_OF_THOUGHT}")       # 128798
print(f"Agent think token: {DEEPSEEK_V3_AGENT_TOKENS.THINK}")                # 128905
```

### Python (Llama 3 models)

```python
from splintr import Tokenizer, LLAMA3_AGENT_TOKENS

# Load Llama 3 tokenizer (includes all special tokens up to Llama 3.3)
tokenizer = Tokenizer.from_pretrained("llama3")

# Encode Llama 3 chat format
chat = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
tokens = tokenizer.encode_with_special(chat)

# Check for official Meta tokens
assert LLAMA3_AGENT_TOKENS.BEGIN_OF_TEXT in tokens  # 128000
assert LLAMA3_AGENT_TOKENS.EOT_ID in tokens         # 128009

# Use agent tokens with Llama 3
text = "<|think|>Let me reason...<|/think|>The answer is 42."
tokens = tokenizer.encode_with_special(text)
print(f"THINK token ID: {LLAMA3_AGENT_TOKENS.THINK}")      # 128305
print(f"IMAGE token ID: {LLAMA3_AGENT_TOKENS.IMAGE}")      # 128256 (official Meta 3.2-Vision)
```

### Rust

```rust
use splintr::{Tokenizer, cl100k_agent_tokens, CL100K_BASE_PATTERN};

// Access token constants
let think_id = cl100k_agent_tokens::THINK;           // 100282
let function_id = cl100k_agent_tokens::FUNCTION;     // 100292

// Use in your agent implementation
fn extract_thinking(tokens: &[u32]) -> Option<(usize, usize)> {
    let start = tokens.iter().position(|&t| t == cl100k_agent_tokens::THINK)?;
    let end = tokens.iter().position(|&t| t == cl100k_agent_tokens::THINK_END)?;
    Some((start, end))
}
```

---

## Python API Reference

### CL100K_AGENT_TOKENS

```python
from splintr import CL100K_AGENT_TOKENS

# Conversation
CL100K_AGENT_TOKENS.SYSTEM          # 100277
CL100K_AGENT_TOKENS.USER            # 100278
CL100K_AGENT_TOKENS.ASSISTANT       # 100279
CL100K_AGENT_TOKENS.IM_START        # 100280
CL100K_AGENT_TOKENS.IM_END          # 100281

# Thinking
CL100K_AGENT_TOKENS.THINK           # 100282
CL100K_AGENT_TOKENS.THINK_END       # 100283

# ReAct
CL100K_AGENT_TOKENS.PLAN            # 100284
CL100K_AGENT_TOKENS.PLAN_END        # 100285
CL100K_AGENT_TOKENS.STEP            # 100286
CL100K_AGENT_TOKENS.STEP_END        # 100287
CL100K_AGENT_TOKENS.ACT             # 100288
CL100K_AGENT_TOKENS.ACT_END         # 100289
CL100K_AGENT_TOKENS.OBSERVE         # 100290
CL100K_AGENT_TOKENS.OBSERVE_END     # 100291

# Tool/Function
CL100K_AGENT_TOKENS.FUNCTION        # 100292
CL100K_AGENT_TOKENS.FUNCTION_END    # 100293
CL100K_AGENT_TOKENS.RESULT          # 100294
CL100K_AGENT_TOKENS.RESULT_END      # 100295
CL100K_AGENT_TOKENS.ERROR           # 100296
CL100K_AGENT_TOKENS.ERROR_END       # 100297

# Code
CL100K_AGENT_TOKENS.CODE            # 100298
CL100K_AGENT_TOKENS.CODE_END        # 100299
CL100K_AGENT_TOKENS.OUTPUT          # 100300
CL100K_AGENT_TOKENS.OUTPUT_END      # 100301
CL100K_AGENT_TOKENS.LANG            # 100302
CL100K_AGENT_TOKENS.LANG_END        # 100303

# RAG
CL100K_AGENT_TOKENS.CONTEXT         # 100304
CL100K_AGENT_TOKENS.CONTEXT_END     # 100305
CL100K_AGENT_TOKENS.QUOTE           # 100306
CL100K_AGENT_TOKENS.QUOTE_END       # 100307
CL100K_AGENT_TOKENS.CITE            # 100308
CL100K_AGENT_TOKENS.CITE_END        # 100309
CL100K_AGENT_TOKENS.SOURCE          # 100310
CL100K_AGENT_TOKENS.SOURCE_END      # 100311

# Memory
CL100K_AGENT_TOKENS.MEMORY          # 100312
CL100K_AGENT_TOKENS.MEMORY_END      # 100313
CL100K_AGENT_TOKENS.RECALL          # 100314
CL100K_AGENT_TOKENS.RECALL_END      # 100315

# Control
CL100K_AGENT_TOKENS.PAD             # 100316
CL100K_AGENT_TOKENS.STOP            # 100317
CL100K_AGENT_TOKENS.SEP             # 100318

# Multimodal
CL100K_AGENT_TOKENS.IMAGE           # 100319
CL100K_AGENT_TOKENS.IMAGE_END       # 100320
CL100K_AGENT_TOKENS.AUDIO           # 100321
CL100K_AGENT_TOKENS.AUDIO_END       # 100322
CL100K_AGENT_TOKENS.VIDEO           # 100323
CL100K_AGENT_TOKENS.VIDEO_END       # 100324

# Document
CL100K_AGENT_TOKENS.TITLE           # 100325
CL100K_AGENT_TOKENS.TITLE_END       # 100326
CL100K_AGENT_TOKENS.SECTION         # 100327
CL100K_AGENT_TOKENS.SECTION_END     # 100328
CL100K_AGENT_TOKENS.SUMMARY         # 100329
CL100K_AGENT_TOKENS.SUMMARY_END     # 100330
```

### O200K_AGENT_TOKENS

Same structure as above, with IDs starting at 200019.

### DEEPSEEK_V3_AGENT_TOKENS

```python
from splintr import DEEPSEEK_V3_AGENT_TOKENS

# Native DeepSeek tokens
DEEPSEEK_V3_AGENT_TOKENS.BOS                   # 0 (begin_of_sentence)
DEEPSEEK_V3_AGENT_TOKENS.EOS                   # 1 (end_of_sentence)
DEEPSEEK_V3_AGENT_TOKENS.BEGIN_OF_THOUGHT      # 128798
DEEPSEEK_V3_AGENT_TOKENS.END_OF_THOUGHT        # 128799
DEEPSEEK_V3_AGENT_TOKENS.FIM_BEGIN             # 128800
DEEPSEEK_V3_AGENT_TOKENS.FIM_HOLE              # 128801
DEEPSEEK_V3_AGENT_TOKENS.FIM_END               # 128802
DEEPSEEK_V3_AGENT_TOKENS.DEEPSEEK_USER         # 128803
DEEPSEEK_V3_AGENT_TOKENS.DEEPSEEK_ASSISTANT    # 128804
DEEPSEEK_V3_AGENT_TOKENS.END_OF_TURN           # 128805
DEEPSEEK_V3_AGENT_TOKENS.TOOL_CALLS_BEGIN      # 128806
DEEPSEEK_V3_AGENT_TOKENS.TOOL_CALLS_END        # 128807
DEEPSEEK_V3_AGENT_TOKENS.TOOL_CALL_BEGIN       # 128808
DEEPSEEK_V3_AGENT_TOKENS.TOOL_CALL_END         # 128809
DEEPSEEK_V3_AGENT_TOKENS.TOOL_OUTPUTS_BEGIN    # 128810
DEEPSEEK_V3_AGENT_TOKENS.TOOL_OUTPUTS_END      # 128811
DEEPSEEK_V3_AGENT_TOKENS.TOOL_OUTPUT_BEGIN     # 128812
DEEPSEEK_V3_AGENT_TOKENS.TOOL_OUTPUT_END       # 128813
DEEPSEEK_V3_AGENT_TOKENS.TOOL_SEP              # 128814

# Conversation
DEEPSEEK_V3_AGENT_TOKENS.SYSTEM                # 128900
DEEPSEEK_V3_AGENT_TOKENS.USER                  # 128901
DEEPSEEK_V3_AGENT_TOKENS.ASSISTANT             # 128902
DEEPSEEK_V3_AGENT_TOKENS.IM_START              # 128903
DEEPSEEK_V3_AGENT_TOKENS.IM_END                # 128904

# Thinking
DEEPSEEK_V3_AGENT_TOKENS.THINK                 # 128905
DEEPSEEK_V3_AGENT_TOKENS.THINK_END             # 128906

# ReAct
DEEPSEEK_V3_AGENT_TOKENS.PLAN                  # 128907
DEEPSEEK_V3_AGENT_TOKENS.PLAN_END              # 128908
DEEPSEEK_V3_AGENT_TOKENS.STEP                  # 128909
DEEPSEEK_V3_AGENT_TOKENS.STEP_END              # 128910
DEEPSEEK_V3_AGENT_TOKENS.ACT                   # 128911
DEEPSEEK_V3_AGENT_TOKENS.ACT_END               # 128912
DEEPSEEK_V3_AGENT_TOKENS.OBSERVE               # 128913
DEEPSEEK_V3_AGENT_TOKENS.OBSERVE_END           # 128914

# Tool/Function
DEEPSEEK_V3_AGENT_TOKENS.FUNCTION              # 128915
DEEPSEEK_V3_AGENT_TOKENS.FUNCTION_END          # 128916
DEEPSEEK_V3_AGENT_TOKENS.RESULT                # 128917
DEEPSEEK_V3_AGENT_TOKENS.RESULT_END            # 128918
DEEPSEEK_V3_AGENT_TOKENS.ERROR                 # 128919
DEEPSEEK_V3_AGENT_TOKENS.ERROR_END             # 128920

# Code
DEEPSEEK_V3_AGENT_TOKENS.CODE                  # 128921
DEEPSEEK_V3_AGENT_TOKENS.CODE_END              # 128922
DEEPSEEK_V3_AGENT_TOKENS.OUTPUT                # 128923
DEEPSEEK_V3_AGENT_TOKENS.OUTPUT_END            # 128924
DEEPSEEK_V3_AGENT_TOKENS.LANG                  # 128925
DEEPSEEK_V3_AGENT_TOKENS.LANG_END              # 128926

# RAG
DEEPSEEK_V3_AGENT_TOKENS.CONTEXT               # 128927
DEEPSEEK_V3_AGENT_TOKENS.CONTEXT_END           # 128928
DEEPSEEK_V3_AGENT_TOKENS.QUOTE                 # 128929
DEEPSEEK_V3_AGENT_TOKENS.QUOTE_END             # 128930
DEEPSEEK_V3_AGENT_TOKENS.CITE                  # 128931
DEEPSEEK_V3_AGENT_TOKENS.CITE_END              # 128932
DEEPSEEK_V3_AGENT_TOKENS.SOURCE                # 128933
DEEPSEEK_V3_AGENT_TOKENS.SOURCE_END            # 128934

# Memory
DEEPSEEK_V3_AGENT_TOKENS.MEMORY                # 128935
DEEPSEEK_V3_AGENT_TOKENS.MEMORY_END            # 128936
DEEPSEEK_V3_AGENT_TOKENS.RECALL                # 128937
DEEPSEEK_V3_AGENT_TOKENS.RECALL_END            # 128938

# Control
DEEPSEEK_V3_AGENT_TOKENS.PAD                   # 128939
DEEPSEEK_V3_AGENT_TOKENS.STOP                  # 128940
DEEPSEEK_V3_AGENT_TOKENS.SEP                   # 128941

# Multimodal
DEEPSEEK_V3_AGENT_TOKENS.IMAGE                 # 128942
DEEPSEEK_V3_AGENT_TOKENS.IMAGE_END             # 128943
DEEPSEEK_V3_AGENT_TOKENS.AUDIO                 # 128944
DEEPSEEK_V3_AGENT_TOKENS.AUDIO_END             # 128945
DEEPSEEK_V3_AGENT_TOKENS.VIDEO                 # 128946
DEEPSEEK_V3_AGENT_TOKENS.VIDEO_END             # 128947

# Document
DEEPSEEK_V3_AGENT_TOKENS.TITLE                 # 128948
DEEPSEEK_V3_AGENT_TOKENS.TITLE_END             # 128949
DEEPSEEK_V3_AGENT_TOKENS.SECTION               # 128950
DEEPSEEK_V3_AGENT_TOKENS.SECTION_END           # 128951
DEEPSEEK_V3_AGENT_TOKENS.SUMMARY               # 128952
DEEPSEEK_V3_AGENT_TOKENS.SUMMARY_END           # 128953
```

### LLAMA3_AGENT_TOKENS

```python
from splintr import LLAMA3_AGENT_TOKENS

# Official Meta tokens
LLAMA3_AGENT_TOKENS.BEGIN_OF_TEXT       # 128000
LLAMA3_AGENT_TOKENS.END_OF_TEXT         # 128001
LLAMA3_AGENT_TOKENS.FINETUNE_RIGHT_PAD_ID # 128004 (Llama 3.1+)
LLAMA3_AGENT_TOKENS.STEP_ID             # 128005 (Llama 3.2-Vision)
LLAMA3_AGENT_TOKENS.START_HEADER_ID     # 128006
LLAMA3_AGENT_TOKENS.END_HEADER_ID       # 128007
LLAMA3_AGENT_TOKENS.EOM_ID              # 128008 (Llama 3.1+)
LLAMA3_AGENT_TOKENS.EOT_ID              # 128009
LLAMA3_AGENT_TOKENS.PYTHON_TAG          # 128010 (Llama 3.1+)

# Conversation
LLAMA3_AGENT_TOKENS.SYSTEM              # 128300
LLAMA3_AGENT_TOKENS.USER                # 128301
LLAMA3_AGENT_TOKENS.ASSISTANT           # 128302
LLAMA3_AGENT_TOKENS.IM_START            # 128303
LLAMA3_AGENT_TOKENS.IM_END              # 128304

# Thinking
LLAMA3_AGENT_TOKENS.THINK               # 128305
LLAMA3_AGENT_TOKENS.THINK_END           # 128306

# ReAct
LLAMA3_AGENT_TOKENS.PLAN                # 128307
LLAMA3_AGENT_TOKENS.PLAN_END            # 128308
LLAMA3_AGENT_TOKENS.STEP                # 128309
LLAMA3_AGENT_TOKENS.STEP_END            # 128310
LLAMA3_AGENT_TOKENS.ACT                 # 128311
LLAMA3_AGENT_TOKENS.ACT_END             # 128312
LLAMA3_AGENT_TOKENS.OBSERVE             # 128313
LLAMA3_AGENT_TOKENS.OBSERVE_END         # 128314

# Tool/Function
LLAMA3_AGENT_TOKENS.FUNCTION            # 128315
LLAMA3_AGENT_TOKENS.FUNCTION_END        # 128316
LLAMA3_AGENT_TOKENS.RESULT              # 128317
LLAMA3_AGENT_TOKENS.RESULT_END          # 128318
LLAMA3_AGENT_TOKENS.ERROR               # 128319
LLAMA3_AGENT_TOKENS.ERROR_END           # 128320

# Code
LLAMA3_AGENT_TOKENS.CODE                # 128321
LLAMA3_AGENT_TOKENS.CODE_END            # 128322
LLAMA3_AGENT_TOKENS.OUTPUT              # 128323
LLAMA3_AGENT_TOKENS.OUTPUT_END          # 128324
LLAMA3_AGENT_TOKENS.LANG                # 128325
LLAMA3_AGENT_TOKENS.LANG_END            # 128326

# RAG
LLAMA3_AGENT_TOKENS.CONTEXT             # 128327
LLAMA3_AGENT_TOKENS.CONTEXT_END         # 128328
LLAMA3_AGENT_TOKENS.QUOTE               # 128329
LLAMA3_AGENT_TOKENS.QUOTE_END           # 128330
LLAMA3_AGENT_TOKENS.CITE                # 128331
LLAMA3_AGENT_TOKENS.CITE_END            # 128332
LLAMA3_AGENT_TOKENS.SOURCE              # 128333
LLAMA3_AGENT_TOKENS.SOURCE_END          # 128334

# Memory
LLAMA3_AGENT_TOKENS.MEMORY              # 128335
LLAMA3_AGENT_TOKENS.MEMORY_END          # 128336
LLAMA3_AGENT_TOKENS.RECALL              # 128337
LLAMA3_AGENT_TOKENS.RECALL_END          # 128338

# Control
LLAMA3_AGENT_TOKENS.PAD                 # 128339
LLAMA3_AGENT_TOKENS.STOP                # 128340
LLAMA3_AGENT_TOKENS.SEP                 # 128341

# Multimodal (aligned with official Meta 3.2-Vision)
LLAMA3_AGENT_TOKENS.IMAGE               # 128256
LLAMA3_AGENT_TOKENS.IMAGE_END           # 128257
LLAMA3_AGENT_TOKENS.AUDIO               # 128258
LLAMA3_AGENT_TOKENS.AUDIO_END           # 128259
LLAMA3_AGENT_TOKENS.VIDEO               # 128260
LLAMA3_AGENT_TOKENS.VIDEO_END           # 128261

# Document
LLAMA3_AGENT_TOKENS.TITLE               # 128348
LLAMA3_AGENT_TOKENS.TITLE_END           # 128349
LLAMA3_AGENT_TOKENS.SECTION             # 128350
LLAMA3_AGENT_TOKENS.SECTION_END         # 128351
LLAMA3_AGENT_TOKENS.SUMMARY             # 128352
LLAMA3_AGENT_TOKENS.SUMMARY_END         # 128353
```

### MISTRAL_V1_AGENT_TOKENS

Mistral V1 tokenizers (7B v0.1/v0.2, Mixtral 8x7B) use SentencePiece encoding with agent tokens starting at ID 32000.

```python
from splintr import MISTRAL_V1_AGENT_TOKENS

# Conversation
MISTRAL_V1_AGENT_TOKENS.SYSTEM          # 32000
MISTRAL_V1_AGENT_TOKENS.USER            # 32001
MISTRAL_V1_AGENT_TOKENS.ASSISTANT       # 32002
MISTRAL_V1_AGENT_TOKENS.IM_START        # 32003
MISTRAL_V1_AGENT_TOKENS.IM_END          # 32004

# Thinking
MISTRAL_V1_AGENT_TOKENS.THINK           # 32005
MISTRAL_V1_AGENT_TOKENS.THINK_END       # 32006

# Function/Tools
MISTRAL_V1_AGENT_TOKENS.FUNCTION        # 32015
MISTRAL_V1_AGENT_TOKENS.FUNCTION_END    # 32016

# ... and 48 more tokens up to 32053
```

### MISTRAL_V2_AGENT_TOKENS

Mistral V2 tokenizers (7B v0.3, Mixtral 8x22B, Codestral) use SentencePiece with control tokens and agent tokens starting at ID 32768.

**Note:** V2 includes native control tokens at IDs 3-9 (e.g., [INST], [/INST]), so agent token base is shifted to 32768 instead of 32000.

```python
from splintr import MISTRAL_V2_AGENT_TOKENS

# Control tokens (native, IDs 3-9)
# [INST] (3), [/INST] (4), [TOOL_CALLS] (5), [AVAILABLE_TOOLS] (6), [TOOL_RESULTS] (7-9)

# Agent tokens start at 32768
MISTRAL_V2_AGENT_TOKENS.SYSTEM          # 32768
MISTRAL_V2_AGENT_TOKENS.USER            # 32769
MISTRAL_V2_AGENT_TOKENS.THINK           # 32773

# ... and 51 more tokens up to 32821
```

### MISTRAL_V3_AGENT_TOKENS

Mistral V3/Tekken tokenizers (NeMo, Large 2, Pixtral) use Tiktoken-based encoding (not SentencePiece) with control tokens and agent tokens starting at ID 131072.

**Note:** V3 includes 7 native control tokens at IDs 3-9 (Tekken-specific), so agent token base is shifted to 131072.

```python
from splintr import MISTRAL_V3_AGENT_TOKENS

# Control tokens (native, IDs 3-9)
# [INST] (3), [/INST] (4), [AVAILABLE_TOOLS] (5), [/AVAILABLE_TOOLS] (6),
# [TOOL_RESULTS] (7), [/TOOL_RESULTS] (8), [TOOL_CALLS] (9)

# Agent tokens start at 131072
MISTRAL_V3_AGENT_TOKENS.SYSTEM          # 131072
MISTRAL_V3_AGENT_TOKENS.USER            # 131073
MISTRAL_V3_AGENT_TOKENS.THINK           # 131077

# ... and 51 more tokens up to 131125
```

---

## Rust API Reference

### cl100k_agent_tokens module

```rust
use splintr::cl100k_agent_tokens;

// All constants follow the same naming as Python
cl100k_agent_tokens::SYSTEM          // 100277
cl100k_agent_tokens::THINK           // 100282
cl100k_agent_tokens::FUNCTION        // 100292
// ... etc
```

### o200k_agent_tokens module

```rust
use splintr::o200k_agent_tokens;

o200k_agent_tokens::SYSTEM           // 200019
o200k_agent_tokens::THINK            // 200024
// ... etc
```

### deepseek_v3_agent_tokens module

```rust
use splintr::deepseek_v3_agent_tokens;

// Native DeepSeek tokens
deepseek_v3_agent_tokens::BOS                  // 0
deepseek_v3_agent_tokens::EOS                  // 1
deepseek_v3_agent_tokens::BEGIN_OF_THOUGHT     // 128798
deepseek_v3_agent_tokens::END_OF_THOUGHT       // 128799
deepseek_v3_agent_tokens::FIM_BEGIN            // 128800
deepseek_v3_agent_tokens::FIM_HOLE             // 128801
deepseek_v3_agent_tokens::FIM_END              // 128802
deepseek_v3_agent_tokens::DEEPSEEK_USER        // 128803
deepseek_v3_agent_tokens::DEEPSEEK_ASSISTANT   // 128804
deepseek_v3_agent_tokens::END_OF_TURN          // 128805

// Agent tokens
deepseek_v3_agent_tokens::SYSTEM               // 128900
deepseek_v3_agent_tokens::THINK                // 128905
deepseek_v3_agent_tokens::FUNCTION             // 128915
// ... etc
```

### mistral_v1_agent_tokens module

```rust
use splintr::mistral_v1_agent_tokens;

// Mistral V1 agent tokens starting at ID 32000
mistral_v1_agent_tokens::SYSTEM                // 32000
mistral_v1_agent_tokens::USER                  // 32001
mistral_v1_agent_tokens::THINK                 // 32005
mistral_v1_agent_tokens::FUNCTION              // 32015
// ... etc
```

### mistral_v2_agent_tokens module

```rust
use splintr::mistral_v2_agent_tokens;

// Mistral V2 agent tokens starting at ID 32768 (after control tokens at 3-9)
mistral_v2_agent_tokens::SYSTEM                // 32768
mistral_v2_agent_tokens::USER                  // 32769
mistral_v2_agent_tokens::THINK                 // 32773
mistral_v2_agent_tokens::FUNCTION              // 32783
// ... etc
```

### mistral_v3_agent_tokens module

```rust
use splintr::mistral_v3_agent_tokens;

// Mistral V3/Tekken agent tokens starting at ID 131072 (after control tokens at 3-9)
mistral_v3_agent_tokens::SYSTEM                // 131072
mistral_v3_agent_tokens::USER                  // 131073
mistral_v3_agent_tokens::THINK                 // 131077
mistral_v3_agent_tokens::FUNCTION              // 131087
// ... etc
```

---

## See Also

- [README.md](../README.md) - Project overview and quick start
- [ByteLevel BPE Encoding](bytelevel_bpe.md) - How DeepSeek V3 encodes bytes to tokens
- [ReAct Paper](https://arxiv.org/abs/2210.03629) - ReAct: Synergizing Reasoning and Acting in Language Models
- [ChatML Specification](https://github.com/openai/openai-python/blob/main/chatml.md) - Chat Markup Language
