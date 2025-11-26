# Special Tokens Reference

This document describes the special tokens available in Splintr's `cl100k_base`, `o200k_base`, and `llama3` tokenizers, including the extended agent token vocabulary.

## Table of Contents

- [Overview](#overview)
- [Design Rationale](#design-rationale)
- [Token ID Allocation](#token-id-allocation)
- [OpenAI Standard Tokens](#openai-standard-tokens)
- [Meta Llama 3 Standard Tokens](#meta-llama-3-standard-tokens)
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

Splintr extends the standard OpenAI tokenizer vocabularies with **54 additional special tokens** designed for building modern AI agent systems. These tokens provide semantic structure for:

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

| Model         | Regular Tokens | Reserved Range    | Agent Tokens    | Total   |
| ------------- | -------------- | ----------------- | --------------- | ------- |
| `cl100k_base` | 0-100,255      | 100,257-100,276   | 100,277-100,330 | 100,331 |
| `o200k_base`  | 0-199,997      | 199,999-200,018   | 200,019-200,072 | 200,073 |
| `llama3`      | 0-127,999      | 128,000-128,261   | 128,300-128,353 | 128,354 |

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

| Token                    | ID     | Purpose                        |
| ------------------------ | ------ | ------------------------------ |
| `<\|begin_of_text\|>`    | 128000 | Beginning of sequence          |
| `<\|end_of_text\|>`      | 128001 | End of sequence                |
| `<\|start_header_id\|>`  | 128006 | Start of role header           |
| `<\|end_header_id\|>`    | 128007 | End of role header             |
| `<\|eot_id\|>`           | 128009 | End of turn                    |

#### Added in Llama 3.1

| Token                         | ID     | Purpose                        |
| ----------------------------- | ------ | ------------------------------ |
| `<\|finetune_right_pad_id\|>` | 128004 | Padding token for fine-tuning  |
| `<\|eom_id\|>`                | 128008 | End of message (tool use)      |
| `<\|python_tag\|>`            | 128010 | Code interpreter marker        |

#### Added in Llama 3.2-Vision

| Token           | ID     | Purpose                        |
| --------------- | ------ | ------------------------------ |
| `<\|step_id\|>` | 128005 | Step identifier for vision     |
| `<\|image\|>`   | 128256 | Image content placeholder      |

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

## Agent Token Categories

### 1. Conversation Structure

**Purpose**: Standard ChatML-style tokens for multi-turn conversations.

| Token             | cl100k ID | o200k ID | llama3 ID | Description                                     |
| ----------------- | --------- | -------- | --------- | ----------------------------------------------- |
| `<\|system\|>`    | 100277    | 200019   | 128300    | System instructions defining assistant behavior |
| `<\|user\|>`      | 100278    | 200020   | 128301    | User input/queries                              |
| `<\|assistant\|>` | 100279    | 200021   | 128302    | Assistant responses                             |
| `<\|im_start\|>`  | 100280    | 200022   | 128303    | Generic message start (ChatML)                  |
| `<\|im_end\|>`    | 100281    | 200023   | 128304    | Generic message end (ChatML)                    |

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

| Token          | cl100k ID | o200k ID | llama3 ID | Description              |
| -------------- | --------- | -------- | --------- | ------------------------ |
| `<\|think\|>`  | 100282    | 200024   | 128305    | Start of reasoning block |
| `<\|/think\|>` | 100283    | 200025   | 128306    | End of reasoning block   |

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

| Token            | cl100k ID | o200k ID | llama3 ID | Description                     |
| ---------------- | --------- | -------- | --------- | ------------------------------- |
| `<\|plan\|>`     | 100284    | 200026   | 128307    | High-level strategy formulation |
| `<\|/plan\|>`    | 100285    | 200027   | 128308    | End of plan                     |
| `<\|step\|>`     | 100286    | 200028   | 128309    | Individual step within plan     |
| `<\|/step\|>`    | 100287    | 200029   | 128310    | End of step                     |
| `<\|act\|>`      | 100288    | 200030   | 128311    | Action intent declaration       |
| `<\|/act\|>`     | 100289    | 200031   | 128312    | End of action                   |
| `<\|observe\|>`  | 100290    | 200032   | 128313    | Environment feedback            |
| `<\|/observe\|>` | 100291    | 200033   | 128314    | End of observation              |

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

| Token             | cl100k ID | o200k ID | llama3 ID | Description                 |
| ----------------- | --------- | -------- | --------- | --------------------------- |
| `<\|function\|>`  | 100292    | 200034   | 128315    | Function call specification |
| `<\|/function\|>` | 100293    | 200035   | 128316    | End of function call        |
| `<\|result\|>`    | 100294    | 200036   | 128317    | Successful return value     |
| `<\|/result\|>`   | 100295    | 200037   | 128318    | End of result               |
| `<\|error\|>`     | 100296    | 200038   | 128319    | Execution error             |
| `<\|/error\|>`    | 100297    | 200039   | 128320    | End of error                |

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

| Token           | cl100k ID | o200k ID | llama3 ID | Description           |
| --------------- | --------- | -------- | --------- | --------------------- |
| `<\|code\|>`    | 100298    | 200040   | 128321    | Code block to execute |
| `<\|/code\|>`   | 100299    | 200041   | 128322    | End of code block     |
| `<\|output\|>`  | 100300    | 200042   | 128323    | Execution output      |
| `<\|/output\|>` | 100301    | 200043   | 128324    | End of output         |
| `<\|lang\|>`    | 100302    | 200044   | 128325    | Language identifier   |
| `<\|/lang\|>`   | 100303    | 200045   | 128326    | End of language tag   |

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

| Token            | cl100k ID | o200k ID | llama3 ID | Description             |
| ---------------- | --------- | -------- | --------- | ----------------------- |
| `<\|context\|>`  | 100304    | 200046   | 128327    | Retrieved context block |
| `<\|/context\|>` | 100305    | 200047   | 128328    | End of context          |
| `<\|quote\|>`    | 100306    | 200048   | 128329    | Direct quotation        |
| `<\|/quote\|>`   | 100307    | 200049   | 128330    | End of quote            |
| `<\|cite\|>`     | 100308    | 200050   | 128331    | Citation reference      |
| `<\|/cite\|>`    | 100309    | 200051   | 128332    | End of citation         |
| `<\|source\|>`   | 100310    | 200052   | 128333    | Source metadata         |
| `<\|/source\|>`  | 100311    | 200053   | 128334    | End of source           |

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

| Token           | cl100k ID | o200k ID | llama3 ID | Description         |
| --------------- | --------- | -------- | --------- | ------------------- |
| `<\|memory\|>`  | 100312    | 200054   | 128335    | Store information   |
| `<\|/memory\|>` | 100313    | 200055   | 128336    | End of memory block |
| `<\|recall\|>`  | 100314    | 200056   | 128337    | Retrieved memory    |
| `<\|/recall\|>` | 100315    | 200057   | 128338    | End of recall       |

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

| Token        | cl100k ID | o200k ID | llama3 ID | Description                 |
| ------------ | --------- | -------- | --------- | --------------------------- |
| `<\|pad\|>`  | 100316    | 200058   | 128339    | Padding for batch alignment |
| `<\|stop\|>` | 100317    | 200059   | 128340    | Generation stop signal      |
| `<\|sep\|>`  | 100318    | 200060   | 128341    | Segment separator           |

**Rationale**: These are utility tokens for training and inference:

- **pad**: Aligns sequences in batches (has no semantic meaning)
- **stop**: Alternative to `<|endoftext|>` for stopping generation
- **sep**: Separates segments without implying document boundaries

---

### 9. Multimodal

**Purpose**: Placeholders for non-text content.

| Token          | cl100k ID | o200k ID | llama3 ID | Description   |
| -------------- | --------- | -------- | --------- | ------------- |
| `<\|image\|>`  | 100319    | 200061   | 128256*   | Image content |
| `<\|/image\|>` | 100320    | 200062   | 128257    | End of image  |
| `<\|audio\|>`  | 100321    | 200063   | 128258    | Audio content |
| `<\|/audio\|>` | 100322    | 200064   | 128259    | End of audio  |
| `<\|video\|>`  | 100323    | 200065   | 128260    | Video content |
| `<\|/video\|>` | 100324    | 200066   | 128261    | End of video  |

*Note: Llama 3's `<|image|>` token (128256) is aligned with the official Meta Llama 3.2-Vision token ID for compatibility.

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

| Token            | cl100k ID | o200k ID | llama3 ID | Description            |
| ---------------- | --------- | -------- | --------- | ---------------------- |
| `<\|title\|>`    | 100325    | 200067   | 128348    | Document/section title |
| `<\|/title\|>`   | 100326    | 200068   | 128349    | End of title           |
| `<\|section\|>`  | 100327    | 200069   | 128350    | Semantic section       |
| `<\|/section\|>` | 100328    | 200070   | 128351    | End of section         |
| `<\|summary\|>`  | 100329    | 200071   | 128352    | Content summary        |
| `<\|/summary\|>` | 100330    | 200072   | 128353    | End of summary         |

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

---

## See Also

- [README.md](../README.md) - Project overview and quick start
- [ReAct Paper](https://arxiv.org/abs/2210.03629) - ReAct: Synergizing Reasoning and Acting in Language Models
- [ChatML Specification](https://github.com/openai/openai-python/blob/main/chatml.md) - Chat Markup Language
