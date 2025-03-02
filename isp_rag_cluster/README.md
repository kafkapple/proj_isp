# Emotion Classification System

## Configuration Options

### Model Settings (`model` section in `llm.yaml`)

1. **Model Type** (`type`)
   - `ollama`: Local models via Ollama
   - `openai`: OpenAI API models
   - `upstage`: Upstage API models

2. **Function Calling** (`function_calling`)
   - `true`: Uses OpenAI-style function calling format
     - Structured output with strict emotion labels
     - Better for API-based models (GPT, Claude)
   - `false`: Uses LangChain's output parser
     - Better for local models (Llama, Ollama)

3. **Template Usage** (`use_template`)
   - `true`: Uses predefined emotion-specific prompts
     - Model-specific templates (`llama`, `qwen`, etc.)
     - Detailed emotion definitions and examples
   - `false`: Uses simple generic prompt
     - Basic emotion classification instruction

4. **RAG** (`use_rag`)
   - `true`: Uses Retrieval-Augmented Generation
     - Finds similar examples from dataset
     - Provides context for classification
   - `false`: Direct classification without examples

## Common Configuration Combinations

### 1. Local Model Setup
```yaml
model:
  type: "ollama"
  name: "llama2"
  function_calling: false
  use_template: true
  use_rag: false
```
- Best for: Local Llama models
- Uses: LangChain parser + Detailed template

### 2. API Model Setup
```yaml
model:
  type: "openai"
  name: "gpt-3.5-turbo"
  function_calling: true
  use_template: true
  use_rag: false
```
- Best for: OpenAI/API models
- Uses: Function calling + Detailed template

### 3. RAG-Enhanced Setup
```yaml
model:
  type: "ollama"
  name: "llama2"
  function_calling: false
  use_template: false
  use_rag: true
```
- Best for: Any model type
- Uses: Similar examples as context

### 4. Simple Classification Setup
```yaml
model:
  type: "ollama"
  name: "llama2"
  function_calling: false
  use_template: false
  use_rag: false
```
- Best for: Quick testing
- Uses: Basic prompt only

## Processing Flow

1. **Function Calling = true**
   - Structured JSON output via function definition
   - Strict emotion label validation
   - Better error handling

2. **Function Calling = false**
   - LangChain output parser
   - More flexible but less structured
   - Good for simpler models

3. **Template Usage**
   - With template: Detailed emotion guidelines
   - Without template: Simple classification prompt

4. **RAG Usage**
   - With RAG: Similar examples as context
   - Without RAG: Direct classification

## Recommended Combinations

1. **High Accuracy Setup**
```yaml
model:
  type: "openai"
  function_calling: true
  use_template: true
  use_rag: true
```

2. **Fast Local Setup**
```yaml
model:
  type: "ollama"
  function_calling: false
  use_template: true
  use_rag: false
```

3. **Balanced Setup**
```yaml
model:
  type: "ollama"
  function_calling: false
  use_template: false
  use_rag: true
```