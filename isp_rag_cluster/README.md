# Emotion Classification System

## Environment Setup

### .env File Configuration
```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
UPSTAGE_API_KEY=your_upstage_api_key
```

## Running the System

### Basic Execution
```bash
python main_llm.py
```

### Override Configuration
```bash
python main_llm.py model.type=openai model.name=gpt-4
```

## Main Entry Points

### main_llm.py
- Primary entry point for emotion classification system
- Model initialization and execution
- RAG system integration
- Result logging and metrics storage

Key Features:
1. Model initialization (`initialize_models`)
2. Data loading and preprocessing (`load_data`)
3. RAG system initialization (if configured)
4. Batch processing and result storage
5. Performance metrics calculation and reporting

## Recent Major Changes

### 1. RAG System Improvements
- Threshold-based filtering (threshold: 0.1)
- Similarity score normalization
- Enhanced debugging information

### 2. Prompt System Enhancements
- Template-based prompt management
- RAG-integrated prompt support
- Function calling format support

### 3. Logging System Strengthening
- Detailed debug information
- Sample-by-sample result logging
- Improved error handling

## Configuration Options

### Model Settings (`model` section)

1. **Model Type** (`type`)
   - `ollama`: Local models
   - `openai`: OpenAI API
   - `upstage`: Upstage API
   - `anthropic`: Anthropic API 

2. **Prompt Templates** (`template`)
   - `baseline_prompt`: Baseline emotion classification
   - `zero_shot_prompt`:
   - `few_shot_prompt`:
   - `rag_prompt`: RAG-based classification
   - Custom templates per model support

3. **RAG Settings** (`rag` section)
   - `k_examples`: Number of similar examples (default: 7)
   - `threshold`: Similarity threshold (default: 0.1)
   - `embedding_model`: Embedding model configuration
   - `save_db`: Vector DB save option
   - `load_db`: Existing DB load option

### Logging Settings
- `output_path`: Results storage path
- `log_interval`: Intermediate results save interval

## Recommended Configurations


## Performance Metrics
- Accuracy
- Class-wise performance (Precision, Recall, F1)
- Misclassification statistics
- RAG system performance indicators

## Error Handling
- Model initialization failures
- RAG system initialization failures
- API call errors
- Result parsing errors

## Future Improvements
1. Multi-model ensemble
2. RAG system optimization
3. Real-time processing support
4. Batch processing performance enhancement