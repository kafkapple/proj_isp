from dotenv import load_dotenv, find_dotenv
import os
import pandas as pd
import json
import re
from ollama import chat, ChatResponse
from anthropic import Anthropic  # Import Anthropic package
from datetime import datetime
load_dotenv()
from pathlib import Path
import hydra
from tqdm import tqdm
from openai import OpenAI
from omegaconf import OmegaConf
from langchain_ollama import OllamaLLM

from typing import Dict, Any

from src.logger import safe_log, setup_logging #, SafeStreamHandler
from src.rag import EmotionRAG
from src.eval import save_metrics, save_prediction_stats
from src.data_prep import load_data, clean_json_response

# Add global flag for prompt logging
_PROMPT_LOGGED = False
_INPUT_LOGGED = False  #

def initialize_models(cfg):
    """Initialize LLM and RAG model"""
    client = None
    
    if cfg.model.type == "ollama":
        client = OllamaLLM(
            model=cfg.model.name,
            temperature=cfg.model.temperature
        )
    elif cfg.model.type == "openai":
        base_url = "https://api.openai.com/v1"
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
    elif cfg.model.type == "upstage":
        base_url = "https://api.upstage.ai/v1/solar"
        api_key = os.getenv("UPSTAGE_API_KEY")
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
    elif cfg.model.type == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        client = Anthropic(api_key=api_key)
    
    return client

def sanitize_model_name(model_name: str) -> str:
    """Clean model name for use as file name"""
    return re.sub(r'[^\w\-\.]', '_', model_name.lower())


def get_output_dir_name(model_name: str, cfg) -> str:
    """Create output folder name based on settings"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_parts = [timestamp, model_name]
    
 
    template_name = str(cfg.model.template)
    if isinstance(template_name, str) and template_name.startswith("${prompt."):
        # Extract the actual template name from the reference
        template_name = template_name.split("${prompt.")[1].rstrip("}")
    name_parts.append(template_name)
    
    return "_".join(name_parts)


def get_model_response(text: str, labels: list, client, model: str, temperature: float, cfg, logger=None, rag=None, similar_examples=None, tools=None, count=0) -> dict:
    """Get model response with RAG or template"""
    global _PROMPT_LOGGED
  
    final_prompt = ""  
    template = ""      
    
    try:
     
        try:
            template_name = cfg.model.template
            
            if logger:
                logger.debug("Using template: {}".format(template_name))

            if template_name == "rag_prompt":
                try:
                    if rag is None:
                        raise ValueError("RAG is enabled but rag object is None")
                    if similar_examples is None:
                        raise ValueError("similar_examples is None")
                    final_prompt = str(rag.get_rag_prompt(text, similar_examples))
                    if logger:
                        logger.debug(f"Using RAG prompt with {len(similar_examples)} examples (threshold: {cfg.rag.threshold})")
                except Exception as e:
                    if logger:
                        logger.error("Error in RAG processing: {}".format(str(e)))
            else:
                template = getattr(cfg.prompt, template_name)
                final_prompt = template
        except Exception as e:
            if logger:
                logger.error("Error in template processing: {}".format(str(e)))
    
            # template = "basic_prompt"
            # final_prompt = "Classify the emotion in this text as one of: {}\n\nText: {}".format(', '.join(labels), text)

        final_prompt = "{}\n\nText: {}".format(final_prompt, text)
        
        if logger and count == 0:
            try:
                safe_prompt = final_prompt.encode('ascii', errors='replace').decode('ascii')
                logger.debug("Using prompt: \n{}".format(safe_prompt))
            
            except Exception as e:
                logger.error("Error logging prompt: {}".format(str(e)))

        try:
            if cfg.model.type == "ollama":
                response = client.invoke(final_prompt)
            elif cfg.model.type == "anthropic":
                response = client.messages.create(
                    model=model,
                    max_tokens=cfg.model.max_tokens,
                    messages=[{"role": "user", "content": final_prompt}],
                    temperature=temperature
                )
                response = response.content[0].text
            else:
                messages = [{"role": "system", "content": final_prompt}]
                use_function_call = any(model_name.lower() in model.lower() 
                                      for model_name in cfg.prompt.function_call_models)
                
                if use_function_call and cfg.model.function_calling:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        tools=tools,
                        seed=int(cfg.general.seed)
                    )
                    if response.choices[0].message.tool_calls:
                        tool_call = response.choices[0].message.tool_calls[0]
                        result = json.loads(tool_call.function.arguments)
                        if logger:
                            try:
                                logger.debug("Function call result: {}".format(
                                    json.dumps(result, ensure_ascii=True, indent=2)
                                ))
                            except Exception as e:
                                logger.error("Error logging function call result: {}".format(str(e)))
                        return result, final_prompt
                else:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature
                    )
                    response = response.choices[0].message.content

            if logger:
                try:
                    safe_response = str(response).encode('ascii', errors='replace').decode('ascii')
                    logger.debug("Raw response: \n{}".format(safe_response))
                except Exception as e:
                    logger.error("Error logging response: {}".format(str(e)))
                    
            result = clean_json_response(response)
            if isinstance(result, str):
                result = json.loads(result)

            required_fields = ["emotion", "confidence_score", "explanation"]
            if not all(field in result for field in required_fields):
                if logger:
                    logger.warning("Missing required fields in response, retrying with structured prompt...")
                return retry_emotion_prediction(text, labels, client, model, temperature, int(cfg.general.seed), cfg), final_prompt

            if result["emotion"] not in labels:
                if logger:
                    logger.warning("Invalid emotion label '{}', retrying with structured prompt...".format(
                        str(result["emotion"]).encode('ascii', errors='replace').decode('ascii')
                    ))
                return retry_emotion_prediction(text, labels, client, model, temperature, int(cfg.general.seed), cfg), final_prompt

            return result, final_prompt

        except json.JSONDecodeError:
            if logger:
                logger.warning("JSON parsing failed, retrying with structured prompt...")
            return retry_emotion_prediction(text, labels, client, model, temperature, int(cfg.general.seed), cfg), final_prompt

        except Exception as e:
            if logger:
                logger.error("Error generating response: {}".format(str(e)))
            raise

    except Exception as e:
        if logger:
            logger.error("Error in get_model_response: {}".format(str(e)))
        return retry_emotion_prediction(text, labels, client, model, temperature, int(cfg.general.seed), cfg), final_prompt


def retry_emotion_prediction(text: str, labels: list, client, model: str, temperature: float, seed: int, cfg: dict) -> dict:
    """Retry emotion prediction attempt"""

    use_function_call = any(model_name.lower() in model.lower() for model_name in cfg.prompt.function_call_models)
    
    try:
        if use_function_call and cfg.model.function_calling:
            tools = [{
                "type": "function",
                "function": {
                    "name": "search",
                    "description": f"Strictly classify the emotion as one of these ONLY: {', '.join(labels)}",
                    "parameters": {
                        "properties": {
                            "emotion": {
                                "type": "string",
                                "enum": labels,
                                "description": f"Must be one of: {', '.join(labels)}"
                            },
                            "confidence_score": {
                                "type": "number",
                                "description": "confidence score of the emotion: 0.0-1.0"
                            },
                            "explanation": {
                                "type": "string",
                                "description": "explain why the emotion is decided"
                            },
                        },
                        "required": ["emotion", "confidence_score", "explanation"]
                    }
                }
            }]
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"You MUST classify the emotion as one of these ONLY: {', '.join(labels)}. Do not use any other emotion words."},
                    {"role": "user", "content": text}
                ],
                tools=tools,
                seed=seed,
                temperature=temperature,
                timeout=10
            )
            
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                return json.loads(tool_call.function.arguments)
        else:
            # Function calling을 지원하지 않는 모델을 위한 일반 응답
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"You MUST classify the emotion as one of these ONLY: {', '.join(labels)}. Do not use any other emotion words."},
                    {"role": "user", "content": text}
                ],
                temperature=temperature,
                timeout=10
            )
            return clean_json_response(response.choices[0].message.content)
            
    except Exception as e:
        print(f"Error in retry prediction: {e}")
    
    return {"emotion": "unknown", "confidence_score": 0.0, "explanation": "Failed to get valid prediction"}


@hydra.main(version_base="1.2", config_path='config', config_name='llm')
def main(cfg):
    # 전역 변수 사용 선언
    global _INPUT_LOGGED
    
    # Initialize models
    client = initialize_models(cfg)
    
    # Convert config values to Python basic types first
    model = str(cfg.model.name)
    temperature = float(cfg.model.temperature)
    
    # Set up output directory first
    model_name = sanitize_model_name(model)
    
    output_dir_name = get_output_dir_name(model_name, cfg)
    output_dir = Path('outputs') / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Update cfg output_path
    cfg.general.output_path = str(output_dir)
    
    # Now setup logging with the created output directory
    logger = setup_logging(cfg, output_dir)
    
    # 데이터 로딩 및 전처리
    df_isear, labels = load_data(cfg, logger)
    labels = list(map(str, labels))
    
    # Set n_samples
    n_samples = len(df_isear) if cfg.data.n_samples == -1 else min(cfg.data.n_samples, len(df_isear))
    logger.info(f"Using sanitized model name: {model_name}")
    logger.info(f"Processing {n_samples} samples out of {len(df_isear)} total samples")
    
    # Select first n_samples
    df_isear = df_isear.head(n_samples)
    
    # Set output file path (directory already created)
    output_path = Path(cfg.general.output_path) / f'dataset-{cfg.data.name}_model-{model_name}.csv'
    
    # Function calling tool definition
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Predict the emotion of the text",
                "parameters": {
                    "properties": {
                        "emotion": {
                            "type": "string",
                            "enum": list(labels),
                            "description": f"Classify the emotion of the text as one of given labels: {', '.join(labels)}"
                        },
                        "confidence_score": {
                            "type": "number",
                            "description": "confidence score of the emotion"
                        },
                        "explanation": {
                            "type": "string",
                            "description": "explain why the emotion is decided"
                        },
                    },
                    "required": ["emotion", "confidence_score", "explanation"],
                    "type": "object"
                }
            }
        }
    ]

    # Initialize invalid prediction statistics
    invalid_predictions = {
        "total": 0,
        "invalid_samples": [],
        "by_true_label": {str(label): 0 for label in labels}
    }

    # Initialize RAG
    rag = None
    try:
        if cfg.model.template == "rag_prompt":
            logger.info("Initializing RAG...")
            logger.info(f"RAG Embeding Model: {cfg.rag.embedding_model}")
            try:
                rag = EmotionRAG(cfg)
                rag.create_index(df_isear)
                logger.info("RAG initialization completed")
            except Exception as e:
                logger.error(f"Error initializing RAG: {str(e)}")
                logger.warning("Continuing without RAG...")
    except Exception as e:
        logger.error(f"Error initializing RAG: {str(e)}")
        logger.warning("Continuing without RAG...")

    # 데이터 정제 후에 tqdm을 사용하도록 수정
    total_samples = len(df_isear)  # 실제 처리할 샘플 수
    
    # 로깅을 위한 구분자 정의
    log_separator = "="*80
    
    for index, row in tqdm(df_isear.iterrows(), total=total_samples):
        try:
            similar_examples = None
            if rag:
                rag.exclude_index(index)
                similar_examples = rag.get_similar_examples(
                    str(row.text),
                    query_emotion=str(row.emotion),
                    k=cfg.rag.k_examples,
                    threshold=cfg.rag.threshold
                )
                if logger:
                    logger.debug(f"Using RAG prompt with {len(similar_examples)} examples (threshold: {cfg.rag.threshold})")

            result, _ = get_model_response(
                str(row.text),
                labels,
                client,
                model,
                temperature,
                cfg,
                logger=logger,
                rag=rag,
                similar_examples=similar_examples,
                tools=tools,
                count=index
            )
            
            df_isear.at[index, f'predicted_emotion_{model_name}'] = result["emotion"]
            df_isear.at[index, f'confidence_score_{model_name}'] = result["confidence_score"]
            df_isear.at[index, f'explanation_{model_name}'] = result["explanation"]
            
            # 예측 결과 로깅 (매 샘플마다)
            result_log = f"""
{log_separator}
[Sample Index: {index}] Prediction Result
{log_separator}
Input Text: {str(row.text).encode('ascii', errors='ignore').decode('ascii')}
Ground Truth: {str(row.emotion).encode('ascii', errors='ignore').decode('ascii')}
Predicted Emotion: {str(result['emotion']).encode('ascii', errors='ignore').decode('ascii')}
Confidence Score: {float(result['confidence_score'])}
Explanation: {str(result.get('explanation', '')).encode('ascii', errors='ignore').decode('ascii')}
{log_separator}
"""
            safe_log(logger, 'info', result_log)
            
            if not _INPUT_LOGGED:
                input_log = f"""
{log_separator}
[Initial Sample] Prompt Details

"""
                safe_log(logger, 'debug', input_log)
                _INPUT_LOGGED = True

            if index % cfg.general.logging.log_interval == 0:
                df_isear.to_csv(output_path, index=False)

        except Exception as e:
            print(f"Error processing row {index}: {e}")
           
            df_isear.at[index, f'predicted_emotion_{model_name}'] = "unknown"
            df_isear.at[index, f'confidence_score_{model_name}'] = 0.0
            df_isear.at[index, f'explanation_{model_name}'] = f"Error: {str(e)}"
            
            error_log = f"""
{log_separator}
[Error at Sample Index: {index}]
{log_separator}
Input Text: {str(row.text).encode('ascii', errors='ignore').decode('ascii')}
Ground Truth: {str(row.emotion).encode('ascii', errors='ignore').decode('ascii')}
Error: {str(e)}
{log_separator}
"""
            safe_log(logger, 'error', error_log)
            continue

    # Final result save
    df_isear.to_csv(output_path, index=False)
    
    # 잘못 분류된 샘플 찾기
    mask = df_isear['emotion'] != df_isear[f'predicted_emotion_{model_name}']
    df_misclassified = df_isear[mask].copy()
    df_misclassified.sort_values(by='emotion', ascending=False, inplace=True)
    misclassified_counts = df_misclassified['emotion'].value_counts()
    logger.info(misclassified_counts)

    # Save misclassified samples
    df_misclassified.to_csv(output_dir / f'misclassified_samples_{model_name}.csv', index=False)

    # Save invalid prediction statistics
    invalid_stats = {
        "total_samples": n_samples,
        "invalid_count": invalid_predictions["total"],
        "invalid_percentage": (invalid_predictions["total"] / n_samples) * 100,
        "by_true_label": invalid_predictions["by_true_label"],
        "invalid_samples": invalid_predictions["invalid_samples"]
    }
    
    print(f"\nInvalid prediction statistics:")
    print(f"Total invalid: {invalid_stats['invalid_count']}")
    print(f"Invalid percentage: {invalid_stats['invalid_percentage']:.2f}%")
    print(f"Number of invalid samples collected: {len(invalid_stats['invalid_samples'])}")
    
    # Save statistics
    save_prediction_stats(invalid_stats, output_dir, model_name, labels)
    
    # Calculate and save evaluation metrics
    report, cm = save_metrics(df_isear, cfg, model_name, output_dir)

    # Save RAG search results and analyze performance if available
    if rag and hasattr(rag, 'search_results'):
        # Save search results
        rag_results_path = output_dir / f'rag_search_results_{model_name}.csv'
        rag.save_search_results(str(rag_results_path))
        logger.info(f"Saved RAG search results to {rag_results_path}")
        
        # Analyze retrieval performance
        rag_stats = rag.analyze_retrieval_performance(output_dir)
        
        # Add RAG performance to final log if stats are available
        if rag_stats:
            rag_log = "\n4. RAG Retrieval Performance:"
            rag_log += f"\n- Overall accuracy: {rag_stats['overall']['accuracy']:.2%}"
            rag_log += f"\n- Total queries: {rag_stats['overall']['total_queries']}"
            rag_log += f"\n- Total retrievals: {rag_stats['overall']['total_retrievals']}"
            rag_log += "\n\nClass-wise RAG Performance:"
            for emotion, stats in rag_stats['by_class'].items():
                rag_log += f"\n{emotion}:"
                rag_log += f"\n  - Accuracy: {stats['accuracy']:.2%}"
                rag_log += f"\n  - Queries: {stats['total_queries']}"
                rag_log += f"\n  - Correct matches: {stats['correct_matches']}"
        else:
            logger.warning("No RAG statistics available for analysis")
            rag_log = "\n4. RAG Retrieval Performance: No data available"
    else:
        rag_log = ""

    # Final results log creation
    final_log = f"""
{'='*80}
[Final Results Summary]
{'='*80}

1. Wrong Prediction Statistics:
- Total samples: {invalid_stats['total_samples']}
- Wrong predictions: {invalid_stats['invalid_count']}
- Wrong prediction percentage: {invalid_stats['invalid_percentage']:.2f}%

2. Classification Performance:
- Overall accuracy: {report['overall']['accuracy']:.4f}

3. Class-wise Performance:"""

    # Add per-class performance metrics
    for label, metrics in report['per_class'].items():
        final_log += f"""
{label}:
  - Precision: {metrics['precision']:.4f}
  - Recall: {metrics['recall']:.4f}
  - F1 Score: {metrics['f1-score']:.4f}"""

    final_log += rag_log
    final_log += f"\n{'='*80}\n"
    
    # Log to file
    safe_log(logger, 'info', final_log)
    
    print("\nClassification report summary:")
    print(f"Accuracy: {report['overall']['accuracy']:.4f}")
    print("\nClass-wise metrics:")
    for label, metrics in report['per_class'].items():
        print(f"{label}:")
        print(f"  precision: {metrics['precision']:.4f}")
        print(f"  recall: {metrics['recall']:.4f}")
        print(f"  f1-score: {metrics['f1-score']:.4f}")


if __name__ == "__main__":
    main()

    
