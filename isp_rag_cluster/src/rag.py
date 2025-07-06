from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from pathlib import Path
import pickle
import json

class EmotionRAG:
    def __init__(self, cfg):
        self.cfg = cfg
        self.db_path = Path(cfg.rag.db_path)
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=cfg.rag.embedding_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        self.vectorstore = None
        self.exclude_indices = set()
        
        # 감정 분석 특화 RAG 템플릿 정의
        self.prompt_template = """
        ## Instructions
        You are an expert in emotion analysis.
        Classify the primary emotion in the given text strictly as one of: 'joy', 'fear', 'anger', 'sadness', 'disgust', 'shame', 'guilt'.

        ## Emotion Definitions:
        - joy: happiness, pleasure, delight.
        - fear: anxiety, dread regarding danger or uncertainty.
        - anger: frustration, rage in response to injustice.
        - sadness: sorrow, grief, disappointment.
        - disgust: revulsion, strong dislike.
        - shame: embarrassment, humiliation.
        - guilt: remorse, regret for wrongdoing.
        
        Do not use any other emotions.
        
        ## Retrieved Context:
        Consider the following similar texts and their emotion classifications:
        {examples}


        ## Output Format
        Respond ONLY with a JSON object in this format:
        {
        "emotion": "emotion_name",
        "confidence_score": value_between_0_and_1,
        "explanation": "brief_explanation"
        }
        """
            
        
        # self.prompt_template = PromptTemplate(
        #     input_variables=["examples", "text"],
        #     template=self.emotion_template
        # )
    
    def create_index(self, df):
        """벡터 DB 생성 또는 로드"""
        embedding_model = str(self.cfg.rag.embedding_model)
        embedding_model = embedding_model.split("/")[-1]

        db_file = self.db_path / f"{self.cfg.data.name}_{embedding_model}_vectorstore.pkl"
        
        # 기존 DB 로드 시도
        if self.cfg.rag.load_db and db_file.exists():
            print(f"Loading existing vector DB from {db_file}")
            try:
                with open(db_file, 'rb') as f:
                    self.vectorstore = pickle.load(f)
                return
            except Exception as e:
                print(f"Error loading vector DB: {e}")
        
        # 새로운 DB 생성
        print("Creating new vector DB...")
        texts = df['text'].tolist()
        metadatas = [
            {
                'emotion': row.emotion,
                'index': idx
            } for idx, row in df.iterrows()
        ]
        
        self.vectorstore = FAISS.from_texts(
            texts,
            self.embeddings,
            metadatas=metadatas
        )
        
        # DB 저장
        if self.cfg.rag.save_db:
            print(f"Saving vector DB to {db_file}")
            self.db_path.mkdir(parents=True, exist_ok=True)
            with open(db_file, 'wb') as f:
                pickle.dump(self.vectorstore, f)
    
    def get_similar_examples(self, query: str, query_emotion: str = None, k: int = 3, threshold: float = 0.5) -> list:
        """유사한 예제 검색"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call create_index first.")
            
        print(f"\n=== RAG Search Debug ===")
        print(f"Query: {query}")
        print(f"Query emotion: {query_emotion}")
        print(f"k: {k}")
        print(f"threshold: {threshold}")
        print(f"Excluded indices: {self.exclude_indices}")
            
        # 먼저 필터 없이 검색
        try:
            docs_and_scores = self.vectorstore.similarity_search_with_score(
                query, k=k * 2
            )
            
            # 필터링된 결과
            filtered_results = []
            search_results = []  # CSV 저장용 결과
            
            for doc, score in docs_and_scores:
                doc_index = doc.metadata.get('index')
                # 점수가 임계값보다 높고, 제외 인덱스에 없는 경우만 포함
                if score >= threshold and doc_index not in self.exclude_indices:
                    filtered_results.append((doc, score))
                    # CSV 저장용 결과 추가
                    search_results.append({
                        'query_text': query,
                        'query_emotion': query_emotion,
                        'similar_text': doc.page_content,
                        'retrieved_emotion': doc.metadata.get('emotion', ''),
                        'similarity_score': score,
                        'index': doc_index,
                        'emotion_match': 1 if query_emotion == doc.metadata.get('emotion', '') else 0
                    })
            
            print(f"Initial results: {len(docs_and_scores)}")
            print(f"Filtered results: {len(filtered_results)}")
            if docs_and_scores:
                print(f"Initial score range: {min(score for _, score in docs_and_scores):.3f} - {max(score for _, score in docs_and_scores):.3f}")
            if filtered_results:
                print(f"Filtered score range: {min(score for _, score in filtered_results):.3f} - {max(score for _, score in filtered_results):.3f}")
                
                # 감정 클래스 일치 분석
                if query_emotion:
                    matches = sum(1 for doc, _ in filtered_results[:k] if doc.metadata.get('emotion') == query_emotion)
                    print(f"Emotion class match rate: {matches/len(filtered_results[:k]):.2%}")
                    
            print("=== End RAG Search Debug ===\n")
            
            # 검색 결과 저장
            if hasattr(self, 'search_results'):
                self.search_results.extend(search_results[:k])
            else:
                self.search_results = search_results[:k]
            
            return filtered_results[:k]
            
        except Exception as e:
            print(f"Error in similarity search: {str(e)}")
            return []

    def analyze_retrieval_performance(self, output_dir: str):
        """검색 결과의 감정 클래스 매칭 분석"""
        if not hasattr(self, 'search_results') or not self.search_results:
            print("No search results available for analysis")
            return None
            
        import pandas as pd
        import numpy as np
        
        try:
            df = pd.DataFrame(self.search_results)
            
            # 전체 검색 성능 분석
            total_matches = df['emotion_match'].sum()
            total_queries = len(df['query_text'].unique())
            total_retrievals = len(df)
            
            # 클래스별 성능 분석
            class_performance = {}
            for emotion in df['query_emotion'].unique():
                emotion_df = df[df['query_emotion'] == emotion]
                matches = emotion_df['emotion_match'].sum()
                total = len(emotion_df)
                accuracy = matches / total if total > 0 else 0
                
                class_performance[emotion] = {
                    'total_queries': len(emotion_df['query_text'].unique()),
                    'total_retrievals': total,
                    'correct_matches': matches,
                    'accuracy': accuracy
                }
            
            # 결과 저장
            performance_stats = {
                'overall': {
                    'total_queries': total_queries,
                    'total_retrievals': total_retrievals,
                    'correct_matches': total_matches,
                    'accuracy': total_matches / total_retrievals if total_retrievals > 0 else 0
                },
                'by_class': class_performance
            }
            
            # 통계 저장
            import json
            stats_path = Path(output_dir) / 'rag_retrieval_stats.json'
            with open(stats_path, 'w') as f:
                json.dump(performance_stats, f, indent=2)
                
            # 상세 분석 결과 출력
            print("\nRAG Retrieval Performance Analysis")
            print("="*50)
            print(f"Overall accuracy: {performance_stats['overall']['accuracy']:.2%}")
            print(f"Total queries: {total_queries}")
            print(f"Total retrievals: {total_retrievals}")
            print("\nClass-wise Performance:")
            for emotion, stats in class_performance.items():
                print(f"\n{emotion}:")
                print(f"  Accuracy: {stats['accuracy']:.2%}")
                print(f"  Queries: {stats['total_queries']}")
                print(f"  Retrievals: {stats['total_retrievals']}")
                print(f"  Correct matches: {stats['correct_matches']}")
                
            return performance_stats
            
        except Exception as e:
            print(f"Error analyzing retrieval performance: {str(e)}")
            return None

    def save_search_results(self, output_path: str):
        """검색 결과를 CSV 파일로 저장"""
        if hasattr(self, 'search_results') and self.search_results:
            import pandas as pd
            df = pd.DataFrame(self.search_results)
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"Saved search results to {output_path}")
            # 저장 후 초기화
            self.search_results = []

    def get_rag_prompt(self, query: str, similar_examples: list) -> str:
        """RAG 프롬프트 생성"""
        try:
            print("\n=== Debug Info ===")
            print(f"Query: {query}")
            print(f"Number of similar examples: {len(similar_examples)}")
            # 유사 예제가 있을 때만 첫 번째 예제 정보 출력
            if similar_examples:
                print(f"First example type: {type(similar_examples[0])}")
                print(f"First example content: {similar_examples[0]}")
            else:
                print("No similar examples found")
            # 유사 예제 JSON 포맷팅
            formatted_examples = []
            if similar_examples:
                for idx, example_tuple in enumerate(similar_examples, 1):
                    print(f"\nProcessing example {idx}:")
                    doc, score = example_tuple  # Document 객체와 점수 분리
                    print(f"Document content: {doc.page_content}")
                    print(f"Document metadata: {doc.metadata}")
                    print(f"Similarity score: {score}")
                    
                    confidence = round(1 - (score / 2), 2)  # 유사도 점수를 0-1 범위의 신뢰도로 변환
                    example = f"Example {idx}:\n"
                    example += f"Text: {str(doc.page_content).strip()}\n"
                    example += f"Emotion: {str(doc.metadata['emotion']).strip()}\n"
                    example += f"Confidence: {confidence:.2f}\n"
                    formatted_examples.append(example)
                    print(f"Formatted example {idx}:\n{example}")
                
            # JSON 문자열로 변환하고 들여쓰기 적용
            # examples_text = json.dumps(formatted_examples, indent=2, ensure_ascii=False)
             # 예시들을 하나의 문자열로 결합
            
            examples_text = "\n".join(formatted_examples) if formatted_examples else "No similar examples found."
            print("\nCombined examples text:")
            print(examples_text)
            final_prompt = self.prompt_template.replace("{examples}", examples_text)
            print("\nFinal prompt:")
            print(final_prompt)
            print("=== End Debug Info ===\n")
            
            return final_prompt
            # # 프롬프트 생성
            # prompt = self.prompt_template.format(
            #     examples=examples_text,
            #     text=str(query).strip()
            # )
            
            # return prompt
            
        except Exception as e:
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            raise ValueError(f"Error formatting RAG prompt: {str(e)}") 
    def exclude_index(self, idx: int):
        """특정 인덱스 제외 목록에 추가"""
        self.exclude_indices.add(idx) 