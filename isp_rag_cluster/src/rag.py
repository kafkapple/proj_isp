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
    
    def get_similar_examples(self, query: str, k: int = 3, threshold: float = 0.5) -> list:
        """유사한 예제 검색"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call create_index first.")
            
        print(f"\n=== RAG Search Debug ===")
        print(f"Query: {query}")
        print(f"k: {k}")
        print(f"threshold: {threshold}")
        print(f"Excluded indices: {len(self.exclude_indices)}")
            
        # 먼저 필터 없이 검색
        try:
            docs_and_scores = self.vectorstore.similarity_search_with_score(
                query, k=k * 2
            )
            
            # 필터링된 결과
            filtered_results = []
            for doc, score in docs_and_scores:
                doc_index = doc.metadata.get('index')
                # 점수가 임계값보다 높고, 제외 인덱스에 없는 경우만 포함
                if score >= threshold and doc_index not in self.exclude_indices:
                    filtered_results.append((doc, score))
               
            
            print(f"Initial results: {len(docs_and_scores)}")
            print(f"Filtered results: {len(filtered_results)}")
            if docs_and_scores:
                print(f"Initial score range: {min(score for _, score in docs_and_scores):.3f} - {max(score for _, score in docs_and_scores):.3f}")
            if filtered_results:
                print(f"Filtered score range: {min(score for _, score in filtered_results):.3f} - {max(score for _, score in filtered_results):.3f}")
            print("=== End RAG Search Debug ===\n")
            
            return filtered_results[:k]
            
        except Exception as e:
            print(f"Error in similarity search: {str(e)}")
            return []

      
    
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