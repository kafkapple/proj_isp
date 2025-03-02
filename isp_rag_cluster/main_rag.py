from dotenv import load_dotenv
from pathlib import Path

from tqdm import tqdm
import hydra
from omegaconf import DictConfig

from langchain_core.embeddings import Embeddings
from typing import List
0
from src.data.data_manager import DataManager
from src.models.retriever import Retriever
from src.models.generator import Generator
from src.evaluation.evaluator import Evaluator

from openai import OpenAI
from src.utils.path_manager import get_vector_store_path
from hydra.utils import get_original_cwd
from src.utils.lmstudio_utils import get_lmstudio_model_info

# Load environment variables from .env file
load_dotenv()

class LMStudioEmbeddings(Embeddings):
    def __init__(self, base_url: str, api_key: str = "not-needed"):
        self.base_url = base_url
        self.api_key = api_key
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self._model_info = self._get_model_info()
        
    def _get_model_info(self) -> dict:
        """Get current loaded model info from LMStudio"""
        try:
            response = self.client.models.list()
            if response.data:
                model = response.data[0]  # 첫 번째 로드된 모델 사용
                return {
                    "id": model.id,
                    "created": model.created,
                    "object": model.object,
                    "owned_by": model.owned_by
                }
            return {"id": "unknown", "error": "No models available"}
        except Exception as e:
            print(f"Warning: Failed to fetch model info: {e}")
            return {"id": "unknown", "error": str(e)}

    @property
    def model_info(self) -> dict:
        """Return model information"""
        return {
            "provider": "lmstudio",
            "base_url": self.base_url,
            "model": self._model_info
        }
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        print(f"\nGenerating embeddings using LMStudio model: {self._model_info['id']}")
        
        all_embeddings = []
        for text in tqdm(texts, desc="Generating embeddings"):
            text = text.replace("\n", " ")  # OpenAI 권장사항
            response = self.client.embeddings.create(
                input=[text],
                model=self._model_info["id"]
            )
            embedding = response.data[0].embedding
            all_embeddings.append(embedding)
            
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query"""
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(
            input=[text],
            model=self._model_info["id"]
        )
        return response.data[0].embedding

class EmotionAnalyzer:
    def __init__(self, train_df, cfg):
        self.cfg = cfg
        
        # Initialize vector store path only if RAG is enabled
        if cfg.model.use_rag:
            self.index_path = Path(get_original_cwd()) / get_vector_store_path(cfg)
            print(f"\nVector store path: {self.index_path}")
            
            # Get embedding model info
            if cfg.model.provider == "lmstudio":
                embedding_model_info = get_lmstudio_model_info(
                    base_url=cfg.model.lmstudio.base_url,
                    api_key=cfg.model.lmstudio.api_key,
                    model_type="embedding"
                )
            else:  # openai
                embedding_model_info = {
                    "provider": "openai",
                    "model": cfg.model.openai.embedding_model
                }
            
            # Initialize retriever with embedding model info
            self.retriever = self._initialize_retriever(train_df, embedding_model_info)
        
        # Get chat model info
        if cfg.model.provider == "lmstudio":
            chat_model_info = get_lmstudio_model_info(
                base_url=cfg.model.lmstudio.base_url,
                api_key=cfg.model.lmstudio.api_key,
                model_type="chat"
            )
        else:  # openai
            chat_model_info = {
                "provider": "openai",
                "model": cfg.model.openai.chat_model_name
            }
        
        # Initialize generator with chat model info
        self.generator = Generator(cfg, chat_model_info)
        
        # Store model information
        self.model_info = {
            "provider": cfg.model.provider,
            "use_rag": cfg.model.use_rag,
            "embedding": embedding_model_info if cfg.model.use_rag else None,
            "chat_model": chat_model_info
        }
        
        self._print_debug_info()
    
    def _initialize_retriever(self, train_df, embedding_model_info) -> Retriever:
        """Initialize retriever with vector store management"""
        retriever = Retriever(train_df, self.cfg)
        
        # Check vector store path
        expected_path = self.index_path
        print(f"\nChecking vector store at {expected_path}")
        
        try:
            if expected_path.exists():
                print(f"\nLoading existing vector store from {expected_path}")
                retriever.load_vector_store(expected_path)
            else:
                print(f"\nCreating new vector store at {expected_path}")
                retriever.create_vector_store()  # 먼저 생성
                print(f"Saving vector store to {expected_path}")
                retriever.save_vector_store(expected_path)  # 그 다음 저장
        except Exception as e:
            print(f"Error with vector store: {e}")
            print("Creating new vector store...")
            retriever.create_vector_store()
            print("Saving vector store...")
            retriever.save_vector_store(expected_path)
            
        return retriever
    
    def _print_debug_info(self):
        """Print debug information if enabled"""
        if self.cfg.debug.show_generation:
            print("\nModel Configuration:")
            print(f"Provider: {self.model_info['provider']}")
            print(f"RAG: {'enabled' if self.model_info['use_rag'] else 'disabled'}")
            
            # Get embedding model info only if RAG is enabled
            if self.model_info['use_rag']:
                embedding_info = self.model_info['embedding']
                if self.cfg.model.provider == "lmstudio":
                    print(f"Embedding model: {embedding_info.get('id', 'unknown')}")
                else:  # openai
                    print(f"Embedding model: {embedding_info['model']}")
            
            # Get chat model info
            chat_model = self.model_info['chat_model']
            if self.cfg.model.provider == "lmstudio":
                print(f"Chat model: {chat_model.get('id', 'unknown')}\n")
            else:  # openai
                print(f"Chat model: {chat_model['model']}\n")

    def analyze_emotion(self, text: str) -> str:
        if self.cfg.model.use_rag:
            # Use RAG with retrieved context
            retrieved_context = self.retriever.retrieve(text)
            return self.generator.generate(retrieved_context, text)
        else:
            # Direct generation without RAG
            return self.generator.generate(None, text)

    @property
    def embedding_info(self) -> dict:
        """Return embedding model information"""
        if self.cfg.model.use_rag:
            if self.cfg.model.provider == "lmstudio":
                return self.retriever.embeddings.model_info
            else:  # openai
                return {
                    "provider": "openai",
                    "model": self.cfg.model.openai.embedding_model
                }
        return None

    @property
    def chat_model_info(self) -> dict:
        """Return chat model information"""
        if self.cfg.model.provider == "lmstudio":
            return self.generator.get_model_info()
        else:  # openai
            return {
                "provider": "openai",
                "model": self.cfg.model.openai.chat_model_name
            }

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Load and split data
    data_manager = DataManager(cfg)
    train_df, val_df = data_manager.split_data()
    
    # Initialize emotion analyzer
    analyzer = EmotionAnalyzer(train_df, cfg)
    
    # Evaluate (using validation set only)
    evaluator = Evaluator(analyzer, cfg)
    metrics = evaluator.evaluate(val_df)
    
    print("\nValidation Results:")
    for metric_name, value in metrics["overall"].items():
        print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    main()