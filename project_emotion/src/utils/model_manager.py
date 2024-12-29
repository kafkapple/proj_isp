# src/utils/model_manager.py
import os
from pathlib import Path
from typing import Optional, Dict, Union, Any
import json
import shutil
import logging
from datetime import datetime
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    AutoProcessor,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer
)
from dotenv import load_dotenv
from transformers.utils import logging as tf_logging

class ModelManager:
    """Hugging Face 모델 다운로드 및 관리를 위한 클래스"""
    
    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        self.models_path = self.base_path / "models"
        self.cache_path = self.base_path / "cache"
        self.registry_path = self.base_path / "model_registry.json"
        
        # 필요한 디렉토리 생성
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        tf_logging.set_verbosity_info()
        
        # 레지스트리 로드
        self.registry = self._load_registry()
        
        # Hugging Face 토큰 설정
        load_dotenv()
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if not self.hf_token:
            self.logger.warning("HUGGINGFACE_TOKEN not found in .env file")
            
    def _load_registry(self) -> Dict:
        """모델 레지스트리 로드"""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {}
        
    def _save_registry(self):
        """모델 레지스트리 저장"""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
            
    def _get_model_path(self, model_id: str) -> Path:
        """모델 저장 경로 생성"""
        # model_id에서 유효한 디렉토리 이름 생성
        safe_name = model_id.replace('/', '_')
        return self.models_path / safe_name
        
    def download_model(
        self,
        model_id: str,
        model_type: str = "auto",
        config: Optional[Dict[str, Any]] = None
    ) -> Path:
        """모델 다운로드 또는 로드"""
        save_path = self.models_path / model_id.split('/')[-1]
        
        # 이미 다운로드된 모델인지 확인
        if save_path.exists() and model_id in self.registry:
            self.logger.info(f"Model '{model_id}' already exists at '{save_path}'")
            return save_path
            
        self.logger.info(f"Downloading model '{model_id}' to '{save_path}'...")
        
        try:
            # 기본 설정
            default_config = {
                "torch_dtype": torch.float16,
                "offload_folder": str(self.cache_path / "offload")
            }
            
            # wav2vec2 모델인 경우 특별 처리
            if "wav2vec2" in model_id.lower():
                default_config.pop("device_map", None)  # device_map 설정 제거
            
            # 사용자 설정과 병합
            model_config = {**default_config, **(config or {})}
            
            # wav2vec2 모델인 경우 device_map 설정 제거
            if "wav2vec2" in model_id.lower() and "device_map" in model_config:
                model_config.pop("device_map")
            
            # 모델 타입에 따른 클래스 선택
            model_class = {
                "causal_lm": AutoModelForCausalLM,
                "processor": AutoProcessor,
                "auto": AutoModel
            }.get(model_type, AutoModel)
            
            # 모델, 토크나이저, 설정 다운로드
            model = model_class.from_pretrained(
                model_id,
                use_auth_token=self.hf_token,
                **model_config
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                use_auth_token=self.hf_token
            )
            config = AutoConfig.from_pretrained(
                model_id,
                use_auth_token=self.hf_token
            )
            
            # 모델 저장
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            config.save_pretrained(save_path)
            
            # 레지스트리 업데이트
            self.registry[model_id] = {
                "path": str(save_path),
                "type": model_type,
                "config": model_config
            }
            self._save_registry()
            
            self.logger.info(f"Model '{model_id}' downloaded successfully")
            return save_path
            
        except Exception as e:
            self.logger.error(f"Error downloading model '{model_id}': {e}")
            raise
            
    def load_model(
        self,
        model_id: str,
        **kwargs
    ) -> Union[PreTrainedModel, PreTrainedTokenizer]:
        """
        저장된 모델 로드
        
        Args:
            model_id: 모델 ID 또는 로컬 경로
            **kwargs: from_pretrained에 전달할 추가 인자
        """
        try:
            # 레지스트리에서 모델 정보 확인
            model_info = self.registry.get(model_id)
            if model_info:
                model_path = Path(model_info["local_path"])
                model_type = model_info["model_type"]
            else:
                # 직접 경로가 주어진 경우
                model_path = Path(model_id)
                model_type = kwargs.pop("model_type", "causal_lm")
                
            if not model_path.exists():
                self.logger.info(f"Model not found locally, downloading {model_id}...")
                model_path = self.download_model(model_id, model_type, **kwargs)
                
            # 모델 타입에 따른 로드
            if model_type == "causal_lm":
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    **kwargs
                )
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                return model, tokenizer
            elif model_type == "processor":
                processor = AutoProcessor.from_pretrained(model_path, **kwargs)
                return processor
                
            # 사용 기록 업데이트
            if model_id in self.registry:
                self.registry[model_id]["last_used"] = datetime.now().isoformat()
                self._save_registry()
                
        except Exception as e:
            self.logger.error(f"Error loading model '{model_id}': {e}")
            raise
            
    def remove_model(self, model_id: str):
        """모델 삭제"""
        if model_id in self.registry:
            model_path = Path(self.registry[model_id]["local_path"])
            if model_path.exists():
                shutil.rmtree(model_path)
            del self.registry[model_id]
            self._save_registry()
            self.logger.info(f"Model '{model_id}' removed")
            
    def list_models(self) -> Dict:
        """저장된 모델 목록 반환"""
        return self.registry
        
    def clean_cache(self):
        """캐시 디렉토리 정리"""
        if self.cache_path.exists():
            shutil.rmtree(self.cache_path)
            self.cache_path.mkdir(parents=True)
            self.logger.info("Cache cleaned")