# src/captioning/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Union
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import BartForConditionalGeneration, BartTokenizer
from PIL import Image

class BaseCaptioner(ABC):
    @abstractmethod
    def generate_caption(self, input_data: Any) -> str:
        pass

class AudioCaptioner(BaseCaptioner):
    def __init__(self, config: Dict):
        self.processor = AutoProcessor.from_pretrained("openai/whisper-base")
        self.model = AutoModelForCausalLM.from_pretrained("openai/whisper-base")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def generate_caption(self, audio_tensor: torch.Tensor) -> str:
        inputs = self.processor(
            audio_tensor, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).to(self.device)
        
        generated_ids = self.model.generate(
            inputs["input_features"],
            max_length=100
        )
        
        transcription = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        return transcription

class ImageCaptioner(BaseCaptioner):
    def __init__(self, config: Dict):
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = AutoModelForCausalLM.from_pretrained("Salesforce/blip-image-captioning-base")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def generate_caption(self, image: Union[Image.Image, torch.Tensor]) -> str:
        if isinstance(image, torch.Tensor):
            # Convert tensor to PIL Image if needed
            image = Image.fromarray(
                image.cpu().numpy().transpose(1, 2, 0)
            )
            
        inputs = self.processor(
            images=image, 
            return_tensors="pt"
        ).to(self.device)
        
        generated_ids = self.model.generate(
            **inputs,
            max_length=50
        )
        
        caption = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        return caption

class CaptioningFactory:
    @staticmethod
    def create(modality: str, config: Dict) -> BaseCaptioner:
        if modality == "audio":
            return AudioCaptioner(config)
        elif modality == "image":
            return ImageCaptioner(config)
        else:
            raise ValueError(f"Unknown modality: {modality}")