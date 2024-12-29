# src/models/llava_emotion.py
from typing import Dict, Any
from src.models.base import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

class LLaVAEmotionModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "llava-hf/LLaVA-NeXT-Video-7B-32K-hf",
            torch_dtype=torch.float16
        )
        self.processor = AutoProcessor.from_pretrained(
            "llava-hf/LLaVA-NeXT-Video-7B-32K-hf"
        )
        
    def _process_video(self, video_tensor):
        # Process video frames
        frames = video_tensor.unbind(dim=1)  # Unbind frames
        frame_features = []
        
        for frame in frames:
            inputs = self.processor(
                images=frame,
                return_tensors="pt"
            ).to(self.device)
            frame_features.append(self.model.get_image_features(**inputs))
            
        return torch.stack(frame_features)
        
    def forward(self, batch):
        # Similar to MultiModalLLMEmotionModel but specialized for video
        video_features = self._process_video(batch["video"])
        
        prompt = self._create_prompt(video_features)
        inputs = self.processor(
            text=prompt,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model.generate(**inputs)
        return self._process_outputs(outputs)

# src/models/emotion_llama.py
class EmotionLLaMAModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "ZebangCheng/Emotion-LLaMA",
            torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "ZebangCheng/Emotion-LLaMA"
        )
        
    def _create_prompt(self, text: str) -> str:
        return f"""
        Analyze the emotional content in the following text:
        {text}
        Emotion:
        """
        
    def forward(self, batch):
        # Process text input
        prompts = [
            self._create_prompt(text) 
            for text in batch["text"]
        ]
        
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        outputs = self.model.generate(**inputs)
        return self._process_outputs(outputs)