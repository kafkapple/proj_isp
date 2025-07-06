# src/models/llm_emotion.py
from typing import Dict, List, Any
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import BaseModel

class MultiModalLLMEmotionModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.llm_path,
            use_fast=False
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.model.llm_path,
            torch_dtype=torch.float16
        )
        
        self.audio_captioner = CaptioningFactory.create("audio", config)
        self.image_captioner = CaptioningFactory.create("image", config)
        
        self.emotion_classes = [
            "angry", "disgust", "fear", "happy", 
            "sad", "surprise", "neutral"
        ]
        
    def _create_prompt(
        self,
        audio_text: str = None,
        image_caption: str = None
    ) -> str:
        prompt = "Analyze the emotional content in the following:\n"
        
        if audio_text:
            prompt += f"Speech content: {audio_text}\n"
        if image_caption:
            prompt += f"Visual content: {image_caption}\n"
            
        prompt += "\nWhat is the dominant emotion? Choose from: "
        prompt += ", ".join(self.emotion_classes)
        prompt += "\nEmotion:"
        
        return prompt
        
    def forward(self, batch):
        # Process each modality
        audio_texts = []
        image_captions = []
        
        if "audio" in batch:
            audio_texts = [
                self.audio_captioner.generate_caption(audio)
                for audio in batch["audio"]
            ]
            
        if "image" in batch:
            image_captions = [
                self.image_captioner.generate_caption(image)
                for image in batch["image"]
            ]
            
        # Create prompts for batch
        prompts = [
            self._create_prompt(audio_text, image_caption)
            for audio_text, image_caption 
            in zip(audio_texts, image_captions)
        ]
        
        # Tokenize and generate
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        outputs = self.llm.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True
        )
        
        # Decode predictions
        predictions = self.tokenizer.batch_decode(
            outputs, 
            skip_special_tokens=True
        )
        
        # Extract emotion labels from predictions
        emotion_logits = torch.zeros(
            len(predictions), 
            len(self.emotion_classes)
        ).to(self.device)
        
        for i, pred in enumerate(predictions):
            for j, emotion in enumerate(self.emotion_classes):
                if emotion.lower() in pred.lower():
                    emotion_logits[i, j] = 1.0
                    
        return emotion_logits
        
    def training_step(self, batch, batch_idx):
        logits = self(batch)
        loss = nn.CrossEntropyLoss()(logits, batch["label"])
        
        # Log metrics
        self.log("train_loss", loss)
        accuracy = (logits.argmax(dim=1) == batch["label"]).float().mean()
        self.log("train_accuracy", accuracy)
        
        return loss