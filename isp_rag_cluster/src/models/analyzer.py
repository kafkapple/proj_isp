def analyze_emotion(self, text: str) -> str:
    if self.cfg.model.use_rag:
        retrieved_context = self.retriever.retrieve(text)
    else:
        retrieved_context = []  # RAG를 사용하지 않을 때는 빈 컨텍스트
    return self.generator.generate(retrieved_context, text) 