from openai import OpenAI

def get_lmstudio_model_info(base_url: str, api_key: str = "lm-studio") -> dict:
    """Get current loaded model info from LMStudio"""
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        response = client.models.list()
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