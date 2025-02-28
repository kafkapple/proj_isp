from openai import OpenAI

def get_lmstudio_model_info(base_url: str, api_key: str = "lm-studio", model_type: str = "chat") -> dict:
    """Get current loaded model info from LMStudio"""
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        response = client.models.list()
        
        if not response.data:
            print("\nWarning: No models available in LMStudio")
            return {"id": "unknown", "error": "No models available"}
            
        print(f"\nAvailable LMStudio models:")
        for model in response.data:
            print(f"- {model.id}")
        
        # 현재 로드된 모델 중에서 선택
        available_models = response.data
        if not available_models:
            return {"id": "unknown", "error": "No models available"}
            
        if model_type == "embedding":
            # embedding 모델 찾기
            for model in available_models:
                if "embedding" in model.id.lower():
                    print(f"\nSelected embedding model: {model.id}")
                    return {"id": model.id, "type": "embedding"}
            
            # embedding 모델이 없으면 첫 번째 모델 사용
            print(f"\nWarning: No specific embedding model found, using first available model: {available_models[0].id}")
            return {"id": available_models[0].id, "type": "embedding"}
        else:
            # chat 모델 찾기 (embedding이 아닌 모델)
            for model in available_models:
                if "embedding" not in model.id.lower():
                    print(f"\nSelected chat model: {model.id}")
                    return {"id": model.id, "type": "chat"}
            
            # chat 모델이 없으면 첫 번째 모델 사용
            print(f"\nWarning: No specific chat model found, using first available model: {available_models[0].id}")
            return {"id": available_models[0].id, "type": "chat"}
                
    except Exception as e:
        print(f"Warning: Failed to fetch model info: {e}")
        return {"id": "unknown", "error": str(e)} 