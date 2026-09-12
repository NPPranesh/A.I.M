from ollama import AsyncClient

async def generate_follow_up(candidate_answer: str, eye_contact: int) -> str:
    """Uses a local Llama model via Ollama to generate a push-back question."""
    
    prompt = f"""
    You are an expert Senior Software Engineer conducting a mock interview. 
    
    The candidate just gave the following answer:
    "{candidate_answer}"
    
    Their average eye contact score during this answer was {eye_contact}%.
    
    Ask ONE challenging, technical follow-up question based on their answer. 
    If their eye contact is below 70%, gently remind them to look at the camera.
    Keep it conversational, direct, and under 30 words.
    """
    
    print(f"\n[Ollama] Local AI is thinking... (Eye Contact: {eye_contact}%)")
    
    response = await AsyncClient().generate(
        model='llama3.2',  
        prompt=prompt
    )
    
    return response['response']