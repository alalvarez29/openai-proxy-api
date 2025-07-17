from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import httpx
from typing import Optional, Dict
from dotenv import load_dotenv
import os

load_dotenv()

DEFAULT_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

app = FastAPI(title="OpenAI REST API Integration")

class QuestionRequest(BaseModel):
    question: str

class SuccessResponse(BaseModel):
    response: str

def validate_api_key(x_api_key: Optional[str]) -> str:
    """Validate and return API key"""
    api_key = x_api_key or DEFAULT_API_KEY
    if not api_key:
        raise HTTPException(status_code=401, detail={"message": "API Key required"})
    return api_key

def create_request_payload(question: str) -> Dict:
    """Create standardized request payload"""
    return {
        "model": "openai/gpt-3.5-turbo",
        "messages": [{"role": "user", "content": question}],
        "max_tokens": 1000,
        "temperature": 0.7
    }

def create_headers(api_key: str) -> Dict:
    """Create standardized headers"""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

def handle_response_error(status_code: int):
    """Handle API response errors uniformly"""
    error_map = {
        401: "API Key required",
        402: "API Key required",
        429: "Rate limit exceeded"
    }
    
    if status_code in error_map:
        raise HTTPException(status_code=status_code, detail={"message": error_map[status_code]})
    else:
        raise HTTPException(
            status_code=status_code,
            detail={"message": f"OpenAI API error: {status_code}"}
        )

def extract_response_content(json_data: Dict) -> str:
    """Extract response content from API response"""
    try:
        return json_data['choices'][0]['message']['content'].strip()
    except KeyError:
        raise HTTPException(status_code=500, detail={"message": "Unexpected response format"})

async def call_openrouter_api(api_key: str, question: str) -> str:
    """Core API call logic"""
    headers = create_headers(api_key)
    payload = create_request_payload(question)
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(OPENROUTER_API_URL, headers=headers, json=payload)
            
            if not response.is_success:
                handle_response_error(response.status_code)
            
            json_data = response.json()
            return extract_response_content(json_data)
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail={"message": "Request timeout"})
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail={"message": "Connection error"})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"message": f"Internal server error: {str(e)}"})

@app.post("/ask", response_model=SuccessResponse)
async def ask_openai(
    request: QuestionRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """Main endpoint - orchestrates the flow"""
    api_key = validate_api_key(x_api_key)
    openai_response = await call_openrouter_api(api_key, request.question)
    return SuccessResponse(response=openai_response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)