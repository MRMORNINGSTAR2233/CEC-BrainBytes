from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from market import get_market_insights
import uvicorn

app = FastAPI(
    title="Agricultural Market Analysis API",
    description="API for analyzing agricultural market prices and providing recommendations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MarketRequest(BaseModel):
    items: List[str]
    location: str = "default"  # Optional location parameter for future use

class MarketResponse(BaseModel):
    market_data: Dict[str, Any]
    recommendations: str
    status: str = "success"

@app.get("/")
async def root():
    return {
        "message": "Welcome to Agricultural Market Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "/analyze": "POST - Analyze market prices and get recommendations",
            "/categories": "GET - Get available item categories"
        }
    }

@app.get("/categories")
async def get_categories():
    """Get all available item categories and their items"""
    from market import MarketScraper
    scraper = MarketScraper()
    return {
        "categories": scraper.categories,
        "status": "success"
    }

@app.post("/analyze", response_model=MarketResponse)
async def analyze_market(request: MarketRequest):
    """
    Analyze market prices and get recommendations for the specified items
    """
    try:
        # Get market insights
        insights = get_market_insights(request.items)
        
        return MarketResponse(
            market_data=insights["market_data"],
            recommendations=insights["recommendations"],
            status="success"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing market data: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from RagBot.main import AgriculturalAssistant
import uvicorn
from io import BytesIO
import os

app = FastAPI(
    title="Agricultural Expert Assistant API",
    description="API for an agricultural expert chatbot that provides farming advice using web search and voice input",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize the assistant
assistant = AgriculturalAssistant()

class ChatRequest(BaseModel):
    question: str

class VoiceResponse(BaseModel):
    transcription: str
    language: str
    response: str
    confidence: float

class ChatResponse(BaseModel):
    response: str
    history: List[tuple]

class ChatHistory(BaseModel):
    messages: List[dict]

@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "name": "Agricultural Expert Assistant API",
        "version": "1.0.0",
        "description": "API for an agricultural expert chatbot that provides farming advice using web search and voice input"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint that processes text questions and returns responses."""
    try:
        response = assistant.get_response(request.question)
        return ChatResponse(response=response, history=assistant.chat_history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/voice", response_model=VoiceResponse)
async def voice_chat(audio_file: UploadFile = File(...)):
    """Voice chat endpoint that processes voice input and returns responses in the native language."""
    try:
        # Read the audio file
        audio_content = await audio_file.read()
        audio_io = BytesIO(audio_content)
        
        # Process voice input and get response
        result = assistant.handle_voice_input(audio_io)
        
        return VoiceResponse(
            transcription=result["transcription"],
            language=result["language"],
            response=result["response"],
            confidence=result["confidence"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/history", response_model=ChatHistory)
async def get_chat_history():
    """Get the current chat history."""
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": content}
        for i, (question, response) in enumerate(assistant.chat_history)
        for content in [question, response]
    ]
    return ChatHistory(messages=messages)

@app.post("/chat/clear")
async def clear_chat():
    """Clear the chat history."""
    assistant.clear_history()
    return {"message": "Chat history cleared successfully"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
from farm_news_agent import process_news_query
import uvicorn

app = FastAPI(
    title="Farm News API",
    description="API for fetching and analyzing farm-related news using LangGraph and Groq AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NewsQuery(BaseModel):
    query: str
    days: Optional[int] = 7

class NewsResponse(BaseModel):
    query: str
    articles_count: int
    articles: List[Dict]
    analysis: str
    timestamp: str

@app.get("/")
async def root():
    return {
        "message": "Welcome to Farm News API",
        "endpoints": {
            "/news": "POST - Get farm-related news and analysis",
            "/health": "GET - Check API health"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/news", response_model=NewsResponse)
async def get_farm_news(query: NewsQuery):
    try:
        result = process_news_query(query.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
from scheme_agent import process_schemes_query
import uvicorn

app = FastAPI(
    title="Government Farming Schemes API",
    description="API for fetching and analyzing government farming schemes using LangGraph and Groq AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SchemeResponse(BaseModel):
    central_schemes_count: int
    state_schemes_count: int
    central_schemes: List[Dict]
    state_schemes: List[Dict]
    analysis: str
    timestamp: str

@app.get("/")
async def root():
    return {
        "message": "Welcome to Government Farming Schemes API",
        "endpoints": {
            "/schemes": "GET - Get government farming schemes and analysis",
            "/health": "GET - Check API health"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/schemes", response_model=SchemeResponse)
async def get_schemes():
    try:
        result = process_schemes_query()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001) 