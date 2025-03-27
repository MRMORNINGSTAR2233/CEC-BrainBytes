from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from features.market import get_market_insights, MarketScraper
from features.bot import AgriculturalAssistant
from features.news import process_news_query
from features.scheme import process_schemes_query
import uvicorn
from io import BytesIO
from datetime import datetime

app = FastAPI(
    title="Agricultural Expert System API",
    description="Comprehensive API for agricultural market analysis, expert assistance, news, and government schemes",
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

# Initialize the assistant
assistant = AgriculturalAssistant()

# Request/Response Models
class MarketRequest(BaseModel):
    items: List[str]
    location: str = "default"

class MarketResponse(BaseModel):
    market_data: Dict[str, Any]
    recommendations: str
    status: str = "success"

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

class NewsQuery(BaseModel):
    query: str
    days: Optional[int] = 7

class NewsResponse(BaseModel):
    query: str
    articles_count: int
    articles: List[Dict]
    analysis: str
    timestamp: str

class SchemeResponse(BaseModel):
    central_schemes_count: int
    state_schemes_count: int
    central_schemes: List[Dict]
    state_schemes: List[Dict]
    analysis: str
    timestamp: str

# Root endpoint
@app.get("/")
async def root():
    return {
        "name": "Agricultural Expert System API",
        "version": "1.0.0",
        "description": "Comprehensive API for agricultural market analysis, expert assistance, news, and government schemes",
        "endpoints": {
            "/market": {
                "analyze": "POST - Analyze market prices and get recommendations",
                "categories": "GET - Get available item categories"
            },
            "/chat": {
                "text": "POST - Chat with agricultural expert",
                "voice": "POST - Voice chat with agricultural expert",
                "history": "GET - Get chat history",
                "clear": "POST - Clear chat history"
            },
            "/news": "POST - Get farm-related news and analysis",
            "/schemes": "GET - Get government farming schemes and analysis"
        }
    }

# Market Analysis Endpoints
@app.get("/market/categories")
async def get_categories():
    """Get all available item categories and their items"""
    scraper = MarketScraper()
    return {
        "categories": scraper.categories,
        "status": "success"
    }

@app.post("/market/analyze", response_model=MarketResponse)
async def analyze_market(request: MarketRequest):
    """Analyze market prices and get recommendations for the specified items"""
    try:
        insights = get_market_insights(request.items)
        return MarketResponse(
            market_data=insights["market_data"],
            recommendations=insights["recommendations"],
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing market data: {str(e)}")

# Chat Endpoints
@app.post("/chat/text", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint that processes text questions and returns responses"""
    try:
        response = assistant.get_response(request.question)
        return ChatResponse(response=response, history=assistant.chat_history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/voice", response_model=VoiceResponse)
async def voice_chat(audio_file: UploadFile = File(...)):
    """Voice chat endpoint that processes voice input and returns responses"""
    try:
        audio_content = await audio_file.read()
        audio_io = BytesIO(audio_content)
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
    """Get the current chat history"""
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": content}
        for i, (question, response) in enumerate(assistant.chat_history)
        for content in [question, response]
    ]
    return ChatHistory(messages=messages)

@app.post("/chat/clear")
async def clear_chat():
    """Clear the chat history"""
    assistant.clear_history()
    return {"message": "Chat history cleared successfully"}

# News Endpoints
@app.post("/news", response_model=NewsResponse)
async def get_farm_news(query: NewsQuery):
    """Get farm-related news and analysis"""
    try:
        result = process_news_query(query.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Schemes Endpoints
@app.get("/schemes", response_model=SchemeResponse)
async def get_schemes():
    """Get government farming schemes and analysis"""
    try:
        result = process_schemes_query()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health Check Endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 