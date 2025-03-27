from typing import Dict, List, Tuple, Any
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from features.utils import setup_web_search, get_web_search_results
import tempfile
import numpy as np
import soundfile as sf
from transformers import pipeline
import torch

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatGroq(
    model_name="llama-3.2-3b-preview",
    temperature=0.7,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Initialize Whisper model from Hugging Face
whisper_model = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
    device=0 if torch.cuda.is_available() else -1  # Use GPU if available, else CPU
)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an agricultural expert assistant helping farmers with their queries.
    Use the following context to answer the farmer's question. If you don't know the answer, say so.
    Make sure to provide accurate and relevant information based on the web search results.
    You can understand and respond in multiple languages. If the question is in a different language,
    respond in the same language.
    
    Context: {context}
    
    Current question: {question}"""),
    ("human", "{question}")
])

# Define the chain
chain = (
    {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

class AgriculturalAssistant:
    def __init__(self):
        self.web_search_tool = setup_web_search()
        self.chat_history: List[Tuple[str, str]] = []
    
    def get_response(self, question: str) -> str:
        """Get response for a question."""
        # Get web search results
        context = get_web_search_results(question, self.web_search_tool)
        
        # Generate response
        response = chain.invoke({
            "context": context,
            "question": question
        })
        
        # Update chat history
        self.chat_history.append((question, response))
        
        return response
    
    def clear_history(self):
        """Clear the chat history."""
        self.chat_history = []
    
    def process_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """Process audio data and convert it to text using Whisper from Hugging Face.
        
        Args:
            audio_data: The audio data in bytes
            
        Returns:
            Dict containing:
                - text: Transcribed text
                - language: Detected language code
                - confidence: Confidence score of the transcription
        """
        try:
            # Create a temporary file to store the audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                # Write the audio data to the temporary file
                temp_file.write(audio_data)
                temp_file.flush()
                
                # Transcribe using Whisper with language detection
                result = whisper_model(
                    temp_file.name,
                    task="transcribe",
                    return_timestamps=True
                )
                
                # Clean up the temporary file
                os.unlink(temp_file.name)
                
                return {
                    "text": result["text"].strip(),
                    "language": result.get("language", "en"),  # Default to English if not detected
                    "confidence": result.get("confidence", 1.0)
                }
        except Exception as e:
            raise Exception(f"Error processing audio: {str(e)}")
    
    def handle_voice_input(self, audio_data: bytes) -> Dict[str, Any]:
        """Handle voice input, transcribe it, and get response from the assistant.
        
        Args:
            audio_data: The audio data in bytes
            
        Returns:
            Dict containing:
                - transcription: The transcribed text
                - language: The detected language
                - response: The assistant's response
                - confidence: Confidence score of the transcription
        """
        # Process the audio to get transcription and language
        audio_result = self.process_audio(audio_data)
        
        # Get response from the assistant
        response = self.get_response(audio_result["text"])
        
        return {
            "transcription": audio_result["text"],
            "language": audio_result["language"],
            "response": response,
            "confidence": audio_result["confidence"]
        }
