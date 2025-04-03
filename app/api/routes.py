from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..rag.chat import ChatBot
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
chatbot = None

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@router.post("/initialize")
async def initialize():
    """Initialize the chatbot with documents."""
    global chatbot
    try:
        logger.info("Starting chatbot initialization...")
        chatbot = ChatBot()
        logger.info("Processing documents from data/pdfs and web content...")
        chatbot.initialize(doc_dir="./data/pdfs", web_url="https://www.angelone.in/support")
        logger.info("Chatbot initialization completed successfully")
        return {"message": "Chatbot initialized successfully"}
    except Exception as e:
        logger.error(f"Error initializing chatbot: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message and return a response."""
    if not chatbot:
        raise HTTPException(status_code=400, detail="Chatbot not initialized. Please call /initialize first.")
    
    try:
        logger.info(f"Processing chat message: {request.message}")
        response = chatbot.get_response(request.message)
        logger.info(f"Generated response: {response}")
        return ChatResponse(response=response)
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 