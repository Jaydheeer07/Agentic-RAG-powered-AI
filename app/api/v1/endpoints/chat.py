from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.db.session import get_db
from app.schemas.chat_models import ChatRequest, ChatResponse, Message
from app.core.rag import get_relevant_chunks, generate_response
from app.core.clients import openai_client

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Process a chat message using RAG
    """
    # Get relevant chunks based on the query
    chunks = await get_relevant_chunks(request.message, db)
    
    # Generate response using chunks and chat history
    response = await generate_response(
        query=request.message,
        chunks=chunks,
        chat_history=request.chat_history
    )
    
    return ChatResponse(
        message=response,
        sources=[chunk.document.url for chunk in chunks]
    )

@router.get("/chat/history", response_model=List[Message])
async def get_chat_history(db: Session = Depends(get_db)):
    """
    Get chat history
    """
    # Implement chat history retrieval
    pass
