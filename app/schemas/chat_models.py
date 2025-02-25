from pydantic import BaseModel
from typing import List, Optional

class Message(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class ChatRequest(BaseModel):
    message: str
    chat_history: Optional[List[Message]] = []

class ChatResponse(BaseModel):
    message: str
    sources: List[str]  # List of URLs where the information came from
