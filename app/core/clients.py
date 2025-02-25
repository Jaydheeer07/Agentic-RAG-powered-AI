from openai import AsyncOpenAI
from app.config import settings

# Initialize the OpenAI client 
openai_client = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY,
    organization=settings.OPENAI_ORGANIZATION
)
