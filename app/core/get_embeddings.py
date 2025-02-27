from typing import List, Optional

from app.core.clients import openai_client


async def get_embedding(chunk_content: str, title: Optional[str] = None) -> List[float]:
    """
    Get embedding vector from OpenAI.
    
    Args:
        chunk_content: The content text to embed
        title: Optional title to prepend to the content for richer context
        
    Returns:
        List of floats representing the embedding vector
    """
    try:
        # Combine title and content if title is provided
        if title:
            combined_text = f"{title}: {chunk_content}"
        else:
            combined_text = chunk_content
            
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=combined_text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error