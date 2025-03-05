import asyncio
import json
from typing import List, Dict
from unittest.mock import AsyncMock, patch, MagicMock

from app.core.chunks.chunk_main import chunk_text
from app.core.title_summary import get_title_and_summary


async def process_document_with_titles_and_summaries(
    document: str, url: str
) -> List[Dict[str, str]]:
    """
    Process a document by:
    1. Chunking it into smaller pieces
    2. Getting title and summary for each chunk
    3. Returning the enhanced chunks
    
    Args:
        document: The document text to process
        url: The source URL of the document
        
    Returns:
        List of dictionaries containing chunk text, title, and summary
    """
    # Step 1: Chunk the document
    chunks = chunk_text(document)
    
    # Step 2: Get title and summary for each chunk
    enhanced_chunks = []
    for chunk in chunks:
        # Get title and summary
        metadata = await get_title_and_summary(chunk, url)
        
        # Create enhanced chunk
        enhanced_chunk = {
            "text": chunk,
            "title": metadata["title"],
            "summary": metadata["summary"],
            "url": url
        }
        
        enhanced_chunks.append(enhanced_chunk)
    
    return enhanced_chunks


async def demo_title_summary_pipeline():
    """Demonstrate the full pipeline with a sample document"""
    # Sample document
    sample_document = """# Understanding Vector Databases
    
    Vector databases are specialized database systems designed to store, manage, and search vector embeddings efficiently. These embeddings are numerical representations of data (text, images, audio, etc.) that capture semantic meaning.
    
    ## Why Vector Databases Matter
    
    Traditional databases struggle with similarity searches across high-dimensional vectors. Vector databases solve this by implementing:
    
    1. Approximate Nearest Neighbor (ANN) algorithms
    2. Specialized indexing techniques
    3. Optimized distance calculations
    
    ## Popular Vector Database Options
    
    Several vector databases have emerged in recent years:
    
    - **Pinecone**: Fully managed vector database with simple API
    - **Weaviate**: Open-source vector search engine
    - **Milvus**: Distributed vector database with scalability focus
    - **Qdrant**: Open-source vector similarity search engine
    - **Chroma**: Lightweight embedding database for AI applications
    
    ## Key Features to Consider
    
    When choosing a vector database, consider:
    
    - Query speed and scalability
    - Filtering capabilities
    - Metadata support
    - Hosting options (cloud vs. self-hosted)
    - Integration with ML frameworks
    """
    
    # Mock OpenAI responses for different chunks
    mock_responses = [
        {
            "title": "Understanding Vector Databases", 
            "summary": "Introduction to vector databases as specialized systems for storing and searching vector embeddings."
        },
        {
            "title": "Importance of Vector Databases", 
            "summary": "Why vector databases matter for similarity searches using ANN algorithms and specialized indexing."
        },
        {
            "title": "Popular Vector Database Options", 
            "summary": "Overview of popular vector databases including Pinecone, Weaviate, Milvus, Qdrant, and Chroma."
        },
        {
            "title": "Key Features for Vector Database Selection", 
            "summary": "Important factors to consider when choosing a vector database including speed, filtering, and hosting options."
        }
    ]
    
    # Create mock response objects
    mock_response_objects = []
    for resp in mock_responses:
        mock_resp = MagicMock()
        mock_resp.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(resp)
                )
            )
        ]
        mock_response_objects.append(mock_resp)
    
    # Create a side effect function to return different responses for different calls
    call_count = 0
    
    async def mock_side_effect(*args, **kwargs):
        nonlocal call_count
        response = mock_response_objects[min(call_count, len(mock_response_objects) - 1)]
        call_count += 1
        return response
    
    # Process the document with mocked OpenAI responses
    with patch("app.core.title_summary.openai_client.chat.completions.create", 
               new=AsyncMock(side_effect=mock_side_effect)):
        
        url = "https://example.com/vector-databases"
        enhanced_chunks = await process_document_with_titles_and_summaries(sample_document, url)
        
        # Print the results
        print(f"\nProcessed document into {len(enhanced_chunks)} enhanced chunks:")
        for i, chunk in enumerate(enhanced_chunks):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Title: {chunk['title']}")
            print(f"Summary: {chunk['summary']}")
            print(f"Text length: {len(chunk['text'])} characters")
            print(f"Text preview: {chunk['text'][:100]}...")
        
        return enhanced_chunks


if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_title_summary_pipeline())
