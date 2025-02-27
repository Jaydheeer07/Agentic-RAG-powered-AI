import asyncio
import time
from typing import List, Dict

from app.core.chunking import chunk_text, detect_content_type
from app.core.title_summary import get_title_and_summary


async def process_document(document: str, url: str) -> List[Dict]:
    """
    Process a document by:
    1. Detecting content type
    2. Chunking it appropriately
    3. Getting title and summary for each chunk
    
    Args:
        document: The document text to process
        url: The source URL of the document
        
    Returns:
        List of dictionaries with chunk, title, and summary
    """
    # Detect content type
    content_type = detect_content_type(document)
    print(f"Detected content type: {content_type}")
    
    # Chunk the document
    chunks = chunk_text(document)
    print(f"Document split into {len(chunks)} chunks")
    
    # Process each chunk
    results = []
    for i, chunk in enumerate(chunks):
        print(f"\nProcessing chunk {i+1}/{len(chunks)}")
        print(f"Chunk length: {len(chunk)} characters")
        
        # Get title and summary
        start_time = time.time()
        metadata = await get_title_and_summary(chunk, url)
        elapsed = time.time() - start_time
        
        print(f"API call completed in {elapsed:.2f} seconds")
        print(f"Title: {metadata['title']}")
        print(f"Summary: {metadata['summary']}")
        
        # Store result
        result = {
            "chunk": chunk,
            "title": metadata["title"],
            "summary": metadata["summary"],
            "url": url
        }
        results.append(result)
    
    return results


async def test_document_processing():
    """Test the full document processing pipeline with a real document"""
    # Sample document
    sample_document = """# Vector Databases for AI Applications

Vector databases have become an essential component in modern AI systems, particularly for applications involving semantic search, recommendation systems, and natural language processing.

## What are Vector Databases?

Vector databases are specialized database systems designed to store, manage, and search vector embeddings efficiently. These embeddings are numerical representations of data (text, images, audio, etc.) that capture semantic meaning.

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

## Integration with LLMs

Vector databases play a crucial role in Retrieval-Augmented Generation (RAG) systems:

```python
# Example of using a vector database in a RAG system
def retrieve_relevant_context(query, vector_db):
    # Convert query to embedding
    query_embedding = embed_text(query)
    
    # Search vector database for similar content
    results = vector_db.search(
        vector=query_embedding,
        top_k=5
    )
    
    # Format results as context
    context = "\\n".join([r.text for r in results])
    return context

def generate_response(query, context):
    # Prompt the LLM with context and query
    prompt = f"Context: {context}\\n\\nQuestion: {query}\\n\\nAnswer:"
    response = llm.generate(prompt)
    return response
```

This approach helps ground LLM responses in factual information and reduces hallucinations.

## Conclusion

Choosing the right vector database depends on your specific requirements around:
- Query speed and scalability
- Filtering capabilities
- Metadata support
- Hosting options (cloud vs. self-hosted)
- Integration with ML frameworks

As AI applications continue to evolve, vector databases will remain a critical infrastructure component for efficient similarity search and retrieval operations.
"""
    
    url = "https://example.com/vector-databases-guide"
    
    print("Testing full document processing pipeline")
    print("=" * 60)
    
    # Process the document
    results = await process_document(sample_document, url)
    
    # Print summary of results
    print("\n" + "=" * 60)
    print(f"Document processed into {len(results)} chunks with titles and summaries")
    
    return results


async def main():
    """Run all tests"""
    await test_document_processing()


if __name__ == "__main__":
    asyncio.run(main())
