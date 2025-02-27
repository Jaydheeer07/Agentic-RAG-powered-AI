import asyncio
import time
from typing import List, Dict

from app.core.chunking import chunk_text, detect_content_type
from app.core.title_summary import get_title_and_summary
from app.core.get_embeddings import get_embedding


async def process_document(document: str, url: str) -> List[Dict]:
    """
    Process a document by:
    1. Detecting content type
    2. Chunking it appropriately
    3. Getting title and summary for each chunk
    4. Getting embeddings for each chunk
    
    Args:
        document: The document text to process
        url: The source URL of the document
        
    Returns:
        List of dictionaries with chunk, title, summary, and embedding
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
        title_and_summary = await get_title_and_summary(chunk, url)
        elapsed = time.time() - start_time
        print(f"Title/Summary API call completed in {elapsed:.2f} seconds")
        
        # Get embedding
        start_time = time.time()
        embedding = await get_embedding(chunk, title=title_and_summary["title"])
        elapsed = time.time() - start_time
        print(f"Embedding API call completed in {elapsed:.2f} seconds")
        print(f"Embedding vector length: {len(embedding)}")
        
        results.append({
            "chunk": chunk,
            "title": title_and_summary["title"],
            "summary": title_and_summary["summary"],
            "embedding": embedding,
            "url": url
        })
    
    return results


async def test_document_processing():
    """Test the full document processing pipeline with a real document"""
    # Sample document
    sample_document = """# Vector Databases for AI Applications

Vector databases have become an essential component in modern AI systems, particularly for applications involving semantic search, recommendation systems, and natural language processing.

## What are Vector Databases?

Vector databases are specialized database systems designed to store, manage, and search vector embeddings efficiently. These embeddings are numerical representations of data (text, images, audio, etc.) that capture semantic meaning."""

    url = "https://example.com/vector-databases-guide"
    
    print("\nTesting full document processing pipeline")
    print("=" * 60)
    
    # Process the document
    results = await process_document(sample_document, url)
    
    # Print summary of results
    print("\n" + "=" * 60)
    print(f"Document processed into {len(results)} chunks with titles, summaries, and embeddings")
    print(f"Each chunk has an embedding vector of length {len(results[0]['embedding'])}")
    
    return results


async def main():
    """Run all tests"""
    await test_document_processing()


if __name__ == "__main__":
    asyncio.run(main())
