import asyncio
import time
from typing import Dict, List

from app.core.title_summary import get_title_and_summary


async def test_edge_cases():
    """Test how get_title_and_summary handles various edge cases"""
    edge_cases = [
        {
            "name": "Very short content",
            "content": "This is a very short piece of text with minimal information.",
            "url": "https://example.com/short"
        },
        {
            "name": "Content with special characters",
            "content": "# Special Characters Test\n\nThis content has special characters: ñ, é, ç, ß, 你好, こんにちは, مرحبا, 😀 🚀 💻",
            "url": "https://example.com/special"
        },
        {
            "name": "Very long content",
            "content": "# " + "Very long title " * 20 + "\n\n" + "This is a test of very long content. " * 100,
            "url": "https://example.com/long"
        },
        {
            "name": "Code-heavy content",
            "content": """
            ```python
            def process_document(document, chunk_size=1000, chunk_overlap=200):
                \"\"\"
                Process a document by chunking it and then processing each chunk.
                
                Args:
                    document: The document to process
                    chunk_size: The size of each chunk
                    chunk_overlap: The overlap between chunks
                    
                Returns:
                    List of processed chunks
                \"\"\"
                # Split the document into chunks
                chunks = split_into_chunks(document, chunk_size, chunk_overlap)
                
                # Process each chunk
                results = []
                for chunk in chunks:
                    result = process_chunk(chunk)
                    results.append(result)
                    
                return results
            
            class VectorDatabase:
                def __init__(self, embedding_model):
                    self.embedding_model = embedding_model
                    self.vectors = []
                    self.metadata = []
                
                def add_document(self, document, metadata=None):
                    embedding = self.embedding_model.embed(document)
                    self.vectors.append(embedding)
                    self.metadata.append(metadata or {})
                
                def search(self, query, top_k=5):
                    query_embedding = self.embedding_model.embed(query)
                    results = self._find_nearest(query_embedding, top_k)
                    return results
                
                def _find_nearest(self, vector, top_k):
                    # Simplified nearest neighbor search
                    distances = []
                    for i, v in enumerate(self.vectors):
                        distance = self._calculate_distance(vector, v)
                        distances.append((i, distance))
                    
                    # Sort by distance (ascending)
                    sorted_results = sorted(distances, key=lambda x: x[1])
                    
                    # Return top_k results with metadata
                    return [
                        {"id": idx, "distance": dist, "metadata": self.metadata[idx]}
                        for idx, dist in sorted_results[:top_k]
                    ]
                
                def _calculate_distance(self, v1, v2):
                    # Simplified cosine distance
                    return 1 - sum(a*b for a, b in zip(v1, v2))
            ```
            """,
            "url": "https://example.com/code-heavy"
        },
        {
            "name": "No clear title",
            "content": "This content doesn't have a clear title or heading. It's just a paragraph of text discussing various topics without any clear structure or organization. The lack of headings and structure makes it harder to automatically extract a meaningful title.",
            "url": "https://example.com/no-title"
        },
        {
            "name": "Multiple potential titles",
            "content": """# First Heading
            
            Some content under the first heading.
            
            ## Second Heading
            
            More content under the second heading.
            
            # Another Top-Level Heading
            
            This could also be considered a title.
            """,
            "url": "https://example.com/multiple-titles"
        }
    ]
    
    results = []
    
    for case in edge_cases:
        print(f"\n=== Testing with {case['name']} ===")
        print(f"Content preview: {case['content'][:100]}...")
        
        start_time = time.time()
        result = await get_title_and_summary(case['content'], case['url'])
        elapsed = time.time() - start_time
        
        print(f"API call completed in {elapsed:.2f} seconds")
        print(f"Title: {result['title']}")
        print(f"Summary: {result['summary']}")
        
        results.append({
            "case": case['name'],
            "title": result['title'],
            "summary": result['summary'],
            "time": elapsed
        })
    
    return results


async def main():
    """Run all tests"""
    print("Testing get_title_and_summary with edge cases")
    print("=" * 60)
    
    results = await test_edge_cases()
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary of results:")
    print(f"Total cases tested: {len(results)}")
    
    avg_time = sum(r['time'] for r in results) / len(results)
    print(f"Average processing time: {avg_time:.2f} seconds")
    
    # Check if any titles or summaries contain error messages
    errors = [r for r in results if "error" in r['title'].lower() or "error" in r['summary'].lower()]
    if errors:
        print(f"Found {len(errors)} cases with error messages")
    else:
        print("No error messages detected in titles or summaries")


if __name__ == "__main__":
    asyncio.run(main())
