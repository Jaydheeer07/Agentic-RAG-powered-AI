import asyncio
import time

from app.core.title_summary import get_title_and_summary


async def test_simple_title_summary():
    """
    Simple test for the get_title_and_summary function with real API calls.
    This test will make actual API calls to OpenAI.
    """
    # Test samples with different content
    samples = [
        {
            "name": "Documentation",
            "content": """# Understanding Vector Databases
            
            Vector databases are specialized database systems designed to store, manage, and search vector embeddings efficiently. 
            These embeddings are numerical representations of data (text, images, audio, etc.) that capture semantic meaning.""",
            "url": "https://example.com/vector-databases"
        },
        {
            "name": "Code",
            "content": """
            def process_document(document, chunk_size=1000):
                \"\"\"
                Process a document by chunking it and then processing each chunk.
                
                Args:
                    document: The document to process
                    chunk_size: The size of each chunk
                    
                Returns:
                    List of processed chunks
                \"\"\"
                # Split the document into chunks
                chunks = split_into_chunks(document, chunk_size)
                
                # Process each chunk
                results = []
                for chunk in chunks:
                    result = process_chunk(chunk)
                    results.append(result)
                    
                return results
            """,
            "url": "https://example.com/code"
        },
        {
            "name": "Article",
            "content": """
            # The Evolution of Machine Learning
            
            In the early days of artificial intelligence, machine learning was primarily rule-based. 
            Researchers would manually code rules and decision trees to solve specific problems. 
            This approach, while effective for well-defined tasks, lacked the ability to generalize to new situations.
            """,
            "url": "https://example.com/article"
        }
    ]
    
    # Test each sample
    for sample in samples:
        print(f"\n=== Testing with {sample['name']} content ===")
        print(f"Content preview: {sample['content'][:100]}...")
        
        start_time = time.time()
        result = await get_title_and_summary(sample['content'], sample['url'])
        elapsed = time.time() - start_time
        
        print(f"API call completed in {elapsed:.2f} seconds")
        print(f"Title: {result['title']}")
        print(f"Summary: {result['summary']}")


async def main():
    """Run the test"""
    print("Testing get_title_and_summary with live OpenAI API calls")
    print("Note: This will make actual API calls and incur costs")
    print("=" * 60)
    
    await test_simple_title_summary()


if __name__ == "__main__":
    asyncio.run(main())
