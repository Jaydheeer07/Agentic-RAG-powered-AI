# Title and Summary Extraction Testing

This directory contains tests for the `get_title_and_summary` function in the `app/core/title_summary.py` module. This function uses OpenAI's GPT-4o-mini model to extract titles and summaries from document chunks.

## Function Overview

The `get_title_and_summary` function:
- Takes a text chunk and a URL as input
- Uses GPT-4o-mini to generate a title and summary for the chunk
- Returns a dictionary with 'title' and 'summary' keys
- Handles errors gracefully

## Test Files

1. **test_title_summary.py**: Unit tests with mocked OpenAI API calls
   - Tests basic functionality
   - Tests error handling
   - Tests integration with chunking
   - Tests truncation of long chunks

2. **test_title_summary_simple.py**: Simple tests with real OpenAI API calls
   - Tests the function with different content types
   - Minimal setup required

3. **test_title_summary_with_chunking.py**: Integration tests with chunking
   - Tests the full document processing pipeline
   - Shows how to chunk a document and get titles/summaries for each chunk

4. **test_title_summary_edge_cases.py**: Tests for edge cases
   - Very short content
   - Content with special characters
   - Very long content
   - Code-heavy content
   - Content without clear titles
   - Content with multiple potential titles

5. **test_title_summary_integration.py**: Demo of a complete pipeline
   - Shows how to process a document end-to-end
   - Includes mocked examples for demonstration

## Running the Tests

### Unit Tests (No API Calls)

```bash
python -m app.tests.test_title_summary
```

### Live Tests (Makes API Calls)

```bash
python -m app.tests.test_title_summary_simple
python -m app.tests.test_title_summary_with_chunking
python -m app.tests.test_title_summary_edge_cases
```

### Integration Demo

```bash
python -m app.tests.test_title_summary_integration
```

## Example Usage

Here's a simple example of how to use the `get_title_and_summary` function in your code:

```python
import asyncio
from app.core.chunking import chunk_text
from app.core.title_summary import get_title_and_summary

async def process_document(document, url):
    # Split document into chunks
    chunks = chunk_text(document)
    
    # Get title and summary for each chunk
    enhanced_chunks = []
    for chunk in chunks:
        metadata = await get_title_and_summary(chunk, url)
        enhanced_chunk = {
            "text": chunk,
            "title": metadata["title"],
            "summary": metadata["summary"],
            "url": url
        }
        enhanced_chunks.append(enhanced_chunk)
    
    return enhanced_chunks

# Usage
async def main():
    document = "Your document text here..."
    url = "https://example.com/document"
    results = await process_document(document, url)
    
    for i, result in enumerate(results):
        print(f"Chunk {i+1}:")
        print(f"Title: {result['title']}")
        print(f"Summary: {result['summary']}")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
```

## Notes and Best Practices

1. **API Usage**: The live tests make actual API calls to OpenAI, which will incur costs.

2. **Error Handling**: The function includes error handling to prevent failures if the API call fails.

3. **Chunk Size**: The function only sends the first 1000 characters of each chunk to the API to reduce token usage.

4. **Response Format**: The function specifies `response_format={"type": "json_object"}` to ensure the API returns properly formatted JSON.

5. **Performance**: Consider implementing caching for frequently accessed chunks to reduce API calls.

6. **Batch Processing**: For large documents, consider implementing batch processing to avoid rate limits.
