import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.core.title_summary import get_title_and_summary
from app.core.chunking import chunk_text


@pytest.mark.asyncio
async def test_get_title_and_summary_success():
    """Test successful title and summary extraction"""
    # Mock data
    test_chunk = "# FastAPI Dependency Injection\n\nFastAPI provides a powerful dependency injection system that helps you share logic, database connections, and enforce security."
    test_url = "https://example.com/docs"
    
    # Expected response from OpenAI
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps({
                    "title": "FastAPI Dependency Injection", 
                    "summary": "Overview of FastAPI's dependency injection system for sharing logic and enforcing security."
                })
            )
        )
    ]
    
    # Patch the OpenAI client
    with patch("app.core.title_summary.openai_client.chat.completions.create", 
               new=AsyncMock(return_value=mock_response)):
        
        # Call the function
        result = await get_title_and_summary(test_chunk, test_url)
        
        # Verify results
        assert isinstance(result, dict)
        assert "title" in result
        assert "summary" in result
        assert result["title"] == "FastAPI Dependency Injection"
        assert "dependency injection system" in result["summary"]


@pytest.mark.asyncio
async def test_get_title_and_summary_exception():
    """Test error handling when OpenAI API fails"""
    # Mock data
    test_chunk = "Some test content"
    test_url = "https://example.com/docs"
    
    # Patch the OpenAI client to raise an exception
    with patch("app.core.title_summary.openai_client.chat.completions.create", 
               new=AsyncMock(side_effect=Exception("API Error"))):
        
        # Call the function
        result = await get_title_and_summary(test_chunk, test_url)
        
        # Verify error handling
        assert isinstance(result, dict)
        assert "title" in result
        assert "summary" in result
        assert result["title"] == "Error processing title"
        assert result["summary"] == "Error processing summary"


@pytest.mark.asyncio
async def test_get_title_and_summary_with_real_chunks():
    """Test title and summary extraction with real chunked content"""
    # Sample documentation text
    sample_docs = """# Python Asyncio
    
    Asyncio is a library to write concurrent code using the async/await syntax.
    
    Asyncio is used as a foundation for multiple Python asynchronous frameworks that provide high-performance network and web-servers, database connection libraries, distributed task queues, etc.
    
    ## Key Features
    
    * Coroutines and Tasks to run concurrent Python code
    * Event loops for handling async operations
    * Synchronization primitives
    """
    
    # Generate chunks
    chunks = chunk_text(sample_docs)
    assert len(chunks) > 0, "Chunking failed"
    
    # Mock OpenAI response
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps({
                    "title": "Python Asyncio Overview", 
                    "summary": "Introduction to Python's asyncio library for concurrent programming with async/await syntax."
                })
            )
        )
    ]
    
    # Test with the first chunk
    with patch("app.core.title_summary.openai_client.chat.completions.create", 
               new=AsyncMock(return_value=mock_response)):
        
        result = await get_title_and_summary(chunks[0], "https://example.com/python-docs")
        
        # Verify results
        assert isinstance(result, dict)
        assert "title" in result
        assert "summary" in result


@pytest.mark.asyncio
async def test_get_title_and_summary_truncation():
    """Test that long chunks are properly truncated before sending to OpenAI"""
    # Create a very long chunk (over 1000 chars)
    long_chunk = "A" * 2000
    test_url = "https://example.com/docs"
    
    # Create a mock to capture the actual API call
    mock_create = AsyncMock()
    mock_create.return_value.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps({
                    "title": "Test Title", 
                    "summary": "Test Summary"
                })
            )
        )
    ]
    
    # Patch the OpenAI client
    with patch("app.core.title_summary.openai_client.chat.completions.create", mock_create):
        
        # Call the function
        await get_title_and_summary(long_chunk, test_url)
        
        # Verify that the chunk was truncated in the API call
        # Get the content from the user message
        call_args = mock_create.call_args
        messages = call_args[1]["messages"]
        user_message = next(msg for msg in messages if msg["role"] == "user")
        
        # The content should contain the URL and truncated chunk with "..."
        assert test_url in user_message["content"]
        assert "..." in user_message["content"]
        # The chunk should be truncated to 1000 chars plus the URL and other text
        assert len(user_message["content"]) < 1100  # Allow some extra for the URL and formatting


if __name__ == "__main__":
    import asyncio
    
    # Run the tests
    asyncio.run(test_get_title_and_summary_success())
    asyncio.run(test_get_title_and_summary_exception())
    asyncio.run(test_get_title_and_summary_with_real_chunks())
    asyncio.run(test_get_title_and_summary_truncation())
    
    print("All tests passed!")
