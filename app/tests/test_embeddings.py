import asyncio
import time
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Union

from app.core.chunks.chunk_main import chunk_text, detect_content_type
from app.core.title_summary import get_title_and_summary
from app.core.get_embeddings import get_embedding
from app.core.chunks.ast_chunker import CodeChunk, format_chunk_with_context


def get_chunk_text(chunk: Union[str, CodeChunk]) -> str:
    """Get the text content from a chunk, handling both string and CodeChunk objects."""
    try:
        if isinstance(chunk, CodeChunk):
            return format_chunk_with_context(chunk)
        return str(chunk)
    except Exception as e:
        print(f"Error in get_chunk_text: {str(e)}")
        print("Chunk type:", type(chunk))
        print("Chunk repr:", repr(chunk))
        traceback.print_exc()
        raise


def save_chunk_details(results: List[Dict], output_dir: str = "test_outputs"):
    """
    Save chunk details to a text file for verification
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"chunk_details_{timestamp}.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== Document Processing Results ===\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"Chunk #{i}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Source URL: {result['url']}\n")
            f.write(f"Title: {result['title']}\n")
            f.write(f"Summary: {result['summary']}\n")
            f.write("\nOriginal Text:\n")
            f.write("-" * 30 + "\n")
            # Handle both string and CodeChunk objects
            chunk = result['chunk']
            if isinstance(chunk, CodeChunk):
                f.write(f"# Code Chunk (lines {chunk.start_line}-{chunk.end_line})\n")
                if chunk.imports:
                    f.write("\nImports:\n")
                    for imp in sorted(chunk.imports):
                        f.write(f"{imp}\n")
                if chunk.global_deps:
                    f.write("\nGlobal Dependencies:\n")
                    for dep in sorted(chunk.global_deps):
                        f.write(f"- {dep}\n")
                f.write("\nContent:\n")
                f.write(chunk.content)
            else:
                f.write(str(chunk))
            f.write("\n" + "-" * 30 + "\n")
            f.write(f"\nEmbedding Vector (first 5 dimensions): {result['embedding'][:5]}\n")
            f.write(f"Embedding Vector Length: {len(result['embedding'])}\n")
            f.write("\n" + "=" * 50 + "\n\n")
            
        print(f"\nChunk details saved to: {output_file}")


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
    try:
        # Detect content type
        content_type = detect_content_type(document)
        print(f"\nProcessing Document:")
        print(f"Input text length: {len(document)} characters")
        print(f"Content type: {content_type}")
        
        # Chunk the document
        chunks = chunk_text(document)
        print(f"\nChunking Results:")
        print(f"Number of chunks: {len(chunks)}")
        print(f"Chunk sizes: {[len(get_chunk_text(c)) for c in chunks]}")
        print(f"Average chunk size: {sum(len(get_chunk_text(c)) for c in chunks) / len(chunks):.2f} characters")
        
        # Process each chunk
        results = []
        for i, chunk in enumerate(chunks):
            print(f"\nProcessing chunk {i+1}/{len(chunks)}")
            print("Chunk type:", type(chunk))
            
            try:
                # Get the text content for processing
                processed_text = get_chunk_text(chunk)
                print(f"Successfully converted chunk to text, length: {len(processed_text)}")
            except Exception as e:
                print(f"Error getting chunk text: {str(e)}")
                print("Chunk:", repr(chunk))
                traceback.print_exc()
                continue
            
            # Initialize title and summary
            title_and_summary = {"title": "", "summary": ""}
            
            try:
                # Get title and summary
                start_time = time.time()
                title_and_summary = await get_title_and_summary(processed_text, url)
                elapsed = time.time() - start_time
                print(f"Title/Summary API call completed in {elapsed:.2f} seconds")
                print(f"Generated Title: {title_and_summary['title']}")
            except Exception as e:
                print(f"Error getting title and summary: {str(e)}")
                if isinstance(chunk, CodeChunk):
                    title = f"Code Chunk (lines {chunk.start_line}-{chunk.end_line})"
                    summary = f"Python code block with {len(chunk.imports)} imports and {len(chunk.global_deps)} global dependencies"
                else:
                    title = "Error processing title"
                    summary = "Error processing summary"
                title_and_summary = {"title": title, "summary": summary}
            
            try:
                # Get embedding
                start_time = time.time()
                embedding = await get_embedding(processed_text, title=title_and_summary["title"])
                elapsed = time.time() - start_time
                print(f"Embedding API call completed in {elapsed:.2f} seconds")
                print(f"Embedding vector length: {len(embedding)}")
                
                # Only append results if we successfully got the embedding
                results.append({
                    "chunk": chunk,
                    "title": title_and_summary["title"],
                    "summary": title_and_summary["summary"],
                    "embedding": embedding,
                    "url": url
                })
            except Exception as e:
                print(f"Error getting embedding: {str(e)}")
                traceback.print_exc()
                continue
        
        return results
    except Exception as e:
        print(f"Error in process_document: {str(e)}")
        traceback.print_exc()
        raise


async def test_document_processing():
    """Test the full document processing pipeline with a real document"""
    # Sample document
    sample_document = """
# RAG Beyond the basics.
"This mostly works but how do I make it better?"

This is a [living document](https://gist.github.com/Donavan/62e238aa0a40ca88191255a070e356a2) that will contain advice, tips and best practices for people who are trying to go beyond the basics with RAG or "Questions and answers with documents".  

## Segmentation
There is not a one size fits all approach to segmentation.  Depending on the content different size metric

- The size and boundary conditions for your segments can have a HUGE impact.
- The goal is to maximize the usefulness  of any given segment for a specific topic and not muddy the waters when added to the context.  
- Use the formatting to your advantage.  A heading type of element typically indicates a change in topic/focus so they should never be in the middle of a segment. 
- If your content has tables you really need to take care.My segmentation logic goes something like this for tables: 
    - Can I fit the entire table in this segment? 
    - Well can I fit the entire table in a segment all by itself?  
    - Ok, then add the table header and as many rows as we can to this segment 
    - then start a new segment and include the table header as the first line and chunk it up until we're out of rows
- Same logic for numbered lists and bulleted lists.
- If the table.list was preceded by a header or some other identifier repeat that in the sub segments for context.

## Processing
- How you format your text for vector retrieval vs how it should be presented to GPT are different. 
    - Each segment should have a copy of the content that has been ran through a cleaning process (downcase, lemmatization, etc).
        - This is used to for the vectors that get indexed.  
        - Our goal with this content is to reduce the possible token space down so that we maximize our chances of finding them.
        - Not even GPT will see this content so any trick to optiimze for retrieval is on the table.  Things like pronoun/nickname disambiguation can have an impact for some content.   
    - Each segment has metadata indicating where it came from as well as where it was sequentially within the source document.



## Embeddings, context and stuffing.
When we're talking about GPT embeddings are only used as a way to index and retrieve text.  GPT never sees the embeddings, nor could it understand them if given them.  That's why I suggest having two copies of the text, one optimized for vectorization and the other for presenting to GPT.

If you're using the stock Q&A stuff from LangChain it's doing something like this:

- Convert the user query to a vector and query the vector store.
- For each relevant segment concatenate it to a context string with a newline separator.
- Add a block of text to the prompt that says something like

```
Use the following pieces of contextual information to answer the user query:

chunk 1
chunk 2
chunk 3
```

These segments are typically ordered with the most relevant segments first with zero attention paid to the original ordering.  However, you can do your own stuffing and arrive at something like this :

```
## Use the following pieces of contextual information to answer the user query.  
## The contextual information is presented in the order in which it appeeared in the source with the most relevant source first:

**source 1**
chunk 7
chunk 3
chunk 2
chunk 8

**source 2**
chunk 1
chunk 9
chunk 4

**source 3**
chunk 5
chunk 6

```

You ensure that the context provided to the model isn't inconrgruent. This gist contains a more up to date approach but even that should be using XML instead of distinct tokens according to Open AI. https://gist.github.com/Donavan/d62d98ec75d611b35c516b7410a63a52


### Tips

- Make sure you're not providing it a bunch of tokens it doesn't need. 
    - You need to be using a minimum relevancy value to limit how many segments get returned.  
    - Tokens that  that don't contribute to the answer impact that ability of the model to find the ones that do.
"""

    url = "https://example.com/rag"
    
    print("\nTesting full document processing pipeline")
    print("=" * 60)
    
    try:
        # Process the document
        results = await process_document(sample_document, url)
        
        # Save detailed results to file
        save_chunk_details(results)
        
        # Print summary of results
        print("\n" + "=" * 60)
        print(f"Document processed into {len(results)} chunks with titles, summaries, and embeddings")
        print(f"Each chunk has an embedding vector of length {len(results[0]['embedding'])}")
        
        return results
    except Exception as e:
        print(f"Error in test_document_processing: {str(e)}")
        traceback.print_exc()
        raise


async def main():
    """Run all tests"""
    await test_document_processing()


if __name__ == "__main__":
    asyncio.run(main())
