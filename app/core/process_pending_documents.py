import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.chunks.ast_chunker import CodeChunk, format_chunk_with_context
from app.core.chunks.chunk_main import chunk_text, detect_content_type
from app.core.db_manager import DatabaseManager
from app.core.get_embeddings import get_embedding
from app.core.title_summary import get_title_and_summary
from app.models.database import Chunk, Document


class ChunkProcessor:
    """Handles processing and storing document chunks."""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    async def process_and_store_document(
        self,
        document_text: str,
        url: str,
        title: Optional[str] = None,
        batch_size: int = 50,
    ) -> Dict[str, any]:
        """
        Process a document and store its chunks in the database.

        Args:
            document_text: Document text to process
            url: Source URL of the document
            title: Optional document title
            batch_size: Number of chunks to process in each batch

        Returns:
            Dict containing processing statistics
        """
        stats = {
            "total_chunks": 0,
            "successful_chunks": 0,
            "failed_chunks": 0,
            "processing_time": 0,
        }

        start_time = time.time()

        try:
            # Process document to get chunks
            chunks = chunk_text(document_text)
            stats["total_chunks"] = len(chunks)

            # Create document record
            with self.db_manager.get_session() as session:
                doc = Document(
                    url=url,
                    title=title,
                    content=document_text,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )
                session.add(doc)
                session.commit()

                # Process chunks in batches
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i : i + batch_size]
                    await self._process_chunk_batch(session, doc.id, batch, stats)

        except Exception as e:
            print(f"Error processing document: {str(e)}")
            traceback.print_exc()
            stats["error"] = str(e)

        stats["processing_time"] = time.time() - start_time
        return stats

    async def _process_chunk_batch(
        self,
        session: Session,
        document_id: int,
        chunks: List[Union[str, CodeChunk]],
        stats: Dict,
    ):
        """Process and store a batch of chunks."""
        try:
            for chunk in chunks:
                try:
                    # Get chunk text
                    processed_text = get_chunk_text(chunk)

                    # Get embedding
                    embedding = await get_embedding(processed_text)

                    # Create metadata
                    metadata = {}
                    if isinstance(chunk, CodeChunk):
                        metadata = {
                            "type": "code",
                            "start_line": chunk.start_line,
                            "end_line": chunk.end_line,
                            "imports": list(chunk.imports),
                            "global_deps": list(chunk.global_deps),
                        }

                    # Create chunk record
                    db_chunk = Chunk(
                        document_id=document_id,
                        content=processed_text,
                        embedding=embedding,
                        metadata=metadata,
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc),
                    )
                    session.add(db_chunk)
                    stats["successful_chunks"] += 1

                except Exception as e:
                    print(f"Error processing chunk: {str(e)}")
                    stats["failed_chunks"] += 1
                    continue

            # Commit the batch
            session.commit()

        except SQLAlchemyError as e:
            print(f"Database error processing batch: {str(e)}")
            session.rollback()
            stats["failed_chunks"] += len(chunks)
        except Exception as e:
            print(f"Unexpected error processing batch: {str(e)}")
            session.rollback()
            stats["failed_chunks"] += len(chunks)


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
        print("\nProcessing Document:")
        print(f"Input text length: {len(document)} characters")
        print(f"Content type: {content_type}")

        # Chunk the document
        chunks = chunk_text(document)
        print("\nChunking Results:")
        print(f"Number of chunks: {len(chunks)}")
        print(f"Chunk sizes: {[len(get_chunk_text(c)) for c in chunks]}")
        print(
            f"Average chunk size: {sum(len(get_chunk_text(c)) for c in chunks) / len(chunks):.2f} characters"
        )

        # Process each chunk
        results = []
        for i, chunk in enumerate(chunks):
            print(f"\nProcessing chunk {i + 1}/{len(chunks)}")
            print("Chunk type:", type(chunk))

            try:
                # Get the text content for processing
                processed_text = get_chunk_text(chunk)
                print(
                    f"Successfully converted chunk to text, length: {len(processed_text)}"
                )
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
                embedding = await get_embedding(
                    processed_text, title=title_and_summary["title"]
                )
                elapsed = time.time() - start_time
                print(f"Embedding API call completed in {elapsed:.2f} seconds")
                print(f"Embedding vector length: {len(embedding)}")

                # Only append results if we successfully got the embedding
                results.append(
                    {
                        "chunk": chunk,
                        "title": title_and_summary["title"],
                        "summary": title_and_summary["summary"],
                        "embedding": embedding,
                        "url": url,
                    }
                )
            except Exception as e:
                print(f"Error getting embedding: {str(e)}")
                traceback.print_exc()
                continue

        return results
    except Exception as e:
        print(f"Error in process_document: {str(e)}")
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
            chunk = result["chunk"]
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
            f.write(
                f"\nEmbedding Vector (first 5 dimensions): {result['embedding'][:5]}\n"
            )
            f.write(f"Embedding Vector Length: {len(result['embedding'])}\n")
            f.write("\n" + "=" * 50 + "\n\n")

        print(f"\nChunk details saved to: {output_file}")
