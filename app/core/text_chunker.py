"""Text-based chunking strategies."""
import os
import nltk
from typing import List



# Set NLTK data path to the project directory
nltk_data_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "nltk_data"
)
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download required NLTK data
nltk.download("punkt", download_dir=nltk_data_dir, quiet=True)


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using NLTK with fallback to simple splitting.
    
    Args:
        text: The text to split
        
    Returns:
        List of sentences
    """
    try:
        return nltk.sent_tokenize(text)
    except Exception:
        # Fallback to simple splitting if NLTK fails
        return text.split(". ")


def chunk_plain_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split plain text into chunks using sentence boundaries."""
    # Split into sentences
    sentences = split_into_sentences(text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        # If adding this sentence would exceed chunk size, save current chunk
        if current_length + sentence_length > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            
            # Keep some sentences for overlap
            if chunk_overlap > 0:
                # Calculate how many sentences to keep for overlap
                overlap_length = 0
                overlap_sentences = []
                
                for s in reversed(current_chunk):
                    if overlap_length + len(s) > chunk_overlap:
                        break
                    overlap_length += len(s)
                    overlap_sentences.insert(0, s)
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            else:
                current_chunk = []
                current_length = 0
        
        current_chunk.append(sentence)
        current_length += sentence_length
    
    # Add the last chunk if there is one
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks
