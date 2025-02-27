import os
import re
from typing import List

import nltk

from app.config import settings
from app.core.ast_chunker import chunk_python_code, format_chunk_with_context

# Set NLTK data path to the project directory
nltk_data_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "nltk_data"
)
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download required NLTK data
nltk.download("punkt", download_dir=nltk_data_dir, quiet=True)


def is_python_code(text: str) -> bool:
    """
    Determine if the text is likely Python code.
    
    Args:
        text: Text to analyze
        
    Returns:
        bool: True if text appears to be Python code
    """
    # Check for common Python patterns
    python_patterns = [
        r"def\s+\w+\s*\(",  # Function definition
        r"class\s+\w+",  # Class definition
        r"import\s+\w+",  # Import statement
        r"from\s+\w+\s+import",  # From import
        r"@\w+",  # Decorator
        r"with\s+\w+",  # With statement
        r"try:\s*\n",  # Try block
        r"if\s+__name__\s*==\s*['\"]__main__['\"]",  # Main block
    ]
    
    # Count how many Python patterns we find
    pattern_matches = sum(1 for pattern in python_patterns if re.search(pattern, text))
    
    # If we find multiple Python patterns, it's likely Python code
    return pattern_matches >= 2


def detect_content_type(text: str) -> str:
    """
    Detect if the content is code, text, or mixed.
    
    Args:
        text: The text to analyze
        
    Returns:
        str: 'code', 'text', or 'mixed'
    """
    # Check for code block markers
    code_markers = len(re.findall(r"```[\s\S]*?```", text))
    
    # Check for common code patterns
    code_patterns = [
        r"def\s+\w+\s*\(",  # Python function
        r"class\s+\w+",  # Python class
        r"import\s+\w+",  # Python import
        r"from\s+\w+\s+import",  # Python from import
        r"<[^>]+>",  # HTML tags
        r"\w+\s*=\s*function",  # JavaScript function
    ]
    
    code_pattern_matches = sum(
        1 for pattern in code_patterns if re.search(pattern, text)
    )
    
    if code_markers > 0 or code_pattern_matches >= 2:
        return "code" if code_markers > 0 else "mixed"
    return "text"


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
        # Fallback to simple sentence splitting
        simple_splits = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in simple_splits if s.strip()]


def get_chunk_params(
    content_type: str, max_chunk_size: int, max_chunk_overlap: int
) -> tuple[int, int]:
    """
    Get appropriate chunk size and overlap based on content type and maximum limits.
    
    Args:
        content_type: Type of content ('code', 'text', or 'mixed')
        max_chunk_size: Maximum allowed chunk size from settings
        max_chunk_overlap: Maximum allowed chunk overlap from settings
        
    Returns:
        Tuple of (chunk_size, chunk_overlap)
    """
    if content_type == "code":
        # For code: use larger chunks with no overlap
        chunk_size = min(max_chunk_size, max(512, max_chunk_size))
        chunk_overlap = 0
    elif content_type == "mixed":
        # For mixed: use medium chunks with minimal overlap
        chunk_size = min(max_chunk_size, max(300, int(max_chunk_size * 0.6)))
        chunk_overlap = min(max_chunk_overlap, int(chunk_size * 0.1))
    else:  # text
        # For text: use smaller chunks with more overlap
        chunk_size = min(max_chunk_size, max(200, int(max_chunk_size * 0.4)))
        chunk_overlap = min(
            max_chunk_overlap,
            max(
                int(chunk_size * 0.1),  # minimum 10% overlap
                min(
                    int(chunk_size * 0.2),  # maximum 20% overlap
                    max_chunk_overlap,
                ),
            ),
        )
    
    return chunk_size, chunk_overlap


def extract_code_blocks(text: str) -> tuple[str, List[tuple[str, str]]]:
    """
    Extract code blocks from text and replace with placeholders.
    
    Args:
        text: Text containing code blocks
        
    Returns:
        Tuple of (text with placeholders, list of (placeholder, code block))
    """
    code_blocks = []
    text_without_code = text
    
    # Extract code blocks and replace with placeholders
    code_pattern = r"```(?:python)?\n([\s\S]*?)```"
    for i, match in enumerate(re.finditer(code_pattern, text)):
        code = match.group(1).strip()
        placeholder = f"CODE_BLOCK_{i}"
        code_blocks.append((placeholder, code))
        text_without_code = text_without_code.replace(match.group(0), placeholder)
    
    return text_without_code, code_blocks


def chunk_plain_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split plain text into chunks using sentence boundaries."""
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        # If adding this sentence would exceed chunk size
        if current_length + sentence_length > chunk_size and current_chunk:
            # Save current chunk
            chunks.append(" ".join(current_chunk))
            # Keep last part for overlap
            overlap_size = 0
            overlap_chunk = []
            for s in reversed(current_chunk):
                if overlap_size + len(s) > chunk_overlap:
                    break
                overlap_size += len(s)
                overlap_chunk.insert(0, s)
            current_chunk = overlap_chunk
            current_length = overlap_size
        
        current_chunk.append(sentence)
        current_length += sentence_length
    
    # Add the last chunk if there is one
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def chunk_text(
    text: str,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping chunks while preserving structure and semantic boundaries.
    
    The chunk_size and chunk_overlap parameters from settings represent maximum allowed values.
    Actual chunk sizes and overlaps are adjusted based on content type:
    - Code: Larger chunks (512+ chars) with no overlap
    - Mixed: Medium chunks with minimal overlap
    - Text: Smaller chunks with 10-20% overlap
    
    For Python code, uses AST-based chunking to preserve logical boundaries.
    
    Args:
        text: The text to split into chunks
        chunk_size: Maximum chunk size (default: from settings)
        chunk_overlap: Maximum chunk overlap (default: from settings)
    
    Returns:
        List of text chunks
    """
    # Detect content type
    content_type = detect_content_type(text)
    
    # Get appropriate chunk parameters
    chunk_size, chunk_overlap = get_chunk_params(content_type, chunk_size, chunk_overlap)
    
    # For Python code, use AST-based chunking
    if is_python_code(text):
        return chunk_python_code(text)
        
    # For mixed content, handle code blocks separately
    if content_type == "mixed":
        # Extract code blocks
        text_with_placeholders, code_blocks = extract_code_blocks(text)
        
        # Chunk the text
        chunks = chunk_plain_text(text_with_placeholders, chunk_size, chunk_overlap)
        
        # Restore code blocks
        final_chunks = []
        for chunk in chunks:
            chunk_with_code = chunk
            for placeholder, code in code_blocks:
                if placeholder in chunk:
                    if is_python_code(code):
                        chunk_with_code = chunk_with_code.replace(
                            placeholder, 
                            "\n".join(chunk_python_code(code))
                        )
                    else:
                        chunk_with_code = chunk_with_code.replace(
                            placeholder, 
                            f"```\n{code}\n```"
                        )
            final_chunks.append(chunk_with_code)
        return final_chunks
    
    # For plain text, use sentence-based chunking
    return chunk_plain_text(text, chunk_size, chunk_overlap)
