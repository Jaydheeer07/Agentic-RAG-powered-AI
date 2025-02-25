import os
import re
from typing import List

import nltk

from app.config import settings

# Set NLTK data path to the project directory
nltk_data_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "nltk_data"
)
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download required NLTK data
nltk.download("punkt", download_dir=nltk_data_dir, quiet=True)


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


def chunk_text(
    text: str,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping chunks while preserving markdown structure and semantic boundaries.

    The chunk_size and chunk_overlap parameters from settings represent maximum allowed values.
    Actual chunk sizes and overlaps are adjusted based on content type:
    - Code: Larger chunks (512+ chars) with no overlap
    - Mixed: Medium chunks with minimal overlap
    - Text: Smaller chunks with 10-20% overlap

    Args:
        text: The text to split into chunks
        chunk_size: Maximum chunk size (default: from settings)
        chunk_overlap: Maximum chunk overlap (default: from settings)

    Returns:
        List of text chunks
    """
    # Detect content type
    content_type = detect_content_type(text)

    # Get appropriate chunk parameters based on content type
    adjusted_size, adjusted_overlap = get_chunk_params(
        content_type, chunk_size, chunk_overlap
    )

    # Preserve code blocks
    code_blocks = []
    text_without_code = text

    # Extract code blocks and replace with placeholders
    code_pattern = r"```[\s\S]*?```"
    for i, match in enumerate(re.finditer(code_pattern, text)):
        placeholder = f"CODE_BLOCK_{i}"
        code_blocks.append(match.group())
        text_without_code = text_without_code.replace(match.group(), placeholder)

    # Split on semantic boundaries
    splits = []
    current_chunk = ""
    current_size = 0

    # Split on paragraphs and headers first
    paragraphs = re.split(r"\n\s*\n|\n#{1,6}\s", text_without_code)

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # For text content, split into sentences
        if content_type == "text":
            sentences = split_into_sentences(paragraph)
        else:
            sentences = [paragraph]

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If adding this sentence would exceed chunk size
            if current_size + len(sentence) > adjusted_size:
                if current_chunk:
                    splits.append(current_chunk.strip())
                current_chunk = sentence
                current_size = len(sentence)
            else:
                if current_chunk:
                    current_chunk += " "
                current_chunk += sentence
                current_size += len(sentence) + 1  # +1 for space

    if current_chunk:
        splits.append(current_chunk.strip())

    # Create overlapping chunks
    chunks = []
    for i in range(len(splits)):
        chunk = splits[i]

        # Add overlap from next chunk if available and if not code
        if i < len(splits) - 1 and adjusted_overlap > 0 and content_type != "code":
            next_chunk = splits[i + 1]
            # For text content, try to overlap at sentence boundaries
            if content_type == "text":
                next_sentences = split_into_sentences(next_chunk)
                overlap_text = ""
                for sent in next_sentences:
                    if len(overlap_text) + len(sent) <= adjusted_overlap:
                        overlap_text += " " + sent
                    else:
                        break
                if overlap_text:
                    chunk += "\n\n" + overlap_text.strip()
            else:
                overlap_text = next_chunk[:adjusted_overlap]
                chunk += "\n\n" + overlap_text

        # Restore code blocks
        for j, code_block in enumerate(code_blocks):
            chunk = chunk.replace(f"CODE_BLOCK_{j}", code_block)

        chunks.append(chunk)

    return chunks
