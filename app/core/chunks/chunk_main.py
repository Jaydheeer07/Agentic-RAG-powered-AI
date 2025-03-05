"""Main chunking module for handling different content types."""

import re
from typing import List, Tuple

from app.config import settings
from app.core.chunks.ast_chunker import chunk_python_code
from app.core.chunks.text_chunker import chunk_plain_text


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

    # If we have markdown code blocks, it's mixed content
    if code_markers > 0:
        return "mixed"
    # If we have multiple code patterns but no markers, it might be pure code
    elif code_pattern_matches >= 2:
        return "code"
    return "text"


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


def extract_code_blocks(text: str) -> Tuple[str, List[str]]:
    """Extract code blocks from markdown text."""
    # Split on code block markers, preserving the markers
    parts = re.split(r"(```(?:.*?)\n[\s\S]*?```)", text)

    text_parts = []
    code_blocks = []

    for part in parts:
        if part.startswith("```"):
            # Store code block with markers
            code_blocks.append(part)
            # Add a placeholder for reconstruction
            text_parts.append(f"[CODE_BLOCK_{len(code_blocks) - 1}]")
        else:
            text_parts.append(part)

    return "".join(text_parts), code_blocks


def find_list_boundary(text: str, start_idx: int) -> int:
    """Find the end of a list starting at start_idx."""
    lines = text[start_idx:].split("\n")
    list_patterns = [
        r"^\s*[\-\*]\s",  # Bullet points
        r"^\s*\d+[\.\)]\s",  # Numbered lists
        r"^\s*[a-zA-Z][\.\)]\s",  # Letter lists
    ]

    end_idx = start_idx
    in_list = False
    list_indent = 0

    for i, line in enumerate(lines):
        # Skip empty lines at start
        if not in_list and not line.strip():
            end_idx += len(line) + 1
            continue

        # Check if line is list item
        is_list_item = any(re.match(pattern, line) for pattern in list_patterns)

        if is_list_item and not in_list:
            # Start of list
            in_list = True
            list_indent = len(line) - len(line.lstrip())
        elif in_list:
            # Check if we're still in list
            if not line.strip():
                # Empty line might be list continuation
                continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent < list_indent and not is_list_item:
                # End of list
                break
        elif not is_list_item and line.strip():
            # Non-list content
            break

        end_idx += len(line) + 1

    return end_idx


def chunk_text(
    text: str,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
) -> List[str]:
    """Split text into chunks while preserving semantic boundaries."""
    content_type = detect_content_type(text)
    chunk_size, chunk_overlap = get_chunk_params(
        content_type, chunk_size, chunk_overlap
    )

    if content_type == "mixed":
        text_without_code, code_blocks = extract_code_blocks(text)

        # Split into paragraphs first
        paragraphs = re.split(r"\n\s*\n", text_without_code)

        chunks = []
        current_chunk = []
        current_size = 0

        for paragraph in paragraphs:
            # Check if paragraph contains a list
            list_match = re.match(r"^(\s*(?:[\-\*]|\d+[\.\)]|\w[\.\)])\s)", paragraph)
            if list_match:
                # This is a list - try to keep it together
                if current_size + len(paragraph) > chunk_size and current_chunk:
                    # List won't fit in current chunk, start a new one
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_size = 0

                current_chunk.append(paragraph)
                current_size += len(paragraph)
                continue

            # Check if paragraph contains a code block placeholder
            if "[CODE_BLOCK_" in paragraph:
                block_idx = int(re.search(r"\[CODE_BLOCK_(\d+)\]", paragraph).group(1))
                code_block = code_blocks[block_idx]

                # Try to keep code with its context
                context_size = (
                    len(paragraph) - len(f"[CODE_BLOCK_{block_idx}]") + len(code_block)
                )

                if current_size + context_size > chunk_size and current_chunk:
                    # Code and context won't fit, save current chunk
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # Replace placeholder with actual code
                current_chunk.append(
                    paragraph.replace(f"[CODE_BLOCK_{block_idx}]", code_block)
                )
                current_size += context_size
                continue

            # Regular paragraph
            if current_size + len(paragraph) > chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(paragraph)
            current_size += len(paragraph)

        # Add the last chunk if there is one
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    elif content_type == "code":
        code_chunks = chunk_python_code(text, chunk_size)
        return [chunk.content for chunk in code_chunks]
    else:
        return chunk_plain_text(text, chunk_size, chunk_overlap)
