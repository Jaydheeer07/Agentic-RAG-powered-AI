# Text Chunking Strategy for RAG-powered AI Agent

This document outlines the comprehensive strategy for text chunking in our RAG-powered AI agent, including recommendations for implementation and optimization.

## Core Strategies

### 1. Code Block Chunking
- **Chunk Size**: 512-1024 characters
- **Key Approaches**:
  - AST-based splitting for programming languages
  - Split at logical points (functions, classes, loops)
  - Preserve imports and dependencies
  - No overlap between chunks to maintain syntax integrity

### 2. Regular Text Chunking
- **Chunk Size**: 200-500 characters
- **Key Approaches**:
  - Split at sentence/paragraph boundaries
  - Preserve semantic meaning
  - Handle lists and bullet points as units
  - Use HTML/CSS tags for structural hints
  - 10-20% overlap between chunks

### 3. Overlap Strategy
- **Text Content**: 
  - 10-20% overlap
  - Align with sentence boundaries
  - Preserve context between chunks
- **Code Content**:
  - Zero overlap when using syntax-aware chunking
  - Preserve function/class integrity
- **Mixed Content**:
  - Special handling for text with inline code
  - Preserve documentation structure

### 4. Content-Aware Processing
- **Detection Methods**:
  - HTML tags (`<pre>`, `<code>`, `<p>`)
  - Syntax pattern recognition
  - ML classifiers for ambiguous content
- **Special Cases**:
  - Code comments
  - Documentation strings
  - Mixed content sections

## Implementation Tools

### Code Processing
- **Primary Tools**:
  - `tree-sitter`: AST parsing
  - `pygments`: Syntax detection
  - Custom regex patterns

### Text Processing
- **Primary Tools**:
  - `spaCy` or `nltk`: Sentence splitting
  - `langchain.text_splitter`: Advanced text splitting
  - SBERT: Semantic grouping

## Processing Workflow

1. **Content Analysis**
   - Detect content type
   - Identify structure and boundaries
   - Choose appropriate chunking strategy

2. **Preprocessing**
   - Extract code blocks
   - Identify natural text boundaries
   - Handle special cases (lists, tables)

3. **Chunking Execution**
   - Apply appropriate chunk size
   - Maintain structural integrity
   - Handle overlaps based on content type

4. **Post-processing**
   - Validate chunk integrity
   - Ensure context preservation
   - Handle edge cases

## Optimization Considerations

- **Performance**:
  - Implement caching for frequent patterns
  - Batch processing for large documents
  - Parallel processing where possible

- **Quality**:
  - Regular validation of chunk quality
  - Monitoring of AI agent performance
  - Feedback loop for refinement

## Edge Cases to Handle

1. **Mixed Content**
   - Documentation with code examples
   - Markdown files with multiple content types
   - Complex HTML structures

2. **Special Formats**
   - Tables and structured data
   - Mathematical formulas
   - Code comments and docstrings

3. **Language-Specific**
   - Different programming languages
   - Multiple natural languages
   - Special characters and encodings

## Future Improvements

- Implement machine learning for boundary detection
- Add support for more programming languages
- Optimize chunk sizes based on performance metrics
- Enhance handling of mixed content types
- Add support for more document formats

## Documentation for Current Chunking Strategy and Planned Improvements

### Current Implementation

#### Overview
Our chunking system implements a content-aware strategy that adapts its behavior based on the type of content being processed (code, text, or mixed). This approach ensures optimal chunk sizes while preserving context and structure.

#### Configuration
- **Model**: OpenAI's text-embedding-3-small
  - Context window: 8,191 tokens
  - Output dimensions: 1,536
- **Maximum Parameters**:
  - `CHUNK_SIZE`: 16,000 characters (≈50% of max token limit)
  - `CHUNK_OVERLAP`: 3,200 characters (20% of chunk size)

#### Content-Type Based Adjustments

1. **Code Content**:
   - Minimum size: 512 characters
   - Maximum size: 16,000 characters
   - No overlap to preserve code block integrity
   - Preserves function and class boundaries

2. **Mixed Content**:
   - Size: 60% of max (≈9,600 characters)
   - Overlap: 10% of chunk size
   - Balances code preservation with context maintenance

3. **Text Content**:
   - Size: 40% of max (≈6,400 characters)
   - Overlap: 10-20% of chunk size
   - Prioritizes semantic boundaries using NLTK

#### Key Features
1. **Content Detection**:
   - Pattern-based identification of code blocks
   - Recognition of common code markers
   - Fallback mechanisms for edge cases

2. **Boundary Detection**:
   - NLTK-based sentence boundary detection for text
   - Code block preservation
   - Paragraph and header recognition

3. **Error Handling**:
   - Graceful fallback for NLTK failures
   - Robust handling of malformed input
   - Preservation of markdown structure

#### Validation Results
Tests with real-world examples show effective chunking across different content types:

1. **Documentation (FastAPI)**:
   - Single chunk: 3,798 characters
   - Preserved code examples and context
   - Maintained markdown structure

2. **Pure Code (asyncio Queue)**:
   - Split into appropriate chunks (1,873-2,825 characters)
   - Preserved class and function integrity
   - Maintained docstring context

3. **Mixed Content (Technical Blog)**:
   - Balanced handling of text and code
   - Preserved code examples
   - Maintained readability

### Planned Improvements

#### 1. AST-Based Code Chunking
**Priority: High**
- Parse code into Abstract Syntax Trees
- Split at logical boundaries (functions, classes)
- Preserve imports and dependencies
- Maintain code context across chunks

#### 2. Semantic Text Chunking
**Priority: Medium**
- Use NLP to identify topic boundaries
- Implement semantic similarity checks
- Improve overlap selection based on content
- Handle multi-language content

#### 3. ML-Based Content Detection
**Priority: Medium**
- Train model to classify content types
- Improve accuracy of boundary detection
- Handle edge cases and mixed content better
- Support more programming languages

#### 4. Enhanced Overlap Strategy
**Priority: Low**
- Dynamic overlap based on content similarity
- Context-aware boundary selection
- Improved handling of cross-references
- Better preservation of semantic relationships

### Next Steps
The recommended next improvement is **AST-Based Code Chunking** because:
1. It provides the most immediate value for code-heavy content
2. It's a deterministic approach with reliable results
3. It builds naturally on our current implementation
4. It addresses the most common edge cases in our current system

### Implementation Plan for AST-Based Chunking:
1. Add Python's `ast` module for code parsing
2. Implement AST visitor pattern for boundary detection
3. Preserve import statements and dependencies
4. Add tests with complex code structures
5. Integrate with existing content-type detection

This improvement will significantly enhance our handling of code-heavy content while maintaining our current capabilities for text and mixed content.
