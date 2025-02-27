"""AST-based code chunking for Python source code.

This module provides functionality to chunk Python code using Abstract Syntax Trees (AST),
ensuring that code blocks are split at logical boundaries while preserving context
and dependencies.
"""

import ast
from dataclasses import dataclass
from typing import List, Set, Optional


@dataclass
class CodeChunk:
    """Represents a chunk of code with its dependencies."""
    content: str
    imports: Set[str]
    global_deps: Set[str]
    start_line: int
    end_line: int


class ImportCollector(ast.NodeVisitor):
    """Collects all import statements from an AST."""
    
    def __init__(self):
        self.imports: Set[str] = set()
        self.import_lines: Set[int] = set()

    def visit_Import(self, node: ast.Import):
        """Handle simple imports: import foo, bar"""
        for name in node.names:
            self.imports.add(f"import {name.name}")
            self.import_lines.add(node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Handle from imports: from foo import bar, baz"""
        module = node.module or ''
        names = ', '.join(name.name for name in node.names)
        self.imports.add(f"from {module} import {names}")
        self.import_lines.add(node.lineno)


class DependencyCollector(ast.NodeVisitor):
    """Collects global dependencies (variables, functions, classes) used in a code block."""
    
    def __init__(self):
        self.dependencies: Set[str] = set()
        self.defined_names: Set[str] = set()
        
    def visit_Name(self, node: ast.Name):
        """Record any global names that are used."""
        if isinstance(node.ctx, ast.Load) and node.id not in self.defined_names:
            self.dependencies.add(node.id)
            
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Record function definitions and visit their bodies."""
        self.defined_names.add(node.name)
        self.generic_visit(node)
        
    def visit_ClassDef(self, node: ast.ClassDef):
        """Record class definitions and visit their bodies."""
        self.defined_names.add(node.name)
        self.generic_visit(node)


def get_node_source(node: ast.AST, source_lines: List[str]) -> str:
    """Extract source code for a given AST node."""
    if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
        return '\n'.join(source_lines[node.lineno - 1:node.end_lineno])
    return ''


def chunk_python_code(code: str, max_chunk_size: int = 5000) -> List[CodeChunk]:
    """
    Split Python code into chunks using AST analysis.
    
    Args:
        code: Python source code to chunk
        max_chunk_size: Maximum size for each chunk
        
    Returns:
        List of CodeChunk objects containing the chunked code and its dependencies
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # If code can't be parsed as Python, return it as a single chunk
        return [CodeChunk(
            content=code,
            imports=set(),
            global_deps=set(),
            start_line=1,
            end_line=len(code.splitlines())
        )]
    
    # Collect all imports
    import_collector = ImportCollector()
    import_collector.visit(tree)
    
    # Split code into lines for source extraction
    source_lines = code.splitlines()
    
    chunks: List[CodeChunk] = []
    current_chunk = []
    current_size = 0
    current_start_line = 1
    
    # Helper function to create a chunk
    def create_chunk(nodes: List[ast.AST], start_line: int) -> Optional[CodeChunk]:
        if not nodes:
            return None
            
        # Get source code for all nodes
        chunk_code = '\n'.join(get_node_source(node, source_lines) for node in nodes)
        
        # Collect dependencies for this chunk
        dep_collector = DependencyCollector()
        for node in nodes:
            dep_collector.visit(node)
            
        # Find the last line number
        end_line = max(
            getattr(node, 'end_lineno', start_line) 
            for node in nodes
        )
        
        return CodeChunk(
            content=chunk_code,
            imports=import_collector.imports,
            global_deps=dep_collector.dependencies - dep_collector.defined_names,
            start_line=start_line,
            end_line=end_line
        )
    
    # Process each top-level node
    for node in tree.body:
        node_source = get_node_source(node, source_lines)
        node_size = len(node_source)
        
        # If this single node exceeds max size, it needs its own chunk
        if node_size > max_chunk_size:
            # First, create a chunk from accumulated nodes if any
            if current_chunk:
                chunk = create_chunk(current_chunk, current_start_line)
                if chunk:
                    chunks.append(chunk)
                current_chunk = []
                current_size = 0
            
            # Create a chunk for this large node
            chunk = create_chunk([node], node.lineno)
            if chunk:
                chunks.append(chunk)
            current_start_line = node.end_lineno + 1 if hasattr(node, 'end_lineno') else node.lineno + 1
            
        # If adding this node would exceed max size, create a new chunk
        elif current_size + node_size > max_chunk_size and current_chunk:
            chunk = create_chunk(current_chunk, current_start_line)
            if chunk:
                chunks.append(chunk)
            current_chunk = [node]
            current_size = node_size
            current_start_line = node.lineno
            
        # Add node to current chunk
        else:
            current_chunk.append(node)
            current_size += node_size
    
    # Create final chunk from any remaining nodes
    if current_chunk:
        chunk = create_chunk(current_chunk, current_start_line)
        if chunk:
            chunks.append(chunk)
    
    return chunks


def format_chunk_with_context(chunk: CodeChunk) -> str:
    """
    Format a code chunk with its necessary context (imports and dependencies).
    
    Args:
        chunk: CodeChunk object containing code and its dependencies
        
    Returns:
        Formatted code string with necessary context
    """
    # Start with imports
    lines = []
    if chunk.imports:
        lines.extend(sorted(chunk.imports))
        lines.append('')  # Empty line after imports
    
    # Add the main content
    lines.append(chunk.content)
    
    return '\n'.join(lines)
