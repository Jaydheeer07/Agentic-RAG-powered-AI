from app.config import settings
from app.core.chunking import chunk_text, detect_content_type


def print_chunk_stats(chunks: list, content_type: str):
    """Print statistics about the chunks"""
    total_chars = sum(len(chunk) for chunk in chunks)
    avg_chunk_size = total_chars / len(chunks) if chunks else 0
    print("\nChunk Statistics:")
    print(f"Content type detected: {content_type}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Average chunk size: {avg_chunk_size:.0f} characters")
    print(f"Smallest chunk: {min(len(chunk) for chunk in chunks)} characters")
    print(f"Largest chunk: {max(len(chunk) for chunk in chunks)} characters")
    print("-" * 50)


def test_general_chunking():
    """
    Test general chunking functionality with real-world examples.
    Tests the basic chunking strategy with:
    - Mixed content (FastAPI docs)
    - Large code files (asyncio queue)
    - Technical blog posts
    """
    print("\nTesting General Chunking Strategy...")
    print(f"CHUNK_SIZE = {settings.CHUNK_SIZE}")
    print(f"CHUNK_OVERLAP = {settings.CHUNK_OVERLAP}")
    print("=" * 50)

    # Test case 1: Long documentation (FastAPI's dependency injection docs)
    print("\n1. FastAPI Documentation (mixed content)")
    fastapi_docs = """# Dependency Injection in FastAPI

FastAPI provides a powerful dependency injection system that helps you:
- Share logic (dependencies)
- Share database connections
- Enforce security, authentication, role requirements, etc.
- And many other things

## What is "Dependency Injection"

"Dependency Injection" means that there's a way for your code to declare things that it requires to work and use: "dependencies".

And then, that system (FastAPI) will take care of doing whatever is needed to provide your code with those needed dependencies ("inject" the dependencies).

This is very useful when you need to:

* Share logic (the same code logic) between multiple parts of the application
* Share database connections
* Enforce security, authentication, role requirements, etc.
* And many other things...

All these, while minimizing code repetition.

## First Steps

### Create a dependency

Let's create a basic dependency.

In this case, let's say you have a dependency that just returns a dictionary:

```python
from fastapi import Depends, FastAPI

app = FastAPI()

async def common_parameters(q: str | None = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}

@app.get("/items/")
async def read_items(commons: dict = Depends(common_parameters)):
    return commons

@app.get("/users/")
async def read_users(commons: dict = Depends(common_parameters)):
    return commons
```

In this example, both path operations share the same common parameters using the dependency.

The key points are:
* You declare your dependency as a function
* Your dependency can have parameters
* Dependencies can have sub-dependencies
* A dependency can be used in multiple places

## Classes as Dependencies

You can also use Python classes as dependencies. This is especially useful when your dependency requires some setup or cleanup:

```python
from fastapi import Depends, FastAPI
from typing import Annotated

app = FastAPI()

class CommonQueryParams:
    def __init__(self, q: str | None = None, skip: int = 0, limit: int = 100):
        self.q = q
        self.skip = skip
        self.limit = limit

@app.get("/items/")
async def read_items(commons: Annotated[CommonQueryParams, Depends(CommonQueryParams)]):
    return {"q": commons.q, "skip": commons.skip, "limit": commons.limit}
```

## Sub-dependencies

You can create dependencies that themselves depend on other dependencies:

```python
from fastapi import Cookie, Depends, FastAPI
from typing import Annotated

app = FastAPI()

def query_extractor(q: str | None = None):
    return q

def query_or_cookie_extractor(
    q: Annotated[str, Depends(query_extractor)],
    last_query: Annotated[str | None, Cookie()] = None,
):
    if not q and last_query:
        return last_query
    return q

@app.get("/items/")
async def read_items(query_or_default: Annotated[str, Depends(query_or_cookie_extractor)]):
    return {"q": query_or_default}
```

## Dependencies in path operation decorators

You can also add dependencies to the path operation decorator:

```python
from fastapi import Depends, FastAPI, Header, HTTPException
from typing import Annotated

app = FastAPI()

async def verify_token(x_token: Annotated[str, Header()]):
    if x_token != "fake-super-secret-token":
        raise HTTPException(status_code=400, detail="X-Token header invalid")

async def verify_key(x_key: Annotated[str, Header()]):
    if x_key != "fake-super-secret-key":
        raise HTTPException(status_code=400, detail="X-Key header invalid")
    return x_key

@app.get("/items/", dependencies=[Depends(verify_token), Depends(verify_key)])
async def read_items():
    return [{"item": "Foo"}, {"item": "Bar"}]
```

This is especially useful for:
* Security requirements
* Database session handling
* Response compression
* And many other cases...
"""

    content_type = detect_content_type(fastapi_docs)
    chunks = chunk_text(fastapi_docs)
    print_chunk_stats(chunks, content_type)

    # Test case 2: Complex Python module (asyncio queue implementation)
    print("\n2. Python Module - asyncio Queue (pure code)")
    python_code = '''"""A queue implementation for asyncio."""

__all__ = ['Queue', 'PriorityQueue', 'LifoQueue', 'QueueFull', 'QueueEmpty']

import collections
import heapq
import warnings
import asyncio
from asyncio import events
from asyncio import locks

class QueueEmpty(Exception):
    """Exception raised when Queue.get_nowait() is called on an empty Queue."""
    pass

class QueueFull(Exception):
    """Exception raised when Queue.put_nowait() is called on a full Queue."""
    pass

class Queue:
    """A queue, useful for coordinating producer and consumer coroutines.

    If maxsize is less than or equal to zero, the queue size is infinite. If it
    is an integer greater than 0, then "await put()" will block when the
    queue reaches maxsize, until an item is removed by get().

    Unlike the standard library Queue, you can reliably know this Queue's size
    with qsize(), since your single-threaded asyncio application won't be
    interrupted between calling qsize() and doing an operation on the Queue.
    """

    def __init__(self, maxsize=0, *, loop=None):
        """Initialize a queue with a given maximum size.

        If maxsize is less than or equal to zero, the queue size is infinite.
        """
        warnings.warn("The loop argument is deprecated since Python 3.8, "
                     "and scheduled for removal in Python 3.10.",
                     DeprecationWarning, stacklevel=2)
        if loop is None:
            loop = events.get_event_loop()
        self._maxsize = maxsize
        self._loop = loop
        self._init(maxsize)

        # Futures for waiting puts and gets
        self._getters = collections.deque()
        self._putters = collections.deque()

        # Lock for protecting the queue data structure
        self._unfinished_tasks = 0
        self._finished = locks.Event()
        self._finished.set()

    def _init(self, maxsize):
        """Initialize the queue's storage."""
        self._queue = collections.deque()

    def _get(self):
        """Remove and return an item from the queue."""
        return self._queue.popleft()

    def _put(self, item):
        """Put an item into the queue."""
        self._queue.append(item)

    def empty(self):
        """Return True if the queue is empty, False otherwise."""
        return not self._queue

    def full(self):
        """Return True if there are maxsize items in the queue.

        Note: if the Queue was initialized with maxsize=0 (the default),
        then full() is never True.
        """
        if self._maxsize <= 0:
            return False
        return self.qsize() >= self._maxsize

    def qsize(self):
        """Number of items in the queue."""
        return len(self._queue)

    @property
    def maxsize(self):
        """Number of items allowed in the queue."""
        return self._maxsize

    def put_nowait(self, item):
        """Put an item into the queue without blocking.

        If no free slot is immediately available, raise QueueFull.
        """
        if self.full():
            raise QueueFull
        self._put(item)
        self._unfinished_tasks += 1
        self._finished.clear()

    def get_nowait(self):
        """Remove and return an item from the queue without blocking.

        If no item is immediately available, raise QueueEmpty.
        """
        if self.empty():
            raise QueueEmpty
        return self._get()

    async def put(self, item):
        """Put an item into the queue.

        If the queue is full, wait until a free slot is available
        before adding the item.
        """
        while self.full():
            putter = self._loop.create_future()
            self._putters.append(putter)
            try:
                await putter
            except:
                putter.cancel()  # Just in case putter is not done yet
                try:
                    self._putters.remove(putter)
                except ValueError:
                    pass
                if not self.full() and not putter.cancelled():
                    self._wakeup_next(self._putters)
                raise
        return self.put_nowait(item)

    async def get(self):
        """Remove and return an item from the queue.

        If queue is empty, wait until an item is available.
        """
        while self.empty():
            getter = self._loop.create_future()
            self._getters.append(getter)
            try:
                await getter
            except:
                getter.cancel()  # Just in case getter is not done yet
                try:
                    self._getters.remove(getter)
                except ValueError:
                    pass
                if not self.empty() and not getter.cancelled():
                    self._wakeup_next(self._getters)
                raise
        return self.get_nowait()
'''
    content_type = detect_content_type(python_code)
    chunks = chunk_text(python_code)
    print_chunk_stats(chunks, content_type)

    # Test case 3: Technical blog post
    print("\n3. Technical Blog Post (mixed content)")
    tech_blog = '''# Understanding Python Decorators: A Comprehensive Guide

Python decorators are a powerful way to modify or enhance functions or classes without directly changing their source code. They follow the principle of open-closed: code should be open for extension but closed for modification.

## What are Decorators?

At their core, decorators are just functions that take another function as an argument and return a modified version of that function. Here's a simple example:

```python
def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

# When we call say_hello(), we get:
# Something is happening before the function is called.
# Hello!
# Something is happening after the function is called.
```

## Practical Examples

### 1. Timing Functions

One common use case is measuring execution time:

```python
import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
    return wrapper

@timing_decorator
def slow_function():
    time.sleep(1)
    return "Function finished!"
```

### 2. Caching Results

Another practical use is caching function results:

```python
def memoize(func):
    cache = {}
    @wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

@memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

## Advanced Decorator Patterns

### Class Decorators

Decorators can also be used with classes:

```python
def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class DatabaseConnection:
    def __init__(self):
        print("Creating database connection...")
```

### Decorators with Arguments

You can create decorators that accept arguments:

```python
def repeat(times):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(times=3)
def greet(name):
    print(f"Hello {name}")
```

## Best Practices

1. **Use functools.wraps**: Always use @wraps when creating decorators to preserve the original function's metadata.

2. **Keep it Simple**: Decorators should do one thing and do it well. If you need complex functionality, consider using multiple decorators.

3. **Handle Arguments Properly**: Make sure your decorators can handle various argument patterns:

```python
def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # This pattern allows any number of arguments
        return func(*args, **kwargs)
    return wrapper
```

4. **Document Your Decorators**: Like any other function, decorators should be well-documented:

```python
def retry(max_attempts=3, delay_seconds=1):
    """
    Retry a function multiple times if it raises an exception.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay_seconds: Delay between attempts in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        raise e
                    time.sleep(delay_seconds)
            return None
        return wrapper
    return decorator
```

## Common Pitfalls

1. **Modifying Function Signatures**: Be careful not to modify the original function's signature unless that's explicitly your intention.

2. **Order of Decorators**: When using multiple decorators, remember they are applied from bottom to top:

```python
@decorator1
@decorator2
def function():
    pass

# This is equivalent to:
# function = decorator1(decorator2(function))
```

3. **Performance Impact**: Remember that each decorator adds a function call overhead. For performance-critical code, measure the impact.

Remember that decorators are a powerful tool in Python, but with great power comes great responsibility. Use them wisely to make your code more maintainable and elegant.'''
    content_type = detect_content_type(tech_blog)
    chunks = chunk_text(tech_blog)
    print_chunk_stats(chunks, content_type)


def test_ast_chunking():
    """
    Test AST-based chunking with Python code.
    Tests the advanced chunking strategy with:
    - Python classes and methods
    - Multiple functions with imports
    - Dependency preservation
    """
    print("\nTesting AST-based Chunking Strategy...")
    print("-" * 50)

    # Test case 1: Python class with methods
    print("\n1. Python Class Definition")
    class_code = '''
class DataProcessor:
    """A class to process data with various methods."""
    
    def __init__(self, data):
        self.data = data
        self.processed = None
        
    def process(self):
        """Process the data."""
        if not self.data:
            raise ValueError("No data to process")
        self.processed = [x * 2 for x in self.data]
        return self.processed
        
    def validate(self):
        """Validate the data."""
        if not isinstance(self.data, list):
            raise TypeError("Data must be a list")
        if not all(isinstance(x, (int, float)) for x in self.data):
            raise TypeError("All elements must be numbers")
        
    def get_stats(self):
        """Get statistics about the data."""
        if not self.processed:
            self.process()
        return {
            'mean': sum(self.processed) / len(self.processed),
            'max': max(self.processed),
            'min': min(self.processed)
        }
'''
    chunks = chunk_text(class_code)
    print("Python class chunking:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(chunk)
        print("-" * 30)

    # Test case 2: Multiple functions with imports
    print("\n2. Multiple Functions with Dependencies")
    functions_code = '''
import numpy as np
from typing import List, Dict, Optional
import pandas as pd

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the input data."""
    # Remove missing values
    data = data.dropna()
    
    # Normalize numeric columns
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_cols] = (data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std()
    
    return data

def train_model(X: np.ndarray, y: np.ndarray) -> Dict:
    """Train a simple model."""
    # Calculate weights using numpy
    weights = np.linalg.inv(X.T @ X) @ X.T @ y
    
    # Calculate predictions and error
    y_pred = X @ weights
    mse = np.mean((y - y_pred) ** 2)
    
    return {
        'weights': weights,
        'mse': mse
    }

def predict(X: np.ndarray, model: Dict) -> np.ndarray:
    """Make predictions using the trained model."""
    return X @ model['weights']
'''
    chunks = chunk_text(functions_code)
    print("\nMultiple functions chunking:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(chunk)
        print("-" * 30)


def test_mixed_content():
    """
    Test chunking with mixed markdown and code content.
    Tests how the system handles:
    - Markdown text with embedded code blocks
    - Python code block detection and processing
    - Context preservation in mixed content
    """
    print("\nTesting Mixed Content Strategy...")
    print("-" * 50)

    mixed_content = '''# Python Data Processing Guide

This guide explains how to process data using Python.

## Basic Data Processing

Here's a simple example of data processing:

```python
import pandas as pd
import numpy as np

def process_data(df):
    # Remove missing values
    df = df.dropna()
    
    # Normalize numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    
    return df
```

## Advanced Processing

For more advanced processing, you might want to use custom classes:

```python
class DataProcessor:
    def __init__(self, data):
        self.data = data
        
    def process(self):
        return self.data * 2
```

Remember to always validate your data before processing!
'''
    chunks = chunk_text(mixed_content)
    print("Mixed content chunking:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(chunk)
        print("-" * 30)


if __name__ == "__main__":
    # Run all tests in sequence
    test_general_chunking()  # Test basic chunking strategy
    test_ast_chunking()      # Test Python-specific AST chunking
    test_mixed_content()     # Test mixed content handling
