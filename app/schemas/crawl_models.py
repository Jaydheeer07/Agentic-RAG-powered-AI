from enum import Enum
from pydantic import BaseModel, HttpUrl
from typing import Dict, Any, Optional
from datetime import datetime

class CrawlStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    RUNNING = "running"

class CrawlRequest(BaseModel):
    url: HttpUrl
    max_concurrent: Optional[int] = None

class CrawlResponse(BaseModel):
    status: CrawlStatus
    message: str
    stats: Optional[Dict[str, Any]] = None
    document_id: Optional[int] = None
