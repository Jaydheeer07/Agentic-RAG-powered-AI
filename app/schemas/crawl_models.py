from pydantic import BaseModel, HttpUrl
from typing import Dict, Any, Optional
from datetime import datetime

class CrawlRequest(BaseModel):
    sitemap_url: HttpUrl
    max_concurrent: Optional[int] = None

class CrawlResponse(BaseModel):
    status: str
    message: str
    task_id: str

class CrawlStatus(BaseModel):
    status: str  # "running", "completed", or "failed"
    message: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    stats: Optional[Dict[str, Any]] = None
