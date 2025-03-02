from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import settings
from app.core.crawl_url import crawl_parallel, get_sitemap_urls
from app.db.session import get_db
from app.models.database import Document
from app.schemas.crawl_models import CrawlRequest, CrawlResponse, CrawlStatus

router = APIRouter()

