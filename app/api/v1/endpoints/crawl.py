from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from app.db.session import get_db
from app.schemas.crawl_models import CrawlRequest, CrawlResponse, CrawlStatus
from app.core.crawl_url import crawl_parallel, get_sitemap_urls
from app.models.database import Document
from app.config import settings

router = APIRouter()

# Store crawl tasks status
crawl_tasks = {}

async def _run_crawl(task_id: str, sitemap_url: str, max_concurrent: int, db: Session):
    try:
        urls = await get_sitemap_urls(sitemap_url)
        if not urls:
            crawl_tasks[task_id] = {
                "status": "failed",
                "message": "No URLs found in sitemap",
                "completed_at": datetime.now(datetime.timezone.utc)
            }
            return

        stats = await crawl_parallel(urls, max_concurrent=max_concurrent)
        
        crawl_tasks[task_id] = {
            "status": "completed",
            "message": f"Successfully crawled {stats['success_count']} URLs",
            "stats": stats,
            "completed_at": datetime.now(datetime.timezone.utc)
        }
    except Exception as e:
        crawl_tasks[task_id] = {
            "status": "failed",
            "message": str(e),
            "completed_at": datetime.now(datetime.timezone.utc)
        }

@router.post("/crawl", response_model=CrawlResponse)
async def crawl_urls(
    request: CrawlRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Start a crawling task for the given sitemap URL.
    The crawling will run in the background.
    """
    task_id = f"crawl_{datetime.now(datetime.timezone.utc).timestamp()}"
    max_concurrent = request.max_concurrent or settings.MAX_CONCURRENT_CRAWLS

    # Initialize task status
    crawl_tasks[task_id] = {
        "status": "running",
        "message": "Crawling started",
        "started_at": datetime.now(datetime.timezone.utc)
    }

    # Start crawling in background
    background_tasks.add_task(
        _run_crawl,
        task_id,
        str(request.sitemap_url),
        max_concurrent,
        db
    )

    return CrawlResponse(
        status="accepted",
        message="Crawling task started",
        task_id=task_id
    )

@router.get("/crawl/{task_id}", response_model=CrawlStatus)
async def get_crawl_status(task_id: str):
    """
    Get the status of a crawling task
    """
    if task_id not in crawl_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return CrawlStatus(**crawl_tasks[task_id])

@router.get("/documents", response_model=List[str])
async def list_crawled_urls(db: Session = Depends(get_db)):
    """
    List all crawled URLs in the database
    """
    documents = db.query(Document.url).all()
    return [doc.url for doc in documents]
