from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import settings
from app.core.crawl_url import crawl_parallel, get_sitemap_urls
from app.core.chunk_pipeline import DocumentProcessor
from app.core.db_manager import DatabaseManager
from app.db.session import get_db, get_db_manager
from app.models.database import Document
from app.schemas.crawl_models import CrawlRequest, CrawlResponse, CrawlStatus

router = APIRouter()


@router.post("/crawl", response_model=CrawlResponse)
async def crawl_urls(
    urls: List[str],
    max_concurrent: int = 5,
    db: Session = Depends(get_db)
) -> CrawlResponse:
    """
    Crawl multiple URLs or sitemaps in parallel and store their content in the database.
    If a URL contains 'sitemap.xml', it will be treated as a sitemap and all URLs from it will be crawled.
    """
    try:
        all_urls = []
        
        # Process each URL - if it's a sitemap, get all URLs from it
        for url in urls:
            if "sitemap.xml" in url.lower():
                sitemap_urls = await get_sitemap_urls(url)
                if sitemap_urls:
                    all_urls.extend(sitemap_urls)
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to fetch URLs from sitemap: {url}"
                    )
            else:
                all_urls.append(url)
        
        if not all_urls:
            raise HTTPException(
                status_code=400,
                detail="No valid URLs found to crawl"
            )
        
        # Crawl all collected URLs
        stats = await crawl_parallel(all_urls, max_concurrent=max_concurrent)
        
        # Get the last crawled document from database if it was a single URL
        document_id = None
        if len(urls) == 1 and len(all_urls) == 1:
            document = db.query(Document).filter(
                Document.url == urls[0]
            ).order_by(Document.created_at.desc()).first()
            document_id = document.id if document else None
        
        return CrawlResponse(
            status=CrawlStatus.SUCCESS if stats["success_count"] > 0 else CrawlStatus.FAILED,
            message=f"Crawled {stats['success_count']} URLs successfully, {stats['fail_count']} failed",
            stats=stats,
            document_id=document_id
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error crawling URLs: {str(e)}"
        )

@router.post("/process-pending", response_model=CrawlResponse)
async def process_pending_documents(
    background_tasks: BackgroundTasks,
    db_manager: DatabaseManager = Depends(get_db_manager)
) -> CrawlResponse:
    """
    Process pending documents in the background.
    This endpoint triggers the chunking process for documents that were crawled but not yet processed.
    """
    try:
        processor = DocumentProcessor(db_manager)
        background_tasks.add_task(processor.process_pending_documents)
        
        return CrawlResponse(
            status=CrawlStatus.SUCCESS,
            message="Processing of pending documents started",
            stats={"status": "processing"}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error starting document processing: {str(e)}"
        )

@router.get("/document/{document_id}/status")
async def get_document_status(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    Get the processing status of a document.
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )
    
    return {
        "id": document.id,
        "status": document.status,
        "url": document.url,
        "title": document.title,
        "chunk_count": len(document.chunks) if document.chunks else 0,
        "crawled_at": document.crawled_at
    }
