import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List
from xml.etree import ElementTree

import aiohttp
import psutil
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

from app.config import settings
from app.core.chunking import chunk_text
from app.db.session import SessionLocal
from app.models.database import Chunk, Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Set environment variables for better compatibility
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "xterm-256color"


class MemoryTracker:
    """Track memory usage during crawling"""

    def __init__(self):
        self.peak_memory = 0
        self.process = psutil.Process(os.getpid())
        self.start_time = datetime.now()

    def log_memory(self, prefix: str = "") -> Dict[str, Any]:
        """Log current memory usage and return stats"""
        current_mem = self.process.memory_info().rss
        if current_mem > self.peak_memory:
            self.peak_memory = current_mem

        duration = datetime.now() - self.start_time
        memory_mb = current_mem / (1024 * 1024)
        peak_memory_mb = self.peak_memory / (1024 * 1024)

        stats = {
            "current_memory_mb": round(memory_mb, 2),
            "peak_memory_mb": round(peak_memory_mb, 2),
            "duration_seconds": duration.total_seconds(),
        }

        if prefix:
            logger.info(
                f"{prefix} - Memory: {memory_mb:.2f}MB, Peak: {peak_memory_mb:.2f}MB"
            )

        return stats


async def crawl_parallel(urls: List[str], max_concurrent: int = 3) -> Dict[str, Any]:
    """
    Crawl multiple URLs in parallel with memory monitoring and error handling.

    Args:
        urls: List of URLs to crawl
        max_concurrent: Maximum number of concurrent crawling tasks

    Returns:
        Dict containing crawling statistics and results
    """
    memory_tracker = MemoryTracker()
    success_count = 0
    fail_count = 0

    try:
        # Configure crawler
        browser_config = BrowserConfig(
            cache_mode=CacheMode.MEMORY,
            headless=True,
            ignore_https_errors=True,
            timeout=30000,
        )

        crawler = AsyncWebCrawler(
            run_config=CrawlerRunConfig(
                max_concurrent=max_concurrent,
                browser_config=browser_config,
            )
        )

        async def process_url(url: str) -> bool:
            """Process a single URL and store results in database"""
            nonlocal success_count, fail_count

            try:
                # Crawl the URL
                result = await crawler.crawl_url(url)
                if not result or not result.text:
                    logger.warning(f"No content found for {url}")
                    fail_count += 1
                    return False

                # Create document in database
                db = SessionLocal()
                try:
                    document = Document(
                        url=url,
                        title=result.title or "Untitled",
                        content=result.text,
                        crawled_at=datetime.now(datetime.timezone.utc),
                    )
                    db.add(document)
                    db.commit()
                    db.refresh(document)

                    # Split content into chunks
                    chunks = chunk_text(
                        result.text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP
                    )

                    # Store chunks
                    for i, chunk_contents in enumerate(chunks):
                        chunk = Chunk(
                            document_id=document.id,
                            content=chunk_contents,
                            sequence=i,
                        )
                        db.add(chunk)

                    db.commit()
                    success_count += 1
                    return True

                finally:
                    db.close()

            except Exception as e:
                logger.error(f"Error processing {url}: {str(e)}")
                fail_count += 1
                return False

        # Process URLs in parallel
        tasks = [process_url(url) for url in urls]
        await asyncio.gather(*tasks)

        # Log final stats
        stats = memory_tracker.log_memory("Crawl completed")
        stats.update(
            {
                "success_count": success_count,
                "fail_count": fail_count,
                "total_urls": len(urls),
            }
        )

        return stats

    except Exception as e:
        logger.error(f"Fatal error in crawl_parallel: {str(e)}")
        raise


async def get_sitemap_urls(sitemap_url: str) -> List[str]:
    """
    Asynchronously fetch URLs from a sitemap with error handling and validation.
    """
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(sitemap_url) as response:
                response.raise_for_status()
                content = await response.text()

                # Parse the XML
                root = ElementTree.fromstring(content)
                namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
                urls = [loc.text for loc in root.findall(".//ns:loc", namespace)]

                logger.info(f"Found {len(urls)} URLs in sitemap")
                return urls

        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching sitemap: {e}")
        except ElementTree.ParseError as e:
            logger.error(f"XML parsing error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")

        return []
