import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List
from xml.etree import ElementTree

import aiohttp
import psutil
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

from app.db.session import SessionLocal
from app.models.database import Document

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
    logger.info(
        f"Starting parallel crawl of {len(urls)} URLs with max_concurrent={max_concurrent}"
    )
    memory_tracker = MemoryTracker()

    # Configure browser for optimal performance
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=[
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--no-sandbox",
            "--disable-extensions",
            "--disable-notifications",
        ],
    )

    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Initialize statistics
    stats = {"success_count": 0, "fail_count": 0, "errors": {}}

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        for batch_num, i in enumerate(range(0, len(urls), max_concurrent), 1):
            batch = urls[i : i + max_concurrent]
            tasks = []

            for j, url in enumerate(batch):
                session_id = f"session_{batch_num}_{j}"
                task = crawler.arun(url=url, config=crawl_config, session_id=session_id)
                tasks.append(task)

            # Log memory before batch
            mem_stats = memory_tracker.log_memory(
                f"Batch {batch_num}/{(len(urls) + max_concurrent - 1) // max_concurrent}"
            )

            # Process batch results
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Save results and track errors
            for url, result in zip(batch, results):
                if isinstance(result, Exception):
                    error_type = type(result).__name__
                    stats["errors"][error_type] = stats["errors"].get(error_type, 0) + 1
                    stats["fail_count"] += 1
                    logger.error(f"Error crawling {url}: {result}")
                    continue

                if not result or not result.success:
                    logger.warning(f"Failed to crawl {url}: No content retrieved")
                    stats["fail_count"] += 1
                    continue

                try:
                    db = SessionLocal()
                    # Store the document with status "pending"
                    document = Document(
                        url=url,
                        title="Untitled",  # Using URL as title since CrawlResult doesn't provide a title attribute
                        content=result.markdown or result.html or "",
                        status="pending",  # Mark as pending for later chunking
                    )
                    db.add(document)
                    db.commit()
                    db.refresh(document)
                    stats["success_count"] += 1
                finally:
                    db.close()

            # Log progress
            total = stats["success_count"] + stats["fail_count"]
            logger.info(
                f"Progress: {total}/{len(urls)} URLs processed "
                f"({stats['success_count']} successful, {stats['fail_count']} failed)"
            )

    finally:
        logger.info("Closing crawler...")
        await crawler.close()

        # Final memory stats
        final_stats = memory_tracker.log_memory("Final stats")

    return stats


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
