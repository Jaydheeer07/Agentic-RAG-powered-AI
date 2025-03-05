import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from xml.etree import ElementTree

import aiohttp
import psutil
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

# Setup paths
__location__ = Path(__file__).parent.absolute()
__output__ = __location__ / "output"
__output__.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(__output__ / "crawler.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Set environment variables for better compatibility
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "xterm-256color"


class MemoryTracker:
    def __init__(self):
        self.peak_memory = 0
        self.process = psutil.Process(os.getpid())
        self.start_time = datetime.now()

    def log_memory(self, prefix: str = "") -> Dict[str, Any]:
        current_mem = self.process.memory_info().rss
        if current_mem > self.peak_memory:
            self.peak_memory = current_mem

        stats = {
            "current_mb": current_mem // (1024 * 1024),
            "peak_mb": self.peak_memory // (1024 * 1024),
            "cpu_percent": self.process.cpu_percent(),
            "elapsed_time": str(datetime.now() - self.start_time),
        }

        logger.info(
            f"{prefix} Memory: {stats['current_mb']}MB (Peak: {stats['peak_mb']}MB) "
            f"CPU: {stats['cpu_percent']}% Time: {stats['elapsed_time']}"
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
    stats = {"success_count": 0, "fail_count": 0, "errors": {}, "memory_stats": []}

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
            stats["memory_stats"].append(mem_stats)

            # Process batch results
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Save results and track errors
            for url, result in zip(batch, results):
                if isinstance(result, Exception):
                    error_type = type(result).__name__
                    stats["errors"][error_type] = stats["errors"].get(error_type, 0) + 1
                    stats["fail_count"] += 1
                    logger.error(f"Error crawling {url}: {result}")
                elif result.success:
                    stats["success_count"] += 1
                    # Save content to file
                    if result.markdown:
                        output_file = __output__ / f"page_{stats['success_count']}.md"
                        output_file.write_text(result.markdown, encoding="utf-8")
                else:
                    stats["fail_count"] += 1
                    logger.warning(f"Failed to crawl {url}: No content retrieved")

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
        stats["memory_stats"].append(final_stats)

        # Save summary report
        summary = (
            f"Crawling Summary\n"
            f"================\n"
            f"Total URLs: {len(urls)}\n"
            f"Successful: {stats['success_count']}\n"
            f"Failed: {stats['fail_count']}\n"
            f"Error types:\n"
            + "\n".join(f"- {k}: {v}" for k, v in stats["errors"].items())
            + f"\n\nPeak memory usage: {final_stats['peak_mb']}MB\n"
            f"Total time: {final_stats['elapsed_time']}\n"
        )
        (__output__ / "summary.txt").write_text(summary, encoding="utf-8")

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


async def main():
    """
    Main execution function with improved error handling and logging.
    """
    try:
        urls = await get_sitemap_urls("https://htmlburger.com/sitemap.xml")
        if not urls:
            logger.error("No URLs found to crawl")
            return

        logger.info(f"Starting crawl of {len(urls)} URLs")
        stats = await crawl_parallel(urls, max_concurrent=5)

        logger.info(
            f"Crawl completed. Success: {stats['success_count']}, "
            f"Failed: {stats['fail_count']}"
        )
        logger.info(f"Results saved in: {__output__}")

    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
