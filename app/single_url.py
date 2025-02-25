import asyncio
import logging
import os
import sys
from crawl4ai import AsyncWebCrawler

# Force UTF-8 encoding for stdout
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7 doesn't have reconfigure
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

# Set environment variables
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["NO_COLOR"] = "1"  # Disable colored output
os.environ["TERM"] = "xterm-256color"  # Set terminal type for Git Bash

# Set up logging with a file handler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawler.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

async def main():
    try:
        async with AsyncWebCrawler() as crawler:
            logger.info("Starting to crawl...")
            result = await crawler.arun(
                url="https://ai.pydantic.dev/",
                max_pages=1  # Limit to single page for testing
            )
            if result and result.markdown:
                # Save the crawled content to a file
                with open('crawled_content.md', 'w', encoding='utf-8') as f:
                    f.write(result.markdown)
                logger.info("Content saved to crawled_content.md")
            else:
                logger.warning("No content was retrieved")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Crawling interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.info("Check crawler.log for details")