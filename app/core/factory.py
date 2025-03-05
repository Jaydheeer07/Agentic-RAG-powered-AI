from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.db.session import init_db, engine
from app.api.v1.endpoints import crawl
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    - Startup: Initialize database tables
    - Shutdown: Close database connections
    """
    # Startup
    try:
        logger.info("Starting up...")
        logger.info("Initializing database...")
        init_db()
        logger.info("Database initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise
    
    # Shutdown
    finally:
        logger.info("Shutting down...")
        try:
            engine.dispose()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")

def create_app() -> FastAPI:
    """
    Factory function to create and configure the FastAPI application
    """
    app = FastAPI(
        title=settings.APP_TITLE,
        description=settings.APP_DESCRIPTION,
        version=settings.APP_VERSION,
        lifespan=lifespan
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
    )

    # Include routers
    app.include_router(crawl.router, prefix="/api/v1", tags=["crawl"])

    return app
