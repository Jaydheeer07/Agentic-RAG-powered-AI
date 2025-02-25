from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.endpoints import crawl
from app.db.session import init_db

# Initialize FastAPI app
app = FastAPI(
    title="Agentic RAG API",
    description="API for RAG-powered document search and chat",
    version="1.0.0",
)

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(crawl.router, prefix="/api/v1", tags=["crawl"])


@app.on_event("startup")
async def startup_event():
    """Initialize database and other startup tasks"""
    init_db()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "Agentic RAG API is running",
        "docs_url": "/docs",
        "openapi_url": "/openapi.json",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
