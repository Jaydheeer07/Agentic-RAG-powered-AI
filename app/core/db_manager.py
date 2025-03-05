"""Database connection and session management."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool

class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.engine = create_engine(
            db_url,
            poolclass=NullPool,  # Disable connection pooling for better async compatibility
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def create_tables(self):
        """Create database tables if they don't exist."""
        from app.models.database import Base
        Base.metadata.create_all(self.engine)
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
