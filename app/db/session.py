from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config import settings

# Create engine
engine = create_engine(settings.DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_manager():
    """Get database manager instance"""
    from app.core.db_manager import DatabaseManager
    return DatabaseManager(settings.DATABASE_URL)

# Create all tables
def init_db():
    """Initialize database tables"""
    from app.models.database import Base
    Base.metadata.create_all(bind=engine)
