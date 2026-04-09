"""
database.py — PostgreSQL Database Connection
=============================================
Sets up the SQLAlchemy engine and session factory.
All other modules import 'SessionLocal' and 'Base' from here.
Configuration is read from the DATABASE_URL environment variable.
Set it in your .env file:
    DATABASE_URL=postgresql://nids:password@localhost:5432/nids_db
For quick local testing without PostgreSQL, you can use SQLite:
    DATABASE_URL=sqlite:///./nids.db
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./nids.db"          
)
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,            
)
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)
Base = declarative_base()
def get_db():
    """
    FastAPI dependency that provides a DB session to each request.
    Automatically closes the session when the request is done.
    Usage in a route:
        @router.get("/example")
        def example(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
