
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from abc import ABC, ABCMeta, abstractmethod
from app.core.config import settings

engine = create_engine(
   settings.SQLALCHEMY_DATABASE_URI, 
   pool_pre_ping=True,
   future=True,
   connect_args={"check_same_thread": False},
   native_datetime=True
)

SessionLocal = sessionmaker(
   autocommit=False, autoflush=False, bind=engine,
   future=True
)
