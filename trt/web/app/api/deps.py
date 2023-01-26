from app.db.session import SessionLocal
from contextlib import contextmanager

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
@contextmanager
def get_conn():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()