from sqlalchemy import Column, ForeignKey, String, Integer, JSON, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.db.base_class import Base
from app.utils.guid_type import GUID

class Model(Base):
    __tablename__ = 'model'

    id = Column(GUID, primary_key=True, index=True)
    production_id = Column(GUID)

    name = Column(String)
    classes = Column(JSON)
    desc = Column(String)
    tags = Column(JSON)
    location = Column(String)
    status = Column(String)
    capacity = Column(Integer)
    
    version = Column(String)
    platform = Column(String)
    framework = Column(String)
    precision = Column(String)
    
    updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created = Column(DateTime(timezone=True), server_default=func.now())