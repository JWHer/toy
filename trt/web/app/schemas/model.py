from urllib.parse import urlparse
import mlflow
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List, Dict, Union
from uuid import UUID

from app.core.config import settings

class ModelBase(BaseModel):
    name: str
    classes: List[Union[str, Dict]]
    desc: str
    tags: List[str]
    location: Optional[str]
    capacity: Optional[int]
    version: Optional[str] = ''
    platform: Optional[str] = ''
    framework: Optional[str] = ''
    precision: Optional[str] = ''
    
class ModelCreateUser(ModelBase):
    production_id: Optional[UUID]
    base_id: UUID                                 

class ModelCreate(ModelBase):
    id: UUID
    production_id: UUID            
    status: str

class ModelUpdate(ModelBase):
    name: Optional[str]
    classes: Optional[List[Union[str, Dict]]]
    desc: Optional[str]
    status: Optional[str]

class ModelDB(ModelBase):
    id: UUID
    production_id: Optional[UUID]
    status: str
    
    updated: datetime
    created: datetime

    @property
    def model_cfg(self) -> str:
        return self.model_dir+"model.json"
    
    @property
    def model_ckpt(self) -> str:
        return self.model_dir+"model.pth"

    @property
    def model_dir(self) -> str:
        if 'base' in self.tags:
            return f"{urlparse(self.location).path}/"        
        return f"{settings.train}/{self.exp_id}/{str(self.id).replace('-', '')}/artifacts/model/data/"

    @property
    def exp_id(self) -> str:
        if 'base' in self.tags:
            raise ValueError(f"Model[{self.id}] has no exp_id (base model)")
        try:
            run = mlflow.get_run(self.id)
            exp_id = run.info.experiment_id
        except:
            exp_id = "0"
        return exp_id
    
    class Config:
        orm_mode = True
