import os
from pwd import getpwuid
from typing import Any, Optional
from pydantic import BaseModel
from datetime import datetime

class MinioObject(BaseModel):
    bucket_name: str
    object_name: str
    last_modified: Optional[datetime]
    etag: Optional[Any]
    size: Optional[Any]
    metadata: Optional[Any]
    version_id: Optional[Any]
    is_latest: Optional[Any]
    storage_class: Optional[Any]
    owner_id: Optional[Any]
    owner_name: Optional[Any]
    content_type: Optional[Any]
    is_delete_marker: Optional[bool]

class Storage(BaseModel):
    dir_name: str
    file_name: Optional[str]
    type: str
    last_modified: Optional[datetime]
    size: Optional[int]
    owner: Optional[Any]

    @property
    def path(self)->str:
        return os.path.join(self.dir_name, self.file_name)

    @classmethod
    def fromFileSystem(cls, abs_path:str):
        return cls(
            dir_name = os.path.dirname(abs_path),
            file_name = os.path.basename(abs_path) if os.path.isfile(abs_path) else None,
            type = 'file' if os.path.isfile(abs_path) else 'directory',
            last_modified = os.path.getmtime(abs_path),
            size = os.path.getsize(abs_path),
            owner = getpwuid(os.stat(abs_path).st_uid).pw_name
        )

    @classmethod
    def fromMinioObject(cls, minio:MinioObject):
        return cls(
            dir_name = os.path.dirname(minio.object_name),
            file_name = os.path.basename(minio.object_name) if hasattr(minio, 'size') and minio.size else None,
            type = 'file' if hasattr(minio, 'size') and minio.size else 'directory',
            last_modified = minio.last_modified,
            size = minio.size if hasattr(minio, 'size') and minio.size else None,
            owner = minio.owner_name if hasattr(minio, 'owner_name') else None
        )

class StorageUpdate(BaseModel):
    uri: str
    owner: Optional[Any]